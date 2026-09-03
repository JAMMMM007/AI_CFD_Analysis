"""The closed 2-D contour that bounds the solid body the flow is solved around.

Every geometry source -- NACA generator, circle, square, DXF import -- produces a
:class:`Contour`, and the mesher consumes nothing else. That keeps the mesher
independent of where the shape came from.
"""

from __future__ import annotations

import numpy as np

# Consecutive points closer than this fraction of the bounding-box diagonal are
# treated as duplicates. Duplicates give zero-length segments, which produce NaN
# tangents and degenerate mesh cells.
_DUPLICATE_FRAC = 1e-10


class ContourError(ValueError):
    """Raised when a point set cannot form a valid solver boundary."""


class Contour:
    """An ordered, closed, non-self-intersecting loop of points in the xy-plane.

    The closing segment is implicit: ``points[-1]`` joins back to ``points[0]``,
    so the first point is never repeated at the end.

    Points are stored **counter-clockwise** (positive signed area). That is the
    standard mathematical orientation and what shapely treats as positive, so it
    is the least surprising representation for a lone polygon.

    The mesher, however, marches outward from the wall, and for the resulting
    (i, j) curvilinear system to be right-handed the wall must be traversed with
    the *fluid* on its left -- clockwise around the solid. :meth:`as_wall_line`
    returns that ordering, and is deliberately the only place the flip happens.
    """

    def __init__(self, points, *, name: str = "body", reference_length: float | None = None):
        xy = np.asarray(points, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ContourError(f"points must have shape (N, 2), got {xy.shape}")
        if len(xy) < 3:
            raise ContourError(f"a closed contour needs at least 3 points, got {len(xy)}")
        if not np.all(np.isfinite(xy)):
            raise ContourError("points contain NaN or infinity")

        xy = _drop_duplicates(xy)
        if len(xy) < 3:
            raise ContourError("fewer than 3 distinct points remain after removing duplicates")

        # Canonical orientation: counter-clockwise.
        if _signed_area(xy) < 0.0:
            xy = xy[::-1].copy()

        self._xy = xy
        self.name = name
        self._reference_length = reference_length

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def points(self) -> np.ndarray:
        """(N, 2) loop points, counter-clockwise, with no repeated closing point."""
        return self._xy

    @property
    def x(self) -> np.ndarray:
        return self._xy[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self._xy[:, 1]

    def __len__(self) -> int:
        return len(self._xy)

    def __repr__(self) -> str:
        return f"Contour({self.name!r}, n={len(self)}, L_ref={self.reference_length:.6g})"

    @property
    def signed_area(self) -> float:
        """Enclosed area; positive by construction, since points are counter-clockwise."""
        return _signed_area(self._xy)

    @property
    def area(self) -> float:
        return abs(self.signed_area)

    @property
    def perimeter(self) -> float:
        return float(np.sum(self.segment_lengths()))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)."""
        lo = self._xy.min(axis=0)
        hi = self._xy.max(axis=0)
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    @property
    def centroid(self) -> np.ndarray:
        """Area centroid of the enclosed region (not the mean of the points)."""
        p = self._xy
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        return np.sum((p + q) * cross[:, None], axis=0) / (6.0 * self.signed_area)

    @property
    def reference_length(self) -> float:
        """Length scale used for the Reynolds number and the force coefficients.

        Set by whichever generator built the contour -- chord for an aerofoil,
        diameter for a circle, side for a square. Falls back to the bounding-box
        x-extent, the conventional choice for an imported profile.
        """
        if self._reference_length is not None:
            return self._reference_length
        xmin, _, xmax, _ = self.bounds
        return xmax - xmin

    @reference_length.setter
    def reference_length(self, value: float) -> None:
        if not value > 0.0:
            raise ContourError(f"reference_length must be positive, got {value}")
        self._reference_length = float(value)

    # ------------------------------------------------------------------
    # Differential geometry (everything is periodic in the loop index)
    # ------------------------------------------------------------------

    def segment_lengths(self) -> np.ndarray:
        """(N,) length of the segment leaving each point, wrapping at the end."""
        d = np.roll(self._xy, -1, axis=0) - self._xy
        return np.hypot(d[:, 0], d[:, 1])

    def arclength(self) -> np.ndarray:
        """(N,) cumulative arclength at each point, starting from zero."""
        return np.concatenate(([0.0], np.cumsum(self.segment_lengths())[:-1]))

    def tangents(self) -> np.ndarray:
        """(N, 2) unit tangents, central-differenced with periodic wrap.

        Central differencing keeps the tangent second-order accurate along smooth
        stretches. At a genuine corner it bisects the two edges, which is the
        sensible average direction there.
        """
        t = np.roll(self._xy, -1, axis=0) - np.roll(self._xy, 1, axis=0)
        return _normalise_rows(t)

    def outward_normals(self) -> np.ndarray:
        """(N, 2) unit normals pointing out of the solid, into the fluid.

        For a counter-clockwise loop the outward normal is the tangent rotated by
        -90 degrees, ``n = (t_y, -t_x)``. Check it on a circle traversed as
        ``(cos, sin)``: the tangent is ``(-sin, cos)``, and the rule returns
        ``(cos, sin)`` -- the outward radial direction.
        """
        t = self.tangents()
        return np.column_stack((t[:, 1], -t[:, 0]))

    def curvature(self) -> np.ndarray:
        """(N,) signed discrete curvature from the circumcircle of each point triple.

        ``kappa = 2 * cross(b - a, c - b) / (|b-a| |c-b| |c-a|)``, which is ``1/R``
        for the circle through the three points. Positive where the contour turns
        counter-clockwise, i.e. where it is locally convex given our orientation.
        Collinear triples return exactly zero instead of dividing by zero.
        """
        return _circumcircle_curvature(self._xy, stride=1)

    def turning_angles(self) -> np.ndarray:
        """(N,) absolute angle in radians between the incoming and outgoing edge.

        Zero along a straight run, pi/2 at a square corner. Used to find the
        vertices that a resampling pass must not round off.
        """
        incoming = _normalise_rows(self._xy - np.roll(self._xy, 1, axis=0))
        outgoing = _normalise_rows(np.roll(self._xy, -1, axis=0) - self._xy)
        dot = np.clip(np.sum(incoming * outgoing, axis=1), -1.0, 1.0)
        return np.arccos(dot)

    # ------------------------------------------------------------------
    # Transforms -- each returns a new Contour, leaving this one untouched
    # ------------------------------------------------------------------

    def translated(self, dx: float, dy: float) -> "Contour":
        return self._derive(self._xy + np.array([dx, dy]))

    def scaled(self, factor: float, *, about: np.ndarray | None = None) -> "Contour":
        if not factor > 0.0:
            raise ContourError(f"scale factor must be positive, got {factor}")
        origin = np.zeros(2) if about is None else np.asarray(about, dtype=float)
        ref = None if self._reference_length is None else self._reference_length * factor
        return Contour(
            origin + (self._xy - origin) * factor,
            name=self.name,
            reference_length=ref,
        )

    def rotated(self, angle_deg: float, *, about: np.ndarray | None = None) -> "Contour":
        """Rotate counter-clockwise by ``angle_deg`` about ``about`` (default origin).

        Angle of attack is applied by rotating the *body* by ``-alpha`` rather than
        tilting the freestream, so the circular far-field boundary stays aligned
        with the mesh and the plots stay upright.
        """
        origin = np.zeros(2) if about is None else np.asarray(about, dtype=float)
        c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
        rot = np.array([[c, -s], [s, c]])
        return self._derive((self._xy - origin) @ rot.T + origin)

    def normalised(self, target_length: float = 1.0) -> "Contour":
        """Scale so :attr:`reference_length` becomes ``target_length``, then centre it."""
        scaled = self.scaled(target_length / self.reference_length)
        xmin, ymin, _, ymax = scaled.bounds
        return scaled.translated(-xmin, -0.5 * (ymin + ymax))

    def _derive(self, xy: np.ndarray) -> "Contour":
        """New contour with the same name and reference length (rigid motions only)."""
        return Contour(xy, name=self.name, reference_length=self._reference_length)

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(
        self,
        n: int,
        *,
        max_turn_deg: float = 4.0,
        max_refinement: float = 25.0,
        corner_angle_deg: float = 25.0,
        max_spacing_ratio: float = 1.2,
        smoothing_passes: int = 6,
        min_spacing: float | None = None,
    ) -> "Contour":
        """Redistribute to ``n`` points, clustering them where the contour curves.

        This is the entry point for CAD geometry, whose point distribution was
        chosen by whoever drew it rather than by what the flow needs. The analytic
        generators already emit a well-graded distribution, so there is no reason
        to run them through here.

        The method builds a *sizing field* -- a target wall spacing ``h(s)`` -- and
        then places points at equal increments of ``integral(ds / h)``. The rule for
        ``h`` is geometric: no cell may turn through more than ``max_turn_deg``, so

            h(s) = clamp( max_turn / kappa(s),  h_uniform / max_refinement,  h_uniform )

        Only the *shape* of ``h`` matters, since the integral is renormalised to
        land exactly ``n`` points; the clamps set the dynamic range.

        Corners are handled by splitting rather than snapping. The loop is cut at
        every detected corner and each piece receives a share of the points
        proportional to its share of the integral, with a point landing exactly on
        each corner by construction. Snapping the nearest point onto a corner
        instead -- the obvious approach -- leaves a squashed cell beside it.

        Straight-line interpolation is used rather than a spline, because a spline
        would round off the genuine corners this is trying to preserve.

        Parameters
        ----------
        n
            Number of points in the result.
        max_turn_deg
            Surface turning permitted across one cell. Smaller values cluster
            harder into curved regions.
        max_refinement
            Cap on how much finer than uniform the spacing may become. Prevents a
            near-singular curvature spike from swallowing the whole point budget.
        corner_angle_deg
            Turning angle above which a vertex counts as a hard corner and is
            reproduced exactly.
        max_spacing_ratio
            Cap on the size ratio between neighbouring wall cells. Abrupt jumps in
            wall spacing propagate outward through the mesh and inflate the
            finite-volume truncation error. The cap binds on the continuous sizing
            field, so the ratio actually realised between two cells lands a little
            above it -- typically within 10% -- because a cell's length samples
            the field over its own width and because each corner-to-corner
            segment receives a whole number of points. Treat it as a smoothness
            dial, not a guarantee: 1.2 here yields realised ratios around 1.3,
            comfortably inside what counts as a well-graded mesh.
        smoothing_passes
            Diffusion passes applied to the sizing field, in log space, before it
            is integrated.
        min_spacing
            Floor on the wall spacing, normally the wall-normal first-layer
            thickness. Curvature clustering below this buys nothing and costs a
            great deal.

            Tangential and wall-normal spacing are chosen independently -- one
            from the point budget, the other from a ``y+`` target -- and nothing
            otherwise stops them contradicting each other. Cluster the surface
            below the first layer thickness and the wall cells come out taller
            than they are wide, which is where the hyperbolic marcher fails:
            measured on a NACA 2412 trailing edge, where the outward normal turns
            through 42 degrees between adjacent points, the march completes
            whenever the first layer is no larger than the tightest spacing and
            collapses once it is several times larger.

                first layer / tightest spacing      layers marched (of 30)
                                    0.97                        30
                                    2.19                        29
                                    5.80                         2
                                   12.58                         3

            The same request -- 240 points at ``y+`` 100 -- goes from an unusable
            mesh to a complete one on 120 points, purely because the trailing-edge
            cluster thins out. So the floor is not a tuning parameter; it is the
            missing constraint between two settings that were never coupled.
        """
        corners_s = self._corner_arclengths(corner_angle_deg)
        if n < max(3, len(corners_s)):
            raise ContourError(
                f"resample needs at least {max(3, len(corners_s))} points "
                f"({len(corners_s)} corners must each keep one), got {n}"
            )

        # Shift the arclength origin onto the first corner so every segment lies
        # inside [0, perimeter] and none of them has to wrap around the seam.
        offset = corners_s[0] if len(corners_s) else 0.0
        n_dense = max(20 * n, 2000)
        u_dense = np.linspace(0.0, self.perimeter, n_dense, endpoint=False)
        dense = self._interpolate_at(offset + u_dense)

        if min_spacing is not None and min_spacing > 0.0:
            # Expressed through the refinement cap the sizing field already has,
            # so there is one mechanism limiting clustering rather than two.
            # Never *coarsens* below what was asked for: a body whose uniform
            # spacing is already at the floor is left alone.
            max_refinement = min(max_refinement, self.perimeter / n / min_spacing)
            max_refinement = max(max_refinement, 1.0)

        h = self._sizing_field(
            dense, n, max_turn_deg, max_refinement, smoothing_passes, max_spacing_ratio
        )

        # W(u) counts points: equal increments of W are one cell apart.
        du = self.perimeter / n_dense
        w_grid = np.concatenate(([0.0], np.cumsum(du / h)))
        u_grid = np.concatenate((u_dense, [self.perimeter]))

        u_bounds = np.concatenate(
            (np.sort((corners_s - offset) % self.perimeter), [self.perimeter])
            if len(corners_s)
            else ([0.0], [self.perimeter])
        )
        w_bounds = np.interp(u_bounds, u_grid, w_grid)
        counts = _allocate_points(np.diff(w_bounds), n)

        targets = np.concatenate(
            [
                w_bounds[k] + np.arange(count) * (w_bounds[k + 1] - w_bounds[k]) / count
                for k, count in enumerate(counts)
                if count > 0
            ]
        )
        u_new = np.interp(targets, w_grid, u_grid)

        return Contour(
            self._interpolate_at(offset + u_new),
            name=self.name,
            reference_length=self._reference_length,
        )

    def _corner_arclengths(self, corner_angle_deg: float) -> np.ndarray:
        """Arclengths of vertices sharp enough to count as genuine corners."""
        idx = np.flatnonzero(self.turning_angles() > np.radians(corner_angle_deg))
        return self.arclength()[idx]

    def _sizing_field(
        self,
        dense: np.ndarray,
        n: int,
        max_turn_deg: float,
        max_refinement: float,
        smoothing_passes: int,
        max_spacing_ratio: float,
    ) -> np.ndarray:
        """Target wall spacing at each dense sample: small where the contour turns fast.

        Curvature is measured across a stencil roughly one target cell wide rather
        than between adjacent dense samples. Pointwise curvature on a polyline is
        meaningless -- it is zero along every facet and near-infinite at every
        vertex -- so a circle drawn as a 512-gon would read as 512 sharp corners
        and the sizing field would cluster points onto the drawing's vertices
        instead of onto the geometry. Measuring at the cell scale averages the
        faceting out while still resolving real corners.
        """
        h_uniform = self.perimeter / n
        stride = max(1, int(round(len(dense) / n)))
        kappa = np.abs(_circumcircle_curvature(dense, stride))

        h = np.divide(
            np.radians(max_turn_deg),
            kappa,
            out=np.full_like(kappa, h_uniform),
            where=kappa > 0.0,
        )
        h = np.clip(h, h_uniform / max_refinement, h_uniform)

        # Smooth in log space so the field stays positive and the smoothing acts
        # on ratios, which is what the growth limit below is expressed in.
        log_h = np.log(h)
        for _ in range(smoothing_passes):
            log_h = 0.25 * (np.roll(log_h, 1) + 2.0 * log_h + np.roll(log_h, -1))

        return _limit_growth(
            np.exp(log_h), max_spacing_ratio, self.perimeter / len(dense)
        )

    def _uniform_by_arclength(self, n: int) -> np.ndarray:
        return self._interpolate_at(np.linspace(0.0, self.perimeter, n, endpoint=False))

    def _interpolate_at(self, s: np.ndarray) -> np.ndarray:
        """Linear interpolation of the polyline at the given arclengths."""
        s_nodes = np.concatenate((self.arclength(), [self.perimeter]))
        closed = np.vstack((self._xy, self._xy[:1]))
        s_wrapped = s % self.perimeter
        return np.column_stack(
            (
                np.interp(s_wrapped, s_nodes, closed[:, 0]),
                np.interp(s_wrapped, s_nodes, closed[:, 1]),
            )
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ContourError` if this contour cannot be meshed.

        A self-intersecting or degenerate boundary sends the marching mesher into
        folded cells with negative volumes, so it is rejected here, where the
        error message can still point at the cause.
        """
        from shapely.geometry import LinearRing  # local import: only needed here

        if self.area <= 0.0:
            raise ContourError("contour encloses no area")

        ring = LinearRing(np.vstack((self._xy, self._xy[:1])))
        if not ring.is_valid:
            raise ContourError("contour is not a valid ring")
        if not ring.is_simple:
            raise ContourError(
                "contour self-intersects; the boundary must be a single simple loop"
            )
        if np.any(self.segment_lengths() <= 0.0):
            raise ContourError("contour has zero-length segments")

    def as_wall_line(self) -> np.ndarray:
        """(N, 2) wall points ordered clockwise, i.e. with the fluid on the left.

        The mesher marches outward in ``j``. With the wall traversed clockwise the
        (i, j) pair is right-handed, so cell volumes come out positive and the
        finite-volume face normals point the way the discretisation assumes.
        Reversing here rather than at each point of use keeps exactly one place
        where the orientation convention gets applied.
        """
        return self._xy[::-1].copy()


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _circumcircle_curvature(xy: np.ndarray, stride: int) -> np.ndarray:
    """Signed curvature from the circle through each point and its two neighbours.

    ``kappa = 2 * cross(b - a, c - b) / (|b-a| |c-b| |c-a|)``, which is ``1/R`` of
    the circumcircle of the triple. Positive where the loop turns counter-
    clockwise. Collinear triples give exactly zero rather than dividing by zero.

    ``stride`` sets how far apart the neighbours are taken, and so the length
    scale the curvature is measured over.
    """
    a = np.roll(xy, stride, axis=0)
    b = xy
    c = np.roll(xy, -stride, axis=0)

    ab, bc, ca = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = (
        np.linalg.norm(ab, axis=1)
        * np.linalg.norm(bc, axis=1)
        * np.linalg.norm(ca, axis=1)
    )
    return np.divide(2.0 * cross, denom, out=np.zeros_like(denom), where=denom > 0.0)


def _signed_area(xy: np.ndarray) -> float:
    """Shoelace formula; positive for a counter-clockwise loop."""
    q = np.roll(xy, -1, axis=0)
    return 0.5 * float(np.sum(xy[:, 0] * q[:, 1] - q[:, 0] * xy[:, 1]))


def _drop_duplicates(xy: np.ndarray) -> np.ndarray:
    """Remove consecutive coincident points, including a repeated closing point."""
    span = float(np.hypot(*(xy.max(axis=0) - xy.min(axis=0))))
    tol = max(span * _DUPLICATE_FRAC, 1e-15)
    step = np.roll(xy, -1, axis=0) - xy
    return xy[np.hypot(step[:, 0], step[:, 1]) > tol]


def _normalise_rows(v: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving exact zeros alone rather than NaN."""
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0.0)


def _limit_growth(h: np.ndarray, max_ratio: float, ds: float) -> np.ndarray:
    """Impose a gradation limit on a periodic spacing field sampled every ``ds``.

    Enforces the Lipschitz condition ``h(s2) <= h(s1) + beta |s2 - s1|`` with
    ``beta = max_ratio - 1``, only ever *reducing* h, so refinement asked for
    anywhere survives and merely spreads into its neighbourhood.

    This is the classical mesh-gradation constraint, and the point of it is that
    it is scale-free: two points a distance ``h`` apart -- one cell -- can differ
    by at most a factor ``1 + beta``, whatever ``h`` happens to be there. A cap
    expressed per *sample* instead cannot do this, because the number of samples
    spanned by one cell varies across the field, so coarse regions accumulate
    more growth than fine ones and the realised ratio drifts above the cap.

    Implementation is a prefix scan. With ``g[i] = h[i] - i beta ds`` the
    constraint ``h[i] <= h[i-1] + beta ds`` reads ``g[i] <= g[i-1]``: g must be
    non-increasing, which is one ``np.minimum.accumulate``. A reversed pass
    imposes the mirror constraint. The field is tripled first so both scans see
    the periodic wrap without a Python loop.
    """
    if max_ratio <= 1.0:
        return h

    n = len(h)
    tiled = np.tile(h, 3)
    ramp = np.arange(3 * n) * (max_ratio - 1.0) * ds

    forward = np.minimum.accumulate(tiled - ramp) + ramp
    # The mirror pass runs on the reversed field but against the *same*
    # increasing ramp; reversing the ramp as well would point the constraint the
    # wrong way and let the cap through unenforced.
    limited = (np.minimum.accumulate(forward[::-1] - ramp) + ramp)[::-1]

    return limited[n : 2 * n]


def _allocate_points(weights: np.ndarray, total: int) -> np.ndarray:
    """Split ``total`` points between segments in proportion to ``weights``.

    Largest-remainder rounding, so the counts sum to exactly ``total``. Every
    segment gets at least one point, because each segment starts at a corner and
    that corner has to be reproduced.
    """
    share = np.asarray(weights, dtype=float)
    share = share / share.sum() * total

    counts = np.maximum(np.floor(share).astype(int), 1)
    # Hand out what rounding left over, to the segments that lost the most.
    for idx in np.argsort(-(share - counts))[: total - counts.sum()]:
        counts[idx] += 1
    # If the minimum-of-one rule overshot, take back from the largest segments.
    while counts.sum() > total:
        counts[np.argmax(counts)] -= 1
    return counts
