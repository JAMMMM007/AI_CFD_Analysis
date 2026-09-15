"""Finite-volume metrics for a structured O-grid.

The solver never touches node coordinates. It works entirely in terms of cell
volumes, face area-vectors and centroid positions, which is what this module
computes once up front.

Indexing, fixed here and assumed everywhere downstream:

* Cells are ``(i, j)`` for ``i`` in ``0 .. Ni-1`` and ``j`` in ``0 .. Nj-1``.
  ``i`` wraps around the body and is periodic; ``j`` runs outward from the wall.
* ``face_i[i, j]`` is the face between cell ``(i-1, j)`` and cell ``(i, j)``, with
  its area-vector pointing in the ``+i`` direction. There are ``Ni`` of them per
  row, and ``face_i[0]`` is the wrap-around face.
* ``face_j[i, j]`` is the face between cell ``(i, j-1)`` and cell ``(i, j)``, with
  its area-vector pointing in ``+j``. There are ``Nj + 1`` per column;
  ``face_j[:, 0]`` is the wall and ``face_j[:, Nj]`` the far field.

Area-vectors carry the face length in their magnitude, so a flux is simply
``u . S`` with no separate length factor to forget. In 2-D the "area" is a length,
the depth being unity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Metrics:
    """Geometric quantities the finite-volume discretisation needs.

    Attributes
    ----------
    volume
        ``(Ni, Nj)`` cell volumes -- areas, for unit depth.
    centroid
        ``(Ni, Nj, 2)`` cell centroids. These are true area centroids, not means
        of the four corners; on the stretched cells of a boundary-layer mesh the
        two differ enough to matter to the gradient reconstruction.
    face_i_area, face_j_area
        ``(Ni, Nj, 2)`` and ``(Ni, Nj+1, 2)`` face area-vectors, pointing towards
        increasing ``i`` and ``j``.
    face_i_centre, face_j_centre
        Face midpoints, matching shapes.
    wall_distance
        ``(Ni, Nj)`` shortest distance from each cell centroid to the body
        surface. The k-omega SST model needs this in its blending functions and
        its wall boundary condition.
    """

    volume: np.ndarray
    centroid: np.ndarray
    face_i_area: np.ndarray
    face_j_area: np.ndarray
    face_i_centre: np.ndarray
    face_j_centre: np.ndarray
    wall_distance: np.ndarray
    #: Whether the ``i`` direction wraps.
    #:
    #: True for the O-grid, where the surface is a closed loop and cell ``Ni-1``
    #: neighbours cell ``0``. False for a topology with two open ends -- a flat
    #: plate, a channel -- where those are boundaries carrying conditions of
    #: their own.
    #:
    #: The two cases differ in the *shape* of ``face_i_area``, and that is the
    #: honest way to tell them apart. A closed loop has exactly as many i-faces
    #: as cells, because the last face is shared with the first cell. An open one
    #: has one more, which is the relationship ``face_j_area`` has always had
    #: with ``volume`` in the ``j`` direction:
    #:
    #:     periodic    face_i_area (Ni,   Nj, 2)    all interior
    #:     open        face_i_area (Ni+1, Nj, 2)    Ni-1 interior + two boundaries
    #:
    #: So this is not a new mechanism. It gives ``i`` the structure ``j`` already
    #: has, and every place that special-cases a ``j`` boundary is the template
    #: for the ``i`` one.
    periodic_i: bool = True

    @property
    def shape(self) -> tuple[int, int]:
        return self.volume.shape

    @property
    def total_volume(self) -> float:
        return float(self.volume.sum())

    def wall_face_area(self) -> np.ndarray:
        """``(Ni, 2)`` area-vectors of the wall faces, pointing *into* the solid.

        ``face_j`` points towards increasing ``j``, which is away from the wall,
        so the wall's outward-from-the-fluid normal is the negative of it. Forces
        on the body are integrated with this, so the sign convention matters:
        pressure acting on the body pushes along ``-face_j``.
        """
        return -self.face_j_area[:, 0]


def compute_metrics(
    nodes: np.ndarray,
    wall_samples: int = 8,
    periodic_i: bool = True,
    wall_mask: np.ndarray | None = None,
) -> Metrics:
    """Build the finite-volume metrics for a node array.

    Parameters
    ----------
    nodes
        Grid nodes. ``(Ni, Nj+1, 2)`` and wrapping in ``i`` for the O-grid, as
        :func:`fluidsolver.mesh.ogrid.build_ogrid` produces; ``(Ni+1, Nj+1, 2)``
        with two open ends when ``periodic_i`` is False, so that ``Ni+1`` node
        lines bound ``Ni`` cells.
    wall_samples
        Sub-samples per wall segment used when measuring wall distance. The
        surface is a polyline; measuring only to its vertices overestimates the
        distance for cells sitting opposite the middle of a long segment.
    periodic_i
        Whether the ``i`` direction wraps. See :attr:`Metrics.periodic_i`.
    wall_mask
        Which ``j = 0`` faces are solid, for the wall distance. ``None`` means all
        of them. A flat plate's row is part symmetry plane, and SST's blending
        functions need the distance to the *plate*: measured to the whole row, a
        cell just ahead of the leading edge would sit a first-layer height from a
        "wall" that is really a slip plane, and ``F1`` would switch on k-omega in
        a stream with no boundary layer in it.

    The two topologies differ only in which node line follows the last one -- the
    first, or the one after it -- so both are written through a single pair of
    ``base`` and ``following`` arrays rather than as two code paths. On the
    periodic branch those reduce to exactly the expressions this used before,
    which is what keeps the O-grid bit-identical.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 3 or nodes.shape[2] != 2 or nodes.shape[1] < 2:
        raise ValueError(f"nodes must have shape (Ni, Nj+1, 2), got {nodes.shape}")
    if not periodic_i and nodes.shape[0] < 2:
        raise ValueError(
            f"an open i direction needs at least two node lines to bound one "
            f"cell, got {nodes.shape[0]}"
        )

    base = nodes if periodic_i else nodes[:-1]
    following = np.roll(nodes, -1, axis=0) if periodic_i else nodes[1:]

    # Corners of every cell.
    volume, centroid = _polygon_volume_and_centroid(
        [base[:, :-1], following[:, :-1], following[:, 1:], base[:, 1:]]
    )

    # A face in the i-direction runs along j: from node (i, j) to node (i, j+1).
    # Rotating that edge by -90 degrees gives a normal pointing towards +i. There
    # is one per *node line*, which is one per cell when the loop closes and one
    # more than that when it does not.
    edge_i = nodes[:, 1:] - nodes[:, :-1]
    face_i_area = np.stack((edge_i[..., 1], -edge_i[..., 0]), axis=-1)
    face_i_centre = 0.5 * (nodes[:, 1:] + nodes[:, :-1])

    # A face in the j-direction runs along i: from node (i, j) to node (i+1, j).
    # Rotating by +90 degrees gives a normal pointing towards +j, i.e. outward.
    edge_j = following - base
    face_j_area = np.stack((-edge_j[..., 1], edge_j[..., 0]), axis=-1)
    face_j_centre = 0.5 * (following + base)

    return Metrics(
        volume=volume,
        centroid=centroid,
        face_i_area=face_i_area,
        face_j_area=face_j_area,
        face_i_centre=face_i_centre,
        face_j_centre=face_j_centre,
        wall_distance=_wall_distance(
            centroid, nodes[:, 0], wall_samples, closed=periodic_i,
            segments=wall_mask,
        ),
        periodic_i=periodic_i,
    )


def _polygon_volume_and_centroid(corners: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Signed area and area centroid of quadrilaterals given corner-by-corner.

    Both come from the same shoelace sum, so they are computed together:

        A   = 1/2 sum (x_k y_{k+1} - x_{k+1} y_k)
        C   = 1/(6A) sum (p_k + p_{k+1}) (x_k y_{k+1} - x_{k+1} y_k)

    Positive area means the corners were given counter-clockwise, which for this
    grid's ``(i, j)`` ordering is the correct, non-inverted orientation.
    """
    area = np.zeros(corners[0].shape[:-1])
    moment = np.zeros(corners[0].shape)

    for current, following in zip(corners, corners[1:] + corners[:1]):
        cross = (
            current[..., 0] * following[..., 1] - following[..., 0] * current[..., 1]
        )
        area += cross
        moment += (current + following) * cross[..., None]

    area *= 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = moment / (6.0 * area[..., None])
    return area, centroid


def _wall_distance(
    centroid: np.ndarray,
    wall: np.ndarray,
    samples: int,
    closed: bool = True,
    segments: np.ndarray | None = None,
) -> np.ndarray:
    """Shortest distance from each cell centroid to the body surface.

    Uses the true minimum distance to the surface polyline rather than the
    marching distance carried by the grid's ``j`` index. Those agree only where
    the grid is orthogonal and the surface is smooth; near a convex corner the
    marching distance is larger than the real one, and feeding that to the SST
    blending functions would mis-place the switch between its two model regimes.

    **The distance to a polyline is computed to the segments, not to samples
    along them.** This used to sub-sample each segment at ``samples`` points and
    query a KD-tree of those, which makes the answer a function of the sampling
    density: a centroid a distance ``y1`` off the wall, opposite the middle of a
    gap of length ``h_s / samples``, reads ``sqrt(y1^2 + (h_s / 2 samples)^2)``
    instead of ``y1``. At eight samples on the NACA 2412 mesh that was 17.4% too
    large in the worst first-row cell, and 3.8% at the worst cell anywhere --
    measured by re-running the same mesh at 32, 128 and 512 samples, where the
    answer converges.

    The error is at the *trailing* edge and along the long flat segments, not at
    the nose: the leading-edge cells came out identical to 512 samples, because
    that is where the surface resampler clusters points most finely. That is the
    opposite of where it would be looked for.

    Nothing measurable moved when this was corrected, and the reason is worth
    recording rather than treating as luck. The distance enters the SST blending
    functions through arguments like ``500 nu / (d^2 omega)``, and an 8% move in
    that argument at the worst cell changes nothing because ``F1`` is saturated at
    1 throughout the region where the error lives -- ``tanh(x^4)`` is 1 either
    way. It would matter if the blending functions were ever evaluated where they
    are not saturated, and it costs nothing to remove the question.

    The KD-tree still does the searching, over one point per segment; it only no
    longer does the *measuring*. Each centroid takes the exact perpendicular
    distance to the nearest segment and to that segment's two neighbours, which
    covers the case where the nearest midpoint and the nearest segment differ.

    ``samples`` is retained for callers that pass it and now selects how many
    points per segment seed the search rather than how finely the answer is
    quantised. The answer no longer depends on it.
    """
    from scipy.spatial import cKDTree

    # A closed surface has a segment from the last point back to the first; an
    # open one does not, and inventing it would put a spurious wall across the
    # domain -- for a flat plate, straight from the trailing edge to the leading
    # one.
    line = np.vstack((wall, wall[:1])) if closed else wall
    start = line[:-1]
    edge = np.diff(line, axis=0)
    if segments is not None:
        # Only the solid segments are wall. The neighbour search below steps to
        # adjacent entries of this *selected* list, which are adjacent segments
        # wherever the solid part is one contiguous run -- a plate -- and are
        # still genuine wall segments where it is not, so the minimum taken over
        # them is never a distance to something that is not wall.
        start = start[segments]
        edge = edge[segments]
    n_segments = len(start)

    # Seed the search with points along each segment, and remember which segment
    # each seed came from.
    fractions = np.linspace(0.0, 1.0, max(samples, 1), endpoint=False)
    seeds = (start[:, None, :] + fractions[None, :, None] * edge[:, None, :]).reshape(-1, 2)
    owner = np.repeat(np.arange(n_segments), len(fractions))

    points = centroid.reshape(-1, 2)
    _, nearest = cKDTree(seeds).query(points)
    segment = owner[nearest]

    # The nearest seed's segment, and its neighbours either side: the true
    # nearest segment can be adjacent to the one carrying the nearest seed,
    # which is exactly the case sub-sampling used to paper over.
    candidates = segment[:, None] + np.array([-1, 0, 1])[None, :]
    candidates = (
        candidates % n_segments if closed else np.clip(candidates, 0, n_segments - 1)
    )

    a = start[candidates]
    d = edge[candidates]
    offset = points[:, None, :] - a
    length_squared = np.sum(d * d, axis=-1)
    projection = np.clip(
        np.sum(offset * d, axis=-1) / np.where(length_squared > 0.0, length_squared, 1.0),
        0.0,
        1.0,
    )
    closest = a + projection[..., None] * d
    distance = np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)
    return distance.reshape(centroid.shape[:-1])


def cell_face_areas(metrics: Metrics) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four *outward* area-vectors of every cell, as (west, east, south, north).

    ``face_i`` and ``face_j`` are stored once per face and shared between the two
    cells either side of it, pointing consistently towards increasing index. A
    cell's own outward normals therefore flip sign on its low-index faces.
    Summing these four gives zero for a closed polygon, which is the identity the
    tests check.
    """
    west = -metrics.face_i_area
    east = np.roll(metrics.face_i_area, -1, axis=0)
    south = -metrics.face_j_area[:, :-1]
    north = metrics.face_j_area[:, 1:]
    return west, east, south, north
