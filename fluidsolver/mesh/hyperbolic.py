"""Hyperbolic marching generation of a body-fitted O-grid.

The grid is grown outward from the wall one layer at a time. Each layer is found
by imposing two conditions on the new points:

    orthogonality      x_xi x_eta + y_xi y_eta = 0
    cell area          x_xi y_eta - y_xi x_eta = V

Together these say "step off the surface at right angles, far enough to sweep out
the area asked for". Written for the whole layer at once and linearised about the
layer below, they form a hyperbolic system in the marching direction, hence the
name.

Why this rather than the obvious alternatives:

* *Marching along the surface normal* is the pointwise solution of the same two
  equations, and it folds. Normals converge wherever the body is concave and fan
  out at convex corners, so grid lines cross within a few layers.
* *Transfinite interpolation* between the body and a circle needs a point-to-point
  correspondence between them. On anything as elongated as an aerofoil that
  correspondence is badly distorted, and the result needs heavy elliptic
  smoothing to become usable.

The coupling that stops the folding comes from solving the layer implicitly: the
``x_xi`` terms are evaluated on the layer being solved for, not the one below, so
every point's step depends on its neighbours'. The discrete system is block
tridiagonal and periodic -- the O-grid wraps -- and is solved directly.

That coupling is necessary but not sufficient. The metric ``r_xi`` is a central
difference, blind to a sawtooth alternating between neighbouring points, so that
mode grows unchecked until adjacent points collide. Damping it is the job of the
fourth-difference dissipation in :func:`_d4`, which separates the sawtooth from
the grid itself by nine orders of magnitude in eigenvalue.

The march is at its most useful near the wall and least useful far from it, where
cells are enormous and the body's shape no longer means anything. It is not
expected to reach the far field: :mod:`fluidsolver.mesh.ogrid` stops it once cell
quality starts to degrade and builds the remainder analytically.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fluidsolver.geometry.contour import Contour


class MeshError(RuntimeError):
    """Raised when a valid grid cannot be marched from the given geometry."""


def hyperbolic_grid(
    contour: Contour,
    thicknesses: np.ndarray,
    *,
    dissipation: float = 0.003,
    second_difference: float = 0.0,
    dissipation_growth: float = 1.0,
    area_relaxation: float = 0.0,
    max_retries: int = 6,
    allow_partial: bool = False,
    max_width_ratio: float = 3.0,
) -> tuple[np.ndarray, int]:
    """March an O-grid outward from ``contour``.

    Parameters
    ----------
    contour
        The body. Its :meth:`~fluidsolver.geometry.Contour.as_wall_line` ordering
        is used, which puts the fluid on the left so that the (i, j) system comes
        out right-handed and every cell area is positive.
    thicknesses
        Layer thicknesses from :mod:`fluidsolver.mesh.spacing`, innermost first.
    dissipation
        Fourth-difference smoothing, the term that holds the sawtooth down. Small
        by design: useful values are thousandths, since the stencil's own weights
        amplify it and anything larger starts smoothing the geometry rather than
        the noise.
    second_difference
        Second-difference smoothing, off by default. It damps the sawtooth only
        by damping the solution with it, so it is a last resort for stubborn
        re-entrant geometry rather than a routine control.
    dissipation_growth
        Factor by which the smoothing rises between the first layer and the last.
    area_relaxation
        How strongly the target cell area is equalised around the layer as the
        march proceeds. Left at 0 by default: inflating a clustered cell's target
        area while its thickness is fixed forces its points apart tangentially,
        which fights the orthogonality condition and folds the grid at corners.
        Equalising the far field is the analytic blend's job instead.
    max_retries
        Attempts per layer. A layer that produces a negative cell area is redone
        with the dissipation multiplied up.

    max_width_ratio
        Under ``allow_partial``, stop once neighbouring cells within a layer
        differ in width by more than this. Only meaningful with
        ``allow_partial``, since without it there is nothing to hand over to.
    allow_partial
        Return the layers that did succeed instead of raising when one folds, or
        when ``max_width_ratio`` is breached.
        The caller is told how many were completed and can take over from there,
        which is how :mod:`fluidsolver.mesh.ogrid` hands the far field off to its
        analytic blend.

    Returns
    -------
    nodes
        ``(Ni, Nj + 1, 2)`` node coordinates. ``i`` wraps around the body and is
        periodic; ``j`` runs from 0 at the wall outward. Rows beyond the number
        of completed layers are undefined.
    completed
        How many layers were actually marched; equal to ``len(thicknesses)``
        unless ``allow_partial`` stopped the march early.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if thicknesses.ndim != 1 or len(thicknesses) < 1:
        raise MeshError("thicknesses must be a non-empty 1-D array")
    if np.any(thicknesses <= 0.0):
        raise MeshError("every layer thickness must be positive")

    wall = contour.as_wall_line()
    n_i, n_j = len(wall), len(thicknesses)

    nodes = np.empty((n_i, n_j + 1, 2))
    nodes[:, 0] = wall

    # Seeded with the surface normal; thereafter each layer reuses the direction
    # the previous one actually took, which is a far better guess.
    step = _outward_normals(wall) * thicknesses[0]

    for j in range(n_j):
        # Ramp on the layer index, not the marched distance. With geometric
        # growth the distance stays a negligible fraction of the total until the
        # last handful of layers, so a distance-based ramp is effectively no ramp
        # at all over the part of the mesh that needs it.
        fraction = (j + 1) / n_j
        ramp = 1.0 + (dissipation_growth - 1.0) * fraction
        epsilon = dissipation * ramp
        equalisation = area_relaxation * fraction

        for attempt in range(max_retries):
            new_layer = _march_one_layer(
                nodes[:, j], step, thicknesses[j], equalisation, epsilon,
                second_difference,
            )
            areas = _cell_areas(nodes[:, j], new_layer)
            if areas.min() > 0.0:
                break
            # Folded. More smoothing straightens the layer out; this is the
            # documented recovery path for a re-entrant profile.
            epsilon *= 4.0
        else:
            if allow_partial:
                return nodes, j
            worst = int(np.argmin(areas))
            raise MeshError(
                f"layer {j + 1} of {n_j} folded at i={worst} "
                f"(x={nodes[worst, j, 0]:.4g}, y={nodes[worst, j, 1]:.4g}) "
                f"after {max_retries} attempts at increasing smoothing. "
                "The profile is probably re-entrant on a scale finer than the "
                "wall spacing; simplify the geometry, coarsen the surface, or "
                "raise the dissipation."
            )

        # A folded cell is the end state, not the first symptom. The sawtooth
        # that eventually inverts a cell squashes it for many layers first, and a
        # layer carrying cells forty times their neighbours is worthless even
        # though every area is still positive. Stop while the grid is good and
        # let the caller's far field take over.
        if allow_partial and j > 0:
            width = np.hypot(*_d_xi(new_layer).T)
            ratio = np.maximum(
                np.roll(width, -1) / width, width / np.roll(width, -1)
            ).max()
            if ratio > max_width_ratio:
                return nodes, j

        nodes[:, j + 1] = new_layer
        step = new_layer - nodes[:, j]
        if j + 1 < n_j:
            # Rescale the direction to the next layer's thickness so it is a
            # prediction of the coming step, not a repeat of the last one.
            step *= thicknesses[j + 1] / thicknesses[j]

    return nodes, n_j


# ----------------------------------------------------------------------
# One marching step
# ----------------------------------------------------------------------


def _march_one_layer(
    previous: np.ndarray,
    step: np.ndarray,
    thickness: float,
    equalisation: float,
    epsilon: float,
    second_difference: float,
    iterations: int = 4,
) -> np.ndarray:
    """Newton-solve the orthogonality/area system for the next layer.

    Writing ``f = (x_xi x_eta + y_xi y_eta,  x_xi y_eta - y_xi x_eta)``, the two
    conditions are ``f = (0, V)``. The Jacobian splits into the two derivative
    blocks

        A = df/dr_xi  = [[x_eta,  y_eta], [ y_eta, -x_eta]]
        B = df/dr_eta = [[x_xi,   y_xi ], [-y_xi,   x_xi ]]

    and a Newton step from the current iterate reads
    ``A r_xi + B r_eta = (0, V) - f + A r_xi + B r_eta``. Both ``A r_xi`` and
    ``B r_eta`` evaluate to ``(f1, f2)``, so the whole right-hand side collapses
    to ``(f1, V + f2) + B r_old``. Note the plus: row 1 of ``A r_xi`` is
    ``x_xi y_eta - y_xi x_eta``, which is ``+f2``. Taking it as ``-f2`` still
    yields a well-conditioned system, but one whose solution marches *inward*.

    Central-differencing ``r_xi`` on the layer being solved for turns each point
    into

        (-A/2) r_{i-1} + B r_i + (A/2) r_{i+1} = rhs

    to which a dissipation term is added on the left. The result is periodic block
    tridiagonal with 2x2 blocks.

    The dissipation smooths the marching *increment* ``r_new - r_old``, not the
    position. Smoothing positions is the obvious choice and it is wrong: the
    discrete Laplacian of a curved grid line is not zero -- it is of order
    ``R d_xi^2`` -- so an ``eps grad^2 r`` term injects a spurious source
    proportional to the body's curvature, and a circle marched with it comes out
    visibly non-circular and folds. The increment's Laplacian is of order
    ``step d_xi^2`` instead, which near the wall is smaller by the ratio of step
    to radius, while still damping exactly the wiggles in marching direction that
    cause folding.

    A single linearisation leaves a relative error of order ``(step / radius)^2``
    per layer. That is negligible near the wall but not in the far field, where
    geometric growth makes the layers comparable to the local radius of
    curvature, and it accumulates over the march. Re-linearising a few times
    drives each layer onto the true nonlinear solution instead.
    """
    xi_old = _d_xi(previous)
    current = previous + step

    for _ in range(iterations):
        # Evaluate the cell width halfway between the two layers, not on either
        # one. The cell's area is its thickness times its *mid* width, and using
        # the outer width instead overstates the area by one factor of
        # (1 + step / radius) -- which forces the layer to advance short by the
        # same factor. At the wall that is invisible; in the far field, where
        # geometric growth makes the step comparable to the local radius, it
        # costs over a tenth of the marched distance.
        mid = 0.5 * (xi_old + _d_xi(current))
        r_eta = current - previous
        x_xi, y_xi = mid[:, 0], mid[:, 1]
        x_eta, y_eta = r_eta[:, 0], r_eta[:, 1]

        f1 = x_xi * x_eta + y_xi * y_eta
        f2 = x_xi * y_eta - y_xi * x_eta

        # Target area is rebuilt from the current mid width each pass. At
        # convergence the area condition then reads |mid| * |step| = thickness *
        # |mid|, so the layer advances by exactly the thickness asked for.
        target_area = thickness * _blend_to_mean(np.hypot(x_xi, y_xi), equalisation)

        # The smoothing coefficients carry a length, to stay dimensionally
        # consistent with B. The local cell width is the obvious choice and it is
        # the wrong one: cells are smallest exactly where points are clustered
        # into a corner, so scaling by width makes the smoothing weakest at the
        # one place the march is about to fold. What actually destabilises the
        # scheme is the step outgrowing the cell it is stepping off, so the
        # length used is whichever of the two is larger.
        scale = np.maximum(np.hypot(x_xi, y_xi), thickness)
        eps2 = second_difference * scale
        eps4 = epsilon * scale

        # Half of r_xi is carried implicitly by the matrix; the other half is the
        # layer below, and belongs on the right.
        explicit = np.column_stack(
            (
                x_eta * xi_old[:, 0] + y_eta * xi_old[:, 1],
                y_eta * xi_old[:, 0] - x_eta * xi_old[:, 1],
            )
        )

        rhs = np.empty_like(previous)
        rhs[:, 0] = f1 + x_xi * previous[:, 0] + y_xi * previous[:, 1]
        rhs[:, 1] = (target_area + f2) - y_xi * previous[:, 0] + x_xi * previous[:, 1]
        rhs -= 0.5 * explicit

        # Both dissipations act on the marching *increment*, not the position.
        # The matrix carries the increment's implicit half; what is left is the
        # same stencil applied to the layer below, which moves to the right.
        rhs += eps2[:, None] * _d2(previous) + eps4[:, None] * _d4(previous)

        matrix = _assemble(x_xi, y_xi, x_eta, y_eta, eps2, eps4)
        solution = spla.spsolve(matrix, rhs.ravel())
        if not np.all(np.isfinite(solution)):
            raise MeshError("the marching system produced a non-finite solution")
        current = solution.reshape(-1, 2)

    return current


def _d_xi(layer: np.ndarray) -> np.ndarray:
    """Central difference around the loop, the discrete ``dr/dxi`` with ``dxi = 1``."""
    return 0.5 * (np.roll(layer, -1, axis=0) - np.roll(layer, 1, axis=0))


def _d2(f: np.ndarray) -> np.ndarray:
    """Negated second difference, stencil ``[-1, 2, -1]``."""
    return 2.0 * f - np.roll(f, 1, axis=0) - np.roll(f, -1, axis=0)


def _d4(f: np.ndarray) -> np.ndarray:
    """Fourth difference, stencil ``[1, -4, 6, -4, 1]``.

    This is the operator that actually holds the march together. The metric
    ``r_xi`` is a central difference, which skips the point it is centred on and
    is therefore completely blind to a sawtooth alternating between neighbours.
    Left alone that mode grows until adjacent points collide and the cell between
    them inverts -- the failure shows up as inverted cells on every *other* index.

    A second difference damps it, but only by also damping the smooth solution:
    forced hard enough to control the sawtooth it drives ``grad^2 (increment)``
    to zero, which for a closed loop means the layer stops deforming and simply
    translates. The fourth difference separates the two. Its eigenvalue on the
    sawtooth is 16, and on the smoothest resolved mode ``(2 pi / N)^4`` -- around
    5e-9 for a 240-point loop -- so it can be applied strongly enough to kill the
    oscillation while leaving the grid itself untouched.
    """
    return (
        6.0 * f
        - 4.0 * (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0))
        + np.roll(f, 2, axis=0)
        + np.roll(f, -2, axis=0)
    )


def _assemble(
    x_xi: np.ndarray,
    y_xi: np.ndarray,
    x_eta: np.ndarray,
    y_eta: np.ndarray,
    eps2: np.ndarray,
    eps4: np.ndarray,
) -> sp.csr_matrix:
    """Build the periodic block-banded matrix for one marching step.

    Unknowns are interleaved as ``[x_0, y_0, x_1, y_1, ...]``, so point ``i``
    owns rows ``2i`` and ``2i+1``. The stencil spans ``i-2 .. i+2``: the
    orthogonality/area terms are tridiagonal, and the fourth-difference
    dissipation widens it by one on each side.
    """
    n = len(x_xi)
    index = np.arange(n)

    identity = np.zeros((n, 2, 2))
    identity[:, 0, 0] = 1.0
    identity[:, 1, 1] = 1.0

    # B, plus the diagonal weight of both dissipation stencils
    # (second difference [-1, 2, -1], fourth difference [1, -4, 6, -4, 1]).
    diagonal = np.stack(
        [
            np.stack([x_xi, y_xi], axis=-1),
            np.stack([-y_xi, x_xi], axis=-1),
        ],
        axis=1,
    ) + identity * (2.0 * eps2 + 6.0 * eps4)[:, None, None]

    # A quarter of A, not a half: only half of the mid-layer r_xi is implicit,
    # the other half being the fixed layer below, so the central difference that
    # reaches the neighbours carries a factor of one half with it.
    quarter_a = np.stack(
        [
            np.stack([0.25 * x_eta, 0.25 * y_eta], axis=-1),
            np.stack([0.25 * y_eta, -0.25 * x_eta], axis=-1),
        ],
        axis=1,
    )
    adjacent = identity * (eps2 + 4.0 * eps4)[:, None, None]
    outer = identity * eps4[:, None, None]

    blocks = [
        (index, diagonal),
        ((index - 1) % n, -quarter_a - adjacent),
        ((index + 1) % n, quarter_a - adjacent),
        ((index - 2) % n, outer),
        ((index + 2) % n, outer),
    ]

    rows, cols, data = [], [], []
    for neighbour, block in blocks:
        for row in range(2):
            for col in range(2):
                rows.append(2 * index + row)
                cols.append(2 * neighbour + col)
                data.append(block[:, row, col])

    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(2 * n, 2 * n),
    )


def _blend_to_mean(width: np.ndarray, equalisation: float) -> np.ndarray:
    """Blend a layer's cell widths towards their mean.

    Leaving the widths alone gives every cell in the layer the same thickness and
    so carries the wall's point clustering outward unchanged. Forty chords from an
    aerofoil that is pure waste -- the leading-edge refinement is still there,
    resolving nothing.

    Pulling the widths towards the layer mean spends those points more usefully
    and, with the orthogonality condition, is what makes the outer layers relax
    towards uniform spacing. The blend preserves the mean, so the layer still
    advances by the thickness it was given.
    """
    blend = float(np.clip(equalisation, 0.0, 1.0))
    return (1.0 - blend) * width + blend * width.mean()


def _outward_normals(wall: np.ndarray) -> np.ndarray:
    """Unit normals pointing into the fluid, for a wall traversed clockwise.

    The contour's own normals are defined for its counter-clockwise storage
    order; the wall line is the reverse of that, so the rotation flips sign to
    ``n = (-t_y, t_x)``.
    """
    tangent = np.roll(wall, -1, axis=0) - np.roll(wall, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.column_stack((-tangent[:, 1], tangent[:, 0]))


def _cell_areas(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Signed areas of the quadrilaterals between two node layers.

    Positive means the cell is correctly oriented. A non-positive value is a
    folded or inverted cell, which no amount of solver robustness can recover
    from, so the marcher retries the layer rather than passing it on.
    """
    p00 = inner
    p10 = np.roll(inner, -1, axis=0)
    p11 = np.roll(outer, -1, axis=0)
    p01 = outer
    return 0.5 * (
        _cross(p00, p10) + _cross(p10, p11) + _cross(p11, p01) + _cross(p01, p00)
    )


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
