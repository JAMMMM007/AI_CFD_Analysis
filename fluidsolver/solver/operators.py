"""Discrete gradient, convection and diffusion operators.

These assemble the coefficients of a general scalar transport equation

    div(rho u phi) - div(Gamma grad phi) = S

into the five-band form :mod:`fluidsolver.solver.linalg` solves. Momentum,
pressure correction, ``k`` and ``omega`` all pass through here; only their
diffusivity and source terms differ.

Two decisions govern the accuracy of everything downstream.

**Convection uses deferred correction.** First-order upwind is unconditionally
bounded and gives a diagonally dominant matrix, but it is far too diffusive to
predict separation -- on a mesh this size the numerical viscosity would swamp the
turbulent viscosity being so carefully modelled. A second-order scheme alone is
unbounded and oscillates. So upwind goes in the matrix, and the difference
between the high-order face value and the upwind one goes in the source, lagged
one iteration. At convergence the correction is exact and the scheme is fully
second order; along the way the matrix keeps upwind's stability.

**Diffusion is split for non-orthogonality.** A body-fitted mesh has faces whose
normals are not parallel to the line joining the cells either side. The part of
the flux along that line is a two-point coupling and goes in the matrix; the
remainder needs the full face gradient, which is not two-point, so it is lagged
in the source. On the orthogonal circle mesh that correction is identically zero.
"""

from __future__ import annotations

import numpy as np

from fluidsolver.solver.faces import FaceGeometry
from fluidsolver.solver.linalg import Coefficients

# Interpolation schemes for the high-order half of the deferred correction.
SCHEMES = ("upwind", "linear", "linear_upwind", "limited_linear")


def face_values_i(field: np.ndarray, faces: FaceGeometry) -> np.ndarray:
    """Linearly interpolate a cell field onto the ``i`` faces. Shape ``(Ni, Nj)``."""
    return faces.i_faces.interpolate(field, np.roll(field, 1, axis=0))


def face_values_j(
    field: np.ndarray, faces: FaceGeometry, wall: np.ndarray, far_field: np.ndarray
) -> np.ndarray:
    """Interpolate onto the ``j`` faces, with boundary values supplied.

    Returns ``(Ni, Nj+1)``: column 0 is the wall, column ``Nj`` the far field, and
    the interior columns are interpolated between neighbouring cells.
    """
    interior = faces.j_faces.interpolate(field[:, 1:], field[:, :-1])
    return np.concatenate((wall[:, None], interior, far_field[:, None]), axis=1)


class Gradient:
    """Weighted least-squares cell gradient.

    For each cell the gradient is the one that best reproduces the differences to
    its four neighbours,

        minimise  sum_N w_N [ (grad phi . d_N) - (phi_N - phi_P) ]^2

    which for a linear field is satisfied exactly, on *any* mesh, however skewed
    or stretched. The 2x2 normal-equation matrix depends only on geometry, so it
    is inverted once here and reused.

    The obvious alternative, Green-Gauss, is unusable on this mesh. It is exact
    only if the face values are exact, and linear interpolation places its value
    on the centroid-to-centroid line rather than at the face centre. That
    skewness error then enters the gradient divided by the cell volume: on a
    boundary-layer cell with an aspect ratio in the hundreds, ``|S|/V`` is of
    order ``1e5``, so a micron of skew becomes a gradient error of hundreds.
    Measured on a NACA 2412 mesh, Green-Gauss returned a gradient of magnitude
    112 for a linear field whose true gradient was 3, and iterating the skewness
    correction made it worse, not better.

    Inverse-square distance weighting is used. Without it the distant neighbour
    across a stretched cell dominates the fit, which is exactly backwards.
    """

    def __init__(self, faces: FaceGeometry):
        centroid = faces.metrics.centroid

        # Four stencil directions. On the boundary rows the missing cell
        # neighbour is replaced by the boundary face itself, which keeps the
        # stencil rectangular and needs no special-casing downstream.
        south = np.empty_like(centroid)
        south[:, 1:] = centroid[:, :-1] - centroid[:, 1:]
        south[:, 0] = faces.wall.centre - centroid[:, 0]

        north = np.empty_like(centroid)
        north[:, :-1] = centroid[:, 1:] - centroid[:, :-1]
        north[:, -1] = faces.far_field.centre - centroid[:, -1]

        self._offsets = np.stack(
            (
                np.roll(centroid, 1, axis=0) - centroid,
                np.roll(centroid, -1, axis=0) - centroid,
                south,
                north,
            ),
            axis=2,
        )

        squared = np.sum(self._offsets**2, axis=-1)
        self._weights = 1.0 / np.where(squared > 0.0, squared, 1.0)

        moment = np.einsum(
            "ijn,ijna,ijnb->ijab", self._weights, self._offsets, self._offsets
        )
        determinant = (
            moment[..., 0, 0] * moment[..., 1, 1] - moment[..., 0, 1] * moment[..., 1, 0]
        )
        if np.any(np.abs(determinant) <= 0.0):
            raise ValueError(
                "a cell's gradient stencil is degenerate -- its neighbours are "
                "collinear. The mesh has a collapsed cell."
            )
        self._inverse = (
            np.stack(
                (
                    np.stack((moment[..., 1, 1], -moment[..., 0, 1]), axis=-1),
                    np.stack((-moment[..., 1, 0], moment[..., 0, 0]), axis=-1),
                ),
                axis=-2,
            )
            / determinant[..., None, None]
        )

    def __call__(
        self, field: np.ndarray, wall: np.ndarray, far_field: np.ndarray
    ) -> np.ndarray:
        """Gradient of a cell field, ``(Ni, Nj, 2)``.

        ``wall`` and ``far_field`` are the values on those boundary faces. For a
        zero-gradient condition pass the adjacent cell values: the difference is
        then zero, which is precisely what a vanishing normal gradient asserts.
        """
        south = np.empty_like(field)
        south[:, 1:] = field[:, :-1] - field[:, 1:]
        south[:, 0] = wall - field[:, 0]

        north = np.empty_like(field)
        north[:, :-1] = field[:, 1:] - field[:, :-1]
        north[:, -1] = far_field - field[:, -1]

        differences = np.stack(
            (
                np.roll(field, 1, axis=0) - field,
                np.roll(field, -1, axis=0) - field,
                south,
                north,
            ),
            axis=2,
        )

        rhs = np.einsum(
            "ijn,ijn,ijna->ija", self._weights, differences, self._offsets
        )
        return np.einsum("ijab,ijb->ija", self._inverse, rhs)


def add_diffusion(
    coefficients: Coefficients,
    faces: FaceGeometry,
    diffusivity: np.ndarray,
    field_gradient: np.ndarray,
    *,
    wall_value: np.ndarray | None,
    far_field_value: np.ndarray | None,
    wall_diffusivity: np.ndarray | None = None,
    far_field_diffusivity: np.ndarray | None = None,
    wall_active: np.ndarray | None = None,
    far_field_active: np.ndarray | None = None,
) -> None:
    """Add ``-div(Gamma grad phi)`` to the coefficients, in place.

    ``wall_value`` and ``far_field_value`` of ``None`` mean a zero-gradient
    (Neumann) condition on that boundary, which contributes nothing: the
    diffusive flux through the face is zero by definition.

    The ``_active`` masks allow the condition to differ face by face along one
    boundary, which the far field needs -- it is Dirichlet where flow enters and
    zero-gradient where it leaves, and which is which changes as the solution
    develops.
    """
    gamma_i = face_values_i(diffusivity, faces)
    gamma_j = faces.j_faces.interpolate(diffusivity[:, 1:], diffusivity[:, :-1])

    # Interior i faces. Each face is shared: it is the east face of cell (i-1)
    # and the west face of cell (i), so it lands in two rows with opposite signs.
    coupling = gamma_i * faces.i_faces.diffusion_factor
    coefficients.centre += coupling + np.roll(coupling, -1, axis=0)
    coefficients.west -= coupling
    coefficients.east -= np.roll(coupling, -1, axis=0)

    # Interior j faces, present only between j-1 and j for j >= 1.
    coupling_j = gamma_j * faces.j_faces.diffusion_factor
    coefficients.centre[:, 1:] += coupling_j
    coefficients.centre[:, :-1] += coupling_j
    coefficients.south[:, 1:] -= coupling_j
    coefficients.north[:, :-1] -= coupling_j

    # Non-orthogonal correction: the part of the flux the two-point coupling
    # cannot represent, evaluated on the lagged gradient.
    face_gradient_i = faces.i_faces.interpolate(
        field_gradient, np.roll(field_gradient, 1, axis=0)
    )
    # The cross term sits on the left of the equation, so it enters the source
    # with its sign reversed. A cell is the owner of its west face and the
    # neighbour of its east face, and picks up opposite signs from the two.
    cross_i = gamma_i * np.sum(face_gradient_i * faces.i_faces.cross, axis=-1)
    coefficients.source += np.roll(cross_i, -1, axis=0) - cross_i

    face_gradient_j = faces.j_faces.interpolate(
        field_gradient[:, 1:], field_gradient[:, :-1]
    )
    cross_j = gamma_j * np.sum(face_gradient_j * faces.j_faces.cross, axis=-1)
    coefficients.source[:, 1:] -= cross_j
    coefficients.source[:, :-1] += cross_j

    _add_boundary_diffusion(
        coefficients, faces.wall, 0, wall_value,
        _boundary_diffusivity(wall_diffusivity, diffusivity[:, 0]),
        field_gradient[:, 0], wall_active,
    )
    _add_boundary_diffusion(
        coefficients, faces.far_field, -1, far_field_value,
        _boundary_diffusivity(far_field_diffusivity, diffusivity[:, -1]),
        field_gradient[:, -1], far_field_active,
    )


def _boundary_diffusivity(supplied, fallback) -> np.ndarray:
    return fallback if supplied is None else supplied


def _add_boundary_diffusion(
    coefficients: Coefficients,
    face,
    column: int,
    value: np.ndarray | None,
    diffusivity: np.ndarray,
    cell_gradient: np.ndarray,
    active: np.ndarray | None,
) -> None:
    """Diffusive flux through a boundary face with a fixed face value.

    A ``None`` value is a zero-gradient condition and contributes nothing, as does
    any face switched off by ``active``.
    """
    if value is None:
        return
    coupling = diffusivity * face.diffusion_factor
    cross = diffusivity * np.sum(cell_gradient * face.cross, axis=-1)
    if active is not None:
        # Both halves of the flux, not just the implicit one. A face switched off
        # by ``active`` is zero-gradient, and a zero-gradient face carries no
        # diffusive flux at all -- leaving the non-orthogonal correction behind
        # puts one there anyway.
        coupling = np.where(active, coupling, 0.0)
        cross = np.where(active, cross, 0.0)
    coefficients.centre[:, column] += coupling
    coefficients.source[:, column] += coupling * value + cross


def add_convection(
    coefficients: Coefficients,
    faces: FaceGeometry,
    flux_i: np.ndarray,
    flux_j: np.ndarray,
    field: np.ndarray,
    field_gradient: np.ndarray,
    *,
    far_field_value: np.ndarray | None,
    wall_value: np.ndarray | None = None,
    scheme: str = "linear",
) -> None:
    """Add ``div(rho u phi)`` to the coefficients, in place.

    ``flux_i`` and ``flux_j`` are face mass fluxes ``rho u . S``, signed towards
    increasing index. ``flux_j`` spans ``(Ni, Nj+1)`` and includes both
    boundaries.

    The wall entry is zero in any real case -- no mass crosses a solid surface --
    but it is carried through the same inflow/outflow logic as the far field
    rather than assumed away, so that the operator can be verified against a
    manufactured solution whose velocity is not tangent to the boundary.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown convection scheme {scheme!r}; expected one of {SCHEMES}")

    # Upwind, implicit. For a face carrying flux F out of the owner, the owner
    # supplies the face value when F > 0 and the neighbour when F < 0.
    outflow_i = np.maximum(flux_i, 0.0)
    inflow_i = np.maximum(-flux_i, 0.0)
    coefficients.centre += inflow_i + np.roll(outflow_i, -1, axis=0)
    coefficients.west -= outflow_i
    coefficients.east -= np.roll(inflow_i, -1, axis=0)

    interior_j = flux_j[:, 1:-1]
    outflow_j = np.maximum(interior_j, 0.0)
    inflow_j = np.maximum(-interior_j, 0.0)
    coefficients.centre[:, 1:] += inflow_j
    coefficients.centre[:, :-1] += outflow_j
    coefficients.south[:, 1:] -= outflow_j
    coefficients.north[:, :-1] -= inflow_j

    # Boundaries. Outflow takes the interior value, which is simply another
    # diagonal contribution; inflow brings in a known value, so it is a source.
    # The wall's outward direction is the negative of face_j's.
    far_flux = flux_j[:, -1]
    coefficients.centre[:, -1] += np.maximum(far_flux, 0.0)
    if far_field_value is not None:
        coefficients.source[:, -1] += np.maximum(-far_flux, 0.0) * far_field_value

    wall_flux = -flux_j[:, 0]
    coefficients.centre[:, 0] += np.maximum(wall_flux, 0.0)
    if wall_value is not None:
        coefficients.source[:, 0] += np.maximum(-wall_flux, 0.0) * wall_value

    # Remove the part of the convective term that only exists because continuity
    # is not yet satisfied.
    #
    #     sum_f F_f phi_f  =  sum_f F_f (phi_f - phi_P)  +  phi_P sum_f F_f
    #
    # The second piece is proportional to the cell's mass imbalance. It vanishes
    # at convergence, so subtracting it changes no converged answer -- but during
    # the run it acts as a source proportional to phi itself, which is to say
    # exponential growth. A uniform field cannot then stay uniform: measured on a
    # NACA 0012, k in the *freestream*, where both production terms are zero and
    # destruction is positive, grew to 570 times its inlet value before the run
    # diverged.
    coefficients.centre -= divergence(flux_i, flux_j, faces)

    if scheme != "upwind":
        _add_deferred_correction(
            coefficients, faces, flux_i, flux_j, field, field_gradient, scheme
        )


def _add_deferred_correction(
    coefficients: Coefficients,
    faces: FaceGeometry,
    flux_i: np.ndarray,
    flux_j: np.ndarray,
    field: np.ndarray,
    field_gradient: np.ndarray,
    scheme: str,
) -> None:
    """Move the high-order minus upwind difference into the source."""
    correction_i = flux_i * _face_correction(
        faces.i_faces,
        flux_i,
        owner=field,
        neighbour=np.roll(field, 1, axis=0),
        owner_gradient=field_gradient,
        neighbour_gradient=np.roll(field_gradient, 1, axis=0),
        scheme=scheme,
    )
    coefficients.source -= np.roll(correction_i, -1, axis=0) - correction_i

    correction_j = flux_j[:, 1:-1] * _face_correction(
        faces.j_faces,
        flux_j[:, 1:-1],
        owner=field[:, 1:],
        neighbour=field[:, :-1],
        owner_gradient=field_gradient[:, 1:],
        neighbour_gradient=field_gradient[:, :-1],
        scheme=scheme,
    )
    # The owner of a j face is the cell on its high-j side, so the face's outward
    # normal from that cell points towards *decreasing* j and the correction
    # enters with the opposite sign to the neighbour's. The i direction above
    # works out the other way round because a cell owns its low-i face.
    coefficients.source[:, 1:] += correction_j
    coefficients.source[:, :-1] -= correction_j


def _face_correction(
    face,
    flux: np.ndarray,
    *,
    owner: np.ndarray,
    neighbour: np.ndarray,
    owner_gradient: np.ndarray,
    neighbour_gradient: np.ndarray,
    scheme: str,
) -> np.ndarray:
    """High-order face value minus the upwind one.

    The area vector points from neighbour to owner, so a *positive* flux flows
    into the owner and the upwind cell is the neighbour. Getting that backwards
    leaves the deferred correction pointing the wrong way, which does not merely
    fail to raise the order -- it makes the scheme worse than the plain upwind it
    was meant to improve.
    """
    from_owner = flux < 0.0
    upwind = np.where(from_owner, owner, neighbour)
    downwind = np.where(from_owner, neighbour, owner)

    if scheme == "linear":
        return face.interpolate(owner, neighbour) - upwind

    upwind_gradient = np.where(from_owner[..., None], owner_gradient, neighbour_gradient)
    to_face = np.where(
        from_owner[..., None], face.to_face_from_owner, face.to_face_from_neighbour
    )
    extrapolated = np.sum(upwind_gradient * to_face, axis=-1)

    if scheme == "linear_upwind":
        return extrapolated

    # limited_linear: a van Leer limiter on the same extrapolation, which keeps
    # the face value between the two cell values. k and omega must stay positive,
    # and an unlimited scheme will happily undershoot them through zero next to a
    # stagnation point or a separation line.
    jump = downwind - upwind
    safe = np.where(np.abs(jump) > 1e-300, jump, 1e-300)
    ratio = 2.0 * extrapolated / safe
    limiter = (ratio + np.abs(ratio)) / (1.0 + np.abs(ratio))
    return np.where(np.abs(jump) > 1e-300, 0.5 * limiter * jump, 0.0)


def divergence(
    flux_i: np.ndarray, flux_j: np.ndarray, faces: FaceGeometry
) -> np.ndarray:
    """Net outflow from each cell, ``sum_f F_f``.

    This is the continuity imbalance the pressure correction has to remove, and
    its magnitude is the headline convergence measure for the SIMPLE loop.
    """
    return (
        np.roll(flux_i, -1, axis=0)
        - flux_i
        + flux_j[:, 1:]
        - flux_j[:, :-1]
    )
