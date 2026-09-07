"""Forces, surface distributions and the diagnostics that judge a solution.

The flow field is the intermediate result; these are what the case was run for.
The force integration in particular is where a sign convention error would hide
most comfortably, so the geometry is spelled out.

The wall face area vectors from :mod:`fluidsolver.solver.faces` point *out of the
fluid*, which is *into the solid*. Pressure pushes on the body along exactly that
direction, so the pressure force is a straight sum of ``p * area``. Shear drags the
body along with the near-wall flow, so the viscous force follows the direction of
the tangential velocity in the first cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluidsolver.solver.faces import FaceGeometry
from fluidsolver.solver.fields import State
from fluidsolver.solver.fluid import Fluid, Freestream


@dataclass(frozen=True)
class SurfaceData:
    """Distributions along the body surface, ordered as the wall line is."""

    x: np.ndarray
    y: np.ndarray
    arclength: np.ndarray
    pressure_coefficient: np.ndarray
    skin_friction_coefficient: np.ndarray
    y_plus: np.ndarray
    wall_shear: np.ndarray


@dataclass(frozen=True)
class Forces:
    """Integrated loads on the body, split by mechanism.

    Keeping pressure and friction separate is diagnostic, not decorative: on an
    attached aerofoil drag is mostly friction, and once it separates it is mostly
    pressure. Which one grew tells you what the solution thinks happened.
    """

    pressure_force: np.ndarray
    viscous_force: np.ndarray
    moment: float
    reference_length: float
    dynamic_pressure: float

    @property
    def total(self) -> np.ndarray:
        return self.pressure_force + self.viscous_force

    @property
    def lift(self) -> float:
        """Force perpendicular to the freestream, which runs along +x here."""
        return float(self.total[1])

    @property
    def drag(self) -> float:
        return float(self.total[0])

    @property
    def lift_coefficient(self) -> float:
        return self.lift / (self.dynamic_pressure * self.reference_length)

    @property
    def drag_coefficient(self) -> float:
        return self.drag / (self.dynamic_pressure * self.reference_length)

    @property
    def pressure_drag_coefficient(self) -> float:
        return float(self.pressure_force[0]) / (
            self.dynamic_pressure * self.reference_length
        )

    @property
    def friction_drag_coefficient(self) -> float:
        return float(self.viscous_force[0]) / (
            self.dynamic_pressure * self.reference_length
        )

    @property
    def moment_coefficient(self) -> float:
        """Pitching moment, nose-up positive, about :attr:`moment_reference`.

        The sign convention is the aerodynamic one -- Anderson section 1.5, and
        the one Abbott and von Doenhoff's tabulated ``Cm_ac`` values are quoted
        in -- because comparison against published section data is what this
        number is for. It is not the ``+z`` component of ``r x F`` in the mesh
        frame, which is its negative; see :func:`compute_forces`.
        """
        return self.moment / (self.dynamic_pressure * self.reference_length**2)


def wall_shear_stress(
    state: State, faces: FaceGeometry, fluid: Fluid, boundaries=None
) -> tuple[np.ndarray, np.ndarray]:
    """Wall shear traction vector and its magnitude.

    With ``boundaries`` supplied the magnitude is ``tau_w = rho u_tau^2`` from the
    blended wall treatment, and the direction is taken from the tangential
    velocity in the first cell. That is the only formulation valid across the
    whole range of near-wall spacings: the alternative below assumes a linear
    profile through the first cell, which is the truth inside the viscous
    sublayer and an overestimate anywhere else.

    Without it the linear form is used, ``tau_w = mu U1 / y1``, which is what
    this did unconditionally and is still what a laminar case wants -- there the
    profile through a well-resolved first cell really is linear and there is no
    friction velocity to speak of.

    Molecular viscosity in that branch, not the effective one: at the wall ``k``
    vanishes and so does the eddy viscosity, so the whole stress is viscous.
    """
    normal = faces.wall.normal
    velocity = state.velocity[:, 0]
    tangential = velocity - np.sum(velocity * normal, axis=-1)[:, None] * normal
    distance = faces.wall.wall_normal_distance

    if boundaries is None:
        traction = fluid.viscosity * tangential / distance[:, None]
        return traction, np.linalg.norm(traction, axis=-1)

    speed = np.linalg.norm(tangential, axis=-1)
    direction = tangential / np.maximum(speed, 1e-30)[:, None]
    magnitude = boundaries.wall_shear(state.u, state.v)
    return direction * magnitude[:, None], magnitude


def wall_pressure(state: State, faces: FaceGeometry) -> np.ndarray:
    """Pressure on the wall *faces*, extrapolated along the wall normal.

    What the force integral needs is ``p`` on the wall face; what the solver holds
    is ``p`` at the first cell centre, a distance ``y1`` away. Taking one for the
    other is a zeroth-order extrapolation, and it was the largest remaining
    first-order term in a scheme claimed to be second order.

    **Why ``dp/dn = 0`` does not rescue it.** The exact normal momentum balance at
    a solid wall, with ``u = 0`` there, is ``dp/dn = mu (grad^2 u) . n``, because
    the convective term vanishes identically at a no-slip surface. So ``dp/dn = 0``
    at the wall is the boundary-layer approximation, not an identity -- and the
    derivative that matters is the one at the *cell centre*, which sits inside the
    layer where ``u_t != 0`` and is not zero even when the wall value is. Taylor:

        p_face = p_cell - y1 (dp/dn)|_cell + O(y1^2),

    so using the cell value drops a term of order ``y1 (dp/dn)|_cell = O(h)``.
    Integrated round the body it does not cancel, because ``dp/dn`` at the first
    cell centre scales with the local ``u_t^2`` and is largest at the shoulder, so
    it is not fore-aft symmetric. The curvature term usually blamed for this is
    not the mechanism: on the Re 40 cylinder ``rho u1^2 y0 / R`` is 3.4e-06 of a
    dynamic head against the 4.0e-03 actually present, three orders too small.

    **The obvious construction does not work, and the measurement is why this one
    is here instead.** The natural fix, and the one the audit proposes, is
    ``p_face = p_P + (grad p)_P . d`` with the solver's own least-squares
    gradient, iterated once because the gradient depends on the wall value it is
    computing. Implemented and measured against a manufactured field on three mesh
    families, that is **order 1.13**, not 2:

        uniform mesh, 48 / 96 / 192 points        error        observed order
          cell value                   8.07e-02 4.05e-02 2.03e-02    1.00
          p_P + (grad p)_P . d         2.85e-02 1.21e-02 5.53e-03    1.13
          p_P + (grad p exact) . d     3.68e-03 8.53e-04 2.04e-04    2.06
          this: linear along n         1.15e-02 2.62e-03 6.21e-04    2.08

    The third row isolates the cause: with an exact gradient the formula is second
    order, so the formula is right and the gradient is not. The least-squares
    gradient **in the wall row does not converge at all** -- its error measured
    5.16e-01, 5.17e-01, 5.17e-01 across the same refinement, an observed order of
    0.00, and 0.12 after the fixed-point pass. The reason is structural: the
    stencil weights go as ``1/|d|^2``, and the wall face is the *nearest* stencil
    point, so a wall value asserting zero normal gradient is the most heavily
    weighted member of the fit. The reconstruction would be inheriting the error
    of the very assumption it exists to remove.

    So the face value is extrapolated along the wall normal through the first two
    cell centres instead, which needs no gradient:

        p_face = p_0 + (p_0 - p_1) y_0 / (y_1 - y_0)

    with ``y_j`` the perpendicular distance from the wall face to centre ``j``.
    Measured order 2.08 on the uniform family, 2.13 on the stretched one and 2.05
    on the sheared one.

    A Lagrange quadratic through the first three centres was also measured. It is
    better on the smooth families -- order 2.94 and 3.25 -- and *worse* on the
    sheared one at 1.95, where the third cell centre is far enough off the normal
    that the extra point costs more than it buys. The audit's own re-integration
    of the converged Re 40 cylinder found the linear and quadratic reconstructions
    agreeing to 8e-05 in ``Cd_pressure``, so on a real case the extra order buys
    nothing; the linear form is taken for being the more robust of two answers
    that agree.
    """
    normal = faces.wall.normal
    centroid = faces.metrics.centroid
    face = faces.wall.centre

    y0 = np.abs(np.sum((centroid[:, 0] - face) * normal, axis=-1))
    y1 = np.abs(np.sum((centroid[:, 1] - face) * normal, axis=-1))

    # A mesh one cell deep has nothing to extrapolate through; the cell value is
    # then the only information there is.
    gap = y1 - y0
    return np.where(
        gap > 0.0,
        state.pressure[:, 0]
        + (state.pressure[:, 0] - state.pressure[:, 1]) * y0 / np.where(gap > 0.0, gap, 1.0),
        state.pressure[:, 0],
    )


def compute_forces(
    state: State,
    faces: FaceGeometry,
    fluid: Fluid,
    freestream: Freestream,
    reference_length: float,
    moment_reference: np.ndarray,
    boundaries=None,
) -> Forces:
    """Integrate pressure and friction over the body."""
    area = faces.wall.area
    length = faces.wall.length

    face_pressure = wall_pressure(state, faces)
    pressure_force = np.sum(face_pressure[:, None] * area, axis=0)

    traction, _ = wall_shear_stress(state, faces, fluid, boundaries)
    viscous_force = np.sum(traction * length[:, None], axis=0)

    # Nose-up positive, which is the negative of the mesh-frame z moment.
    #
    # `Forces.lift` is `total[1]` and `Forces.drag` is `total[0]`, so lift is +y,
    # the freestream runs +x, and the leading edge is at smaller x. The sum
    # `r_x F_y - r_y F_x` is then the +z component of `r x F` with z out of the
    # page. Take a unit lift applied one length *ahead* of the reference,
    # r = (-1, 0) and F = (0, 1): that expression gives -1, and lift acting ahead
    # of the moment reference is a nose-up moment, which the standard convention
    # reports as +1. The two differ by a sign uniformly, for every contribution.
    #
    # Checked against the case: on the NACA 2412 at 5 degrees this reported
    # Cm = -0.0825, and a NACA 2412 with Cm_ac about -0.05, taken about a
    # reference roughly 0.42c aft of the aerodynamic centre at Cl = 0.75, should
    # read about -0.05 + 0.17 x 0.75 = +0.08. The magnitude agreed and the sign
    # did not.
    lever = faces.wall.centre - moment_reference
    element = face_pressure[:, None] * area + traction * length[:, None]
    moment = -float(np.sum(lever[:, 0] * element[:, 1] - lever[:, 1] * element[:, 0]))

    return Forces(
        pressure_force=pressure_force,
        viscous_force=viscous_force,
        moment=moment,
        reference_length=reference_length,
        dynamic_pressure=freestream.dynamic_pressure(fluid),
    )


def surface_data(
    state: State, faces: FaceGeometry, fluid: Fluid, freestream: Freestream,
    boundaries=None,
) -> SurfaceData:
    """Pressure coefficient, skin friction and ``y+`` along the surface."""
    dynamic = freestream.dynamic_pressure(fluid)
    centre = faces.wall.centre

    _, shear = wall_shear_stress(state, faces, fluid, boundaries)
    friction_velocity = np.sqrt(shear / fluid.density)
    y_plus = (
        fluid.density * friction_velocity * faces.wall.wall_normal_distance
        / fluid.viscosity
    )

    edges = np.linalg.norm(np.diff(centre, axis=0, prepend=centre[-1:]), axis=1)

    return SurfaceData(
        x=centre[:, 0],
        y=centre[:, 1],
        arclength=np.cumsum(edges) - edges[0],
        # The same reconstruction the force integral uses, so a plotted Cp and
        # the Cd it integrates to describe the same wall pressure.
        pressure_coefficient=wall_pressure(state, faces) / dynamic,
        skin_friction_coefficient=shear / dynamic,
        y_plus=y_plus,
        wall_shear=shear,
    )


def vorticity(state: State, gradient) -> np.ndarray:
    """``dv/dx - du/dy``, for visualising shear layers and the wake."""
    grad_u = gradient(state.u, state.u[:, 0] * 0.0, state.u[:, -1])
    grad_v = gradient(state.v, state.v[:, 0] * 0.0, state.v[:, -1])
    return grad_v[..., 0] - grad_u[..., 1]


def _wall_gradient_sign(
    state: State, faces: FaceGeometry, tangent: np.ndarray
) -> np.ndarray:
    """Sign of ``du_t/dy`` at the wall, from a quadratic through the first two cells.

    Fitting ``u_t = a y + b y^2`` through the no-slip wall and the tangential
    velocity at the first two cell centres and returning ``sign(a)``. The
    no-slip condition supplies the third point for free, which is why a quadratic
    costs only two cells.

    Falls back to the first-cell value on a mesh with a single wall-normal cell,
    where there is no second point to fit through and nothing better is available.
    """
    normal = faces.wall.normal

    def tangential(row):
        velocity = state.velocity[:, row]
        along_normal = np.sum(velocity * normal, axis=-1)[:, None] * normal
        return np.sum((velocity - along_normal) * tangent, axis=-1)

    if state.u.shape[1] < 2:
        return np.sign(tangential(0))

    y1 = faces.wall.wall_normal_distance
    y2 = y1 + np.abs(
        np.sum((faces.metrics.centroid[:, 1] - faces.metrics.centroid[:, 0]) * normal, axis=-1)
    )
    u1, u2 = tangential(0), tangential(1)

    denominator = y1 * y2**2 - y2 * y1**2
    slope = np.where(
        np.abs(denominator) > 0.0,
        (u1 * y2**2 - u2 * y1**2) / np.where(denominator != 0.0, denominator, 1.0),
        u1,
    )
    return np.sign(slope)


def separation_points(
    state: State, faces: FaceGeometry, fluid: Fluid, boundaries=None
) -> np.ndarray:
    """Surface positions where the wall shear changes sign.

    Separation is where the near-wall flow reverses, so the tangential component
    of the wall traction along the surface passes through zero. Interpolating
    between the two faces either side locates it to better than one cell.

    **The sign comes from a one-sided wall gradient, not from the first cell.**
    The traction direction that :func:`wall_shear_stress` returns is taken from
    the tangential velocity at the first cell centre, and in a separating
    boundary layer that is not the same thing as the wall shear: the profile is
    inflected, so ``du_t/dy`` changes sign *at the wall* before ``u_t(y1)`` does,
    because the reversed region grows outward from the surface. The disagreement
    is ``O(y1)`` -- first order in the wall spacing. Measured on the converged
    Re 40 cylinder, the same field gives 53.71710 degrees from the first-cell
    velocity and 53.97023 from a one-sided wall gradient, a difference of 0.253
    degrees on a quantity the gate was printing to three decimals.

    So the sign is taken here from a quadratic through the wall and the first two
    cell centres, ``u_t = a y + b y^2``, whose slope at the wall is

        a = ( u_t1 y2^2 - u_t2 y1^2 ) / ( y1 y2^2 - y2 y1^2 )

    which is ``O(h^2)`` where the first-cell value is ``O(h)``. Only the sign is
    taken from it; the traction *magnitude* still comes from
    :func:`wall_shear_stress`, which for a turbulent run is the blended wall
    treatment and has no better one-sided estimate available.

    The crossing test is ``along * following < 0`` rather than a comparison of
    ``np.sign``. ``np.sign`` returns 0 for an exact zero, so a face where the
    traction vanishes identically registered as *two* crossings rather than one.
    On a symmetric body the stagnation faces can hit that exactly;
    ``separation_angle`` filters them by angle afterwards, which works on a
    cylinder and would not on an aerofoil.
    """
    _, magnitude = wall_shear_stress(state, faces, fluid, boundaries)
    centre = faces.wall.centre

    tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    along = magnitude * _wall_gradient_sign(state, faces, tangent)

    following = np.roll(along, -1)
    crossing = np.flatnonzero(along * following < 0.0)
    if len(crossing) == 0:
        return np.empty((0, 2))

    weight = along[crossing] / (along[crossing] - following[crossing])
    return centre[crossing] + weight[:, None] * (
        np.roll(centre, -1, axis=0)[crossing] - centre[crossing]
    )
