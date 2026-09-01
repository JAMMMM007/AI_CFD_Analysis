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
        return self.moment / (self.dynamic_pressure * self.reference_length**2)


def wall_shear_stress(
    state: State, faces: FaceGeometry, fluid: Fluid
) -> tuple[np.ndarray, np.ndarray]:
    """Wall shear traction vector and its magnitude.

    ``tau_w = mu du_t/dn`` at the surface, evaluated from the first cell: the wall
    velocity is zero, so the tangential velocity there divided by the
    perpendicular distance is the near-wall gradient. That one-sided difference is
    only accurate if the first cell sits inside the viscous sublayer, which is
    the reason the mesher targets ``y+`` of order one.

    Molecular viscosity is used, not the effective one. At the wall ``k`` is zero
    and so is the eddy viscosity; the whole stress there is viscous.
    """
    normal = faces.wall.normal
    velocity = state.velocity[:, 0]
    tangential = velocity - np.sum(velocity * normal, axis=-1)[:, None] * normal

    distance = faces.wall.wall_normal_distance
    traction = fluid.viscosity * tangential / distance[:, None]
    return traction, np.linalg.norm(traction, axis=-1)


def compute_forces(
    state: State,
    faces: FaceGeometry,
    fluid: Fluid,
    freestream: Freestream,
    reference_length: float,
    moment_reference: np.ndarray,
) -> Forces:
    """Integrate pressure and friction over the body."""
    area = faces.wall.area
    length = faces.wall.length

    # Zero normal pressure gradient at a wall, so the face value is the cell value.
    wall_pressure = state.pressure[:, 0]
    pressure_force = np.sum(wall_pressure[:, None] * area, axis=0)

    traction, _ = wall_shear_stress(state, faces, fluid)
    viscous_force = np.sum(traction * length[:, None], axis=0)

    lever = faces.wall.centre - moment_reference
    element = wall_pressure[:, None] * area + traction * length[:, None]
    moment = float(np.sum(lever[:, 0] * element[:, 1] - lever[:, 1] * element[:, 0]))

    return Forces(
        pressure_force=pressure_force,
        viscous_force=viscous_force,
        moment=moment,
        reference_length=reference_length,
        dynamic_pressure=freestream.dynamic_pressure(fluid),
    )


def surface_data(
    state: State, faces: FaceGeometry, fluid: Fluid, freestream: Freestream
) -> SurfaceData:
    """Pressure coefficient, skin friction and ``y+`` along the surface."""
    dynamic = freestream.dynamic_pressure(fluid)
    centre = faces.wall.centre

    _, shear = wall_shear_stress(state, faces, fluid)
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
        pressure_coefficient=state.pressure[:, 0] / dynamic,
        skin_friction_coefficient=shear / dynamic,
        y_plus=y_plus,
        wall_shear=shear,
    )


def vorticity(state: State, gradient) -> np.ndarray:
    """``dv/dx - du/dy``, for visualising shear layers and the wake."""
    grad_u = gradient(state.u, state.u[:, 0] * 0.0, state.u[:, -1])
    grad_v = gradient(state.v, state.v[:, 0] * 0.0, state.v[:, -1])
    return grad_v[..., 0] - grad_u[..., 1]


def separation_points(
    state: State, faces: FaceGeometry, fluid: Fluid
) -> np.ndarray:
    """Surface positions where the wall shear changes sign.

    Separation is where the near-wall flow reverses, so the tangential component
    of the wall traction along the surface passes through zero. Interpolating
    between the two faces either side locates it to better than one cell.
    """
    traction, _ = wall_shear_stress(state, faces, fluid)
    centre = faces.wall.centre

    tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    along = np.sum(traction * tangent, axis=-1)

    following = np.roll(along, -1)
    crossing = np.flatnonzero(np.sign(along) != np.sign(following))
    if len(crossing) == 0:
        return np.empty((0, 2))

    weight = along[crossing] / (along[crossing] - following[crossing])
    return centre[crossing] + weight[:, None] * (
        np.roll(centre, -1, axis=0)[crossing] - centre[crossing]
    )
