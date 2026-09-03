"""A complete case: geometry, mesh, physics and the iteration that solves it.

This is the object the GUI drives and the validation scripts import. Everything
below it is stateless machinery; everything about a particular run lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fluidsolver.geometry.contour import Contour
from fluidsolver.mesh import spacing
from fluidsolver.mesh.metrics import compute_metrics
from fluidsolver.mesh.ogrid import OGrid, build_ogrid
from fluidsolver.mesh.quality import QualityReport, assess
from fluidsolver.solver import post
from fluidsolver.solver.bc import Boundaries
from fluidsolver.solver.faces import build_faces
from fluidsolver.solver.fields import History, Residuals, State
from fluidsolver.solver.fluid import Fluid, Freestream
from fluidsolver.solver.simple import CflRamp, Numerics, PressureVelocityCoupling
from fluidsolver.solver.turbulence import MODELS

# Under-relaxation is eased in over the opening iterations. A cold start puts a
# uniform freestream right up against a no-slip wall, which is a discontinuity;
# the first few corrections to it are large and unrepresentative, and taking them
# at full strength is the commonest way for a case to diverge in its first ten
# iterations and never recover.
_RAMP_ITERATIONS = 100
_RAMP_INITIAL_FRACTION = 0.35


@dataclass
class MeshSettings:
    """How to turn a contour into a mesh."""

    surface_points: int = 240
    target_y_plus: float = 1.0
    far_field_radius_ratio: float = 40.0
    growth: float = 1.15
    transition_distance: float | None = None
    first_layer: float | None = None

    def far_field_radius(self, reference_length: float) -> float:
        return self.far_field_radius_ratio * reference_length

    def resolve_first_layer(
        self, fluid: Fluid, freestream: Freestream, reference_length: float, laminar: bool
    ) -> float:
        """Wall-adjacent cell thickness, from whichever rule applies.

        An explicit value wins. Otherwise the sizing follows the physics being
        solved: a turbulent case is sized by ``y+``, so that the first cell centre
        lands inside the viscous sublayer where the SST wall condition is valid; a
        laminar case is sized by cell count across the Blasius layer, because
        ``y+`` is a turbulent correlation and produces nonsense at low Reynolds
        number.
        """
        if self.first_layer is not None:
            return self.first_layer

        arguments = (
            freestream.velocity, reference_length, fluid.density, fluid.viscosity
        )
        if laminar:
            return spacing.laminar_first_layer(*arguments)
        return spacing.first_layer_thickness(self.target_y_plus, *arguments)


@dataclass
class Case:
    """A meshed, configured, runnable case."""

    grid: OGrid
    fluid: Fluid
    freestream: Freestream
    numerics: Numerics = field(default_factory=Numerics)
    model_name: str = "k-omega-sst"
    moment_reference: np.ndarray | None = None

    def __post_init__(self):
        if self.model_name not in MODELS:
            raise ValueError(
                f"unknown turbulence model {self.model_name!r}; "
                f"expected one of {sorted(MODELS)}"
            )

        self.metrics = compute_metrics(self.grid.nodes)
        self.quality: QualityReport = assess(self.metrics, self.grid.nodes)
        if not self.quality.is_usable:
            raise ValueError(
                "the mesh has inverted cells and cannot be solved on:\n"
                + self.quality.summary()
            )

        self.faces = build_faces(self.metrics)
        self.boundaries = Boundaries(self.faces, self.fluid, self.freestream)
        self.coupling = PressureVelocityCoupling(
            self.faces,
            self.fluid,
            self.boundaries,
            self.numerics,
            reference_length=self.reference_length,
        )
        self.model = MODELS[self.model_name](
            self.faces,
            self.fluid,
            self.boundaries,
            self.numerics,
            reference_length=self.reference_length,
        )

        if self.moment_reference is None:
            self.moment_reference = self.grid.contour.centroid

        self.state = State.uniform(self.faces, self.fluid, self.freestream)
        self.history = History()
        self.iteration = 0
        self.cfl_ramp = CflRamp(self.numerics)

        # Laminar runs carry no eddy viscosity; establish that before the first
        # momentum assembly rather than leaving the freestream estimate in place.
        self.model.update(self.state)

    # ------------------------------------------------------------------

    @property
    def reference_length(self) -> float:
        return self.grid.contour.reference_length

    @property
    def reynolds(self) -> float:
        return self.fluid.reynolds(self.freestream.velocity, self.reference_length)

    def _relaxation_scale(self) -> float:
        """Ramp factor applied to every relaxation factor early on.

        Off when pseudo-transient continuation is running: the CFL ramp already
        eases the opening iterations, and easing the same thing twice from two
        controllers that cannot see each other is how a run ends up crawling for
        reasons nobody can attribute.
        """
        if self.numerics.pseudo_transient:
            return 1.0
        if self.iteration >= _RAMP_ITERATIONS:
            return 1.0
        progress = self.iteration / _RAMP_ITERATIONS
        return _RAMP_INITIAL_FRACTION + (1.0 - _RAMP_INITIAL_FRACTION) * progress

    def step(self) -> Residuals:
        """Advance one SIMPLE outer iteration."""
        if self.numerics.pseudo_transient:
            self.coupling.cfl = self.cfl_ramp.value
            self.model.cfl = self.cfl_ramp.value
        scale = self._relaxation_scale()
        original = (
            self.numerics.relax_velocity,
            self.numerics.relax_pressure,
            self.numerics.relax_turbulence,
        )
        if scale < 1.0:
            self.numerics.relax_velocity *= scale
            self.numerics.relax_pressure *= scale
            self.numerics.relax_turbulence *= scale

        try:
            residual_u, residual_v, residual_p, continuity = self.coupling.iterate(
                self.state
            )
            residual_k, residual_omega = self.model.update(self.state)
        finally:
            (
                self.numerics.relax_velocity,
                self.numerics.relax_pressure,
                self.numerics.relax_turbulence,
            ) = original

        if not self.state.is_finite():
            raise FloatingPointError(
                f"the solution went non-finite at iteration {self.iteration}. "
                "Reduce the relaxation factors, or check the mesh quality report."
            )

        forces = self.forces()
        self.iteration += 1
        residuals = Residuals(
            iteration=self.iteration,
            u=residual_u,
            v=residual_v,
            pressure=residual_p,
            continuity=continuity,
            k=residual_k,
            omega=residual_omega,
            lift_coefficient=forces.lift_coefficient,
            drag_coefficient=forces.drag_coefficient,
        )
        self.history.append(residuals)
        self.cfl_ramp.update(residuals.worst)
        return residuals

    def run(
        self,
        *,
        max_iterations: int | None = None,
        callback: Callable[[Residuals], bool | None] | None = None,
        report_every: int = 1,
    ) -> History:
        """Iterate to convergence, or until told to stop.

        ``callback`` is invoked every ``report_every`` iterations with the latest
        residuals; returning ``False`` from it stops the run, which is how the GUI
        implements its stop button without reaching into the solver.
        """
        limit = max_iterations or self.numerics.max_iterations

        for _ in range(limit):
            residuals = self.step()

            if callback is not None and self.iteration % report_every == 0:
                if callback(residuals) is False:
                    break

            if residuals.has_converged(self.numerics.tolerance):
                break

        return self.history

    # ------------------------------------------------------------------

    def forces(self) -> post.Forces:
        return post.compute_forces(
            self.state,
            self.faces,
            self.fluid,
            self.freestream,
            self.reference_length,
            self.moment_reference,
        )

    def surface(self) -> post.SurfaceData:
        return post.surface_data(self.state, self.faces, self.fluid, self.freestream)

    def separation_points(self) -> np.ndarray:
        return post.separation_points(self.state, self.faces, self.fluid)

    def summary(self) -> str:
        forces = self.forces()
        surface = self.surface()
        lines = [
            f"body            {self.grid.contour.name}",
            f"mesh            {self.grid.shape[0]} x {self.grid.shape[1]}"
            f" = {self.grid.n_cells} cells",
            f"model           {self.model.name}",
            f"Re              {self.reynolds:.3e}",
            f"iterations      {self.iteration}",
            f"Cl              {forces.lift_coefficient:+.5f}",
            f"Cd              {forces.drag_coefficient:+.6f}"
            f"   (pressure {forces.pressure_drag_coefficient:+.6f},"
            f" friction {forces.friction_drag_coefficient:+.6f})",
            f"Cm              {forces.moment_coefficient:+.5f}",
            f"y+              {surface.y_plus.min():.3f} .. {surface.y_plus.max():.3f}",
        ]
        if self.history.entries:
            lines.append(f"final residual  {self.history.entries[-1].worst:.3e}")
        return "\n".join(lines)


def build_case(
    contour: Contour,
    fluid: Fluid,
    freestream: Freestream,
    *,
    mesh_settings: MeshSettings | None = None,
    numerics: Numerics | None = None,
    model_name: str = "k-omega-sst",
    moment_reference: np.ndarray | None = None,
) -> Case:
    """Mesh a body and assemble a case around it.

    Angle of attack is applied here, by rotating the body by ``-alpha``. Tilting
    the freestream instead would leave the circular far-field boundary at an angle
    to the flow for no benefit, and would tilt every plot. Rotating the body keeps
    the freestream along ``+x``, so lift is the ``y`` force and drag the ``x``
    force with no further resolution.
    """
    mesh_settings = mesh_settings or MeshSettings()
    numerics = numerics or Numerics()

    body = contour
    if freestream.angle_of_attack_deg != 0.0:
        pivot = np.array([contour.bounds[0], contour.centroid[1]])
        body = contour.rotated(-freestream.angle_of_attack_deg, about=pivot)

    surface = body.resample(mesh_settings.surface_points)
    reference_length = contour.reference_length

    first_layer = mesh_settings.resolve_first_layer(
        fluid, freestream, reference_length, laminar=model_name == "laminar"
    )
    grid = build_ogrid(
        surface,
        first_layer=first_layer,
        far_field_radius=mesh_settings.far_field_radius(reference_length),
        growth=mesh_settings.growth,
        transition_distance=mesh_settings.transition_distance,
    )

    return Case(
        grid=grid,
        fluid=fluid,
        freestream=freestream,
        numerics=numerics,
        model_name=model_name,
        moment_reference=moment_reference,
    )
