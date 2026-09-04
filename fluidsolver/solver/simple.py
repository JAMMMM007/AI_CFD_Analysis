"""SIMPLE: the pressure-velocity coupling.

Incompressible flow has no equation for pressure. Continuity constrains the
velocity but contains no pressure at all, and pressure appears in the momentum
equations only through its gradient. SIMPLE resolves this by deriving a pressure
equation from the two together: guess a pressure, solve momentum with it, measure
how badly the resulting velocities violate continuity, and solve for the pressure
correction that removes the imbalance.

    1. solve momentum with the current pressure           -> u*, v*
    2. build face mass fluxes from u*, v* (Rhie-Chow)     -> F*
    3. solve for p' such that correcting F* by it conserves mass
    4. correct p, u, v and F
    5. solve the turbulence model
    6. repeat

Two details make the difference between this working and not.

**Rhie-Chow interpolation.** Velocity and pressure both live at cell centres. A
face velocity interpolated straight from the cells is insensitive to a pressure
that alternates cell to cell, so nothing in the discrete system penalises a
checkerboard pressure field, and one duly appears. Rhie-Chow builds the face flux
from a *compact* two-cell pressure difference instead of the interpolated
gradient. The difference between the two acts as a third-derivative damping that
vanishes with mesh refinement but couples adjacent cells directly.

**Implicit under-relaxation.** The pressure correction derived here is not exact:
it neglects the effect of neighbouring velocity corrections. Taking it at face
value diverges. Relaxing it, along with the momentum solutions, is what makes the
iteration contract -- and because the relaxation is applied in Patankar's implicit
form, it changes only the path taken, never the converged answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluidsolver.solver import operators as ops
from fluidsolver.solver.bc import Boundaries
from fluidsolver.solver.faces import FaceGeometry
from fluidsolver.solver.fields import State
from fluidsolver.solver.fluid import Fluid
from fluidsolver.solver.linalg import (
    Coefficients,
    StructuredMatrix,
    incomplete_lu_preconditioner,
    jacobi_preconditioner,
    solve,
)


@dataclass
class Numerics:
    """Discretisation and iteration settings."""

    # Bounded second order, not plain central differencing.
    #
    # Central differencing is unbounded above a cell Peclet number of 2, and an
    # external aerodynamics mesh is nowhere near that outside the boundary
    # layer: measured on a 240x99 cylinder mesh at Re = 2e6 the median cell
    # Peclet number is 3.2e4 and the peak 2.6e7. With ``linear`` the deferred
    # correction there has nothing holding it, and a sawtooth seeded at the
    # marched-to-analytic mesh seam grows until the run dies -- on that case, at
    # iteration 459, having already passed through Cl = 37525. The van Leer
    # limiter costs a little accuracy where the flow is smooth and well resolved,
    # and is the difference between an answer and a NaN everywhere else.
    scheme: str = "limited_linear"
    turbulence_scheme: str = "upwind"
    relax_velocity: float = 0.7
    relax_pressure: float = 0.3
    relax_turbulence: float = 0.7
    relax_eddy_viscosity: float = 0.4
    inner_tolerance: float = 0.1
    pressure_inner_tolerance: float = 0.01
    max_iterations: int = 3000
    tolerance: float = 1e-6

    # Pseudo-transient continuation: a local time step damping each cell by its
    # own convective transit time instead of by a global relaxation factor.
    #
    # Off by default, on the evidence. It was added expecting a robustness win
    # and does not deliver one for a *steady* segregated solver. Measured:
    #
    #   NACA 2412, 5 deg attached, iterations to reach 1e-6
    #       relaxation only 0.7/0.3                    1010
    #       pseudo-transient, cfl_max 10 / 30 / 100 / 300   1155 / 1060 / 1032 / 1023
    #
    #   cylinder Re 2.03e6, SST, 800 iterations, final residual
    #       relaxation only 0.7/0.3                    3.47e-02
    #       relaxation only 0.5/0.2 (hand-tuned)       1.57e-02
    #       pseudo-transient 0.7/0.3                   4.00e-02
    #
    # It converges towards plain relaxation as the ceiling rises and never past
    # it, and on the bluff body it is worse than doing nothing. The reason is
    # structural rather than a tuning miss: the step is convective only (see
    # ops.pseudo_time_diagonal for why it must be), which leaves the near-wall
    # cells essentially undamped -- and on a cylinder the trouble is the shedding
    # mode working through the boundary layer, precisely the region this declines
    # to touch. Nor can it stand alone: relax_velocity = 1.0 diverged at every
    # ceiling tried, down to cfl_max = 2.
    #
    # It is kept because Stage 4 needs exactly this term with a global physical
    # time step in place of the local pseudo one, and because it is verified: the
    # fixed point is preserved to 0.0055% in Cl at a residual of 1e-9.
    pseudo_transient: bool = False
    cfl: float = 1.0
    cfl_max: float = 100.0
    cfl_growth: float = 1.03

    def __post_init__(self):
        for name in ("scheme", "turbulence_scheme"):
            value = getattr(self, name)
            if value not in ops.SCHEMES:
                raise ValueError(
                    f"unknown {name} {value!r}; expected one of {ops.SCHEMES}"
                )
        for name in (
            "relax_velocity", "relax_pressure",
            "relax_turbulence", "relax_eddy_viscosity",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        for name in ("cfl", "cfl_max"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.cfl_growth < 1.0:
            raise ValueError(f"cfl_growth must be at least 1, got {self.cfl_growth}")
        if self.cfl > self.cfl_max:
            raise ValueError(
                f"cfl {self.cfl:g} already exceeds cfl_max {self.cfl_max:g}"
            )


#: Iterations averaged over when deciding whether the run is still settling. One
#: iteration says nothing -- residuals rattle by a factor of two on a healthy run
#: -- so the comparison is between the geometric means of two adjacent windows.
_RAMP_WINDOW = 25
#: How far the trailing mean may rise before the step is judged too large.
_BACKOFF_TRIGGER = 2.0
#: What a back-off does to the CFL, and how low it may go before the run is
#: simply not converging and something other than the step size is wrong.
_BACKOFF_FACTOR = 0.5
_CFL_MIN = 0.05


class CflRamp:
    """Raises the pseudo-time CFL while the run settles; drops it when it stops.

    This is the part that replaces the user guessing a relaxation factor. The
    step starts small, because the opening iterations of a cold start are the
    least representative -- a uniform freestream against a no-slip wall is a
    discontinuity, and the first corrections to it are large and meaningless --
    and grows geometrically while the residual keeps falling. If the residual
    turns and climbs, the step was too big and is halved.

    Nothing here changes the converged answer. Every CFL only scales a term that
    vanishes at convergence, so the ramp changes the path taken and cannot move
    the fixed point.
    """

    def __init__(self, numerics: Numerics):
        self.numerics = numerics
        self.value = numerics.cfl
        self.backoffs = 0
        self._history: list[float] = []
        self._hold = 0

    def update(self, residual: float) -> None:
        """Fold in one iteration's worst residual and adjust the step."""
        if not np.isfinite(residual) or residual <= 0.0:
            return
        self._history.append(residual)
        if len(self._history) > 2 * _RAMP_WINDOW:
            self._history.pop(0)

        if self._hold > 0:
            self._hold -= 1
        elif self._rising():
            self.value = max(self.value * _BACKOFF_FACTOR, _CFL_MIN)
            self.backoffs += 1
            # Let the run answer the smaller step before judging it again.
            self._hold = 2 * _RAMP_WINDOW
            self._history.clear()
            return

        self.value = min(
            self.value * self.numerics.cfl_growth, self.numerics.cfl_max
        )

    def _rising(self) -> bool:
        """Whether the trailing window is worse than the one before it."""
        if len(self._history) < 2 * _RAMP_WINDOW:
            return False
        recent = np.log(self._history[-_RAMP_WINDOW:]).mean()
        earlier = np.log(self._history[:_RAMP_WINDOW]).mean()
        return recent - earlier > np.log(_BACKOFF_TRIGGER)


class PressureVelocityCoupling:
    """One SIMPLE outer iteration, holding everything reusable between them."""

    def __init__(
        self,
        faces: FaceGeometry,
        fluid: Fluid,
        boundaries: Boundaries,
        numerics: Numerics,
        reference_length: float = 1.0,
        wall_model: bool = True,
    ):
        self.faces = faces
        self.fluid = fluid
        self.boundaries = boundaries
        self.numerics = numerics
        self.reference_length = reference_length
        #: Whether a turbulent wall model applies. False for a laminar run,
        #: where the first cell is sized from the Blasius thickness rather than
        #: a y+ target, so the profile through it really is linear.
        self.wall_model = wall_model
        #: Current CFL number. Ramped by the owner of the run, so that it can be
        #: raised as the solution settles and dropped again if it stops settling.
        self.cfl = numerics.cfl
        self.gradient = ops.Gradient(faces)
        self.matrix = StructuredMatrix(faces.shape)
        self.volume = faces.metrics.volume

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def momentum(self, state: State) -> tuple[Coefficients, Coefficients, np.ndarray]:
        """Assemble both momentum components and return them with the diagonal.

        The diagonal is needed afterwards for Rhie-Chow and for the velocity
        correction, so it is returned rather than recomputed. Both components
        share it: they differ only in their sources.
        """
        far_flux = state.flux_j[:, -1]
        wall_u, wall_v = self.boundaries.wall_velocity()
        far_u, far_v = self.boundaries.far_velocity(state.u, state.v, far_flux)
        inflow = self.boundaries.inflow_mask(far_flux)

        viscosity = self.fluid.viscosity + state.eddy_viscosity
        grad_u = self.gradient(state.u, wall_u, far_u)
        grad_v = self.gradient(state.v, wall_v, far_v)

        wall_pressure = state.pressure[:, 0]
        far_pressure = self.boundaries.far_pressure(state.pressure, far_flux)
        grad_p = self.gradient(state.pressure, wall_pressure, far_pressure)

        transpose = self._transpose_stress(viscosity, grad_u, grad_v)

        # The wall face carries a viscosity chosen to deliver the blended wall
        # shear rather than the molecular one. On a y+ ~ 1 mesh the two coincide,
        # so nothing changes there; on a coarser mesh the linear profile the
        # diffusive flux assumes is wrong and this is what corrects it.
        #
        # ``None`` for a laminar run, which falls back to the molecular value.
        # A friction velocity and a log law describe a turbulent boundary layer
        # and a laminar case has neither, so applying the blend there has no
        # basis -- and it is not harmless: the fourth-power blend always exceeds
        # the larger of its two branches, so it returned slightly more than
        # molecular even where the viscous branch was exact, and moved the
        # Re 40 cylinder's converged residual in its third digit.
        wall_viscosity = (
            self.boundaries.wall_viscosity(state.u, state.v)
            if self.wall_model
            else None
        )

        components = []
        for field, gradient, wall, far, index in (
            (state.u, grad_u, wall_u, far_u, 0),
            (state.v, grad_v, wall_v, far_v, 1),
        ):
            coefficients = Coefficients.zeros(self.faces.shape)
            ops.add_convection(
                coefficients, self.faces, state.flux_i, state.flux_j,
                field, gradient,
                far_field_value=far, scheme=self.numerics.scheme,
            )
            ops.add_diffusion(
                coefficients, self.faces, viscosity, gradient,
                wall_value=wall, far_field_value=far,
                wall_diffusivity=wall_viscosity,
                far_field_active=inflow,
            )
            # Pressure gradient and the transpose half of the viscous stress.
            coefficients.source += (
                -grad_p[..., index] + transpose[index]
            ) * self.volume
            components.append(coefficients)

        # Pseudo-time first, then relaxation. The two are alternatives rather
        # than partners -- relaxation defaults to 1 when pseudo-transient is on --
        # but the order matters if both are used: relaxing afterwards scales the
        # pseudo-time term with everything else, which keeps the effective step
        # consistent instead of leaving two damping mechanisms disagreeing.
        pseudo_time = self.pseudo_time_diagonal(state)
        for coefficients, field in zip(components, (state.u, state.v)):
            if pseudo_time is not None:
                coefficients.add_pseudo_time(field, pseudo_time)
            coefficients.under_relax(field, self.numerics.relax_velocity)

        # The relaxed diagonal is what the matrix actually contains, so it is the
        # one the velocity correction must divide by for the two to be consistent.
        return components[0], components[1], components[0].centre

    def pseudo_time_diagonal(self, state: State) -> np.ndarray | None:
        """``rho V / dtau`` at the current CFL, or ``None`` if switched off.

        See :func:`fluidsolver.solver.operators.pseudo_time_diagonal` for why the
        step is convective only.
        """
        if not self.numerics.pseudo_transient:
            return None
        return ops.pseudo_time_diagonal(
            state.flux_i,
            state.flux_j,
            self.volume,
            density=self.fluid.density,
            velocity=self.boundaries.freestream.velocity,
            reference_length=self.reference_length,
            cfl=self.cfl,
        )

    def _transpose_stress(self, viscosity, grad_u, grad_v) -> tuple[np.ndarray, np.ndarray]:
        """``div(mu_eff grad(u)^T)``, the second half of the viscous stress.

        The full stress is ``mu (grad u + grad u^T)``. Only the first half is a
        Laplacian; the transpose is a source. For *constant* viscosity it can be
        dropped -- it collapses to ``mu grad(div u)``, which vanishes for
        incompressible flow -- but ``mu_eff`` is anything but constant here. It
        varies by orders of magnitude across the boundary layer, and that
        variation carries real stress precisely where the shear is largest.

        Expanding the divergence and using ``div u = 0`` leaves

            div(mu grad(u)^T)_m = sum_n (d mu / d x_n)(d u_n / d x_m)

        so it needs one extra gradient and no face sums at all. Written as a face
        sum instead it needs a boundary treatment on the wall and the far field,
        and the two halves of a ``j``-direction divergence are easy to transpose
        by accident -- which is exactly what happened the first time.
        """
        # mu_t vanishes at the wall, where k does. The far field is left
        # zero-gradient, which it effectively is that far out.
        wall_value = np.full(self.faces.shape[0], self.fluid.viscosity)
        grad_mu = self.gradient(viscosity, wall_value, viscosity[:, -1])

        return (
            grad_mu[..., 0] * grad_u[..., 0] + grad_mu[..., 1] * grad_v[..., 0],
            grad_mu[..., 0] * grad_u[..., 1] + grad_mu[..., 1] * grad_v[..., 1],
        )

    # ------------------------------------------------------------------
    # Rhie-Chow face fluxes
    # ------------------------------------------------------------------

    def face_fluxes(
        self, state: State, diagonal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mass fluxes with the Rhie-Chow pressure-velocity coupling.

        ``F = rho [ u_f . S  -  D_f ( (p_N - p_P) g  -  (grad p)_f . S ) ]``

        The bracketed difference is between the compact two-cell pressure
        gradient and the smoothly interpolated one. They agree for a smooth
        pressure field and differ sharply for a checkerboard, which is precisely
        the mode that has to be suppressed. The term is of order ``h^3``, so it
        disappears under refinement without biasing the answer.

        Also returns the two ``D`` coefficients the pressure equation needs.
        """
        far_flux = state.flux_j[:, -1]
        wall_u, wall_v = self.boundaries.wall_velocity()
        far_u, far_v = self.boundaries.far_velocity(state.u, state.v, far_flux)
        far_pressure = self.boundaries.far_pressure(state.pressure, far_flux)

        grad_p = self.gradient(state.pressure, state.pressure[:, 0], far_pressure)
        mobility = self.volume / diagonal

        # --- i faces (all interior, wrapping around the body) ---
        velocity = state.velocity
        interpolated = self.faces.i_faces.interpolate(
            velocity, np.roll(velocity, 1, axis=0)
        )
        d_i = self.faces.i_faces.interpolate(mobility, np.roll(mobility, 1, axis=0))
        compact = (
            state.pressure - np.roll(state.pressure, 1, axis=0)
        ) * self.faces.i_faces.diffusion_factor
        smooth = np.sum(
            self.faces.i_faces.interpolate(grad_p, np.roll(grad_p, 1, axis=0))
            * self.faces.metrics.face_i_area,
            axis=-1,
        )
        flux_i = self.fluid.density * (
            np.sum(interpolated * self.faces.metrics.face_i_area, axis=-1)
            - d_i * (compact - smooth)
        )

        # --- j faces (interior only; boundaries handled below) ---
        d_j = self.faces.j_faces.interpolate(mobility[:, 1:], mobility[:, :-1])
        interpolated_j = self.faces.j_faces.interpolate(velocity[:, 1:], velocity[:, :-1])
        compact_j = (
            state.pressure[:, 1:] - state.pressure[:, :-1]
        ) * self.faces.j_faces.diffusion_factor
        smooth_j = np.sum(
            self.faces.j_faces.interpolate(grad_p[:, 1:], grad_p[:, :-1])
            * self.faces.metrics.face_j_area[:, 1:-1],
            axis=-1,
        )
        interior_j = self.fluid.density * (
            np.sum(interpolated_j * self.faces.metrics.face_j_area[:, 1:-1], axis=-1)
            - d_j * (compact_j - smooth_j)
        )

        # Wall: impermeable. Far field: whatever the boundary velocity carries,
        # rescaled so the domain neither gains nor loses mass overall.
        wall_flux = np.zeros(self.faces.shape[0])
        far = self.fluid.density * (
            far_u * self.faces.far_field.area[:, 0]
            + far_v * self.faces.far_field.area[:, 1]
        )
        far = self.boundaries.enforce_global_mass_balance(far)

        flux_j = np.concatenate(
            (wall_flux[:, None], interior_j, far[:, None]), axis=1
        )
        return flux_i, flux_j, d_i, d_j

    # ------------------------------------------------------------------
    # Pressure correction
    # ------------------------------------------------------------------

    def pressure_correction(
        self,
        state: State,
        flux_i: np.ndarray,
        flux_j: np.ndarray,
        d_i: np.ndarray,
        d_j: np.ndarray,
        diagonal: np.ndarray,
    ) -> tuple[np.ndarray, Coefficients]:
        """Solve for the pressure correction that restores continuity."""
        density = self.fluid.density
        coefficients = Coefficients.zeros(self.faces.shape)

        coupling_i = density * d_i * self.faces.i_faces.diffusion_factor
        coefficients.centre += coupling_i + np.roll(coupling_i, -1, axis=0)
        coefficients.west -= coupling_i
        coefficients.east -= np.roll(coupling_i, -1, axis=0)

        coupling_j = density * d_j * self.faces.j_faces.diffusion_factor
        coefficients.centre[:, 1:] += coupling_j
        coefficients.centre[:, :-1] += coupling_j
        coefficients.south[:, 1:] -= coupling_j
        coefficients.north[:, :-1] -= coupling_j

        # Where the far field holds the pressure at zero, the boundary face
        # contributes a coupling to a known value -- diagonal only. Where it
        # holds the velocity instead, the flux there is already fixed and the
        # correction through that face is zero.
        coefficients.centre[:, -1] += self._far_field_coupling(flux_j, diagonal)

        coefficients.source = -ops.divergence(flux_i, flux_j, self.faces)

        matrix = self.matrix.build(coefficients)
        correction, _ = solve(
            matrix,
            coefficients.source,
            np.zeros(self.faces.shape),
            tolerance=self.numerics.pressure_inner_tolerance,
            max_iterations=400,
            preconditioner=incomplete_lu_preconditioner(matrix),
        )
        return correction, coefficients

    def _far_field_coupling(
        self, flux_j: np.ndarray, diagonal: np.ndarray
    ) -> np.ndarray:
        """``rho D g`` on the far-field faces that hold the pressure, zero elsewhere.

        This is the one term that appears in both halves of the pressure step: it
        is the diagonal entry :meth:`pressure_correction` adds for an outflow
        face, and it is the flux correction :meth:`apply_correction` has to apply
        through that same face for the two to describe the same thing. It lives
        here so they cannot disagree -- which they did. The matrix asserted a
        correction of ``rho D g p'`` leaving through every fixed-pressure face and
        the flux update never made it, so the outer ring of cells was left holding
        exactly that imbalance after every iteration. Measured on a NACA 0012:
        62% of all the mass imbalance left after the pressure correction sat in
        that single row of cells, correlating with the missing term at -0.9996.
        """
        return np.where(
            self.boundaries.far_pressure_is_fixed(flux_j[:, -1]),
            self.fluid.density
            * (self.volume[:, -1] / diagonal[:, -1])
            * self.faces.far_field.diffusion_factor,
            0.0,
        )

    def apply_correction(
        self,
        state: State,
        correction: np.ndarray,
        flux_i: np.ndarray,
        flux_j: np.ndarray,
        d_i: np.ndarray,
        d_j: np.ndarray,
        diagonal: np.ndarray,
    ) -> None:
        """Update pressure, velocity and fluxes with the correction, in place.

        The fluxes are corrected with the same compact operator that built the
        pressure equation, so continuity is satisfied to solver tolerance
        immediately -- the far-field faces that hold the pressure included, via
        :meth:`_far_field_coupling`. The cell velocities are corrected with the
        smooth gradient instead: they are cell quantities, and using the compact
        form on them would reintroduce the decoupling Rhie-Chow just removed.
        """
        density = self.fluid.density
        fixed = self.boundaries.far_pressure_is_fixed(flux_j[:, -1])

        state.pressure += self.numerics.relax_pressure * correction

        correction_gradient = self.gradient(
            correction,
            correction[:, 0],
            np.where(fixed, 0.0, correction[:, -1]),
        )
        mobility = self.volume / diagonal
        state.u -= mobility * correction_gradient[..., 0]
        state.v -= mobility * correction_gradient[..., 1]

        state.flux_i = flux_i - density * d_i * self.faces.i_faces.diffusion_factor * (
            correction - np.roll(correction, 1, axis=0)
        )
        state.flux_j = flux_j.copy()
        state.flux_j[:, 1:-1] -= (
            density
            * d_j
            * self.faces.j_faces.diffusion_factor
            * (correction[:, 1:] - correction[:, :-1])
        )
        # The far field holds p' at zero where it holds the pressure, so the
        # correction through that face is outward and proportional to p' in the
        # cell inside it. Where it holds the velocity instead the coupling is
        # zero, and the flux there stays exactly what the boundary condition set.
        state.flux_j[:, -1] = flux_j[:, -1] + self._far_field_coupling(
            flux_j, diagonal
        ) * correction[:, -1]

    # ------------------------------------------------------------------
    # One outer iteration
    # ------------------------------------------------------------------

    def iterate(self, state: State) -> tuple[float, float, float, float]:
        """Advance the pressure-velocity coupling by one outer iteration.

        Returns the momentum, pressure and continuity residuals.
        """
        coefficients_u, coefficients_v, diagonal = self.momentum(state)

        residual_u = coefficients_u.residual(state.u)
        residual_v = coefficients_v.residual(state.v)

        for coefficients, name in ((coefficients_u, "u"), (coefficients_v, "v")):
            matrix = self.matrix.build(coefficients)
            value, _ = solve(
                matrix,
                coefficients.source,
                getattr(state, name),
                tolerance=self.numerics.inner_tolerance,
                preconditioner=jacobi_preconditioner(matrix),
            )
            setattr(state, name, value)

        flux_i, flux_j, d_i, d_j = self.face_fluxes(state, diagonal)
        imbalance = ops.divergence(flux_i, flux_j, self.faces)

        correction, pressure_coefficients = self.pressure_correction(
            state, flux_i, flux_j, d_i, d_j, diagonal
        )
        # Measured at the correction that was obtained, not at zero. At zero the
        # expression collapses to sum|b| / sum|b| and reports 1.000e+00 on every
        # iteration of every run -- which it did, for as long as this line read
        # ``residual(np.zeros(...))``. What is wanted here is how well the
        # pressure equation was solved; how far continuity still is from being
        # satisfied is the separate ``continuity`` figure below.
        residual_p = pressure_coefficients.residual(correction)

        self.apply_correction(
            state, correction, flux_i, flux_j, d_i, d_j, diagonal
        )

        # Continuity residual, scaled by the mass actually flowing through the
        # domain so that it reads as a fraction rather than as kg/s.
        reference = np.abs(state.flux_j[:, -1]).sum()
        continuity = float(
            np.abs(imbalance).sum() / (reference if reference > 0.0 else 1.0)
        )
        return residual_u, residual_v, residual_p, continuity
