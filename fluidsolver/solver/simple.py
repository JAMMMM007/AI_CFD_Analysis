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

    scheme: str = "linear"
    turbulence_scheme: str = "upwind"
    relax_velocity: float = 0.7
    relax_pressure: float = 0.3
    relax_turbulence: float = 0.7
    relax_eddy_viscosity: float = 0.4
    inner_tolerance: float = 0.1
    pressure_inner_tolerance: float = 0.01
    max_iterations: int = 3000
    tolerance: float = 1e-6

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


class PressureVelocityCoupling:
    """One SIMPLE outer iteration, holding everything reusable between them."""

    def __init__(
        self,
        faces: FaceGeometry,
        fluid: Fluid,
        boundaries: Boundaries,
        numerics: Numerics,
    ):
        self.faces = faces
        self.fluid = fluid
        self.boundaries = boundaries
        self.numerics = numerics
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
                far_field_active=inflow,
            )
            # Pressure gradient and the transpose half of the viscous stress.
            coefficients.source += (
                -grad_p[..., index] + transpose[index]
            ) * self.volume
            components.append(coefficients)

        for coefficients, field in zip(components, (state.u, state.v)):
            coefficients.under_relax(field, self.numerics.relax_velocity)

        # The relaxed diagonal is what the matrix actually contains, so it is the
        # one the velocity correction must divide by for the two to be consistent.
        return components[0], components[1], components[0].centre

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
