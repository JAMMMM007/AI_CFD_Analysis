"""Menter's k-omega SST model, 2003 revision.

Two turbulence models were in wide use before this one and each failed in a way
the other did not. The ``k-omega`` model resolves the viscous sublayer without
wall functions and handles adverse pressure gradients well, but its freestream
value of ``omega`` is arbitrary and the answer depends on it -- unacceptable for
an external flow, where most of the domain *is* freestream. The ``k-epsilon``
model is insensitive to the freestream but needs wall functions and predicts
separation late, which for an aerofoil is the one thing that matters.

SST is the resolution: run ``k-omega`` near the wall, ``k-epsilon`` (rewritten in
``omega`` form) away from it, and blend between them with a function ``F1`` built
from the wall distance. The "SST" part is separate and just as important -- a
limiter on the eddy viscosity that enforces Bradshaw's observation that the shear
stress in a boundary layer is proportional to ``k``. Without it the model
over-predicts eddy viscosity in adverse pressure gradients and separation is
delayed all over again.

The transported equations:

    D(rho k)/Dt     = P~_k - beta* rho k omega
                      + div[(mu + sigma_k mu_t) grad k]

    D(rho omega)/Dt = gamma rho S^2 - beta rho omega^2
                      + div[(mu + sigma_w mu_t) grad omega]
                      + 2(1 - F1) rho sigma_w2 (1/omega) grad k . grad omega

    mu_t = rho a1 k / max(a1 omega, S F2)

The final term in the ``omega`` equation is the cross-diffusion that appears when
the ``k-epsilon`` model is transformed into ``omega`` form; ``(1 - F1)`` switches
it off near the wall, where the model is meant to behave as ``k-omega``.
"""

from __future__ import annotations

import numpy as np

from fluidsolver.solver import operators as ops
from fluidsolver.solver.fields import State
from fluidsolver.solver.linalg import (
    Coefficients,
    StructuredMatrix,
    jacobi_preconditioner,
    solve,
)
from fluidsolver.solver.turbulence.base import TurbulenceModel

# Shared constants.
BETA_STAR = 0.09
A1 = 0.31
KAPPA = 0.41

# Inner (k-omega) and outer (k-epsilon) constant sets. The blended value of any
# constant is F1 * inner + (1 - F1) * outer.
SIGMA_K1, SIGMA_W1, BETA_1 = 0.85, 0.5, 0.075
SIGMA_K2, SIGMA_W2, BETA_2 = 1.0, 0.856, 0.0828

# gamma (Menter's alpha) is not independent: it follows from requiring the model
# to reproduce the log layer, gamma = beta/beta* - sigma_w kappa^2 / sqrt(beta*),
# which gives 0.5532 and 0.4404. Menter 2003 states the rounded 5/9 and 0.44
# instead, and those are the numbers the model was calibrated and published with,
# so they are what is used here. The derivation is left above because it is where
# they come from and because it is what fixes them if any of beta, sigma_w or
# kappa is ever changed.
GAMMA_1 = 5.0 / 9.0
GAMMA_2 = 0.44

# Menter's production limiter, which stops k running away at a stagnation point
# where the strain rate is large but the turbulence is not.
PRODUCTION_LIMIT = 10.0

# Floors, as fractions of the freestream values. k and omega are physically
# positive and the discretisation does not guarantee it, so the result is
# clipped. The omega floor matters most: mu_t goes as k/omega, so omega drifting
# to zero produces an unbounded eddy viscosity and takes momentum with it.
#
# Convection of both fields defaults to first-order upwind (Numerics.
# turbulence_scheme). That is not laziness. Near a wall omega spans six orders of
# magnitude over a handful of cells, and a deferred correction is an *explicit*
# source proportional to that spread: measured on a NACA 0012 at y+ = 1, even a
# limiter-bounded correction came out at thirty times the production term and two
# hundred thousand times the diagonal, and the equation diverged within forty
# iterations. Turbulence quantities tolerate numerical diffusion far better than
# momentum does, and the model constants were calibrated against schemes of
# exactly this kind.
K_FLOOR_FRACTION = 1e-10
OMEGA_FLOOR_FRACTION = 1e-5

# Ceiling on mu_t / mu, as a safety net against a transient excursion.
MAX_VISCOSITY_RATIO = 1e5


class KOmegaSST(TurbulenceModel):
    """The k-omega SST closure."""

    name = "k-omega-sst"

    def __init__(self, faces, fluid, boundaries, numerics, reference_length=1.0):
        super().__init__(faces, fluid, boundaries, numerics, reference_length)
        self.gradient = ops.Gradient(faces)
        self.matrix = StructuredMatrix(faces.shape)
        self.volume = faces.metrics.volume
        self.wall_distance = np.maximum(faces.metrics.wall_distance, 1e-300)

        freestream = boundaries.freestream
        self._k_floor = K_FLOOR_FRACTION * freestream.turbulent_kinetic_energy()
        self._omega_floor = OMEGA_FLOOR_FRACTION * freestream.specific_dissipation(fluid)

    # ------------------------------------------------------------------

    def update(self, state: State) -> tuple[float, float]:
        """Solve both transport equations and refresh the eddy viscosity."""
        far_flux = state.flux_j[:, -1]
        # Both wall conditions now depend on the velocity field: omega through
        # the friction velocity in its logarithmic branch, and k by being zero
        # flux rather than a fixed zero.
        wall_k, wall_omega = self.boundaries.wall_turbulence(state.u, state.v)
        far_k, far_omega = self.boundaries.far_turbulence(state.k, state.omega, far_flux)
        inflow = self.boundaries.inflow_mask(far_flux)

        # ``wall_k`` is None, which add_diffusion reads as zero flux. The
        # gradient operator says the same thing by being handed the adjacent
        # cell value: the difference across the face is then zero, which is
        # precisely what a vanishing normal gradient asserts.
        grad_k = self.gradient(state.k, state.k[:, 0], far_k)
        grad_omega = self.gradient(state.omega, wall_omega, far_omega)
        strain = self.strain_rate(state, self.gradient)

        blend = self._blending_f1(state, grad_k, grad_omega)
        cross_diffusion = self._cross_diffusion(state, grad_k, grad_omega, blend)

        residual_k = self._solve_k(
            state, strain, blend, grad_k, wall_k, far_k, inflow
        )
        residual_omega = self._solve_omega(
            state, strain, blend, cross_diffusion, grad_omega, wall_omega, far_omega, inflow
        )

        # Relax the eddy viscosity rather than replacing it outright. It is the
        # only channel through which turbulence reaches the momentum equations,
        # and it is the product of two fields that are themselves still moving, so
        # it swings much harder than either. Left unrelaxed the coupling sustains
        # a slow oscillation that looks converged for a few hundred iterations and
        # then grows. Being a relaxation, it changes only the path: at
        # convergence the new and old values coincide.
        updated = self._eddy_viscosity(state, strain)
        factor = self.numerics.relax_eddy_viscosity
        state.eddy_viscosity = (
            factor * updated + (1.0 - factor) * state.eddy_viscosity
        )
        return residual_k, residual_omega

    # ------------------------------------------------------------------
    # Blending and the eddy viscosity
    # ------------------------------------------------------------------

    def _blending_f1(self, state, grad_k, grad_omega) -> np.ndarray:
        """``F1``: one near the wall (use k-omega), zero outside (use k-epsilon).

        ``arg1`` takes the maximum of two near-wall length-scale ratios -- the
        turbulent one and the viscous one -- so that either can hold the switch
        on, then a minimum against a third term that forces ``F1`` to zero in a
        free shear layer, where the cross-diffusion is genuinely wanted. Raising
        to the fourth power inside the ``tanh`` makes the transition sharp.
        """
        distance = self.wall_distance
        viscosity = self.fluid.kinematic_viscosity
        omega = np.maximum(state.omega, self._omega_floor)
        k = np.maximum(state.k, 0.0)

        turbulent = np.sqrt(k) / (BETA_STAR * omega * distance)
        viscous = 500.0 * viscosity / (distance**2 * omega)

        # Positive cross-diffusion, floored so the ratio stays finite where
        # grad k and grad omega happen to be orthogonal.
        cross = np.maximum(
            2.0 * self.fluid.density * SIGMA_W2 * np.sum(grad_k * grad_omega, axis=-1)
            / omega,
            1e-10,
        )
        free_shear = 4.0 * self.fluid.density * SIGMA_W2 * k / (cross * distance**2)

        argument = np.minimum(np.maximum(turbulent, viscous), free_shear)
        return np.tanh(np.clip(argument, 0.0, 20.0) ** 4)

    def _blending_f2(self, state, strain) -> np.ndarray:
        """``F2``: one inside a boundary layer, zero outside it.

        Gates the shear-stress limiter, which must act where Bradshaw's relation
        holds -- in a wall boundary layer -- and not in a free shear flow.
        """
        distance = self.wall_distance
        omega = np.maximum(state.omega, self._omega_floor)
        turbulent = 2.0 * np.sqrt(np.maximum(state.k, 0.0)) / (
            BETA_STAR * omega * distance
        )
        viscous = 500.0 * self.fluid.kinematic_viscosity / (distance**2 * omega)
        return np.tanh(np.clip(np.maximum(turbulent, viscous), 0.0, 20.0) ** 2)

    def _eddy_viscosity(self, state, strain) -> np.ndarray:
        """``mu_t = rho a1 k / max(a1 omega, S F2)`` -- the SST limiter.

        The standard ``mu_t = rho k / omega`` is recovered wherever ``a1 omega``
        wins. Where the strain rate is large -- an adverse pressure gradient
        approaching separation -- the second argument takes over and caps the
        eddy viscosity at ``rho a1 k / S``, which is exactly the statement that
        the shear stress cannot exceed ``a1`` times the turbulent kinetic energy.
        This single term is the difference between predicting separation and
        missing it.
        """
        k = np.maximum(state.k, 0.0)
        omega = np.maximum(state.omega, self._omega_floor)
        denominator = np.maximum(A1 * omega, strain * self._blending_f2(state, strain))
        viscosity = self.fluid.density * A1 * k / np.maximum(denominator, 1e-300)

        # A safety cap, not part of the model. Real turbulent eddy viscosities in
        # external aerodynamics reach a few thousand times the molecular value;
        # anything approaching this ceiling means the transport equations have
        # transiently misbehaved, and letting that feed back into momentum turns
        # a recoverable excursion into a divergence.
        return np.minimum(viscosity, MAX_VISCOSITY_RATIO * self.fluid.viscosity)

    def _cross_diffusion(self, state, grad_k, grad_omega, blend) -> np.ndarray:
        """``2 (1 - F1) rho sigma_w2 (1/omega) grad k . grad omega``."""
        omega = np.maximum(state.omega, self._omega_floor)
        return (
            2.0
            * (1.0 - blend)
            * self.fluid.density
            * SIGMA_W2
            * np.sum(grad_k * grad_omega, axis=-1)
            / omega
        )

    @staticmethod
    def _blended(blend, inner, outer):
        return blend * inner + (1.0 - blend) * outer

    # ------------------------------------------------------------------
    # Transport equations
    # ------------------------------------------------------------------

    def _solve_k(
        self, state, strain, blend, grad_k, wall_k, far_k, inflow
    ) -> float:
        """Turbulent kinetic energy."""
        density = self.fluid.density
        omega = np.maximum(state.omega, self._omega_floor)

        # Menter's equation (5): P_k = mu_t dU_i/dx_j (dU_i/dx_j + dU_j/dx_i),
        # which for incompressible flow is identically mu_t S^2, capped at ten
        # times the destruction.
        #
        # The stagnation-point build-up this model is famous for is what the cap
        # is *for*. The obvious alternative is Kato-Launder, mu_t S Omega, which
        # this used to do: it agrees in a shear layer, where S = Omega, and it
        # suppresses the anomaly at its source, because a stagnation point has
        # large strain and almost no rotation. It was the wrong trade. Omega is
        # exactly zero on a stagnation streamline by symmetry, so Kato-Launder
        # puts a line of exactly zero production along it -- measured on a
        # cylinder, k on the stagnation ray came out five orders of magnitude
        # below its value eighteen degrees away, and mu_t through the whole
        # stagnation region sat 50 to 200 times low. The limiter, meanwhile,
        # never activated anywhere in the field, so the mechanism Menter
        # specifies was dead code. With the form below the limiter is active in
        # roughly 9% of cells and does the job it was designed for.
        # The wall-adjacent row takes its strain from the two-layer near-wall
        # profile rather than from the discrete gradient, which is the *average*
        # across the cell and overstates the local one by kappa U+ once the cell
        # leaves the viscous sublayer. See Boundaries.wall_velocity_gradient: it
        # reduces to the resolved value exactly on a wall-resolved mesh, so this
        # changes nothing at y+ ~ 1 and is the difference between converging and
        # not at y+ 30.
        production_strain = strain.copy()
        production_strain[:, 0] = self.boundaries.wall_velocity_gradient(
            state.u, state.v
        )

        production = np.minimum(
            state.eddy_viscosity * production_strain**2,
            PRODUCTION_LIMIT * BETA_STAR * density * np.maximum(state.k, 0.0) * omega,
        )

        coefficients = Coefficients.zeros(self.faces.shape)
        diffusivity = self.fluid.viscosity + self._blended(
            blend, SIGMA_K1, SIGMA_K2
        ) * state.eddy_viscosity

        ops.add_convection(
            coefficients, self.faces, state.flux_i, state.flux_j, state.k, grad_k,
            far_field_value=far_k, scheme=self.numerics.turbulence_scheme,
        )
        ops.add_diffusion(
            coefficients, self.faces, diffusivity, grad_k,
            wall_value=wall_k, far_field_value=far_k, far_field_active=inflow,
        )

        # Destruction is linear in k, so it belongs on the diagonal rather than in
        # the source. Put there it makes the matrix more diagonally dominant and
        # cannot drive k negative; as an explicit source it could.
        coefficients.centre += BETA_STAR * density * omega * self.volume
        coefficients.source += production * self.volume

        return self._solve_and_clip(coefficients, state, "k", self._k_floor)

    def _solve_omega(
        self, state, strain, blend, cross_diffusion, grad_omega, wall_omega,
        far_omega, inflow,
    ) -> float:
        """Specific dissipation rate."""
        density = self.fluid.density
        omega = np.maximum(state.omega, self._omega_floor)

        gamma = self._blended(blend, GAMMA_1, GAMMA_2)
        beta = self._blended(blend, BETA_1, BETA_2)

        coefficients = Coefficients.zeros(self.faces.shape)
        diffusivity = self.fluid.viscosity + self._blended(
            blend, SIGMA_W1, SIGMA_W2
        ) * state.eddy_viscosity

        ops.add_convection(
            coefficients, self.faces, state.flux_i, state.flux_j, state.omega, grad_omega,
            far_field_value=far_omega, scheme=self.numerics.turbulence_scheme,
        )
        # No wall value here: omega is prescribed in the wall-adjacent cell
        # rather than on the face, so the face carries no diffusive flux of its
        # own. See _solve_and_clip.
        ops.add_diffusion(
            coefficients, self.faces, diffusivity, grad_omega,
            wall_value=None, far_field_value=far_omega, far_field_active=inflow,
        )

        # Destruction is quadratic; linearising it as beta rho omega_old * omega
        # keeps it implicit and unconditionally stable.
        coefficients.centre += beta * density * omega * self.volume
        coefficients.source += gamma * density * strain**2 * self.volume

        # Cross-diffusion changes sign. The positive part is a source; the
        # negative part is split off and made implicit, so it can never push
        # omega through zero however large it becomes.
        coefficients.source += np.maximum(cross_diffusion, 0.0) * self.volume
        coefficients.centre += (
            np.maximum(-cross_diffusion, 0.0) / omega
        ) * self.volume

        return self._solve_and_clip(
            coefficients, state, "omega", self._omega_floor, fixed_wall=wall_omega
        )

    def _solve_and_clip(
        self,
        coefficients: Coefficients,
        state: State,
        name: str,
        floor: float,
        fixed_wall: np.ndarray | None = None,
    ) -> float:
        """Relax, solve, and clip the result to stay positive.

        ``fixed_wall`` prescribes the wall-adjacent cell value outright, by
        replacing its row with the identity. This is how Menter's ``omega``
        condition is meant to be applied: ``6 nu / (beta1 d1^2)`` is the
        asymptotic solution evaluated *at the first cell centre*, not a value on
        the surface. Imposing it as a Dirichlet face value instead drives an
        enormous diffusive flux across the twelve-micron gap of a ``y+ = 1``
        mesh, and the equation diverges within a couple of hundred iterations.

        The row is pinned again after relaxation, so the relaxation cannot dilute
        it.

        The residual is measured on the substituted system, which is the one that
        is actually solved. Measuring it before the substitution -- as this did --
        measures the wall row's *unsubstituted* equation, an equation that is then
        thrown away. That row is the stiffest in the mesh, and it dominates:
        measured on a NACA 0012 at ``y+ = 1``, 99.6% of the reported imbalance
        came from it, so the reported figure was 9.2e-2 while the system being
        solved stood at 1.9e-4. Since :attr:`Residuals.worst` includes ``omega``,
        that alone put a floor of order 1e-1 under every run, whatever the physics
        was doing.
        """
        current = getattr(state, name)

        if fixed_wall is not None:
            self._fix_wall_row(coefficients, fixed_wall)

        residual = coefficients.residual(current)

        # Damped with the same local step the momentum equations use, so that k
        # and omega move at the pace of the velocity field driving them. The
        # wall row is pinned again afterwards either way: it is a prescribed
        # value, not a transported one, and must not be relaxed or time-stepped
        # away from what the boundary condition sets.
        pseudo_time = self.pseudo_time_diagonal(state)
        if pseudo_time is not None:
            coefficients.add_pseudo_time(current, pseudo_time)
        coefficients.under_relax(current, self.numerics.relax_turbulence)
        if fixed_wall is not None:
            self._pin_wall_row(coefficients, fixed_wall)

        matrix = self.matrix.build(coefficients)
        value, _ = solve(
            matrix,
            coefficients.source,
            current,
            tolerance=self.numerics.inner_tolerance,
            preconditioner=jacobi_preconditioner(matrix),
        )
        setattr(state, name, np.maximum(value, floor))
        return residual

    @staticmethod
    def _fix_wall_row(coefficients: Coefficients, fixed_wall: np.ndarray) -> None:
        """Replace the wall-adjacent row with the identity, once."""
        for band in (
            coefficients.west, coefficients.east,
            coefficients.south, coefficients.north,
        ):
            band[:, 0] = 0.0
        # The cell above no longer has a neighbour to solve for; fold its
        # coupling into the source so the equation there stays correct.
        coefficients.source[:, 1] -= coefficients.south[:, 1] * fixed_wall
        coefficients.south[:, 1] = 0.0
        KOmegaSST._pin_wall_row(coefficients, fixed_wall)

    @staticmethod
    def _pin_wall_row(coefficients: Coefficients, fixed_wall: np.ndarray) -> None:
        """Restore the identity on the wall row, after something has scaled it.

        Separate from :meth:`_fix_wall_row` because it is idempotent and that one
        is not: folding the ``south`` coupling into the row above may happen once
        and only once.
        """
        coefficients.centre[:, 0] = 1.0
        coefficients.source[:, 0] = fixed_wall
