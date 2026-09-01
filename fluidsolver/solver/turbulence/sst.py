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

# gamma is not independent: it follows from requiring the model to reproduce the
# log layer, gamma = beta/beta* - sigma_w kappa^2 / sqrt(beta*). The familiar
# 5/9 and 0.44 are these numbers rounded.
GAMMA_1 = BETA_1 / BETA_STAR - SIGMA_W1 * KAPPA**2 / np.sqrt(BETA_STAR)
GAMMA_2 = BETA_2 / BETA_STAR - SIGMA_W2 * KAPPA**2 / np.sqrt(BETA_STAR)

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

    def __init__(self, faces, fluid, boundaries, numerics):
        super().__init__(faces, fluid, boundaries, numerics)
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
        wall_k, wall_omega = self.boundaries.wall_turbulence()
        far_k, far_omega = self.boundaries.far_turbulence(state.k, state.omega, far_flux)
        inflow = self.boundaries.inflow_mask(far_flux)

        grad_k = self.gradient(state.k, wall_k, far_k)
        grad_omega = self.gradient(state.omega, wall_omega, far_omega)
        strain = self.strain_rate(state, self.gradient)

        blend = self._blending_f1(state, grad_k, grad_omega)
        cross_diffusion = self._cross_diffusion(state, grad_k, grad_omega, blend)

        vorticity = self.vorticity(state, self.gradient)

        residual_k = self._solve_k(
            state, strain, vorticity, blend, grad_k, wall_k, far_k, inflow
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
        self, state, strain, vorticity, blend, grad_k, wall_k, far_k, inflow
    ) -> float:
        """Turbulent kinetic energy."""
        density = self.fluid.density
        omega = np.maximum(state.omega, self._omega_floor)

        # Kato-Launder production: mu_t S Omega in place of mu_t S^2.
        #
        # The two are identical in a shear layer, where S = Omega, so nothing is
        # lost where the model is calibrated. They differ at a stagnation point,
        # where the strain rate is large but the flow barely rotates. The
        # unmodified form reads that pure strain as turbulence production and
        # manufactures k out of nothing on the stagnation streamline -- and since
        # more k means more mu_t means more turbulent diffusion carrying it
        # further upstream, it feeds itself. Measured on a NACA 0012, this put a
        # plume of 15% turbulence intensity a quarter of a chord *ahead* of the
        # leading edge, in a freestream of 0.1%, and it grew without bound.
        #
        # Menter's limiter below caps production at ten times the destruction,
        # which bounds the stagnation anomaly but does not remove it; using the
        # rotation rate removes its source.
        production = np.minimum(
            state.eddy_viscosity * strain * vorticity,
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
        condition is meant to be applied: ``60 nu / (beta1 d1^2)`` is the
        asymptotic solution evaluated *at the first cell centre*, not a value on
        the surface. Imposing it as a Dirichlet face value instead drives an
        enormous diffusive flux across the twelve-micron gap of a ``y+ = 1``
        mesh, and the equation diverges within a couple of hundred iterations.

        The row is replaced after relaxation, so the relaxation cannot dilute it.
        """
        current = getattr(state, name)
        residual = coefficients.residual(current)

        coefficients.under_relax(current, self.numerics.relax_turbulence)

        if fixed_wall is not None:
            for band in (
                coefficients.west, coefficients.east,
                coefficients.south, coefficients.north,
            ):
                band[:, 0] = 0.0
            coefficients.centre[:, 0] = 1.0
            coefficients.source[:, 0] = fixed_wall
            # The cell above no longer has a neighbour to solve for; fold its
            # coupling into the source so the equation there stays correct.
            coefficients.source[:, 1] -= coefficients.south[:, 1] * fixed_wall
            coefficients.south[:, 1] = 0.0

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
