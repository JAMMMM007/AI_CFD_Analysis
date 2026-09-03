"""Turbulence-model tests.

A turbulence model cannot be checked by looking at a flow field -- almost anything
looks plausible. What can be checked is that each piece does the specific job it
was derived to do: the constants satisfy the relation they were derived from, the
blending functions switch where they are supposed to switch, and the eddy
viscosity reduces to the analytic log-layer result when handed a log layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluidsolver.geometry.primitives import circle
from fluidsolver.mesh.metrics import compute_metrics
from fluidsolver.mesh.ogrid import build_ogrid
from fluidsolver.solver.bc import Boundaries
from fluidsolver.solver.faces import build_faces
from fluidsolver.solver.fields import State
from fluidsolver.solver.fluid import AIR_15C, Freestream
from fluidsolver.solver.simple import Numerics
from fluidsolver.solver.turbulence import MODELS, sst
from fluidsolver.solver.turbulence.laminar import Laminar
from fluidsolver.solver.turbulence.sst import KOmegaSST


@pytest.fixture(scope="module")
def _built():
    """Mesh and model, built once -- they are read-only and expensive."""
    grid = build_ogrid(circle(1.0, 96), first_layer=2.0e-5, far_field_radius=25.0)
    metrics = compute_metrics(grid.nodes)
    faces = build_faces(metrics)
    freestream = Freestream(velocity=30.0, turbulence_intensity=0.001)
    boundaries = Boundaries(faces, AIR_15C, freestream)
    model = KOmegaSST(faces, AIR_15C, boundaries, Numerics())
    return model, faces, metrics, freestream


@pytest.fixture
def rig(_built):
    """A y+ ~ 1 mesh round a cylinder, with a *fresh* state each test.

    The state has to be rebuilt per test: several of these deliberately drive k
    and omega to extremes, and sharing one state lets a later test read whatever
    an earlier one left behind.
    """
    model, faces, metrics, freestream = _built
    state = State.uniform(faces, AIR_15C, freestream)
    return model, state, faces, metrics, freestream


class TestConstants:
    def test_gamma_follows_the_log_layer_relation(self):
        """gamma is not free: it is fixed by requiring the log law to come out.

        ``gamma = beta/beta* - sigma_w kappa^2 / sqrt(beta*)``. The values 5/9 and
        0.44 quoted in the literature are these rounded, so hard-coding those
        instead leaves the model slightly inconsistent with its own derivation.
        """
        assert sst.GAMMA_1 == pytest.approx(5.0 / 9.0, abs=0.005)
        assert sst.GAMMA_2 == pytest.approx(0.44, abs=0.005)

    def test_the_two_constant_sets_are_distinct(self):
        """F1 blends between k-omega and k-epsilon; identical sets would make the
        blending pointless and hide a copy-paste error."""
        assert (sst.SIGMA_K1, sst.SIGMA_W1, sst.BETA_1) != (
            sst.SIGMA_K2, sst.SIGMA_W2, sst.BETA_2
        )

    def test_blending_is_a_convex_combination(self):
        blend = np.array([0.0, 0.25, 1.0])
        result = KOmegaSST._blended(blend, 10.0, 20.0)
        assert np.allclose(result, [20.0, 17.5, 10.0])


class TestBlendingFunctions:
    def test_f1_is_one_at_the_wall_and_zero_in_the_freestream(self, rig):
        """F1 selects k-omega near the wall and k-epsilon away from it.

        Regression: initialising omega uniformly at its freestream value while
        prescribing ~1e8 in the wall cell put a six-order jump across the first
        cell. The resulting huge grad(omega) inflated CD_k-omega, which is the
        *denominator* of F1's third argument, so F1 collapsed to zero at the wall
        -- switching the cross-diffusion on precisely where the model exists to
        switch it off. The run then diverged in the momentum equations, where the
        cause was invisible.
        """
        model, state, faces, metrics, _ = rig
        zero = np.zeros(state.k.shape + (2,))
        blend = model._blending_f1(state, zero, zero)

        near_wall = metrics.wall_distance < 1e-4
        far_away = metrics.wall_distance > 5.0
        assert blend[near_wall].min() > 0.99
        assert blend[far_away].max() < 0.01

    def test_f2_is_one_near_the_wall(self, rig):
        model, state, _, metrics, _ = rig
        strain = np.zeros(state.k.shape)
        assert model._blending_f2(state, strain)[metrics.wall_distance < 1e-4].min() > 0.99

    def test_blending_functions_stay_between_zero_and_one(self, rig):
        model, state, _, _, _ = rig
        rng = np.random.default_rng(0)
        gradient = rng.normal(size=state.k.shape + (2,)) * 1e3
        for blend in (
            model._blending_f1(state, gradient, gradient),
            model._blending_f2(state, np.abs(gradient[..., 0])),
        ):
            assert blend.min() >= 0.0 and blend.max() <= 1.0


class TestEddyViscosity:
    def test_reduces_to_k_over_omega_when_the_limiter_is_inactive(self, rig):
        """Away from strong shear the model must give the standard k-omega result."""
        model, state, _, _, _ = rig
        state.k = np.full(state.k.shape, 0.5)
        state.omega = np.full(state.omega.shape, 1.0e4)
        viscosity = model._eddy_viscosity(state, np.zeros(state.k.shape))
        assert np.allclose(viscosity, AIR_15C.density * 0.5 / 1.0e4)

    def test_the_shear_limiter_caps_it_in_strong_strain(self, rig):
        """``mu_t = rho a1 k / (S F2)`` once ``S F2`` exceeds ``a1 omega``.

        This is Bradshaw's relation, and it is the single term that lets SST
        predict separation where plain k-omega delays it.
        """
        model, state, _, metrics, _ = rig
        state.k = np.full(state.k.shape, 1.0)
        state.omega = np.full(state.omega.shape, 100.0)
        strain = np.full(state.k.shape, 1.0e5)  # far above a1 * omega = 31

        limited = model._eddy_viscosity(state, strain)
        unlimited = AIR_15C.density * 1.0 / 100.0

        near_wall = metrics.wall_distance < 1e-3  # where F2 -> 1
        assert limited[near_wall].max() < unlimited
        assert limited[near_wall] == pytest.approx(
            AIR_15C.density * sst.A1 * 1.0 / 1.0e5, rel=0.02
        )

    def test_log_layer_gives_the_analytic_eddy_viscosity(self, rig):
        """In a log layer SST must reproduce ``nu_t = kappa u_tau y``.

        Substituting the log-layer solution ``k = u_tau^2 / sqrt(beta*)`` and
        ``omega = u_tau / (sqrt(beta*) kappa y)`` into ``mu_t = rho k / omega``
        gives exactly that, so this checks the model's algebra against the
        equilibrium it was constructed to satisfy.
        """
        model, state, _, metrics, _ = rig
        friction_velocity = 1.2
        distance = np.maximum(metrics.wall_distance, 1e-12)

        state.k = np.full(state.k.shape, friction_velocity**2 / np.sqrt(sst.BETA_STAR))
        state.omega = friction_velocity / (np.sqrt(sst.BETA_STAR) * sst.KAPPA * distance)

        # In equilibrium S = du/dy = u_tau / (kappa y), below the limiter.
        strain = friction_velocity / (sst.KAPPA * distance)
        viscosity = model._eddy_viscosity(state, strain)

        expected = AIR_15C.density * sst.KAPPA * friction_velocity * distance
        layer = (metrics.wall_distance > 1e-4) & (metrics.wall_distance < 1e-2)
        assert np.allclose(viscosity[layer], expected[layer], rtol=0.02)

    def test_a_safety_ceiling_is_applied(self, rig):
        model, state, _, _, _ = rig
        state.k = np.full(state.k.shape, 1e6)
        state.omega = np.full(state.omega.shape, 1e-6)
        viscosity = model._eddy_viscosity(state, np.zeros(state.k.shape))
        assert viscosity.max() <= sst.MAX_VISCOSITY_RATIO * AIR_15C.viscosity


class TestWallConditions:
    def test_omega_uses_the_asymptote_not_the_face_form(self, rig):
        """``6 nu / (beta1 d1^2)`` at the first cell, not ``60``.

        The factor of ten belongs to formulations that set omega on the wall
        *face*, where it is formally infinite. Carrying it into a first-cell
        prescription puts omega ten times too high in the stiffest cell there is.
        """
        model, _, faces, _, _ = rig
        _, omega = model.boundaries.wall_turbulence()
        expected = (
            6.0 * AIR_15C.kinematic_viscosity
            / (sst.BETA_1 * faces.wall.wall_normal_distance**2)
        )
        assert np.allclose(omega, expected)

    def test_initial_omega_matches_the_prescribed_wall_value(self, rig):
        """A mismatch here is the six-order discontinuity that breaks F1."""
        model, state, _, _, _ = rig
        _, wall_omega = model.boundaries.wall_turbulence()
        assert state.omega[:, 0] == pytest.approx(wall_omega, rel=0.05)

    def test_k_takes_a_zero_flux_wall_condition(self, rig):
        """Not ``k = 0``, which is right only in the low-Reynolds limit.

        Esch and Menter are explicit that zero flux is what holds in *both* the
        viscous and the logarithmic limit, which is what a wall treatment
        spanning the two has to satisfy. Fixing the face value to zero is correct
        on a y+ ~ 1 mesh, where the first cell really does sit in the sublayer;
        on a coarser one that cell centre carries a substantial turbulent kinetic
        energy, and driving it to zero across the face removes energy the flow
        has. ``None`` is how add_diffusion is told a boundary carries no
        diffusive flux.
        """
        model, _, _, _, _ = rig
        k, _ = model.boundaries.wall_turbulence()
        assert k is None

    def test_the_omega_residual_measures_the_system_that_is_solved(self, rig):
        """The wall row is replaced before the solve, so it must be replaced
        before the residual too.

        Regression, and the reason this solver could never report convergence.
        ``omega`` in the wall cell is prescribed, not solved: its row is
        overwritten with the identity. The residual was taken before that, so it
        measured the wall row's *unsubstituted* equation -- an equation that is
        then discarded, on the stiffest row in the mesh, where ``omega`` is of
        order 1e8. Measured on a NACA 0012 at ``y+ = 1``, 99.6% of the reported
        imbalance came from that one row: the figure printed was 9.2e-2 while the
        system actually being solved stood at 1.9e-4.

        :attr:`Residuals.worst` includes ``omega``, so this alone put a floor of
        order 1e-1 under every run, whatever the physics was doing. It is exactly
        the residual plateau the model was blamed for.
        """
        from fluidsolver.solver.linalg import Coefficients

        model, state, faces, _, _ = rig
        _, wall_omega = model.boundaries.wall_turbulence()
        state.omega = np.full(faces.shape, 1.0e3)
        state.omega[:, 0] = wall_omega

        # An interior that is satisfied exactly, and a wall row that is not:
        # a large diagonal against a source that has nothing to do with it.
        coefficients = Coefficients.zeros(faces.shape)
        coefficients.centre[:] = 1.0
        coefficients.source[:] = state.omega
        coefficients.centre[:, 0] = 1.0e6
        coefficients.source[:, 0] = 0.0

        before = coefficients.residual(state.omega)
        residual = model._solve_and_clip(
            coefficients, state, "omega", model._omega_floor, fixed_wall=wall_omega
        )

        assert before > 0.5, "the unsubstituted wall row should dominate"
        assert residual < 1e-10, f"reported {residual:g} for a satisfied system"


class TestLaminar:
    def test_eddy_viscosity_is_exactly_zero(self, rig):
        _, state, faces, _, freestream = rig
        model = Laminar(faces, AIR_15C, Boundaries(faces, AIR_15C, freestream), Numerics())
        assert model.update(state) == (0.0, 0.0)
        assert np.all(state.eddy_viscosity == 0.0)


class TestStrainRate:
    def test_pure_shear_gives_the_shear_rate(self, rig):
        """For ``u = a y``, ``S = sqrt(2 S_ij S_ij)`` must come out as ``a``."""
        model, state, faces, metrics, _ = rig
        rate = 3.5
        state.u = rate * metrics.centroid[..., 1]
        state.v = np.zeros(state.v.shape)
        state.flux_j = np.zeros(state.flux_j.shape)

        strain = model.strain_rate(state, model.gradient)
        interior = strain[:, 2:-2]
        assert np.median(interior) == pytest.approx(rate, rel=0.02)


class TestRegistry:
    def test_both_models_are_registered(self):
        assert set(MODELS) == {"laminar", "k-omega-sst"}


class TestAutomaticWallTreatment:
    """Esch and Menter (IGTC 2003), equations 15-18, plus the wall-cell strain.

    The property worth guarding above all others is that every piece of this
    reduces *exactly* to the low-Reynolds treatment as the first cell approaches
    the wall. A wall model that quietly perturbs a y+ ~ 1 answer has taken
    something away in exchange for what it gives.
    """

    @staticmethod
    def _case(y_plus, iterations=60):
        from fluidsolver.geometry.naca import naca4
        from fluidsolver.solver.case import MeshSettings, build_case
        from fluidsolver.solver.fluid import AIR_15C, Freestream

        case = build_case(
            naca4("2412"), AIR_15C,
            Freestream(velocity=30.0, angle_of_attack_deg=5.0),
            mesh_settings=MeshSettings(
                surface_points=240, target_y_plus=y_plus, far_field_radius_ratio=40.0
            ),
        )
        for _ in range(iterations):
            case.step()
        return case

    def test_the_friction_velocity_recovers_the_log_law(self):
        """Built from (U1, y1) alone, u_tau must come back out of the profile.

        An earlier version seeded y+ from the viscous branch only, which
        underestimates u_tau, shrinks ln(y+) and therefore *raises* the
        logarithmic branch -- an overestimate that grew with coarsening, reaching
        +21% at y+ 300. It is now iterated to a fixed point.
        """
        from fluidsolver.solver.bc import _KAPPA, _LOG_LAW_CONSTANT

        nu, u_tau = 1.5e-5, 1.0
        for y_plus in (30.0, 100.0, 300.0):
            y1 = y_plus * nu / u_tau
            u1 = u_tau * (np.log(y_plus) / _KAPPA + _LOG_LAW_CONSTANT)

            viscous = np.sqrt(nu * u1 / y1)
            friction = viscous
            for _ in range(5):
                yp = max(friction * y1 / nu, 1.0)
                log = u1 / (np.log(yp) / _KAPPA + _LOG_LAW_CONSTANT)
                friction = (viscous**4 + max(log, 0.0) ** 4) ** 0.25

            assert friction == pytest.approx(u_tau, rel=0.05)

    def test_the_wall_shear_is_what_the_momentum_equation_receives(self):
        """mu_wall * g * U1 has to equal tau_w * A, or the model is decorative."""
        case = self._case(30.0)
        b, st = case.boundaries, case.state
        delivered = (
            b.wall_viscosity(st.u, st.v)
            * case.faces.wall.diffusion_factor
            * b.wall_tangential_velocity(st.u, st.v)
        )
        intended = b.wall_shear(st.u, st.v) * case.faces.wall.length
        assert np.allclose(delivered, intended, rtol=1e-12)

    def test_the_wall_strain_reduces_to_the_resolved_one_when_resolved(self):
        """The whole treatment must be invisible on a y+ ~ 1 mesh."""
        case = self._case(1.0)
        b, st = case.boundaries, case.state
        resolved = b.wall_tangential_velocity(st.u, st.v) / (
            case.faces.wall.wall_normal_distance
        )
        corrected = b.wall_velocity_gradient(st.u, st.v)
        assert np.median(corrected / resolved) == pytest.approx(1.0, rel=0.02)

    def test_the_wall_strain_is_cut_hard_once_the_sublayer_is_unresolved(self):
        """Regression: production went as the *average* gradient across the cell.

        In the log layer that overstates the local gradient by kappa U+, which
        squares into roughly thirty times too much production. Measured on a
        NACA 2412 at y+ 30, k climbed from 4.4e-03 to 7.0e-03 and stuck on the
        solution limiter -- 165 cells clipped on 554 of 600 iterations -- while
        momentum, continuity and omega all converged by two orders.
        """
        case = self._case(30.0)
        b, st = case.boundaries, case.state
        resolved = b.wall_tangential_velocity(st.u, st.v) / (
            case.faces.wall.wall_normal_distance
        )
        corrected = b.wall_velocity_gradient(st.u, st.v)
        assert np.median(corrected / resolved) < 0.4

    def test_a_coarse_wall_mesh_converges_and_leaves_the_limiter_alone(self):
        """Both halves matter. A run that converges onto the limiter is not a run."""
        case = self._case(30.0, iterations=400)
        last = case.history.entries[-1]
        assert last.k < 1.0e-4
        assert case.limiter.is_quiet
