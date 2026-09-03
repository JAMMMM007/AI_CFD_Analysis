"""Solver tests: linear algebra, operators, boundary conditions and forces.

The centrepiece is the method of manufactured solutions. An analytic field is
substituted into the discrete operators and the result compared against the
analytic answer on a sequence of refined meshes. What matters is not the size of
the error but its *order*: a scheme that is second order by construction and
first order in practice has a bug, and this is the only test that says so
regardless of how plausible the flow field looks.

Several tests here are regressions for specific defects found during
development. Each says which, because a test whose reason is forgotten is a test
that gets deleted.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from fluidsolver.geometry.primitives import circle
from fluidsolver.mesh.metrics import compute_metrics
from fluidsolver.mesh.ogrid import build_ogrid
from fluidsolver.solver import operators as ops
from fluidsolver.solver.bc import Boundaries
from fluidsolver.solver.faces import build_faces
from fluidsolver.solver.fields import State
from fluidsolver.solver.fluid import AIR_15C, Fluid, Freestream
from fluidsolver.solver.linalg import (
    Coefficients,
    StructuredMatrix,
    solve,
)
from fluidsolver.solver.post import compute_forces, wall_shear_stress


# ----------------------------------------------------------------------
# Manufactured fields
#
# The velocity comes from a stream function, so the face fluxes derived from it
# are *exactly* divergence-free on the discrete mesh: the flux through any face
# is the difference of the stream function at its two nodes, and the four
# contributions round a cell telescope to zero. Without that, the convection
# operator picks up a spurious `phi * div(F)` term of the same order as the
# answer, and the measured order stalls at one however good the scheme is.
# ----------------------------------------------------------------------


def stream_function(p):
    return np.sin(p[..., 0]) * np.sin(p[..., 1])


def velocity(p):
    return np.stack(
        (
            np.sin(p[..., 0]) * np.cos(p[..., 1]),
            -np.cos(p[..., 0]) * np.sin(p[..., 1]),
        ),
        axis=-1,
    )


def scalar(p):
    return np.sin(1.3 * p[..., 0]) * np.cos(0.7 * p[..., 1]) + 2.0


def scalar_gradient(p):
    return np.stack(
        (
            1.3 * np.cos(1.3 * p[..., 0]) * np.cos(0.7 * p[..., 1]),
            -0.7 * np.sin(1.3 * p[..., 0]) * np.sin(0.7 * p[..., 1]),
        ),
        axis=-1,
    )


def scalar_laplacian(p):
    return -(1.3**2 + 0.7**2) * np.sin(1.3 * p[..., 0]) * np.cos(0.7 * p[..., 1])


def uniform_mesh(surface_points: int, outer: float = 3.0):
    """A circle in a circular far field, with uniform radial spacing."""
    radial = outer - 0.5
    grid = build_ogrid(
        circle(1.0, surface_points),
        first_layer=radial / (surface_points // 3),
        far_field_radius=outer,
        growth=1.0,
    )
    metrics = compute_metrics(grid.nodes)
    return grid, metrics, build_faces(metrics)


def divergence_free_fluxes(nodes):
    psi = stream_function(nodes)
    return psi[:, 1:] - psi[:, :-1], psi - np.roll(psi, -1, axis=0)


def observed_order(errors: list[float]) -> float:
    """Order of accuracy from the last pair of a refinement sequence."""
    return float(np.log2(errors[-2] / errors[-1]))


# ----------------------------------------------------------------------
# Linear algebra
# ----------------------------------------------------------------------


class TestStructuredMatrix:
    @pytest.mark.parametrize("shape", [(6, 5), (7, 3), (60, 24)])
    def test_assembled_matrix_matches_the_operator(self, shape):
        """Regression: the CSR permutation was applied as a scatter, not a gather.

        The result had the right sparsity pattern and the right number of
        non-zeros, and every value in the wrong place -- which no shape or size
        check would have caught.
        """
        rng = np.random.default_rng(0)
        coefficients = Coefficients(*(rng.normal(size=shape) for _ in range(6)))
        coefficients.centre += 10.0
        coefficients.south[:, 0] = 0.0
        coefficients.north[:, -1] = 0.0

        matrix = StructuredMatrix(shape).build(coefficients)
        field = rng.normal(size=shape)

        assert np.allclose(
            coefficients.apply(field), (matrix @ field.ravel()).reshape(shape)
        )

    def test_boundary_rows_have_no_outward_neighbour(self):
        shape = (8, 4)
        coefficients = Coefficients.zeros(shape)
        coefficients.centre += 1.0
        matrix = StructuredMatrix(shape).build(coefficients)
        assert matrix.nnz == shape[0] * shape[1] * 5 - 2 * shape[0]


class TestSolve:
    def test_solve_reduces_the_residual_it_started_from(self):
        """Regression: the inner solve could return success having done nothing.

        SciPy measures ``rtol`` against ``|b|``. Inside a converging SIMPLE run
        the starting residual is already far below that, so the solver declared
        victory immediately, the outer loop stopped advancing, and the case sat
        on a residual plateau that looked like a physics problem.
        """
        rng = np.random.default_rng(1)
        shape = (20, 10)
        coefficients = Coefficients(*(0.1 * rng.normal(size=shape) for _ in range(6)))
        coefficients.centre += 5.0
        coefficients.south[:, 0] = 0.0
        coefficients.north[:, -1] = 0.0
        coefficients.source = 100.0 + rng.normal(size=shape)

        matrix = StructuredMatrix(shape).build(coefficients)
        guess = np.full(shape, 20.0)

        start = np.linalg.norm(matrix @ guess.ravel() - coefficients.source.ravel())
        assert start < 0.1 * np.linalg.norm(coefficients.source)  # the trap

        solution, _ = solve(matrix, coefficients.source, guess, tolerance=0.01)
        finish = np.linalg.norm(matrix @ solution.ravel() - coefficients.source.ravel())
        assert finish < 0.02 * start

    def test_non_finite_result_is_reported_not_returned(self):
        """A diverged solve must raise where it happened, not hand back NaN.

        NaN propagates silently through the outer loop and surfaces hundreds of
        iterations later as an unreadable force coefficient.
        """
        shape = (4, 4)
        coefficients = Coefficients.zeros(shape)
        coefficients.centre += 1.0
        coefficients.centre[2, 2] = np.nan
        matrix = StructuredMatrix(shape).build(coefficients)
        with pytest.raises(FloatingPointError, match="diverged"):
            solve(matrix, np.ones(shape), np.ones(shape))


class TestUnderRelaxation:
    def test_relaxation_leaves_the_converged_solution_unchanged(self):
        """Patankar's implicit form must alter the path, never the destination."""
        rng = np.random.default_rng(2)
        shape = (12, 6)
        coefficients = Coefficients(*(0.2 * rng.normal(size=shape) for _ in range(6)))
        coefficients.centre += 4.0
        coefficients.south[:, 0] = 0.0
        coefficients.north[:, -1] = 0.0
        coefficients.source = rng.normal(size=shape)

        matrix = StructuredMatrix(shape)
        exact = spla.spsolve(
            matrix.build(coefficients).tocsc(), coefficients.source.ravel()
        ).reshape(shape)

        relaxed = Coefficients(
            *(getattr(coefficients, f).copy() for f in
              ("centre", "west", "east", "south", "north", "source"))
        )
        relaxed.under_relax(exact, 0.3)
        again = spla.spsolve(
            matrix.build(relaxed).tocsc(), relaxed.source.ravel()
        ).reshape(shape)

        assert np.allclose(again, exact)


class TestPseudoTime:
    """Local time stepping: the damping must be non-uniform and must not bias."""

    def test_the_pseudo_time_term_leaves_the_converged_solution_unchanged(self):
        """Same guarantee as relaxation: it vanishes where phi equals phi_old."""
        rng = np.random.default_rng(7)
        shape = (12, 6)
        coefficients = Coefficients(*(0.2 * rng.normal(size=shape) for _ in range(6)))
        coefficients.centre += 4.0
        coefficients.south[:, 0] = 0.0
        coefficients.north[:, -1] = 0.0
        coefficients.source = rng.normal(size=shape)

        matrix = StructuredMatrix(shape)
        exact = spla.spsolve(
            matrix.build(coefficients).tocsc(), coefficients.source.ravel()
        ).reshape(shape)

        stepped = Coefficients(
            *(getattr(coefficients, f).copy() for f in
              ("centre", "west", "east", "south", "north", "source"))
        )
        stepped.add_pseudo_time(exact, 3.0 + rng.random(shape))
        again = spla.spsolve(
            matrix.build(stepped).tocsc(), stepped.source.ravel()
        ).reshape(shape)

        assert np.allclose(again, exact)

    def test_the_diagonal_is_the_cell_outflow_over_the_cfl(self):
        """A uniform rightward flux gives each cell one face's worth of outflow."""
        _, metrics, faces = uniform_mesh(32)
        flux_i = np.full(faces.shape, 2.0)
        flux_j = np.zeros((faces.shape[0], faces.shape[1] + 1))

        diagonal = ops.pseudo_time_diagonal(
            flux_i, flux_j, metrics.volume,
            density=1.0, velocity=1.0, reference_length=1.0, cfl=4.0,
        )
        # Every i face carries +2, so each cell sees 2 out of its east face and
        # nothing out of its west; the floor is far below that.
        assert np.allclose(diagonal, 2.0 / 4.0)

    def test_a_stagnant_cell_is_floored_rather_than_left_undamped(self):
        _, metrics, faces = uniform_mesh(32)
        zero_i = np.zeros(faces.shape)
        zero_j = np.zeros((faces.shape[0], faces.shape[1] + 1))

        diagonal = ops.pseudo_time_diagonal(
            zero_i, zero_j, metrics.volume,
            density=2.0, velocity=5.0, reference_length=10.0, cfl=1.0,
        )
        assert np.all(diagonal > 0.0)
        assert np.allclose(diagonal, 2.0 * metrics.volume * 5.0 / 10.0)

    def test_the_damping_is_not_uniform_across_the_mesh(self):
        """The whole point. A convective step damps the far field and not the wall.

        Pair the step with the full spectral radius instead -- convection *and*
        diffusion -- and ``rho V / dtau`` becomes ``a_P / CFL``, whereupon the
        effective relaxation is ``CFL/(1+CFL)`` in every cell and the mechanism
        is global under-relaxation wearing a different hat. This test is what
        stops that regression going unnoticed.
        """
        from fluidsolver.solver.case import MeshSettings, build_case
        from fluidsolver.solver.simple import Numerics

        case = build_case(
            circle(1.0, 96),
            Fluid(density=1.0, viscosity=1.0 / 5000.0),
            Freestream(velocity=1.0),
            mesh_settings=MeshSettings(surface_points=96, far_field_radius_ratio=20.0),
            numerics=Numerics(
                pseudo_transient=True, relax_velocity=1.0, relax_turbulence=1.0
            ),
            model_name="laminar",
        )
        for _ in range(20):
            case.step()

        coupling = case.coupling
        pseudo_time = coupling.pseudo_time_diagonal(case.state)
        coupling.numerics.pseudo_transient = False
        _, _, diagonal = coupling.momentum(case.state)
        coupling.numerics.pseudo_transient = True

        alpha = diagonal / (diagonal + pseudo_time)
        wall = case.metrics.wall_distance
        near = np.median(alpha[wall < np.percentile(wall, 5)])
        far = np.median(alpha[wall > np.percentile(wall, 80)])

        # Barely damped at the wall, damped in the far field, and the two must
        # differ by a wide margin rather than by rounding.
        assert near > 0.9
        assert far < 0.75
        assert near - far > 0.15

    def test_it_is_off_by_default_and_then_costs_nothing(self):
        """Shipped off: it measurably does not help a steady segregated solver."""
        from fluidsolver.solver.simple import Numerics

        assert Numerics().pseudo_transient is False


class TestCflRamp:
    def _ramp(self, **kwargs):
        from fluidsolver.solver.simple import CflRamp, Numerics

        return CflRamp(Numerics(**kwargs))

    def test_it_grows_while_the_residual_falls_and_stops_at_the_ceiling(self):
        ramp = self._ramp(cfl=1.0, cfl_max=4.0, cfl_growth=1.5)
        residual = 1.0
        for _ in range(200):
            residual *= 0.95
            ramp.update(residual)
        assert ramp.value == pytest.approx(4.0)
        assert ramp.backoffs == 0

    def test_it_backs_off_when_the_residual_turns_and_climbs(self):
        ramp = self._ramp(cfl=8.0, cfl_max=64.0, cfl_growth=1.0)
        residual = 1e-6
        for _ in range(400):
            residual *= 1.05
            ramp.update(residual)
        assert ramp.backoffs > 0
        assert ramp.value < 8.0

    def test_a_single_spike_does_not_trigger_a_back_off(self):
        """Residuals rattle. Only a sustained rise counts."""
        ramp = self._ramp(cfl=2.0, cfl_max=2.0, cfl_growth=1.0)
        for i in range(120):
            ramp.update(100.0 if i == 60 else 1e-5)
        assert ramp.backoffs == 0

    def test_a_non_finite_residual_is_ignored_rather_than_poisoning_the_history(self):
        ramp = self._ramp(cfl=2.0, cfl_max=8.0)
        ramp.update(float("nan"))
        ramp.update(float("inf"))
        ramp.update(0.0)
        assert ramp.value > 0.0
        assert ramp.backoffs == 0


# ----------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------


class TestGradient:
    @pytest.mark.parametrize("surface_points", [48, 96])
    def test_exact_for_a_linear_field(self, surface_points):
        """Least squares reproduces a linear field on any mesh, by construction.

        Green-Gauss does not, and its failure is not subtle: on a boundary-layer
        mesh the skewness error enters divided by cell volume, and it returned a
        gradient of magnitude 112 where the true one was 3.
        """
        _, metrics, faces = uniform_mesh(surface_points)
        exact = np.array([2.7, -1.3])

        def field(p):
            return 0.4 + p[..., 0] * exact[0] + p[..., 1] * exact[1]

        gradient = ops.Gradient(faces)(
            field(metrics.centroid), field(faces.wall.centre), field(faces.far_field.centre)
        )
        assert np.abs(gradient - exact).max() < 1e-10

    def test_exact_on_a_stretched_boundary_layer_mesh(self):
        grid = build_ogrid(circle(1.0, 200), first_layer=2.4e-5, far_field_radius=40.0)
        metrics = compute_metrics(grid.nodes)
        faces = build_faces(metrics)
        exact = np.array([1.0, -2.0])

        def field(p):
            return p[..., 0] * exact[0] + p[..., 1] * exact[1]

        gradient = ops.Gradient(faces)(
            field(metrics.centroid), field(faces.wall.centre), field(faces.far_field.centre)
        )
        assert np.abs(gradient - exact).max() < 1e-8


class TestManufacturedSolution:
    """Order-of-accuracy of the discrete operators.

    The operator is evaluated as ``apply(phi) - source``: the implicit half from
    the matrix, plus everything the assembly moved to the right-hand side. Only
    interior cells are measured. Boundary rows use a one-sided gradient which is
    first order there by construction; that is a known and accepted property of
    the scheme, not a defect, and including them would mask the interior order.
    """

    def _operator_error(self, surface_points, convect, diffuse, scheme):
        grid, metrics, faces = uniform_mesh(surface_points)
        phi = scalar(metrics.centroid)
        wall, far = scalar(faces.wall.centre), scalar(faces.far_field.centre)
        gradient = ops.Gradient(faces)(phi, wall, far)

        coefficients = Coefficients.zeros(faces.shape)
        exact = np.zeros(faces.shape)

        if convect:
            flux_i, flux_j = divergence_free_fluxes(grid.nodes)
            assert np.abs(ops.divergence(flux_i, flux_j, faces)).max() < 1e-12
            ops.add_convection(
                coefficients, faces, flux_i, flux_j, phi, gradient,
                far_field_value=far, wall_value=wall, scheme=scheme,
            )
            exact += metrics.volume * np.sum(
                velocity(metrics.centroid) * scalar_gradient(metrics.centroid), axis=-1
            )
        if diffuse:
            ops.add_diffusion(
                coefficients, faces, np.ones(faces.shape), gradient,
                wall_value=wall, far_field_value=far,
            )
            exact -= metrics.volume * scalar_laplacian(metrics.centroid)

        error = (coefficients.apply(phi) - coefficients.source) - exact
        weight = metrics.volume[:, 1:-1]
        norm = np.sqrt((exact**2 * metrics.volume).sum() / metrics.volume.sum())
        return float(np.sqrt((error[:, 1:-1] ** 2 * weight).sum() / weight.sum()) / norm)

    def test_diffusion_is_second_order(self):
        errors = [self._operator_error(n, False, True, "linear") for n in (48, 96, 192)]
        assert observed_order(errors) > 1.8

    def test_upwind_convection_is_first_order(self):
        """Not a defect -- upwind is first order, and confirming it proves the
        harness can tell the two apart."""
        errors = [self._operator_error(n, True, False, "upwind") for n in (48, 96, 192)]
        assert 0.8 < observed_order(errors) < 1.3

    @pytest.mark.parametrize("scheme", ["linear", "linear_upwind"])
    def test_high_order_convection_is_second_order(self, scheme):
        """Regressions for two sign errors in the deferred correction.

        The upwind cell was taken as the face's owner when the flux was positive,
        but a positive flux runs *into* the owner, so the upwind side is the
        neighbour. And the ``j`` direction had the correction's sign reversed,
        because a cell owns its low-``i`` face but its low-``j`` face belongs to
        the cell below. Either one alone reduced the scheme to first order and
        made it slightly worse than the upwind it was meant to improve on.
        """
        errors = [self._operator_error(n, True, False, scheme) for n in (48, 96, 192)]
        assert observed_order(errors) > 1.8

    def test_limited_convection_stays_above_first_order(self):
        """A limiter costs accuracy at extrema; it must not cost all of it."""
        errors = [
            self._operator_error(n, True, False, "limited_linear") for n in (48, 96, 192)
        ]
        assert observed_order(errors) > 1.3

    def test_full_convection_diffusion_is_second_order(self):
        errors = [self._operator_error(n, True, True, "linear") for n in (48, 96, 192)]
        assert observed_order(errors) > 1.8


class TestDivergence:
    def test_uniform_flow_has_no_divergence(self):
        _, metrics, faces = uniform_mesh(64)
        stream = np.array([1.7, 0.9])
        flux_i = np.sum(metrics.face_i_area * stream, axis=-1)
        flux_j = np.sum(metrics.face_j_area * stream, axis=-1)
        imbalance = ops.divergence(flux_i, flux_j, faces)
        assert np.abs(imbalance / metrics.volume).max() < 1e-11


# ----------------------------------------------------------------------
# Boundary conditions
# ----------------------------------------------------------------------


class TestBoundaries:
    @pytest.fixture
    def setup(self):
        _, metrics, faces = uniform_mesh(64)
        freestream = Freestream(velocity=30.0)
        return faces, Boundaries(faces, AIR_15C, freestream), freestream

    def test_omega_at_the_wall_follows_the_asymptote(self, setup):
        """``omega -> 6 nu / (beta1 d1^2)``, with d1 the perpendicular distance.

        Six, not sixty: the factor of ten belongs to formulations that set omega
        on the wall face. See tests/test_turbulence.py for why that matters.
        """
        faces, boundaries, _ = setup
        k, omega = boundaries.wall_turbulence()
        expected = (
            6.0
            * AIR_15C.kinematic_viscosity
            / (0.075 * faces.wall.wall_normal_distance**2)
        )
        assert np.allclose(omega, expected)
        # Zero flux for k, not a fixed zero: see test_turbulence.py.
        assert k is None

    def test_far_field_splits_on_the_sign_of_the_flux(self, setup):
        faces, boundaries, freestream = setup
        flux = boundaries.far_flux_from_freestream()
        entering = boundaries.inflow_mask(flux)

        assert entering.any() and not entering.all()  # a circle has both

        u = np.zeros(faces.shape)
        v = np.zeros(faces.shape)
        far_u, far_v = boundaries.far_velocity(u, v, flux)
        assert np.allclose(far_u[entering], freestream.velocity)
        assert np.allclose(far_u[~entering], 0.0)  # extrapolated from the interior

    def test_pressure_is_pinned_only_where_flow_leaves(self, setup):
        _, boundaries, _ = setup
        flux = boundaries.far_flux_from_freestream()
        pressure = np.full((len(flux), 3), 7.0)
        far = boundaries.far_pressure(pressure, flux)
        assert np.allclose(far[~boundaries.inflow_mask(flux)], 0.0)
        assert np.allclose(far[boundaries.inflow_mask(flux)], 7.0)

    def test_outflow_is_rescaled_to_balance_inflow(self, setup):
        """The pressure equation is solvable only if the boundary fluxes balance."""
        _, boundaries, _ = setup
        flux = boundaries.far_flux_from_freestream()
        flux = flux * np.where(flux > 0, 1.6, 1.0)  # break the balance
        balanced = boundaries.enforce_global_mass_balance(flux)
        assert abs(balanced.sum()) < 1e-10 * np.abs(balanced).sum()


class TestPressureCorrection:
    """That the matrix and the flux update describe the same operator."""

    @pytest.fixture
    def case(self):
        from fluidsolver.solver.case import MeshSettings, build_case

        return build_case(
            circle(1.0, 96),
            Fluid(density=1.0, viscosity=1.0 / 200.0),
            Freestream(velocity=1.0),
            mesh_settings=MeshSettings(surface_points=96, far_field_radius_ratio=20.0),
            model_name="laminar",
        )

    def test_the_corrected_fluxes_satisfy_the_equation_that_produced_them(self, case):
        """``div(F) after correction == A p' - b``, on every cell, boundary included.

        This is an identity, not an approximation: the pressure equation is built
        so that ``div(F')`` *is* ``A p'``, and ``b`` is ``-div(F*)``. It holds only
        if every flux correction the matrix accounts for is actually applied to a
        face.

        Regression. :meth:`pressure_correction` put a diagonal entry on every
        far-field face holding the pressure -- asserting a correction of
        ``rho D g p'`` leaving through it -- and :meth:`apply_correction` never
        applied it. The outer ring of cells was therefore left holding exactly
        that imbalance after every iteration, for ever: on a NACA 0012 it was 62%
        of all the mass error left in the domain, correlating with the missing
        term at -0.9996. It never showed up as a wrong answer, only as a
        continuity residual that would not go below about 1e-3.
        """
        for _ in range(5):
            case.step()

        coupling = case.coupling
        state = case.state
        _, _, diagonal = coupling.momentum(state)
        flux_i, flux_j, d_i, d_j = coupling.face_fluxes(state, diagonal)
        correction, coefficients = coupling.pressure_correction(
            state, flux_i, flux_j, d_i, d_j, diagonal
        )
        coupling.apply_correction(
            state, correction, flux_i, flux_j, d_i, d_j, diagonal
        )

        after = ops.divergence(state.flux_i, state.flux_j, case.faces)
        expected = coefficients.apply(correction) - coefficients.source
        scale = np.abs(coefficients.source).max()
        assert np.abs(after - expected).max() < 1e-10 * scale

    def test_the_outer_row_is_not_where_the_mass_error_lives(self, case):
        """The symptom the identity above explains, stated in the terms it was seen in."""
        for _ in range(20):
            case.step()
        imbalance = np.abs(
            ops.divergence(case.state.flux_i, case.state.flux_j, case.faces)
        )
        assert imbalance[:, -1].sum() < 0.25 * imbalance.sum()

    def test_the_pressure_residual_is_a_measurement(self, case):
        """It must depend on the solution. Reporting 1.0 for ever is not a residual.

        Regression. The residual was evaluated at ``phi = 0``, where the
        expression reduces to ``sum|b| / sum|b|``, so every line of every log
        read ``p 1.000e+00``.
        """
        values = [case.step().pressure for _ in range(6)]
        assert not any(v == pytest.approx(1.0) for v in values)
        assert len(set(values)) > 1


# ----------------------------------------------------------------------
# Forces
# ----------------------------------------------------------------------


class TestForces:
    @pytest.fixture
    def setup(self):
        grid = build_ogrid(circle(1.0, 120), first_layer=1e-3, far_field_radius=20.0)
        metrics = compute_metrics(grid.nodes)
        faces = build_faces(metrics)
        fluid = Fluid(density=2.0, viscosity=1e-3)
        freestream = Freestream(velocity=3.0)
        state = State.uniform(faces, fluid, freestream)
        return grid, faces, fluid, freestream, state

    def test_uniform_pressure_exerts_no_net_force(self, setup):
        """A closed surface at constant pressure: the normals must sum to zero."""
        grid, faces, fluid, freestream, state = setup
        state.pressure[:] = 5.0
        state.u[:] = 0.0
        state.v[:] = 0.0
        forces = compute_forces(
            state, faces, fluid, freestream, 1.0, grid.contour.centroid
        )
        scale = freestream.dynamic_pressure(fluid)
        assert np.abs(forces.total).max() < 1e-9 * scale

    def test_pressure_high_at_the_front_gives_positive_drag(self, setup):
        """Sign convention: wall area vectors point into the solid, so a pressure
        pushing on the upstream face must come out as drag along +x."""
        grid, faces, fluid, freestream, state = setup
        state.u[:] = 0.0
        state.v[:] = 0.0
        state.pressure[:] = -faces.wall.centre[:, 0][:, None] * np.ones(faces.shape)
        forces = compute_forces(
            state, faces, fluid, freestream, 1.0, grid.contour.centroid
        )
        assert forces.drag > 0.0

    def test_wall_shear_follows_the_near_wall_flow(self, setup):
        _, faces, fluid, _, state = setup
        traction, magnitude = wall_shear_stress(state, faces, fluid)
        tangential = state.velocity[:, 0] - np.sum(
            state.velocity[:, 0] * faces.wall.normal, axis=-1
        )[:, None] * faces.wall.normal
        assert np.all(np.sum(traction * tangential, axis=-1) >= -1e-15)
        assert np.allclose(magnitude, np.linalg.norm(traction, axis=-1))


# ----------------------------------------------------------------------
# Fluid model
# ----------------------------------------------------------------------


class TestFluid:
    def test_reynolds_number(self):
        assert AIR_15C.reynolds(30.0, 1.0) == pytest.approx(1.225 * 30.0 / 1.81e-5)

    def test_freestream_turbulence_matches_the_definitions(self):
        freestream = Freestream(
            velocity=30.0, turbulence_intensity=0.01, eddy_viscosity_ratio=5.0
        )
        k = freestream.turbulent_kinetic_energy()
        assert k == pytest.approx(1.5 * (0.01 * 30.0) ** 2)
        # mu_t = rho k / omega, so omega = rho k / (mu * ratio)
        omega = freestream.specific_dissipation(AIR_15C)
        assert AIR_15C.density * k / omega == pytest.approx(5.0 * AIR_15C.viscosity)

    def test_compressibility_warning_appears_only_above_mach_0_3(self):
        assert Freestream(velocity=50.0).compressibility_warning() is None
        assert "Mach" in Freestream(velocity=200.0).compressibility_warning()

    @pytest.mark.parametrize("kwargs", [dict(density=0.0), dict(viscosity=-1.0)])
    def test_impossible_properties_are_refused(self, kwargs):
        with pytest.raises(ValueError, match="positive"):
            Fluid(**{"density": 1.0, "viscosity": 1.0, **kwargs})

    def test_turbulence_intensity_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="fraction"):
            Freestream(velocity=10.0, turbulence_intensity=5.0)


# ----------------------------------------------------------------------
# Guardrails
# ----------------------------------------------------------------------


class TestSolutionLimits:
    @staticmethod
    def _state_and_limits(shape=(8, 4)):
        from fluidsolver.solver.guard import SolutionLimits

        freestream = Freestream(velocity=10.0)
        state = State(
            u=np.full(shape, 10.0),
            v=np.zeros(shape),
            pressure=np.zeros(shape),
            k=np.zeros(shape),
            omega=np.ones(shape),
            eddy_viscosity=np.zeros(shape),
            flux_i=np.zeros(shape),
            flux_j=np.zeros((shape[0], shape[1] + 1)),
        )
        limits = SolutionLimits(AIR_15C, freestream, cells=int(np.prod(shape)))
        return state, limits

    def test_an_ordinary_field_is_left_alone(self):
        state, limits = self._state_and_limits()
        before = state.u.copy()
        report = limits.apply(state)
        assert report.is_quiet
        assert np.array_equal(state.u, before)

    def test_a_runaway_speed_is_held_and_counted(self):
        state, limits = self._state_and_limits()
        state.u[3, 2] = 1.0e6
        report = limits.apply(state)
        assert report.speed == 1
        assert np.hypot(state.u, state.v).max() == pytest.approx(100.0)

    def test_a_cold_start_pressure_spike_passes_through_untouched(self):
        """Regression: the first cap was set inside the healthy band.

        Starting a case puts a uniform field against a no-slip wall, and the
        first pressure correction to that discontinuity is enormous before
        decaying away. Measured peaks over the opening iterations are |Cp| of
        123.5 on the Re 40 cylinder, 50.4 on the cylinder at Re 2e6 and 30.3 on
        a NACA 2412 at 15 degrees -- all of which a cap of 100 dynamic heads
        clipped. A backstop that fires on a run which was always going to
        converge is shaping the answer, which is the one thing it must not do.
        """
        state, limits = self._state_and_limits()
        q = Freestream(velocity=10.0).dynamic_pressure(AIR_15C)
        state.pressure[:] = 150.0 * q  # above the old cap, inside the real one
        report = limits.apply(state)
        assert report.pressure == 0
        assert state.pressure.max() == pytest.approx(150.0 * q)

    def test_clipping_preserves_direction(self):
        """Scale the vector, do not clip the components: the flow still goes
        where it was going, it merely stops accelerating without bound."""
        state, limits = self._state_and_limits()
        state.u[1, 1], state.v[1, 1] = 3.0e5, 4.0e5
        limits.apply(state)
        assert state.v[1, 1] / state.u[1, 1] == pytest.approx(4.0 / 3.0)
        assert np.hypot(state.u[1, 1], state.v[1, 1]) == pytest.approx(100.0)

    def test_activity_accumulates_across_iterations(self):
        state, limits = self._state_and_limits()
        for _ in range(3):
            state.u[0, 0] = 1.0e6
            report = limits.apply(state)
        assert report.iterations_active == 3
        assert "3 iterations" in report.summary()


class TestDivergenceMonitor:
    @staticmethod
    def _monitor():
        from fluidsolver.solver.guard import DivergenceMonitor

        return DivergenceMonitor()

    def test_a_falling_residual_never_trips(self):
        monitor = self._monitor()
        residual = 1.0
        for _ in range(400):
            residual *= 0.98
            assert not monitor.update(residual)

    def test_a_sustained_climb_trips(self):
        monitor = self._monitor()
        residual = 1.0e-2
        tripped = False
        for _ in range(400):
            residual *= 1.05
            if monitor.update(residual):
                tripped = True
                break
        assert tripped

    def test_the_slow_grind_that_the_first_version_missed(self):
        """Regression, and the reason the trigger is 1.5 rather than 10.

        The laminar cylinder at Re = 2e6 does not blow up; it climbs about 1.3%
        per iteration for hundreds of iterations. That is only 1.9x over a
        fifty-iteration window, so a detector demanding a tenfold rise inside one
        window sees nothing. Measured end to end, the first version of this let
        that case run 900 iterations to a residual of 7.7 without objecting --
        the exact failure it had been written to catch.
        """
        monitor = self._monitor()
        residual = 1.0e-4
        tripped_at = None
        for i in range(900):
            residual *= 1.013
            if monitor.update(residual):
                tripped_at = i
                break
        assert tripped_at is not None
        # And it must object early enough to be worth having.
        assert tripped_at < 500

    def test_a_spike_alone_does_not_trip(self):
        monitor = self._monitor()
        for i in range(300):
            assert not monitor.update(50.0 if i == 150 else 1.0e-2)

    def test_a_converged_run_drifting_in_the_ninth_decimal_is_left_alone(self):
        """A rise of ten from 1e-9 is not a divergence, whatever the ratio says."""
        monitor = self._monitor()
        residual = 1.0e-10
        for _ in range(400):
            residual *= 1.02
            assert not monitor.update(residual)

    def test_a_noisy_plateau_is_left_alone(self):
        """Never much better than it is now, so it has not lost ground."""
        monitor = self._monitor()
        rng = np.random.default_rng(3)
        for _ in range(400):
            assert not monitor.update(1.0e-2 * float(np.exp(rng.normal(0.0, 0.4))))

    def test_a_non_finite_residual_trips_immediately(self):
        assert self._monitor().update(float("nan"))
