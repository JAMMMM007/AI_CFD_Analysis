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
        assert np.all(k == 0.0)

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
