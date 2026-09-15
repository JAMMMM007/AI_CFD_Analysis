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

from fluidsolver.geometry.naca import naca4
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
from fluidsolver.solver.simple import Numerics, PressureVelocityCoupling


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
    return grid.nodes, metrics, build_faces(metrics)


#: Refinement base for the two families below. Level ``r`` has ``r`` times the
#: surface points and ``r`` times the layers of the base, so ``h`` halves in both
#: directions at once and an order read off the sequence is an order in ``h``.
_MMS_BASE_POINTS = 48
_MMS_BASE_LAYERS = 16


def stretched_mesh(surface_points: int, outer: float = 3.0, base_growth: float = 1.3):
    """The circle again, with the wall-normal spacing geometrically stretched.

    Isolates stretching from non-orthogonality: the body is still a circle, so
    ``T`` is still zero to rounding, and the only thing that has changed against
    :func:`uniform_mesh` is the expansion ratio -- 1.30 at the base level against
    ``uniform_mesh``'s 1.27 falling to 1.08. Truncation error on a stretched mesh
    loses one order in its leading term unless the stretching is *smooth*, and
    nothing in this file tested that before.

    Refining a stretched family is not the same as refining a uniform one, and
    doing it wrongly is how a family stops being a family. Holding ``growth``
    fixed while halving the first layer adds only ``ln 2 / ln growth`` layers, so
    the radial direction refines logarithmically while the surface refines
    linearly -- measured, that gives 20, 25 and 30 layers against 48, 96 and 192
    points, and an order read off it means nothing. Here the layer count is
    doubled outright and ``growth`` taken to the matching root, ``g^(1/r)``, so
    that the same total thickness is spanned by twice the layers with the same
    *distribution*. The expansion ratio then tends to one as the family refines,
    which is the definition of smooth stretching rather than a way of making it
    disappear: a family whose expansion ratio stayed at 1.30 would have a first
    layer that never shrank.
    """
    ratio = surface_points / _MMS_BASE_POINTS
    layers = int(round(_MMS_BASE_LAYERS * ratio))
    growth = base_growth ** (1.0 / ratio)
    total = outer - 0.5
    first_layer = total * (growth - 1.0) / (growth**layers - 1.0)
    grid = build_ogrid(
        circle(1.0, surface_points),
        first_layer=first_layer,
        far_field_radius=outer,
        growth=growth,
    )
    metrics = compute_metrics(grid.nodes)
    return grid.nodes, metrics, build_faces(metrics)


def sheared_mesh(surface_points: int, outer: float = 3.0, shear: float = 0.45):
    """Concentric circles with the angular coordinate sheared against the radius.

    Isolates non-orthogonality from everything else, at an angle that is *the
    same at every refinement level*. That last property is what a body-fitted
    family cannot supply and why this is built analytically instead: a NACA
    through ``build_ogrid`` refined in both directions was measured at
    non-orthogonality mean 13.85, 14.73, 15.15 degrees and aspect ratio 32, 57,
    103 across three levels, so it drifts on two axes at once and an order read
    off it cannot be attributed. Nodes are placed at

        r_j = 0.5 + (outer - 0.5) j / Nj
        theta_ij = -2 pi i / Ni + shear (r_j - 0.5)

    so the ``j`` grid lines are spirals rather than radii. The sign of the first
    term is not cosmetic: ``build_ogrid`` orders its surface clockwise, and a
    mesh built anticlockwise has negative cell volumes everywhere and is
    rejected by ``quality.assess`` before the solver sees it. The local
    misalignment between the face normal and the centroid-to-centroid vector is
    ``arctan(r shear)``, which runs from 12.7 degrees at the body to 51.5 at the
    outer boundary and does not change when the mesh is refined -- which is
    exactly the property that makes an ``O(h) tan(theta)`` error term survive
    refinement, and therefore the property this family has to have.

    Radial spacing is uniform, the angular sweep is uniform, and the cells are
    convex everywhere for ``shear`` below about 0.8; 0.45 is well inside that.
    """
    layers = int(round(_MMS_BASE_LAYERS * surface_points / _MMS_BASE_POINTS))
    radius = 0.5 + (outer - 0.5) * np.arange(layers + 1) / layers
    angle = (
        -2.0 * np.pi * np.arange(surface_points)[:, None] / surface_points
        + shear * (radius - 0.5)[None, :]
    )
    nodes = np.stack(
        (radius[None, :] * np.cos(angle), radius[None, :] * np.sin(angle)), axis=-1
    )
    metrics = compute_metrics(nodes)
    return nodes, metrics, build_faces(metrics)


def aerofoil_mesh(surface_points: int = 160, first_layer: float = 5.0e-5):
    """A body-fitted mesh with the non-orthogonality the solver actually meets.

    ``uniform_mesh`` is a circle in a circular far field, and a circle meshes at
    exactly 0.0000 degrees of non-orthogonality, mean and peak -- measured. That
    makes it the right mesh for isolating a scheme's interior order and the wrong
    one for anything whose error term is proportional to the non-orthogonal
    remainder ``T = S - g d``, because ``T`` is identically zero there.

    This mesh is a NACA 0012 through the same ``build_ogrid`` the solver uses, so
    the marched-to-analytic seam and the polar blend are both present. Measured:
    non-orthogonality mean 4.05 degrees and peak 61.01, marched 49 layers of 80,
    against the primary NACA 2412 use case's mean 3.9 and peak 60.1. It is a
    small mesh -- 160x80, built in 0.23 s -- but it is the same *kind* of mesh,
    which is the property that matters here.
    """
    grid = build_ogrid(
        naca4("0012", 400).resample(surface_points, min_spacing=first_layer),
        first_layer=first_layer,
        far_field_radius=20.0,
    )
    metrics = compute_metrics(grid.nodes)
    return grid.nodes, metrics, build_faces(metrics)


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


class TestResidualScaling:
    """That the residual measures the system being solved, and only that."""

    @staticmethod
    def _with_a_prescribed_wall_row(shape=(8, 5), wall_value=8.0e6):
        """A plausible ``omega`` system: identity on the wall row, huge value on it.

        The wall row is what ``sst._fix_wall_row`` leaves behind -- ``1 * phi = W``
        with ``W`` of order 1e6 against an interior field near 1e2. That is a
        prescribed value, not a solved equation.
        """
        coefficients = Coefficients.zeros(shape)
        coefficients.centre[:] = 2.0
        coefficients.west[:] = coefficients.east[:] = -0.5
        coefficients.south[:] = coefficients.north[:] = -0.5
        coefficients.source[:] = 1.0

        field = np.full(shape, 100.0)
        for band in (
            coefficients.west, coefficients.east,
            coefficients.south, coefficients.north,
        ):
            band[:, 0] = 0.0
        coefficients.centre[:, 0] = 1.0
        coefficients.source[:, 0] = wall_value
        field[:, 0] = wall_value
        return coefficients, field

    def test_a_prescribed_row_does_not_contribute_to_the_residual(self):
        """Regression. On ``omega`` it supplied most of both sums.

        Measured on the NACA 2412 at iteration 400, the wall row was 56.89% of the
        normaliser and 65.54% of the imbalance -- reproducing the audit's 56.90%
        and 65.32% on an independent run. What it contributed was the distance the
        *boundary condition* had moved since the previous iteration, which is not
        a measure of how well any transport equation was solved.

        The assertion is insensitivity rather than size. Moving the prescribed
        row's own imbalance must not move the reported residual *at all*, which is
        what "does not contribute" means; asserting that the masked figure is
        merely smaller would be weaker and, on this system, not even true --
        removing the row takes more out of the normaliser than out of the
        imbalance, so the masked residual here comes out larger. Which direction
        it moves depends on the case and is not the property being tested.
        """
        coefficients, field = self._with_a_prescribed_wall_row()
        solved = np.ones(field.shape, dtype=bool)
        solved[:, 0] = False

        masked = coefficients.residual(field, solved)
        whole = coefficients.residual(field)

        # Put a large, arbitrary error on the prescribed row alone.
        coefficients.source[:, 0] *= 1.5

        assert coefficients.residual(field, solved) == pytest.approx(masked)
        assert coefficients.residual(field) != pytest.approx(whole)

    def test_masking_does_not_change_a_system_with_nothing_prescribed(self):
        """The mask must be inert where it selects everything."""
        coefficients, field = self._with_a_prescribed_wall_row()
        everything = np.ones(field.shape, dtype=bool)
        assert coefficients.residual(field, everything) == pytest.approx(
            coefficients.residual(field)
        )

    def test_the_residual_still_falls_as_the_field_approaches_the_solution(self):
        """A masked residual is still a residual, not just a smaller number."""
        coefficients, field = self._with_a_prescribed_wall_row()
        solved = np.ones(field.shape, dtype=bool)
        solved[:, 0] = False

        matrix = StructuredMatrix(field.shape).build(coefficients)
        exact, _ = solve(matrix, coefficients.source, field, tolerance=1e-12)

        errors = [
            coefficients.residual(field + (exact - field) * (1.0 - gap), solved)
            for gap in (1.0, 0.5, 0.1)
        ]
        assert errors[0] > errors[1] > errors[2]


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
            flux_i, flux_j, metrics.volume, faces,
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
            zero_i, zero_j, metrics.volume, faces,
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

    @pytest.mark.parametrize(
        "mesh, improvement",
        [(uniform_mesh, None), (sheared_mesh, 1.10)],
        ids=["orthogonal", "sheared"],
    )
    def test_zero_normal_gradient_is_imposed_along_the_normal(self, mesh, improvement):
        """Regression. Passing the adjacent cell value asserts a different thing.

        The stencil minimises ``sum w_N [(grad phi . d_N) - (phi_N - phi_P)]^2``,
        so ``phi_face = phi_P`` drives the fit towards ``grad phi . d = 0`` --
        along the centroid-to-face vector, not along the normal. The two coincide
        only where ``d`` is parallel to ``n``, and the operator's docstring called
        them the same thing.

        Checked against an exact gradient rather than against the other
        condition. ``sheared_mesh`` has a circular wall, so the distance field is
        ``d = r - 0.5`` and its gradient is the radial unit vector exactly, while
        the angular shear makes the boundary offset non-parallel to the face
        normal -- which is the whole condition being tested. ``phi = cos(2 d)``
        then has ``dphi/dn = 0`` at the wall face exactly and a known gradient
        everywhere.

        Measured, as the worst wall-row gradient error relative to the exact
        gradient:

            mesh        n     cell value   tangential   ratio
            uniform    48     3.7601e-02   3.7601e-02   1.00
            uniform   192     6.0099e-03   6.0099e-03   1.00
            sheared    48     6.8223e-02   5.8834e-02   1.16
            sheared   192     1.3518e-02   1.1944e-02   1.13

        Identical on the orthogonal mesh, to every digit, which is the signature
        that this is the term it claims to be and not a general improvement. On
        the sheared one it removes 13 to 16% of the error, consistently across a
        refinement by four.

        What it does *not* do is change the order -- both sequences run at about
        1.3. The wall-row gradient's convergence is limited by something else, and
        ``post.wall_pressure`` records what: the stencil weights go as ``1/|d|^2``
        and the wall face is the nearest stencil point, so whatever is asserted
        there dominates the fit.
        """
        def error(n):
            _, metrics, faces = mesh(n)
            radius = np.linalg.norm(metrics.centroid, axis=-1)
            distance = radius - 0.5
            phi = np.cos(2.0 * distance)
            exact = (-2.0 * np.sin(2.0 * distance))[..., None] * (
                metrics.centroid / radius[..., None]
            )
            operator = ops.Gradient(faces)
            scale = np.abs(exact[:, 0]).max()
            return (
                np.abs(operator(phi, phi[:, 0], phi[:, -1])[:, 0] - exact[:, 0]).max()
                / scale,
                np.abs(operator(phi, None, phi[:, -1])[:, 0] - exact[:, 0]).max()
                / scale,
            )

        for n in (48, 192):
            cell, tangential = error(n)
            if improvement is None:
                assert tangential == pytest.approx(cell, rel=1e-12)
            else:
                assert cell / tangential > improvement

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
    the matrix, plus everything the assembly moved to the right-hand side.

    **Which cells are measured, and why it is now a parameter.** Interior cells
    were the only ones measured, on the argument that the boundary rows use a
    one-sided gradient which is first order there by construction, so including
    them would mask the interior order. The second half is right and the first
    half is not: measured, the boundary truncation error is *zeroth* order, and
    the derivation agrees -- see
    :meth:`test_the_boundary_rows_are_zeroth_order`. Excluding them from the
    interior measurement is still correct, but they are now measured separately
    rather than left unexamined, so that a change in the boundary treatment is
    visible instead of averaged away.

    **Which meshes, and why it is now a parameter.** Every order in this class
    was measured on ``uniform_mesh``: an orthogonal, unstretched circle in a
    circular far field, aspect ratio 2.5, expansion 1.08 to 1.27, and
    non-orthogonality of 0.0000 degrees mean and peak. The mesh ``build_case``
    produces for the primary use case has aspect ratio 453, expansion 4.77 and
    non-orthogonality averaging 3.9 degrees with a peak of 60.1. It is not a
    harder version of the verification mesh; it is a different object, and a
    second-order result on one says nothing about the other. Two more families
    are measured here, each isolating one property -- :func:`stretched_mesh` for
    the expansion ratio and :func:`sheared_mesh` for the non-orthogonality.
    """

    def _operator_error(
        self, surface_points, convect, diffuse, scheme, mesh=uniform_mesh, rows="interior"
    ):
        nodes, metrics, faces = mesh(surface_points)
        phi = scalar(metrics.centroid)
        wall, far = scalar(faces.wall.centre), scalar(faces.far_field.centre)
        gradient = ops.Gradient(faces)(phi, wall, far)

        coefficients = Coefficients.zeros(faces.shape)
        exact = np.zeros(faces.shape)

        if convect:
            flux_i, flux_j = divergence_free_fluxes(nodes)
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
        norm = np.sqrt((exact**2 * metrics.volume).sum() / metrics.volume.sum())

        # The norm is taken over the whole field either way, so the two row
        # selections are measured against the same yardstick and their errors
        # can be compared with each other rather than only within a family.
        if rows == "interior":
            error, weight = error[:, 1:-1], metrics.volume[:, 1:-1]
        elif rows == "boundary":
            error = np.concatenate((error[:, :1], error[:, -1:]), axis=1)
            weight = np.concatenate(
                (metrics.volume[:, :1], metrics.volume[:, -1:]), axis=1
            )
        else:
            raise ValueError(f"unknown row selection {rows!r}")
        return float(np.sqrt((error**2 * weight).sum() / weight.sum()) / norm)

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

    # ------------------------------------------------------------------
    # The meshes the solver actually runs on
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "convect, diffuse, scheme",
        [(False, True, "linear"), (True, False, "linear"), (True, True, "linear")],
        ids=["diffusion", "convection", "both"],
    )
    def test_the_interior_order_survives_stretching(self, convect, diffuse, scheme):
        """Expansion ratio 1.61 at the base level against ``uniform_mesh``'s 1.27.

        Measured, errors and the order from the finer pair:

            diffusion    7.1597e-02  2.4099e-02  6.5743e-03   1.874
            convection   6.8472e-02  2.5944e-02  6.8067e-03   1.930
            both         4.5950e-02  1.5920e-02  4.5842e-03   1.796

        The coarse pair reads 1.40 to 1.57 on all three, so the family is not
        asymptotic at 48 points and the finer pair is the one to read -- which is
        why the threshold is set below the measured value rather than at it.
        """
        errors = [
            self._operator_error(n, convect, diffuse, scheme, mesh=stretched_mesh)
            for n in (48, 96, 192)
        ]
        assert observed_order(errors) > 1.7

    @pytest.mark.parametrize(
        "convect, diffuse, scheme",
        [(False, True, "linear"), (True, False, "linear"), (True, True, "linear")],
        ids=["diffusion", "convection", "both"],
    )
    def test_the_interior_order_survives_non_orthogonality(
        self, convect, diffuse, scheme
    ):
        """36.5 degrees mean and 53.3 peak, held constant across the family.

        This is the measurement that was missing. Every order in this class was
        taken on a mesh with ``T = S - g d`` identically zero, which makes the
        whole non-orthogonal correction path in ``add_diffusion`` dead code under
        test -- the audit's phrase, and it was accurate. Measured now, with the
        path live:

            diffusion    9.2842e-02  2.6914e-02  7.2018e-03   1.902
            convection   3.6641e-02  1.0055e-02  2.6293e-03   1.935
            both         9.7303e-02  2.8181e-02  7.5320e-03   1.904

        A positive result, and worth stating plainly because the audit's F4 could
        easily have been read as implying otherwise: the *operators* are second
        order on a non-orthogonal mesh and the deferred cross-term correction
        does its job. The first-order term F4 identifies is in the flux
        definition, which no manufactured solution here evaluates -- see
        :class:`TestRhieChowConsistency`, which is where that one is caught.
        """
        errors = [
            self._operator_error(n, convect, diffuse, scheme, mesh=sheared_mesh)
            for n in (48, 96, 192)
        ]
        assert observed_order(errors) > 1.7

    @pytest.mark.parametrize(
        "mesh", [uniform_mesh, stretched_mesh, sheared_mesh],
        ids=["uniform", "stretched", "sheared"],
    )
    def test_the_boundary_rows_are_zeroth_order(self, mesh):
        """Not first order. Measured, on all three families and both operators.

        This class's own docstring said the boundary rows "use a one-sided
        gradient which is first order there by construction", and the audit
        repeated it while asking for the rows to be measured rather than
        excluded. Measured, the truncation error there does not fall at all:

            uniform    both   4.0608e-01  4.0576e-01  4.0635e-01   order +0.001
            stretched  both   7.0577e-01  6.9856e-01  6.9281e-01   order +0.012
            sheared    both   6.9594e-01  6.7848e-01  6.7323e-01   order +0.011

        The derivation agrees, which is why this is a correction to the docstring
        rather than a suspected bug. The wall diffusive flux is
        ``Gamma (phi_wall - phi_P) |S| / delta`` with ``delta ~ h/2``, so the
        one-sided difference carries an ``O(h)`` error in the gradient and an
        ``O(h) |S| = O(h^2)`` error in the flux; the operator divides through by
        a volume that is also ``O(h^2)``, and what is left is ``O(1)``. First
        order would have required a second-order boundary gradient.

        **This does not mean the solution is zeroth order at the wall.** For an
        elliptic operator the boundary truncation error is damped rather than
        transported, and a boundary one order below the interior is the classical
        situation in which the global order is still the interior one. It does
        mean that the boundary treatment is the weakest link in the discretisation
        and that nothing here has ever measured it, which is the point of adding
        this.

        The bound is two-sided, in the same spirit as
        :meth:`test_upwind_convection_is_first_order`: a *rise* would be a real
        defect, and an improvement should fail this test and make somebody update
        the number rather than passing silently.
        """
        errors = [
            self._operator_error(n, True, True, "linear", mesh=mesh, rows="boundary")
            for n in (48, 96, 192)
        ]
        assert -0.2 < observed_order(errors) < 0.5


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

    def test_the_bound_vortex_turns_the_right_way(self, setup):
        """Three independent physical checks, because the obvious source has it
        the other way round.

        A point vortex of counter-clockwise strength ``G`` induces
        ``(G/2pi)(-(y-y0), (x-x0))/|r-r0|^2``. Taking ``G = +Gamma`` with
        ``Gamma = Cl U c / 2`` -- which is how the physics audit writes it -- puts
        the induced velocity above a lifting body along ``-x``, so the flow is
        *slower* over the suction side. That is backwards, and all three of these
        fail together with it:

            above the body   u > 0   faster where the pressure is lower
            ahead of it      v > 0   upwash
            behind it        v < 0   downwash

        The bound vortex of a body lifting along ``+y`` in a freestream along
        ``+x`` is clockwise. Getting this wrong would double the error rather than
        remove it, and it would still look plausible: the sensitivity to the
        far-field radius would change, just in the wrong direction.
        """
        faces, boundaries, _ = setup
        boundaries.set_circulation(0.75)
        assert boundaries.circulation > 0.0

        centre = boundaries.vortex_centre
        induced = boundaries.far_vortex_velocity()
        offset = faces.far_field.centre - centre

        above = np.argmax(offset[:, 1])
        ahead = np.argmin(offset[:, 0])
        behind = np.argmax(offset[:, 0])

        assert induced[above, 0] > 0.0
        assert induced[ahead, 1] > 0.0
        assert induced[behind, 1] < 0.0

    def test_the_induced_velocity_has_the_magnitude_kutta_joukowski_gives(self, setup):
        """``|u| = Gamma / (2 pi R) = Cl c / (4 pi R) * U``, analytically.

        The far-field boundary of this mesh is a circle about the body, so every
        face sits at the same radius and the induced speed is the same on all of
        them. That makes the magnitude checkable in closed form rather than by
        comparison against another run.
        """
        faces, boundaries, freestream = setup
        cl, chord = 0.75, boundaries.reference_length
        boundaries.set_circulation(cl)

        induced = boundaries.far_vortex_velocity()
        radius = np.linalg.norm(
            faces.far_field.centre - boundaries.vortex_centre, axis=-1
        )
        expected = cl * chord * freestream.velocity / (4.0 * np.pi * radius)
        assert np.allclose(np.linalg.norm(induced, axis=-1), expected, rtol=1e-12)

    def test_a_non_lifting_case_is_untouched(self, setup):
        """Zero circulation must leave both far-field conditions exactly as they
        were, or every cylinder result in this project moves for no reason."""
        faces, boundaries, _ = setup
        assert boundaries.circulation == 0.0

        assert np.array_equal(
            boundaries.far_vortex_velocity(),
            np.zeros_like(faces.far_field.centre),
        )

        flux = boundaries.far_flux_from_freestream()
        pressure = np.zeros((faces.shape[0], 3))
        assert np.allclose(
            boundaries.far_pressure(pressure, flux)[~boundaries.inflow_mask(flux)], 0.0
        )

    def test_a_boundary_with_outflow_needs_no_compatibility_projection(self, setup):
        """Because it is not a pure Neumann problem, which is the whole point.

        This replaces a test asserting that the outflow was rescaled to balance
        the inflow, on the stated grounds that "the pressure equation is solvable
        only if the boundary fluxes balance". That is the compatibility condition
        of a *pure Neumann* Poisson problem, and this one is not: a Dirichlet
        coupling to ``p' = 0`` sits behind every outflow face. The old test
        asserted the rescaling worked, which it did; what it never asked was
        whether it was needed.

        It was not, and it was still acting at convergence -- measured on the
        Re 40 cylinder, every outflow face rescaled by -0.0252% for ever, which
        makes it a boundary condition rather than a solvability repair.
        """
        _, boundaries, _ = setup
        flux = boundaries.far_flux_from_freestream()
        assert boundaries.far_flux_is_solvable(flux)

        # Break the balance badly. It is still solvable, because what makes it
        # solvable is the presence of a fixed-pressure face and not the balance.
        unbalanced = flux * np.where(flux > 0, 1.6, 1.0)
        assert abs(unbalanced.sum()) > 1e-6 * np.abs(unbalanced).sum()
        assert boundaries.far_flux_is_solvable(unbalanced)

    def test_a_boundary_that_is_inflow_everywhere_does_need_one(self, setup):
        """The one case the compatibility condition is for, and the one the old
        rescaling silently declined to handle: its guard returned the flux
        untouched whenever either total was non-positive."""
        _, boundaries, _ = setup
        entirely_inflow = -np.abs(boundaries.far_flux_from_freestream())
        assert not boundaries.far_flux_is_solvable(entirely_inflow)


def open_rectangle(nx: int = 7, ny: int = 4, dx: float = 0.5, dy: float = 0.25):
    """A uniform rectangle, open in ``i``, with its metrics and faces."""
    x = np.arange(nx + 1) * dx
    y = np.arange(ny + 1) * dy
    nodes = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    metrics = compute_metrics(nodes, periodic_i=False)
    return nodes, metrics, build_faces(metrics)


class TestOpenMatrix:
    """That an open ``i`` removes couplings rather than adding bands.

    This is the claim that makes opening ``i`` cheap and the C-grid expensive. A
    non-periodic ``i`` gives cell ``0`` no west neighbour and cell ``Ni-1`` no
    east one, exactly as ``j`` already has none at its two ends, so the sparsity
    pattern keeps its five diagonals. A C-grid's wake cut would instead make cell
    ``(i, 0)`` a neighbour of ``(Ni-1-i, 0)`` -- far apart in the ``k = i*Nj + j``
    ordering, and a genuine sixth and seventh band.
    """

    @staticmethod
    def _laplacian(shape, periodic_i):
        c = Coefficients.zeros(shape, periodic_i=periodic_i)
        c.centre[:] = 4.0
        c.south[:, 1:] = -1.0
        c.north[:, :-1] = -1.0
        if periodic_i:
            c.west[:] = c.east[:] = -1.0
        else:
            c.west[1:] = -1.0
            c.east[:-1] = -1.0
        return c

    def test_it_still_has_five_bands(self):
        shape = (7, 4)
        matrix = StructuredMatrix(shape, periodic_i=False).build(
            self._laplacian(shape, False)
        )
        assert np.diff(matrix.indptr).max() <= 5

    def test_opening_i_removes_exactly_the_end_couplings(self):
        """Two ends times ``Nj`` cells, and nothing else."""
        shape = (7, 4)
        closed = StructuredMatrix(shape, periodic_i=True).build(
            self._laplacian(shape, True)
        )
        opened = StructuredMatrix(shape, periodic_i=False).build(
            self._laplacian(shape, False)
        )
        assert closed.nnz - opened.nnz == 2 * shape[1]

    def test_the_corner_cell_couples_only_inward(self):
        """Cell (0, 0) has no west and no south neighbour, and must show it.

        The sharpest statement of the whole change: on a periodic mesh this cell
        wraps round to ``Ni-1`` and here it must not.
        """
        shape = (7, 4)
        matrix = StructuredMatrix(shape, periodic_i=False).build(
            self._laplacian(shape, False)
        )
        coupled = {
            (k // shape[1], k % shape[1])
            for k in matrix[0].toarray().ravel().nonzero()[0]
        }
        assert coupled == {(0, 0), (1, 0), (0, 1)}

    @pytest.mark.parametrize("periodic", [True, False])
    def test_apply_agrees_with_the_assembled_matrix(self, periodic):
        """``Coefficients.apply`` and the sparse product must be the same operator.

        ``apply`` exists so that residuals and the velocity reconstruction do not
        have to build a matrix, which means there are two implementations of one
        thing and they can drift. On an open mesh ``apply`` has to stop rolling in
        ``i``, and this is what catches it if it does not.
        """
        shape = (7, 4)
        coefficients = self._laplacian(shape, periodic)
        matrix = StructuredMatrix(shape, periodic_i=periodic).build(coefficients)

        field = np.random.default_rng(0).normal(size=shape)
        direct = coefficients.apply(field)
        through_matrix = (matrix @ field.ravel()).reshape(shape)
        assert np.abs(direct - through_matrix).max() < 1e-14 * np.abs(direct).max()


class TestOpenFaceFamilies:
    """That an open ``i`` gets the boundary families a closed one does not need.

    ``wall`` and ``far_field`` already exist because the ``j`` direction has two
    ends. When ``i`` stops wrapping it has two ends as well, and ``i_start`` and
    ``i_end`` are the same construction for the same reason -- not a new concept,
    which is the point worth checking: they must behave like the two that have
    been in use since the beginning.
    """

    def test_a_periodic_mesh_has_no_i_boundaries(self):
        """None rather than empty: there is no such face, not zero of them."""
        _, _, faces = uniform_mesh(48)
        assert faces.periodic_i
        assert faces.i_start is None and faces.i_end is None
        assert faces.i_faces.area.shape[:2] == faces.shape

    def test_an_open_mesh_splits_i_into_interior_and_two_ends(self):
        _, _, faces = open_rectangle()
        assert not faces.periodic_i
        assert faces.i_faces.area.shape == (6, 4, 2)
        assert faces.i_start.area.shape == (4, 2)
        assert faces.i_end.area.shape == (4, 2)
        # The j family is untouched by any of this.
        assert faces.j_faces.area.shape == (7, 3, 2)
        assert faces.wall.area.shape == (7, 2)

    @pytest.mark.parametrize(
        "family, outward",
        [
            ("i_start", [-1.0, 0.0]),
            ("i_end", [1.0, 0.0]),
            ("wall", [0.0, -1.0]),
            ("far_field", [0.0, 1.0]),
        ],
    )
    def test_every_boundary_normal_points_out_of_its_cell(self, family, outward):
        """The sign convention all four share, and the one to get wrong.

        ``face_i`` points towards increasing ``i``, so at the low end that is
        *into* the fluid and the outward normal is its negative -- the same
        inversion ``wall`` needs against ``face_j``, and for the same reason.
        Getting it backwards would reverse every flux through that boundary and
        the forces with them.

        ``delta`` runs from the cell centroid to the face centre, so ``delta . n``
        positive is the independent check: the face really is on the outside of
        the cell it belongs to.
        """
        _, _, faces = open_rectangle()
        boundary = getattr(faces, family)
        assert np.allclose(boundary.normal, outward)
        assert np.all(np.sum(boundary.delta * boundary.normal, axis=-1) > 0.0)

    def test_the_four_boundaries_and_the_interior_faces_close_every_cell(self):
        """Conservation, assembled the way the operators will assemble it.

        Every cell's outward area vectors must sum to zero, and on an open mesh
        that sum is drawn from up to four different arrays. This is the identity
        that the divergence operator telescoping to the boundary flux rests on,
        and the indexing it needs is exactly what an open ``i`` changes.
        """
        _, metrics, faces = open_rectangle()
        west = np.empty(metrics.centroid.shape)
        east = np.empty(metrics.centroid.shape)
        west[1:] = -faces.i_faces.area
        east[:-1] = faces.i_faces.area
        west[0] = faces.i_start.area
        east[-1] = faces.i_end.area

        south = np.empty(metrics.centroid.shape)
        north = np.empty(metrics.centroid.shape)
        south[:, 1:] = -faces.j_faces.area
        north[:, :-1] = faces.j_faces.area
        south[:, 0] = faces.wall.area
        north[:, -1] = faces.far_field.area

        closure = west + east + south + north
        scale = np.linalg.norm(faces.j_faces.area, axis=-1).max()
        assert np.abs(closure).max() < 1e-14 * scale


class TestSymmetryPlane:
    """A ``j = 0`` row that is part solid wall and part symmetry plane.

    Which is what a flat plate needs: the TMR ``2DZP`` case puts a symmetry
    plane ahead of the leading edge, because without one the plate starts at a
    singularity. The two conditions meet on a face, so the row carries both.

    Symmetry is not a fourth kind of boundary here. It is the wall's condition
    with the tangential target changed from zero to whatever the cell has, so
    what these tests check is that the change is complete -- every quantity the
    wall row feeds, not just the velocity.
    """

    @staticmethod
    def _plate(nx=8, ny=4):
        """An open rectangle whose ``j = 0`` row is symmetry then wall."""
        _, _, faces = open_rectangle(nx=nx, ny=ny)
        mask = np.zeros(nx, dtype=bool)
        mask[nx // 2:] = True
        boundaries = Boundaries(
            faces, AIR_15C, Freestream(velocity=10.0), wall_mask=mask
        )
        return faces, boundaries, mask

    def test_a_uniform_stream_sees_no_wall_on_the_symmetry_half(self):
        """The face value equals the cell value, so the diffusive flux is zero.

        This is the whole condition in one line. A stream parallel to the plane
        must be able to slide along it without the boundary taking any momentum
        out, and the mirrored face value is what makes the coupling
        ``mu g (u_face - u_cell)`` vanish identically rather than nearly.
        """
        faces, boundaries, mask = self._plate()
        u = np.full(faces.shape, 10.0)
        v = np.zeros(faces.shape)

        wall_u, wall_v = boundaries.wall_velocity(u, v)
        assert np.allclose(wall_u[~mask], 10.0)
        assert np.allclose(wall_v[~mask], 0.0)
        # And the solid half is still no slip.
        assert np.allclose(wall_u[mask], 0.0)
        assert np.allclose(wall_v[mask], 0.0)

    def test_the_normal_component_is_the_part_that_gets_removed(self):
        """Not the tangential one. The face normal here is ``-y``."""
        faces, boundaries, mask = self._plate()
        u = np.full(faces.shape, 3.0)
        v = np.full(faces.shape, -2.0)

        wall_u, wall_v = boundaries.wall_velocity(u, v)
        assert np.allclose(wall_u[~mask], 3.0)
        assert np.allclose(wall_v[~mask], 0.0)

    def test_a_symmetry_face_carries_no_shear_and_no_friction_drag(self):
        """A friction velocity there would be finite, and fictitious.

        ``friction_velocity`` takes the tangential speed of the first cell, which
        on a symmetry plane is the full freestream. Left unmasked it produces a
        perfectly plausible wall shear along a boundary that is not a surface,
        and ``compute_forces`` integrates it into the drag.
        """
        faces, boundaries, mask = self._plate()
        u = np.full(faces.shape, 10.0)
        v = np.zeros(faces.shape)

        shear = boundaries.wall_shear(u, v)
        assert np.all(shear[~mask] == 0.0)
        assert np.all(shear[mask] > 0.0)

    def test_the_wall_function_viscosity_is_not_applied_off_the_plate(self):
        """The interior value is, because the flux there is an ordinary one."""
        faces, boundaries, mask = self._plate()
        u = np.full(faces.shape, 10.0)
        v = np.zeros(faces.shape)
        interior = np.full(faces.shape[0], 7.0)

        viscosity = boundaries.wall_viscosity(u, v, interior)
        assert np.allclose(viscosity[~mask], 7.0)
        assert np.all(viscosity[mask] >= AIR_15C.viscosity)

    def test_the_force_integral_counts_only_the_solid_half(self):
        """A symmetry plane carries pressure and is not a surface to push on.

        This is the failure that would be quietest: the plane's pressure is real,
        its area vector is real, and integrating it gives a drag that looks
        plausible and is entirely an artefact of where the domain was cut.
        """
        faces, boundaries, mask = self._plate()
        state = State.uniform(faces, AIR_15C, boundaries.freestream)
        state.pressure[:] = 5.0

        masked = compute_forces(
            state, faces, AIR_15C, boundaries.freestream, 1.0, np.zeros(2),
            boundaries,
        )
        whole = compute_forces(
            state, faces, AIR_15C, boundaries.freestream, 1.0, np.zeros(2),
            Boundaries(faces, AIR_15C, boundaries.freestream),
        )
        area = faces.wall.area
        assert np.allclose(
            masked.pressure_force, 5.0 * area[mask].sum(axis=0)
        )
        assert not np.allclose(masked.pressure_force, whole.pressure_force)

    def test_solid_wall_is_all_true_without_a_mask(self):
        """So that a weight built from it leaves an O-grid's arithmetic alone."""
        _, _, faces = uniform_mesh(48)
        boundaries = Boundaries(faces, AIR_15C, Freestream(velocity=30.0))
        assert boundaries.solid_wall.all()
        assert boundaries.solid_wall.shape == (faces.shape[0],)

    def test_a_masked_boundary_refuses_to_guess_the_velocity(self):
        """Because on the symmetry half the answer depends on the flow."""
        _, boundaries, _ = self._plate()
        with pytest.raises(ValueError, match="needs the velocity"):
            boundaries.wall_velocity()


class TestMixedGradientRow:
    """A gradient whose wall row is Dirichlet on some faces and Neumann on others.

    ``omega`` is the field that needs this: the near-wall asymptote is prescribed
    on the solid part of the row and means nothing on the symmetry part, where
    the normal derivative vanishes instead. Without a per-face choice the
    asymptote -- which is ``6 nu / (beta1 d^2)`` and therefore enormous on a fine
    mesh -- would be asserted on a face that has no boundary layer behind it.
    """

    def test_the_mask_reproduces_both_pure_conditions_at_its_extremes(self):
        _, _, faces = open_rectangle()
        gradient = ops.Gradient(faces)
        field = np.linspace(0.0, 1.0, faces.shape[0] * faces.shape[1]).reshape(
            faces.shape
        )
        wall = np.full(faces.shape[0], 0.25)
        far = field[:, -1]
        ends = (field[0], field[-1])

        dirichlet = gradient(field, wall, far, *ends)
        neumann = gradient(field, None, far, *ends)

        all_solid = gradient(
            field, wall, far, *ends,
            wall_active=np.ones(faces.shape[0], dtype=bool),
        )
        none_solid = gradient(
            field, wall, far, *ends,
            wall_active=np.zeros(faces.shape[0], dtype=bool),
        )
        assert np.allclose(all_solid, dirichlet)
        assert np.allclose(none_solid, neumann)

    def test_each_half_of_a_mixed_row_matches_its_own_pure_condition(self):
        """The two halves must not contaminate each other.

        The gradient is a per-cell least-squares fit, so a cell's answer depends
        only on its own stencil; the mask is therefore a face-by-face choice and
        not a blend. That is worth checking rather than assuming, because the
        seeded pass that produces the zero-normal-gradient value is computed for
        the whole row at once.
        """
        nx = 8
        _, _, faces = open_rectangle(nx=nx)
        gradient = ops.Gradient(faces)
        field = np.linspace(0.0, 1.0, faces.shape[0] * faces.shape[1]).reshape(
            faces.shape
        )
        wall = np.full(nx, 0.25)
        far = field[:, -1]
        ends = (field[0], field[-1])
        mask = np.zeros(nx, dtype=bool)
        mask[nx // 2:] = True

        mixed = gradient(field, wall, far, *ends, wall_active=mask)
        dirichlet = gradient(field, wall, far, *ends)
        neumann = gradient(field, None, far, *ends)

        assert np.allclose(mixed[mask], dirichlet[mask])
        assert np.allclose(mixed[~mask], neumann[~mask])


class TestOpenEnds:
    """The two ``i`` ends of an open mesh, carrying flow in and out.

    They take the far field's characteristic condition rather than a new one:
    each face decides on the sign of ``u . n``. The part that differs is the
    sign convention of ``flux_i``, which is outward at the high end and inward
    at the low one, so most of what is checked here is that nothing downstream
    was told the wrong direction.
    """

    FLUID = Fluid(density=1.0, viscosity=0.01, name="test")

    def _box(self, nx=12, ny=6, velocity=2.0):
        """An open box whose floor is a symmetry plane: nothing to push on."""
        _, _, faces = open_rectangle(nx=nx, ny=ny)
        freestream = Freestream(velocity=velocity)
        boundaries = Boundaries(
            faces, self.FLUID, freestream, wall_mask=np.zeros(nx, dtype=bool)
        )
        coupling = PressureVelocityCoupling(
            faces, self.FLUID, boundaries, Numerics(scheme="linear"),
            wall_model=False,
        )
        state = State.uniform(faces, self.FLUID, freestream)
        return faces, boundaries, coupling, state

    def test_a_uniform_stream_enters_at_the_low_end_and_leaves_at_the_high(self):
        faces, boundaries, _, state = self._box()
        start, end = boundaries.i_inflow_mask(state.flux_i)
        assert start.all()
        assert not end.any()
        start_fixed, end_fixed = boundaries.i_pressure_is_fixed(state.flux_i)
        assert not start_fixed.any() and end_fixed.all()

    def test_a_periodic_mesh_has_no_ends_to_ask_about(self):
        _, _, faces = uniform_mesh(48)
        boundaries = Boundaries(faces, AIR_15C, Freestream(velocity=30.0))
        flux_i = np.zeros(faces.shape)
        assert boundaries.i_inflow_mask(flux_i) == (None, None)
        assert boundaries.i_velocity(flux_i, flux_i, flux_i) == (
            (None, None), (None, None)
        )

    def test_a_uniform_stream_crosses_an_open_box_unchanged(self):
        """The exact fixed point, and every face of the box has to agree to it.

        Inflow imposes the freestream at the low end, the floor is a symmetry
        plane the stream slides along, the top carries no flux and so holds the
        pressure, and the high end extrapolates. Any one of those assembled with
        the wrong sign or the wrong face would push the field off uniform in the
        first iteration, and it would not come back to rounding.
        """
        faces, _, coupling, state = self._box()
        for _ in range(20):
            coupling.iterate(state)

        assert np.abs(state.u - 2.0).max() < 1e-12
        assert np.abs(state.v).max() < 1e-12
        assert np.abs(state.pressure).max() < 1e-12
        # And the mass through the two ends is the same mass.
        assert np.isclose(state.flux_i[0].sum(), state.flux_i[-1].sum(), rtol=1e-13)

    def test_the_corrected_fluxes_satisfy_the_equation_that_produced_them(self):
        """``div(F) after correction == A p' - b``, through the open ends too.

        The same identity the O-grid test checks, and for the same reason: the
        pressure equation puts a Dirichlet coupling on every face that holds the
        pressure, and the flux update has to apply the correction that coupling
        asserts. At the low end the outward direction is ``-i``, so a sign slip
        there leaves an imbalance of exactly ``2 rho D g p'`` in the end column.

        The field is disturbed first so that ``p'`` is not zero, and the high end
        is checked to be holding the pressure, so that the identity is being
        asked something rather than holding vacuously.
        """
        faces, boundaries, coupling, state = self._box()
        x = faces.metrics.centroid[..., 0]
        y = faces.metrics.centroid[..., 1]
        state.u = state.u + 0.3 * np.sin(np.pi * x / x.max()) * np.cos(y)
        state.v = state.v + 0.2 * np.sin(2.0 * np.pi * y / y.max())
        for _ in range(3):
            coupling.iterate(state)

        _, _, diagonal = coupling.momentum(state)
        flux_i, flux_j, d_i, d_j = coupling.face_fluxes(state, diagonal)
        assert boundaries.i_pressure_is_fixed(flux_i)[1].any()

        correction, coefficients, cross_i, cross_j = coupling.pressure_correction(
            state, flux_i, flux_j, d_i, d_j, diagonal
        )
        assert np.abs(correction[-1]).max() > 0.0
        coupling.apply_correction(
            state, correction, flux_i, flux_j, d_i, d_j, diagonal, cross_i, cross_j
        )

        after = ops.divergence(state.flux_i, state.flux_j, faces)
        expected = coefficients.apply(correction) - coefficients.source
        scale = np.abs(coefficients.source).max()
        assert np.abs(after - expected).max() < 1e-10 * scale

    def test_the_pressure_equation_is_not_singular_when_only_an_end_holds_it(self):
        """The top of a flat-plate domain may be inflow everywhere.

        Then no far-field face holds the pressure, and if the i ends were not
        counted the source would be projected to zero mean as though the problem
        were pure Neumann -- which, with the outlet holding it, it is not.
        """
        faces, boundaries, _, state = self._box()
        far_flux = -np.abs(state.flux_j[:, -1]) - 1.0
        assert not boundaries.far_flux_is_solvable(far_flux)
        assert boundaries.far_flux_is_solvable(far_flux, state.flux_i)


class TestRhieChowConsistency:
    """That the pressure-velocity damping vanishes when it is supposed to.

    Rhie-Chow adds the difference between a compact two-cell pressure gradient
    and a smoothly interpolated one. The whole justification for adding it is
    that the difference is a *third* derivative -- it suppresses a checkerboard
    and disappears under refinement without biasing the answer. A term that does
    not vanish for a field with no third derivative is not that term.

    So the test is an identity rather than an order of accuracy. Writing the
    face area as ``S = g d + T`` with ``d`` the centroid-to-centroid vector, and
    taking any exactly linear ``p = G . x``:

        p_N - p_P = G . d,   (grad p)_f = G,

        damping / D_f = g (G . d) + G . T - G . (g d + T) = 0

    identically, on any mesh, for any ``G``. There is no discretisation error to
    allow for and the tolerance is machine precision.

    This is the one measurement that catches an inconsistent compact operator,
    and no manufactured solution in this file can: they exercise the assembled
    convection and diffusion operator against a *prescribed* flux field, while
    the defect lives in the code that builds the flux.
    """

    @staticmethod
    def _damping(faces, metrics, gradient_of_p):
        """The flux ``face_fluxes`` produces for a linear ``p`` and zero velocity.

        Run through the real :meth:`PressureVelocityCoupling.face_fluxes` rather
        than a re-derivation of it, because a test that reimplements the code it
        is testing agrees with it by construction. With the velocity at rest the
        convective part of the flux is exactly zero, so whatever comes back *is*
        the damping.
        """
        from fluidsolver.solver.simple import Numerics, PressureVelocityCoupling

        fluid = Fluid(density=1.0, viscosity=1.0e-3)
        freestream = Freestream(velocity=1.0)
        boundaries = Boundaries(faces, fluid, freestream)
        coupling = PressureVelocityCoupling(
            faces, fluid, boundaries, Numerics(), wall_model=False
        )

        state = State.uniform(faces, fluid, freestream)
        state.u[:] = 0.0
        state.v[:] = 0.0
        state.flux_i[:] = 0.0
        state.flux_j[:] = 0.0
        state.pressure = metrics.centroid @ gradient_of_p

        # Any strictly positive diagonal is a legitimate momentum diagonal here;
        # it scales the damping and cannot create or remove it.
        flux_i, flux_j, _, _ = coupling.face_fluxes(state, np.ones(faces.shape))
        return flux_i, flux_j

    @pytest.mark.parametrize(
        "mesh",
        [lambda: uniform_mesh(96), aerofoil_mesh],
        ids=["near-orthogonal circle", "body-fitted aerofoil"],
    )
    def test_a_linear_pressure_field_produces_no_damping(self, mesh):
        """Fails on both meshes until the compact operator carries the cross term.

        Regression for the missing non-orthogonal correction. As it stands, for a
        linear field the coded ``compact - smooth`` evaluates to
        ``-(grad p)_f . T`` rather than to zero -- verified to 8e-14 against that
        closed form on both of these meshes -- so the damping is proportional to
        the non-orthogonal remainder ``T`` and is spurious in its entirety.

        Measured, as the worst interior face flux against the largest
        ``V |(grad p)_f . S|`` on the same mesh:

            circle 96x32   2.7771e-07     max |T|/|S| 5.8769e-07
            aerofoil       2.5502e-02     max |T|/|S| 1.8046e+00

        Five orders apart, and each tracks its mesh's own ``|T|/|S|`` to within a
        factor of two, which is the closed form above and not a coincidence.
        Note that the circle fails too: ``build_ogrid`` marches even a circular
        body, and the result is orthogonal to about 6e-7 rather than to rounding.
        The defect is one mechanism at two magnitudes, not two different things.

        The tolerance is machine precision because the statement being tested is
        an algebraic identity with no discretisation error in it. It is not sized
        from healthy behaviour -- there is no band here, only zero.

        Both boundary rows are excluded, and not to make the test pass. The
        gradient there is built from the boundary values ``face_fluxes`` chooses
        for its own purposes -- zero-gradient at the wall, and zero on outflow
        faces -- neither of which is the exact value of a manufactured linear
        field, so ``(grad p)_f`` is not exact in those rows and the identity does
        not apply to them. Measured: the gradient is exact to 5.6e-15 in every
        other row and wrong by 0.90 and 69.8 in the two excluded ones. The
        existing manufactured solutions exclude the same two rows for the same
        reason.
        """
        _, metrics, faces = mesh()
        gradient_of_p = np.array([1.7, -0.9])

        flux_i, flux_j = self._damping(faces, metrics, gradient_of_p)

        # The physical term the damping is derived from, carrying the same
        # mobility (here the cell volume, since the diagonal was set to one).
        volume = metrics.volume
        scale = max(
            (np.abs(np.sum(gradient_of_p * metrics.face_i_area, axis=-1)) * volume).max(),
            (
                np.abs(np.sum(gradient_of_p * metrics.face_j_area[:, 1:-1], axis=-1))
                * volume[:, 1:]
            ).max(),
        )
        worst = max(np.abs(flux_i[:, 1:-1]).max(), np.abs(flux_j[:, 2:-2]).max())
        assert worst < 1e-12 * scale, (
            f"a linear pressure field leaves a spurious flux of {worst:.4e}, "
            f"which is {worst / scale:.4e} of the physical pressure term"
        )

    @pytest.mark.parametrize("mesh", [lambda: uniform_mesh(96), aerofoil_mesh],
                             ids=["near-orthogonal circle", "body-fitted aerofoil"])
    def test_the_converged_flux_does_not_depend_on_the_velocity_relaxation(self, mesh):
        """Regression. It did, linearly, and four docstrings said it could not.

        ``momentum`` returns the *under-relaxed* diagonal, so a Rhie-Chow
        mobility built from it is ``alpha_u V / a_P`` and carries the relaxation
        factor into a term that does not vanish at convergence. The damping is now
        built from ``V / a_P`` instead, and the flux it produces is therefore the
        same at any ``alpha_u``.

        This is the algebraic half of the criterion and it is deliberately not the
        whole of it. A test built from the same belief as the code agrees with it
        by construction, so the flux is compared between two relaxation factors on
        an otherwise frozen state rather than being checked against a formula. The
        end-to-end half is a converged cylinder swept over ``alpha_u`` at a
        residual of 1e-9, recorded in
        :meth:`PressureVelocityCoupling.face_fluxes`; a unit test cannot stand in
        for it because it cannot tell a fixed point that is independent of
        ``alpha_u`` from one that is merely reached slowly.
        """
        from fluidsolver.solver.simple import Numerics, PressureVelocityCoupling

        _, metrics, faces = mesh()
        fluid = Fluid(density=1.0, viscosity=1.0e-3)
        freestream = Freestream(velocity=1.0)

        def settled(alpha_u):
            boundaries = Boundaries(faces, fluid, freestream)
            coupling = PressureVelocityCoupling(
                faces, fluid, boundaries,
                Numerics(relax_velocity=alpha_u), wall_model=False,
            )
            state = State.uniform(faces, fluid, freestream)
            # A frozen, non-trivial pressure field, so that the only thing
            # differing between the two runs is the relaxation factor.
            state.pressure = np.sin(2.1 * metrics.centroid[..., 0]) * np.cos(
                1.4 * metrics.centroid[..., 1]
            )
            # The relaxed diagonal, which is what ``momentum`` returns and the
            # only route by which ``alpha_u`` reaches the flux: ``a_P / alpha_u``
            # with the same unrelaxed ``a_P`` in both runs. Passing a diagonal
            # that does not scale with the relaxation would make the test pass
            # for the wrong reason -- and would also make it fail for the wrong
            # reason, since the retention is built to cancel exactly this scaling.
            diagonal = (1.0 + np.abs(metrics.centroid[..., 0])) / alpha_u
            flux_i, flux_j, _, _ = coupling.face_fluxes(state, diagonal)
            return flux_i, flux_j

        slow_i, slow_j = settled(0.4)
        fast_i, fast_j = settled(0.9)

        scale = max(np.abs(fast_i).max(), np.abs(fast_j).max())
        assert np.abs(slow_i - fast_i).max() < 1e-10 * scale
        assert np.abs(slow_j - fast_j).max() < 1e-10 * scale

    def test_the_aerofoil_mesh_is_actually_non_orthogonal(self):
        """The test above is only evidence if its mesh can carry the defect.

        A guard on the guard: if ``aerofoil_mesh`` ever came back orthogonal --
        through a mesher change, or a resample that happened to land on a smooth
        distribution -- the test above would pass while measuring nothing, which
        is exactly how the cylinder gate has been blind to this for the whole
        life of the project.
        """
        _, _, faces = aerofoil_mesh()
        angles = []
        for family in (faces.i_faces, faces.j_faces):
            unit_area = family.area / np.linalg.norm(
                family.area, axis=-1, keepdims=True
            )
            unit_delta = family.delta / np.linalg.norm(
                family.delta, axis=-1, keepdims=True
            )
            angles.append(
                np.degrees(
                    np.arccos(
                        np.clip(np.abs(np.sum(unit_area * unit_delta, axis=-1)), 0.0, 1.0)
                    )
                )
            )
        assert max(a.max() for a in angles) > 30.0
        assert np.mean([a.mean() for a in angles]) > 1.0


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
        correction, coefficients, cross_i, cross_j = coupling.pressure_correction(
            state, flux_i, flux_j, d_i, d_j, diagonal
        )
        coupling.apply_correction(
            state, correction, flux_i, flux_j, d_i, d_j, diagonal, cross_i, cross_j
        )

        after = ops.divergence(state.flux_i, state.flux_j, case.faces)
        expected = coefficients.apply(correction) - coefficients.source
        scale = np.abs(coefficients.source).max()
        assert np.abs(after - expected).max() < 1e-10 * scale

    @pytest.mark.parametrize(
        "mesh, ceiling, floor",
        [(lambda: uniform_mesh(96), 1e-5, 0.0), (aerofoil_mesh, None, 5e-3)],
        ids=["near-orthogonal circle", "body-fitted aerofoil"],
    )
    def test_the_cross_term_is_a_real_part_of_the_correction(self, mesh, ceiling, floor):
        """And is identically zero on an orthogonal mesh, which is why it was missed.

        The flux correction through a face is ``-rho D_f (grad p')_f . S``, which
        splits into ``g (p'_N - p'_P)`` -- the part the matrix holds -- and
        ``(grad p')_f . T``, which it cannot. The second was simply absent.

        Measured here on a *prescribed* smooth correction field rather than on
        whatever a few solver iterations happen to have produced, as the largest
        cross flux against the largest orthogonal correction flux on the same
        face family:

                          i-faces      j-faces
            circle 96x32  3.7653e-07   2.8636e-07
            aerofoil      1.7714e-02   9.1376e-02

        Five orders apart. On the circle the term is at the level of that mesh's
        own residual non-orthogonality -- ``build_ogrid`` marches even a circular
        body, so it is orthogonal to about 6e-7 rather than to rounding -- and on
        the aerofoil it is 1.8% of the correction on the ``i`` faces and 9.1% on
        the ``j`` faces. Either way it was missing entirely.

        The first version of this test ran five solver iterations and measured
        0.224 on the aerofoil, and then read 0.0385 once the far-field vortex
        correction changed the opening transient. Both numbers are true and
        neither is about the cross term: iteration five of a cold start is a
        fragile place to measure a ratio. What the term's size actually depends on
        is the mesh, so the field is prescribed and the measurement is
        deterministic. The real ``_correction_cross_flux`` still does the work.
        """
        from fluidsolver.solver.simple import Numerics, PressureVelocityCoupling

        _, metrics, faces = mesh()
        fluid = Fluid(density=1.0, viscosity=1.0e-3)
        freestream = Freestream(velocity=1.0)
        coupling = PressureVelocityCoupling(
            faces, fluid, Boundaries(faces, fluid, freestream), Numerics(),
            wall_model=False,
        )

        correction = np.sin(1.7 * metrics.centroid[..., 0]) * np.cos(
            1.1 * metrics.centroid[..., 1]
        )
        mobility = metrics.volume
        d_i = faces.i_faces.interpolate(mobility, np.roll(mobility, 1, axis=0))
        d_j = faces.j_faces.interpolate(mobility[:, 1:], mobility[:, :-1])
        fixed = np.zeros(faces.shape[0], dtype=bool)

        cross_i, _ = coupling._correction_cross_flux(correction, d_i, d_j, fixed)
        orthogonal = np.abs(
            fluid.density
            * d_i
            * faces.i_faces.diffusion_factor
            * (correction - np.roll(correction, 1, axis=0))
        )
        fraction = np.abs(cross_i).max() / orthogonal.max()

        if ceiling is not None:
            assert fraction < ceiling
        assert fraction > floor

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

    @pytest.mark.parametrize(
        "mesh", [uniform_mesh, stretched_mesh, sheared_mesh],
        ids=["uniform", "stretched", "sheared"],
    )
    def test_the_wall_pressure_is_second_order(self, mesh):
        """Regression. It was the cell value, which is zeroth-order extrapolation.

        A first-order boundary term in the force integral caps the observed order
        of every coefficient the solver reports, and no amount of second-order
        accuracy in the interior operators recovers it. Measured on the gate's own
        mesh, this term is worth 0.143% of ``Cd``.

        The order is measured against a manufactured field rather than against a
        finer mesh of the same solution, so the exact answer is known at every
        refinement and there is no extrapolation in the test itself.

        The comparison against the cell value is part of the assertion. Without
        it, a reconstruction that happened to be second order for an unrelated
        reason would pass; the point is that this specific term went from first
        order to second.
        """
        def error(n):
            _, metrics, faces = mesh(n)
            state = State.uniform(
                faces, Fluid(density=1.0, viscosity=1.0e-3), Freestream(velocity=1.0)
            )
            state.pressure = scalar(metrics.centroid)
            exact = scalar(faces.wall.centre)
            from fluidsolver.solver.post import wall_pressure

            return (
                np.abs(state.pressure[:, 0] - exact).max(),
                np.abs(wall_pressure(state, faces) - exact).max(),
            )

        rows = [error(n) for n in (48, 96, 192)]
        cell = observed_order([r[0] for r in rows])
        reconstructed = observed_order([r[1] for r in rows])

        assert 0.9 < cell < 1.2
        assert reconstructed > 1.9

    def test_lift_acting_ahead_of_the_reference_is_a_nose_up_moment(self, setup):
        """Regression. Every reported ``Cm`` had the opposite sign to the convention.

        The integrand was ``r_x F_y - r_y F_x``, the ``+z`` component of ``r x F``
        with ``z`` out of the page. Lift is ``+y`` and the freestream runs ``+x``,
        so the leading edge is at smaller ``x``; a unit lift applied one length
        ahead of the reference has ``r = (-1, 0)`` and ``F = (0, 1)``, and that
        expression returns ``-1``. Lift ahead of the moment reference is a
        *nose-up* moment, which the aerodynamic convention reports as ``+1``.

        The sign is not a matter of taste here because the number's purpose is
        comparison against published section data, where ``Cm`` is nose-up
        positive. On the NACA 2412 at 5 degrees the code reported ``Cm = -0.0825``
        where a section with ``Cm_ac`` near -0.05, about a reference roughly
        0.42c aft of the aerodynamic centre at ``Cl = 0.75``, should read about
        ``+0.08``: right magnitude, wrong sign.

        The body here is a circle, and that needs saying because it constrains
        how the test can be built: pressure acts along the surface normal, every
        normal of a circle is radial, so *every* pressure distribution on a circle
        has exactly zero moment about its centre. The first attempt at this test
        loaded the forward upper quadrant and measured a moment of 1.5e-18, which
        is the correct answer to the question it was asking.

        So the reference is moved instead. About any point ``P``, the moment of a
        purely radial load is ``sum (r_i - P) x F_i = -P x F_total``, exactly.
        With the reference at the rear of the body and the whole upper surface in
        suction, the resultant lift acts one radius *ahead* of it, and the
        nose-up moment is ``+0.5 L`` in closed form.
        """
        grid, faces, fluid, freestream, state = setup
        state.u[:] = 0.0
        state.v[:] = 0.0

        centre = grid.contour.centroid
        reference = centre + np.array([0.5, 0.0])

        state.pressure[:] = 0.0
        state.pressure[faces.wall.centre[:, 1] > centre[1], 0] = -1.0

        forces = compute_forces(state, faces, fluid, freestream, 1.0, reference)
        assert forces.lift > 0.0
        assert forces.moment == pytest.approx(0.5 * forces.lift, rel=1e-9)
        assert forces.moment_coefficient > 0.0

    def test_separation_follows_the_wall_gradient_not_the_first_cell(self, setup):
        """Regression. The sign came from ``u_t(y1)``, which reverses too late.

        In a separating boundary layer the profile is inflected: the reversed
        region grows outward from the surface, so ``du_t/dy`` changes sign at the
        wall *before* the first-cell velocity does. Between the two there is a
        band of surface that has separated and does not look separated. The
        disagreement is ``O(y1)``, first order in the wall spacing, and on the
        converged Re 40 cylinder it moved the separation angle by 0.253 degrees --
        53.71710 from the first-cell velocity against 53.97023 from a one-sided
        wall gradient, on a figure the gate prints to three decimals.

        Built here as exactly that band: a velocity field whose first cell is
        still moving forward everywhere while the wall gradient has already
        reversed over an arc. The first-cell test finds no separation at all on
        this field; the wall-gradient test finds it.
        """
        _, faces, fluid, _, state = setup

        centre = faces.wall.centre
        tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)

        # u_t(y1) > 0 everywhere, u_t(y2) large enough that the quadratic through
        # the wall and the two centres has negative slope over part of the body.
        angle = np.arctan2(centre[:, 1], centre[:, 0])
        reversed_arc = np.cos(angle) < -0.3
        first = np.full(centre.shape[0], 0.01)
        second = np.where(reversed_arc, 1.0, 0.02)

        state.u[:] = 0.0
        state.v[:] = 0.0
        state.u[:, 0], state.v[:, 0] = first * tangent[:, 0], first * tangent[:, 1]
        state.u[:, 1], state.v[:, 1] = second * tangent[:, 0], second * tangent[:, 1]

        from fluidsolver.solver.post import _wall_gradient_sign, separation_points

        sign = _wall_gradient_sign(state, faces, tangent)
        assert np.all(first > 0.0), "the first cell must not itself be reversed"
        assert (sign < 0).any(), "the wall gradient must reverse somewhere"
        assert len(separation_points(state, faces, fluid)) > 0

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
