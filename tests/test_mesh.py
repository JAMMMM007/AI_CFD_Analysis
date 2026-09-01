"""Mesh-layer tests: wall spacing, hyperbolic marching, O-grid assembly, metrics.

The circle carries most of the weight here, because it is the one body whose
offset grid is known in closed form: concentric circles, exactly. Anything the
marcher gets wrong shows up against it immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluidsolver.geometry.naca import naca4
from fluidsolver.geometry.primitives import circle, rectangle, square
from fluidsolver.mesh import spacing
from fluidsolver.mesh.hyperbolic import MeshError, hyperbolic_grid
from fluidsolver.mesh.metrics import cell_face_areas, compute_metrics
from fluidsolver.mesh.ogrid import build_ogrid
from fluidsolver.mesh.quality import assess
from fluidsolver.mesh.spacing import SpacingError

# Air at 15 C, 30 m/s over a 1 m chord: Re = 2.0e6, the NACA 0012 validation case.
AIR = dict(velocity=30.0, length=1.0, density=1.225, viscosity=1.81e-5)


@pytest.fixture(scope="module")
def first_layer():
    return spacing.first_layer_thickness(1.0, **AIR)


@pytest.fixture(scope="module")
def bodies():
    return {
        "circle": circle(1.0, 240),
        "square": square(1.0, 240).resample(240),
        "naca2412": naca4("2412", 1201).resample(240),
        "naca0012": naca4("0012", 1201).resample(240),
        "rounded": rectangle(2.0, 1.0, 240, corner_radius=0.05).resample(240),
    }


# ----------------------------------------------------------------------
# Wall spacing
# ----------------------------------------------------------------------


class TestSpacing:
    def test_first_cell_centre_lands_on_the_requested_y_plus(self, first_layer):
        """The centre sits at half the cell height; that factor of two is the point."""
        assert spacing.y_plus_of(first_layer, **AIR) == pytest.approx(1.0)

    def test_y_plus_is_inversely_proportional_to_spacing(self):
        thin = spacing.first_layer_thickness(1.0, **AIR)
        thick = spacing.first_layer_thickness(30.0, **AIR)
        assert thick / thin == pytest.approx(30.0)

    def test_friction_velocity_follows_the_correlation(self):
        """u_tau = U sqrt(Cf / 2) with Cf = 0.026 Re^(-1/7)."""
        reynolds = AIR["density"] * AIR["velocity"] * AIR["length"] / AIR["viscosity"]
        expected = AIR["velocity"] * np.sqrt(0.5 * 0.026 * reynolds ** (-1 / 7))
        assert spacing.friction_velocity(**AIR) == pytest.approx(expected)

    @pytest.mark.parametrize("growth", [1.05, 1.15, 1.3])
    def test_geometric_layers_span_exactly_and_grow_as_asked(self, growth):
        layers = spacing.geometric_layers(1e-5, 40.0, growth)
        assert layers.sum() == pytest.approx(40.0)
        assert layers[0] == pytest.approx(1e-5, rel=1e-9)
        ratios = layers[1:] / layers[:-1]
        assert np.allclose(ratios, ratios[0])
        assert ratios[0] <= growth + 1e-9

    def test_layers_for_count_hits_the_count_and_the_distance(self):
        layers = spacing.layers_for_count(1e-4, 20.0, 60)
        assert len(layers) == 60
        assert layers.sum() == pytest.approx(20.0)

    def test_an_impossible_stretching_request_is_refused(self):
        with pytest.raises(SpacingError, match="growth ratio"):
            spacing.layers_for_count(1e-8, 40.0, 10)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(first=0.0, total=1.0, growth=1.1), "must be positive"),
            (dict(first=2.0, total=1.0, growth=1.1), "thinner than"),
            (dict(first=0.1, total=1.0, growth=0.5), "at least 1"),
        ],
    )
    def test_invalid_distributions_are_refused(self, kwargs, message):
        with pytest.raises(SpacingError, match=message):
            spacing.geometric_layers(**kwargs)


# ----------------------------------------------------------------------
# Hyperbolic marching, against the one case with an exact answer
# ----------------------------------------------------------------------


class TestMarchingAgainstTheExactSolution:
    def test_circle_marches_to_concentric_circles_exactly(self):
        """Offsetting a circle outward gives circles of radius r + distance.

        This is the whole scheme under test at once: the sign of the source term,
        the mid-layer evaluation of the cell width, and the Newton iteration. Get
        the width from the outer layer instead of the midpoint and each layer
        advances short by a factor (1 - step/radius), which is invisible at the
        wall and loses over a tenth of the marched distance in the far field.
        """
        layers = spacing.geometric_layers(1e-4, 20.0, 1.12)
        nodes, completed = hyperbolic_grid(circle(1.0, 240), layers, dissipation=0.0)

        assert completed == len(layers)
        radius = np.linalg.norm(nodes, axis=2)
        exact = 0.5 + np.concatenate(([0.0], np.cumsum(layers)))
        assert np.abs(radius / exact[None, :] - 1.0).max() < 1e-12

    def test_marched_grid_stays_orthogonal_to_the_wall(self, bodies, first_layer):
        layers = spacing.geometric_layers(first_layer, 0.1, 1.15)
        for name, body in bodies.items():
            nodes, _ = hyperbolic_grid(body, layers)
            along = np.roll(nodes[:, 0], -1, axis=0) - np.roll(nodes[:, 0], 1, axis=0)
            outward = nodes[:, 1] - nodes[:, 0]
            cosine = np.abs(np.sum(along * outward, axis=1)) / (
                np.linalg.norm(along, axis=1) * np.linalg.norm(outward, axis=1)
            )
            worst = np.degrees(np.arcsin(np.clip(cosine, 0.0, 1.0))).max()
            assert worst < 2.0, f"{name}: wall angle off by {worst:.2f} deg"

    def test_first_layer_thickness_is_honoured(self, bodies, first_layer):
        layers = spacing.geometric_layers(first_layer, 0.1, 1.15)
        for name, body in bodies.items():
            nodes, _ = hyperbolic_grid(body, layers)
            achieved = np.linalg.norm(nodes[:, 1] - nodes[:, 0], axis=1)
            assert achieved == pytest.approx(first_layer, rel=0.02), name

    def test_partial_march_stops_before_quality_collapses(self):
        """Cells are squashed for many layers before any of them inverts.

        Without a quality trigger the square's last marched layers carry cells
        fifty times their neighbours -- every area still positive, and the mesh
        still useless.
        """
        body = square(1.0, 240).resample(240)
        layers = spacing.geometric_layers(
            spacing.first_layer_thickness(1.0, **AIR), 1.0, 1.15
        )
        _, guarded = hyperbolic_grid(body, layers, allow_partial=True, max_width_ratio=3.0)
        _, unguarded = hyperbolic_grid(
            body, layers, allow_partial=True, max_width_ratio=1e9
        )
        assert 0 < guarded < unguarded

    def test_bad_layer_specifications_are_refused(self):
        with pytest.raises(MeshError, match="positive"):
            hyperbolic_grid(circle(1.0, 64), np.array([1e-3, -1e-3]))


# ----------------------------------------------------------------------
# Assembled O-grid
# ----------------------------------------------------------------------


class TestOGrid:
    def test_every_body_produces_a_valid_grid(self, bodies, first_layer):
        for name, body in bodies.items():
            grid = build_ogrid(body, first_layer=first_layer, far_field_radius=40.0)
            report = assess(compute_metrics(grid.nodes), grid.nodes)
            assert report.is_usable, f"{name}: {report.summary()}"
            assert report.negative_volumes == 0, name
            assert report.max_skewness < 1.0, f"{name}: skewness {report.max_skewness}"
            assert report.max_expansion_ratio < 10.0, name

    def test_outer_boundary_is_the_circle_that_was_asked_for(self, bodies, first_layer):
        for name, body in bodies.items():
            grid = build_ogrid(body, first_layer=first_layer, far_field_radius=35.0)
            radius = np.linalg.norm(grid.far_field - body.centroid, axis=1)
            assert radius == pytest.approx(35.0, rel=1e-9), name

    def test_outer_boundary_is_evenly_spaced(self, bodies, first_layer):
        """The far field must forget the body's clustering, or the freestream
        boundary carries cells hundreds of times each other's size."""
        for name, body in bodies.items():
            grid = build_ogrid(body, first_layer=first_layer, far_field_radius=35.0)
            edges = np.linalg.norm(
                np.roll(grid.far_field, -1, axis=0) - grid.far_field, axis=1
            )
            assert edges.max() / edges.min() < 1.3, name

    def test_wall_nodes_are_the_body(self, first_layer):
        body = naca4("2412", 1201).resample(200)
        grid = build_ogrid(body, first_layer=first_layer, far_field_radius=30.0)
        assert np.allclose(grid.wall, body.as_wall_line())

    def test_a_far_field_inside_the_body_is_refused(self, first_layer):
        with pytest.raises(MeshError, match="not clear of the body"):
            build_ogrid(circle(1.0, 120), first_layer=first_layer, far_field_radius=0.6)

    def test_an_early_handover_is_reported_not_hidden(self, first_layer):
        """A sharp square cannot be marched as far as a smooth body, and the grid
        should say so rather than quietly returning something different."""
        grid = build_ogrid(
            square(1.0, 240).resample(240), first_layer=first_layer, far_field_radius=40.0
        )
        assert grid.marched_layers < len(grid.thicknesses)
        assert any("march stopped" in note for note in grid.notes)


# ----------------------------------------------------------------------
# Finite-volume metrics
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid():
    return build_ogrid(
        naca4("2412", 1201).resample(160),
        first_layer=spacing.first_layer_thickness(1.0, **AIR),
        far_field_radius=40.0,
    )


class TestMetrics:
    def test_face_normals_of_each_cell_sum_to_zero(self, grid):
        """The closure identity. Every conservation property downstream rests on it."""
        metrics = compute_metrics(grid.nodes)
        closure = np.abs(sum(cell_face_areas(metrics))).max()
        scale = np.linalg.norm(metrics.face_j_area, axis=-1).max()
        assert closure / scale < 1e-13

    def test_volume_agrees_with_the_divergence_theorem(self, grid):
        """V = 1/2 sum(face centre . outward normal), independent of the shoelace."""
        metrics = compute_metrics(grid.nodes)
        west, east, south, north = cell_face_areas(metrics)
        gauss = 0.5 * (
            np.sum(metrics.face_i_centre * west, -1)
            + np.sum(np.roll(metrics.face_i_centre, -1, axis=0) * east, -1)
            + np.sum(metrics.face_j_centre[:, :-1] * south, -1)
            + np.sum(metrics.face_j_centre[:, 1:] * north, -1)
        )
        assert np.abs(gauss / metrics.volume - 1.0).max() < 1e-9

    def test_all_volumes_are_positive(self, grid):
        assert (compute_metrics(grid.nodes).volume > 0.0).all()

    def test_total_volume_of_an_annulus_is_exact(self):
        """A circle in a circular far field encloses a known area."""
        grid = build_ogrid(circle(1.0, 720), first_layer=1e-4, far_field_radius=20.0)
        metrics = compute_metrics(grid.nodes)
        exact = np.pi * (20.0**2 - 0.5**2)
        assert metrics.total_volume == pytest.approx(exact, rel=1e-4)

    def test_wall_distance_of_the_first_cell_is_half_its_height(self):
        """The SST wall condition is omega = 60 nu / (beta1 d1^2), so a wrong d1
        goes straight into the turbulence at the wall as an inverse square."""
        first = spacing.first_layer_thickness(1.0, **AIR)
        grid = build_ogrid(circle(1.0, 360), first_layer=first, far_field_radius=40.0)
        distance = compute_metrics(grid.nodes).wall_distance[:, 0]
        assert distance == pytest.approx(0.5 * first, rel=1e-3)

    def test_wall_distance_matches_geometry_on_a_circle(self):
        grid = build_ogrid(circle(1.0, 720), first_layer=1e-3, far_field_radius=20.0)
        metrics = compute_metrics(grid.nodes)
        radial = np.linalg.norm(metrics.centroid, axis=-1) - 0.5
        assert np.abs(metrics.wall_distance - radial).max() < 1e-5

    def test_wall_face_normals_point_into_the_solid(self, grid):
        """Forces are integrated with these, so a sign slip flips lift and drag."""
        metrics = compute_metrics(grid.nodes)
        outward = grid.wall - grid.contour.centroid
        assert (np.sum(metrics.wall_face_area() * outward, axis=-1) < 0.0).all()

    def test_wall_face_lengths_sum_to_the_perimeter(self):
        grid = build_ogrid(circle(1.0, 720), first_layer=1e-3, far_field_radius=20.0)
        metrics = compute_metrics(grid.nodes)
        total = np.linalg.norm(metrics.wall_face_area(), axis=-1).sum()
        assert total == pytest.approx(np.pi, rel=1e-4)

    def test_centroids_are_area_centroids(self):
        """On a boundary-layer cell stretched thousands to one, the area centroid
        and the mean of the corners are not interchangeable."""
        grid = build_ogrid(circle(1.0, 360), first_layer=1e-4, far_field_radius=20.0)
        metrics = compute_metrics(grid.nodes)
        radius = np.linalg.norm(metrics.centroid, axis=-1)
        assert (np.diff(radius, axis=1) > 0.0).all()


class TestQuality:
    def test_a_circle_grid_is_perfectly_orthogonal(self):
        grid = build_ogrid(circle(1.0, 240), first_layer=1e-4, far_field_radius=30.0)
        report = assess(compute_metrics(grid.nodes), grid.nodes)
        assert report.max_non_orthogonality_deg < 0.1
        assert report.warnings == []

    def test_inverted_cells_make_a_mesh_unusable(self):
        grid = build_ogrid(circle(1.0, 120), first_layer=1e-4, far_field_radius=30.0)
        nodes = grid.nodes.copy()
        nodes[5, 3] = nodes[7, 3]  # collapse a cell onto its neighbour
        report = assess(compute_metrics(nodes), nodes)
        assert not report.is_usable
        assert any("inverted" in w for w in report.warnings)

    def test_summary_mentions_every_headline_number(self):
        grid = build_ogrid(circle(1.0, 120), first_layer=1e-4, far_field_radius=30.0)
        text = assess(compute_metrics(grid.nodes), grid.nodes).summary()
        for heading in ("cells", "volume", "non-orthogonality", "skewness", "aspect"):
            assert heading in text
