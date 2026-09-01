"""Geometry-layer tests: contours, analytic shapes, resampling and DXF import.

Values are checked against the analytic definitions of the shapes rather than
against previously-recorded output, so these fail if the maths drifts rather than
merely if the numbers change.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluidsolver.geometry.contour import Contour, ContourError, _limit_growth
from fluidsolver.geometry.naca import naca4
from fluidsolver.geometry.primitives import circle, rectangle, square

SQUARE_CORNERS = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]])


def max_spacing_ratio(contour: Contour) -> float:
    """Largest size ratio between neighbouring cells around the loop."""
    h = contour.segment_lengths()
    return float(np.maximum(np.roll(h, -1) / h, h / np.roll(h, -1)).max())


def corner_error(contour: Contour) -> float:
    """Distance from each unit-square corner to the nearest contour point."""
    d = np.linalg.norm(contour.points[:, None, :] - SQUARE_CORNERS[None, :, :], axis=2)
    return float(d.min(axis=0).max())


# ----------------------------------------------------------------------
# Contour invariants
# ----------------------------------------------------------------------


class TestContour:
    def test_orientation_is_normalised_to_counter_clockwise(self):
        clockwise = [[0, 0], [0, 1], [1, 1], [1, 0]]
        assert Contour(clockwise).signed_area > 0.0

    def test_repeated_closing_point_is_dropped(self):
        assert len(Contour([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])) == 4

    def test_area_and_centroid_of_a_unit_square(self):
        c = square(1.0, 64)
        assert c.area == pytest.approx(1.0, abs=1e-12)
        assert c.centroid == pytest.approx([0.0, 0.0], abs=1e-12)

    def test_centroid_is_the_area_centroid_not_the_point_mean(self):
        # A unit square whose points bunch up on the left edge. The two measures
        # must disagree, and it is the area centroid that belongs at x = 0.5.
        left_edge = np.column_stack((np.zeros(50), np.linspace(1, 0, 50)))
        c = Contour(np.vstack(([[0, 0], [1, 0], [1, 1]], left_edge)))
        assert c.area == pytest.approx(1.0, abs=1e-12)
        assert c.centroid[0] == pytest.approx(0.5, abs=1e-9)
        assert c.points.mean(axis=0)[0] < 0.1

    def test_outward_normals_point_away_from_the_body(self):
        c = circle(2.0, 128)
        radial = c.points / np.linalg.norm(c.points, axis=1, keepdims=True)
        assert np.allclose(c.outward_normals(), radial, atol=1e-12)

    def test_curvature_of_a_circle_is_one_over_radius(self):
        assert circle(2.0, 400).curvature().mean() == pytest.approx(1.0, rel=1e-6)

    def test_turning_angles_find_exactly_the_square_corners(self):
        angles = np.degrees(square(1.0, 200).turning_angles())
        assert (angles > 80.0).sum() == 4

    def test_as_wall_line_reverses_to_put_fluid_on_the_left(self):
        c = circle(1.0, 64)
        wall = Contour(c.as_wall_line())
        # Contour re-orients on construction, so compare the raw arrays instead.
        assert np.allclose(c.as_wall_line(), c.points[::-1])
        assert len(wall) == len(c)

    def test_rigid_transforms_preserve_area(self):
        c = naca4("2412", 200)
        moved = c.rotated(17.0).translated(3.0, -2.0)
        assert moved.area == pytest.approx(c.area, rel=1e-12)

    def test_scaling_scales_the_reference_length(self):
        c = circle(2.0, 1024).scaled(3.0)
        assert c.reference_length == pytest.approx(6.0)
        assert c.area == pytest.approx(np.pi * 9.0, rel=1e-4)

    @pytest.mark.parametrize(
        "points, message",
        [
            ([[0, 0], [1, 0]], "at least 3"),
            ([[0, 0], [1, 0], [2, 0]], "no area"),
            ([[0, 0], [1, 0], [0, 1], [1, 1]], "no area"),  # bow-tie
        ],
    )
    def test_degenerate_input_is_rejected(self, points, message):
        with pytest.raises(ContourError, match=message):
            Contour(points).validate()

    def test_self_intersecting_loop_is_rejected(self):
        # A figure-eight with non-zero net area still has to be refused.
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        pts = np.column_stack((np.sin(t), np.sin(2 * t))) + [[3.0, 0.0]]
        with pytest.raises(ContourError, match="self-intersect|no area|not a valid ring"):
            Contour(pts).validate()


# ----------------------------------------------------------------------
# Sizing field and resampling
# ----------------------------------------------------------------------


class TestGradationLimiter:
    @pytest.mark.parametrize("scale", [1e-2, 1.0, 1e2])
    def test_limit_is_scale_free(self, scale):
        """A spacing field and its rescaling must be graded identically."""
        h = np.full(2000, 0.05 * scale)
        h[1000] = 0.002 * scale
        ds = 0.001 * scale
        out = _limit_growth(h, 1.2, ds)

        worst = max(
            out[(i + max(1, round(out[i] / ds))) % 2000] / out[i] for i in range(2000)
        )
        assert worst <= 1.24  # 1.2 plus half a sample of discretisation
        assert out.min() == pytest.approx(0.002 * scale)  # refinement survives

    def test_limiter_only_reduces_spacing(self):
        rng = np.random.default_rng(0)
        h = rng.uniform(0.01, 0.1, 500)
        assert np.all(_limit_growth(h, 1.2, 1e-3) <= h + 1e-15)


class TestResample:
    @pytest.mark.parametrize("n", [61, 100, 179, 400])
    def test_square_corners_are_reproduced_exactly(self, n):
        assert corner_error(square(1.0, 256).resample(n)) < 1e-12

    @pytest.mark.parametrize("n", [61, 179, 400])
    def test_square_area_is_preserved(self, n):
        assert square(1.0, 256).resample(n).area == pytest.approx(1.0, abs=1e-12)

    def test_resampling_refines_into_a_sharp_corner(self):
        r = square(1.0, 256).resample(400)
        h = r.segment_lengths()
        at_corner = h[int(np.argmin(np.linalg.norm(r.points - SQUARE_CORNERS[0], axis=1)))]
        assert np.median(h) / at_corner > 5.0

    @pytest.mark.parametrize("source_points", [128, 512])
    def test_circle_resamples_uniformly(self, source_points):
        """Constant curvature must give constant spacing.

        This is the regression test for measuring curvature between adjacent
        dense samples: on a polygonal circle that reads as alternating flats and
        corners, and the points pile onto the source polygon's vertices.
        """
        h = circle(1.0, source_points).resample(200).segment_lengths()
        assert h.max() / h.min() < 1.01

    def test_aerofoil_clusters_at_the_leading_edge(self):
        r = naca4("2412", 1201).resample(241)
        h = r.segment_lengths()
        le = h[int(np.argmin(r.x))]
        assert 0.0005 < le < 0.004  # 0.05%-0.4% chord, the usual range
        assert np.median(h) / le > 4.0

    def test_gradation_holds_away_from_the_blunt_trailing_edge(self):
        """The trailing-edge base must span a cell, so it is excluded.

        A 0.25%-chord base sits next to ~1%-chord surface cells; that jump is
        geometry, not a meshing defect.
        """
        r = naca4("2412", 1201).resample(241)
        h = r.segment_lengths()
        ratios = np.maximum(np.roll(h, -1) / h, h / np.roll(h, -1))
        away_from_te = np.abs(r.x - r.x.max()) > 0.02 * r.reference_length
        assert ratios[away_from_te].max() < 1.35

    def test_too_few_points_for_the_corners_is_refused(self):
        with pytest.raises(ContourError, match="corners"):
            square(1.0, 64).resample(3)


# ----------------------------------------------------------------------
# Analytic shapes
# ----------------------------------------------------------------------


class TestNaca:
    def _surfaces(self, contour):
        """Split a section loop into upper and lower surfaces at the leading edge."""
        i_le = int(np.argmin(contour.x))
        return contour.points[: i_le + 1][::-1], contour.points[i_le:]

    @pytest.mark.parametrize("code, thickness", [("0006", 0.06), ("0012", 0.12), ("2415", 0.15)])
    def test_maximum_thickness_matches_the_code(self, code, thickness):
        c = naca4(code, 601)
        upper, lower = self._surfaces(c)
        x = np.linspace(0.01, 0.99, 2000)
        t = np.interp(x, upper[:, 0], upper[:, 1]) - np.interp(x, lower[:, 0], lower[:, 1])
        assert t.max() == pytest.approx(thickness, rel=2e-3)

    def test_maximum_thickness_sits_at_thirty_percent_chord(self):
        c = naca4("0012", 601)
        upper, lower = self._surfaces(c)
        x = np.linspace(0.01, 0.99, 2000)
        t = np.interp(x, upper[:, 0], upper[:, 1]) - np.interp(x, lower[:, 0], lower[:, 1])
        assert x[t.argmax()] == pytest.approx(0.30, abs=0.01)

    @pytest.mark.parametrize("code, camber, position", [("2412", 0.02, 0.4), ("4412", 0.04, 0.4)])
    def test_camber_line_matches_the_code(self, code, camber, position):
        c = naca4(code, 601)
        upper, lower = self._surfaces(c)
        x = np.linspace(0.01, 0.99, 2000)
        mid = 0.5 * (
            np.interp(x, upper[:, 0], upper[:, 1]) + np.interp(x, lower[:, 0], lower[:, 1])
        )
        assert mid.max() == pytest.approx(camber, rel=5e-3)
        assert x[mid.argmax()] == pytest.approx(position, abs=0.015)

    def test_open_trailing_edge_has_the_analytic_base_thickness(self):
        """5t * 0.0021 per surface, so 2 * 0.0105 * t overall."""
        c = naca4("0012", 401)
        i_le = int(np.argmin(c.x))
        base = abs(c.points[0][1] - c.points[2 * i_le][1])
        assert base == pytest.approx(2 * 0.0105 * 0.12, rel=1e-6)

    def test_closed_trailing_edge_really_closes(self):
        c = naca4("0012", 401, closed_trailing_edge=True)
        assert abs(c.points[0][1]) < 1e-15

    def test_symmetric_section_is_symmetric(self):
        c = naca4("0012", 401)
        upper, lower = self._surfaces(c)
        x = np.linspace(0.01, 0.99, 500)
        assert np.allclose(
            np.interp(x, upper[:, 0], upper[:, 1]),
            -np.interp(x, lower[:, 0], lower[:, 1]),
            atol=1e-12,
        )

    def test_chord_scales_the_section_and_the_reference_length(self):
        c = naca4("2412", 301, chord=2.5)
        assert c.reference_length == pytest.approx(2.5)
        assert c.bounds[2] - c.bounds[0] == pytest.approx(2.5, rel=1e-3)

    @pytest.mark.parametrize("code", ["241", "24123", "abcd", "2400"])
    def test_invalid_codes_are_rejected(self, code):
        with pytest.raises(ContourError):
            naca4(code)

    def test_camber_at_the_leading_edge_is_rejected(self):
        with pytest.raises(ContourError, match="position"):
            naca4("2012")


class TestPrimitives:
    def test_circle_area_and_perimeter_converge(self):
        c = circle(1.0, 2048)
        assert c.area == pytest.approx(np.pi / 4, rel=1e-5)
        assert c.perimeter == pytest.approx(np.pi, rel=1e-5)
        assert c.reference_length == pytest.approx(1.0)

    def test_square_is_exact(self):
        c = square(2.0, 200)
        assert c.area == pytest.approx(4.0, abs=1e-12)
        assert c.perimeter == pytest.approx(8.0, abs=1e-12)
        assert c.reference_length == pytest.approx(2.0)

    def test_sharp_square_clusters_towards_its_corners(self):
        h = square(1.0, 200).segment_lengths()
        assert h.max() / h.min() > 5.0

    def test_filleted_corners_do_not_create_a_spacing_jump(self):
        """Clustering into an already-rounded corner leaves a 37:1 jump where the
        arc meets the straight edge; the fillet must switch clustering off."""
        assert max_spacing_ratio(rectangle(2.0, 1.0, 300, corner_radius=0.15)) < 1.1

    def test_fillet_reduces_area_by_the_corner_cutoff(self):
        r = 0.15
        c = rectangle(2.0, 1.0, 400, corner_radius=r)
        expected = 2.0 * 1.0 - (4 - np.pi) * r**2
        assert c.area == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("bad", [-0.1, 0.5, 1.0])
    def test_impossible_fillet_radius_is_rejected(self, bad):
        with pytest.raises(ContourError, match="corner_radius"):
            rectangle(2.0, 1.0, 200, corner_radius=bad)
