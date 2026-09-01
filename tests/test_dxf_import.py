"""DXF import tests.

Fixtures are written by ezdxf at test time rather than checked in as binaries, so
each one documents in code exactly which awkward CAD habit it represents.
"""

from __future__ import annotations

import numpy as np
import pytest

ezdxf = pytest.importorskip("ezdxf")

from fluidsolver.geometry.dxf_import import (  # noqa: E402
    DxfImportError,
    describe_dxf,
    read_contour,
    read_contours,
)
from fluidsolver.geometry.naca import naca4  # noqa: E402

INCH = 0.0254


@pytest.fixture(scope="module")
def drawings(tmp_path_factory):
    """Write one DXF per import hazard and return a name -> path mapping."""
    out = tmp_path_factory.mktemp("dxf")
    paths = {}

    def save(doc, name):
        path = out / f"{name}.dxf"
        doc.saveas(path)
        paths[name] = path

    # A rectangle as CAD actually stores it: one closed four-vertex polyline.
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4  # millimetres
    doc.modelspace().add_lwpolyline([(0, 0), (100, 0), (100, 100), (0, 100)], close=True)
    save(doc, "square_mm")

    # A single CIRCLE entity, dimensioned in inches.
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 1
    doc.modelspace().add_circle((0, 0), radius=2.0)
    save(doc, "circle_inches")

    # An outline exploded into individual LINEs whose endpoints do not quite
    # meet -- the single most common reason a DXF will not close.
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 6
    points = naca4("2412", 121).points
    rng = np.random.default_rng(0)
    msp = doc.modelspace()
    for i in range(len(points)):
        start, end = points[i], points[(i + 1) % len(points)]
        msp.add_line(tuple(start), tuple(end + rng.normal(0, 2e-7, 2)))
    save(doc, "gapped_lines")

    # Two SPLINEs plus a LINE, with one spline drawn in the opposite direction,
    # so the chainer has to reverse a piece to keep a consistent orientation.
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 6
    points = naca4("0012", 201, n_trailing_edge=2).points
    i_le = int(np.argmin(points[:, 0]))
    msp = doc.modelspace()
    msp.add_spline(np.column_stack((points[: i_le + 1], np.zeros(i_le + 1))))
    msp.add_spline(np.column_stack((points[i_le:], np.zeros(len(points) - i_le)))[::-1])
    msp.add_line(tuple(points[-1]), tuple(points[0]))
    save(doc, "reversed_splines")

    # The profile inside a BLOCK, placed by a scaled and rotated INSERT, with
    # annotation on a separate layer that must not be mistaken for geometry.
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4
    doc.blocks.new(name="PROFILE").add_lwpolyline(
        [(0, 0), (40, 0), (40, 20), (0, 20)], close=True
    )
    msp = doc.modelspace()
    msp.add_blockref(
        "PROFILE", (10, 10), dxfattribs={"xscale": 2.0, "yscale": 2.0, "rotation": 30}
    )
    doc.layers.add("NOTES")
    msp.add_text("SECTION A-A", dxfattribs={"layer": "NOTES"}).set_placement((5, 50))
    msp.add_line((-50, -50), (150, -50), dxfattribs={"layer": "NOTES"})
    save(doc, "block_insert")

    # Two independent closed loops in one drawing.
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(-5, -5), (5, -5), (5, 5), (-5, 5)], close=True)
    msp.add_circle((0, 0), radius=1.0)
    save(doc, "two_loops")

    # A genuinely unclosed outline.
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], close=False)
    save(doc, "open_profile")

    # ARCs and LINEs chained into a stadium, to exercise curve flattening.
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    msp.add_line((0, -5), (20, -5))
    msp.add_arc((20, 0), 5, -90, 90)
    msp.add_line((20, 5), (0, 5))
    msp.add_arc((0, 0), 5, 90, 270)
    save(doc, "stadium")

    return paths


class TestUnits:
    def test_millimetre_drawing_is_converted_to_metres(self, drawings):
        c = read_contour(drawings["square_mm"])
        assert c.bounds[2] - c.bounds[0] == pytest.approx(0.1)
        assert c.area == pytest.approx(0.01)

    def test_inch_drawing_is_converted_to_metres(self, drawings):
        c = read_contour(drawings["circle_inches"])
        assert c.area == pytest.approx(np.pi * (2 * INCH) ** 2, rel=1e-3)

    def test_explicit_scale_overrides_the_header(self, drawings):
        c = read_contour(drawings["square_mm"], scale=1.0)
        assert c.bounds[2] - c.bounds[0] == pytest.approx(100.0)

    def test_describe_reports_units_without_importing(self, drawings):
        info = describe_dxf(drawings["circle_inches"])
        assert info["units"] == "inches"
        assert info["metres_per_unit"] == pytest.approx(INCH)
        assert info["entity_counts"] == {"CIRCLE": 1}


class TestChaining:
    def test_gapped_lines_are_chained_into_a_closed_loop(self, drawings):
        reference = naca4("2412", 121)
        imported = read_contour(drawings["gapped_lines"])
        assert imported.area == pytest.approx(reference.area, rel=1e-5)
        assert imported.perimeter == pytest.approx(reference.perimeter, rel=1e-5)

    def test_a_tolerance_below_the_gap_refuses_to_bridge_it(self, drawings):
        with pytest.raises(DxfImportError, match="no closed loop"):
            read_contour(drawings["gapped_lines"], gap_tolerance=1e-12)

    def test_reversed_entities_are_flipped_to_match(self, drawings):
        reference = naca4("0012", 201, n_trailing_edge=2)
        assert read_contour(drawings["reversed_splines"]).area == pytest.approx(
            reference.area, rel=1e-3
        )

    def test_arcs_and_lines_chain_into_a_stadium(self, drawings):
        exact = 0.020 * 0.010 + np.pi * 0.005**2
        assert read_contour(drawings["stadium"]).area == pytest.approx(exact, rel=1e-3)

    def test_an_open_outline_reports_the_likely_cause(self, drawings):
        with pytest.raises(DxfImportError, match="gap between entity endpoints"):
            read_contour(drawings["open_profile"])


class TestSelection:
    def test_four_vertex_rectangles_survive(self, drawings):
        """Regression: a point-count filter throws away the commonest profile."""
        assert len(read_contour(drawings["square_mm"])) == 4

    def test_every_closed_loop_is_returned_largest_first(self, drawings):
        contours = read_contours(drawings["two_loops"])
        assert [round(c.area, 3) for c in contours] == [100.0, pytest.approx(3.137, abs=1e-3)]

    def test_block_references_are_expanded_with_their_transform(self, drawings):
        """A 40x20 mm block at 2x scale encloses 3200 mm^2 however it is rotated."""
        c = read_contour(drawings["block_insert"])
        assert c.area == pytest.approx(3200e-6, rel=1e-6)
        assert c.bounds[2] - c.bounds[0] > 0.08  # rotated, so wider than the block

    def test_annotation_layers_yield_no_geometry(self, drawings):
        assert read_contours(drawings["block_insert"], layers=["NOTES"]) == []

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(DxfImportError, match="file not found"):
            read_contour(tmp_path / "absent.dxf")

    def test_a_non_dxf_file_is_reported_clearly(self, tmp_path):
        path = tmp_path / "not_a_drawing.dxf"
        path.write_text("this is not a DXF")
        with pytest.raises(DxfImportError, match="could not be read"):
            read_contour(path)


class TestMeshReadiness:
    def test_an_imported_profile_can_be_resampled_for_meshing(self, drawings):
        r = read_contour(drawings["gapped_lines"]).resample(241)
        r.validate()
        assert len(r) == 241
        h = r.segment_lengths()
        ratios = np.maximum(np.roll(h, -1) / h, h / np.roll(h, -1))
        # Excluding the blunt trailing edge, which must span at least one cell.
        away_from_te = np.abs(r.x - r.x.max()) > 0.02 * r.reference_length
        assert ratios[away_from_te].max() < 1.4
