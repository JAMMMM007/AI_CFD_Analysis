"""GUI smoke tests.

These do not test how the window looks. They test that the pages build, that the
controls actually write through to the session, and that a case can be set up and
solved from the state the GUI produces -- the wiring, in other words, which is
where a front end usually breaks.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")

from fluidsolver.gui.main_window import MainWindow  # noqa: E402
from fluidsolver.gui.plot_canvas import FIELDS, close_seam, field_values  # noqa: E402
from fluidsolver.gui.worker import SolverWorker  # noqa: E402


@pytest.fixture
def window(qtbot):
    main = MainWindow()
    qtbot.addWidget(main)
    return main


class TestWindow:
    def test_every_page_builds(self, window):
        assert window.pages.count() == 5
        assert window.steps.count() == 5

    def test_stepping_moves_the_stack(self, window):
        for row in range(5):
            window.steps.setCurrentRow(row)
            assert window.pages.currentIndex() == row


class TestSetupPage:
    def test_controls_write_through_to_the_session(self, window):
        setup = window.page_widgets[0]
        setup.velocity.setValue(45.0)
        setup.incidence.setValue(6.0)
        setup.reference_length.setValue(2.0)

        session = window.session
        assert session.freestream.velocity == pytest.approx(45.0)
        assert session.freestream.angle_of_attack_deg == pytest.approx(6.0)
        assert session.shape.reference_length == pytest.approx(2.0)

    def test_reynolds_readout_matches_the_definition(self, window):
        setup = window.page_widgets[0]
        setup.velocity.setValue(30.0)
        setup.reference_length.setValue(1.0)
        assert float(setup.reynolds.text()) == pytest.approx(
            window.session.reynolds, rel=1e-3
        )

    def test_turbulence_intensity_is_entered_as_a_percentage(self, window):
        setup = window.page_widgets[0]
        setup.intensity.setValue(5.0)
        assert window.session.freestream.turbulence_intensity == pytest.approx(0.05)

    def test_a_supersonic_setting_is_flagged(self, window):
        setup = window.page_widgets[0]
        setup.velocity.setValue(300.0)
        assert "Mach" in setup.warning.text()
        setup.velocity.setValue(30.0)
        assert setup.warning.text() == ""


class TestShapePage:
    @pytest.mark.parametrize("index, kind", [(0, "naca"), (1, "circle"), (2, "square")])
    def test_each_analytic_shape_previews(self, window, index, kind):
        shape = window.page_widgets[1]
        shape.kind.setCurrentIndex(index)
        assert window.session.shape.kind == kind
        assert shape.problem.text() == ""
        assert int(shape.points.text()) > 8

    def test_an_invalid_naca_code_is_reported_not_raised(self, window):
        shape = window.page_widgets[1]
        shape.kind.setCurrentIndex(0)
        shape.naca_code.setText("99")
        shape.refresh()
        assert "four digits" in shape.problem.text()

    def test_the_body_is_scaled_to_the_reference_length(self, window):
        window.page_widgets[0].reference_length.setValue(3.0)
        shape = window.page_widgets[1]
        shape.kind.setCurrentIndex(1)  # circle
        shape.refresh()
        assert window.session.contour().reference_length == pytest.approx(3.0)


class TestMeshPage:
    def test_generating_a_mesh_reports_its_quality(self, window):
        mesh = window.page_widgets[2]
        window.page_widgets[1].kind.setCurrentIndex(1)  # circle
        mesh.surface_points.setValue(96)
        mesh.far_field.setValue(20.0)
        mesh.build()

        assert mesh.has_mesh()
        assert "inverted cells        0" in mesh.summary.text()

    def test_changing_a_setting_invalidates_the_mesh(self, window):
        mesh = window.page_widgets[2]
        window.page_widgets[1].kind.setCurrentIndex(1)
        mesh.surface_points.setValue(96)
        mesh.build()
        assert mesh.has_mesh()

        mesh.y_plus.setValue(30.0)
        assert not mesh.has_mesh()
        assert "generate the mesh again" in mesh.summary.text()

    def test_an_impossible_mesh_is_reported_not_raised(self, window):
        mesh = window.page_widgets[2]
        window.page_widgets[1].kind.setCurrentIndex(1)
        mesh.far_field.setValue(5.0)
        mesh.surface_points.setValue(96)
        window.page_widgets[0].reference_length.setValue(1.0)
        mesh.far_field.setValue(5.0)
        mesh.build()
        # Either it built, or it said clearly why not -- never an exception.
        assert mesh.has_mesh() or mesh.problem.text() != ""


class TestNumericsPage:
    def test_choosing_laminar_disables_the_turbulence_controls(self, window):
        numerics = window.page_widgets[3]
        numerics.model.setCurrentIndex(1)  # laminar
        assert window.session.model_name == "laminar"
        assert not numerics.relax_turbulence.isEnabled()

        numerics.model.setCurrentIndex(0)
        assert window.session.model_name == "k-omega-sst"
        assert numerics.relax_turbulence.isEnabled()

    def test_relaxation_controls_write_through(self, window):
        numerics = window.page_widgets[3]
        numerics.relax_velocity.setValue(0.45)
        assert window.session.numerics.relax_velocity == pytest.approx(0.45)


class TestSolvingThroughTheGui:
    """The wiring end to end: session -> case -> a few iterations -> plots."""

    @pytest.fixture
    def solved(self, window):
        window.page_widgets[0].velocity.setValue(1.0)
        window.page_widgets[0].reference_length.setValue(1.0)
        window.page_widgets[1].kind.setCurrentIndex(1)  # circle
        window.page_widgets[3].model.setCurrentIndex(1)  # laminar

        session = window.session
        session.fluid = type(session.fluid)(density=1.0, viscosity=0.05, name="test")
        session.mesh_settings.surface_points = 64
        session.mesh_settings.far_field_radius_ratio = 12.0
        session.numerics.max_iterations = 6

        case = session.build_case()
        case.run(max_iterations=6)
        return window, case

    def test_a_short_run_produces_forces(self, solved):
        _, case = solved
        forces = case.forces()
        assert np.isfinite(forces.drag_coefficient)
        assert case.iteration == 6
        assert len(case.history) == 6

    @pytest.mark.parametrize("field", list(FIELDS))
    def test_every_offered_field_can_be_extracted(self, solved, field):
        _, case = solved
        name, _ = FIELDS[field]
        values = field_values(case, name)
        assert values.shape == case.grid.shape
        assert np.all(np.isfinite(values))

    def test_the_run_page_draws_without_error(self, solved):
        window, _ = solved
        run = window.page_widgets[4]
        run._draw_field()
        run._draw_history()
        run._update_results()
        assert "Cd" in run.results.text()

    def test_surface_export_writes_a_readable_file(self, solved, tmp_path, monkeypatch):
        window, _ = solved
        run = window.page_widgets[4]
        target = tmp_path / "surface.csv"
        monkeypatch.setattr(
            "fluidsolver.gui.pages.run.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(target), ""),
        )
        run._export_surface()

        data = np.loadtxt(target, delimiter=",", skiprows=1)
        assert data.shape[1] == 6
        assert len(data) == window.session.case.grid.shape[0]


class TestWorker:
    def test_the_worker_stops_when_asked(self, window, qtbot):
        session = window.session
        session.fluid = type(session.fluid)(density=1.0, viscosity=0.05, name="test")
        window.page_widgets[0].velocity.setValue(1.0)
        window.page_widgets[1].kind.setCurrentIndex(1)
        window.page_widgets[3].model.setCurrentIndex(1)
        session.mesh_settings.surface_points = 48
        session.mesh_settings.far_field_radius_ratio = 10.0
        session.numerics.max_iterations = 500

        case = session.build_case()
        worker = SolverWorker(case, snapshot_every=5)

        seen = []
        worker.progressed.connect(seen.append)
        worker.progressed.connect(
            lambda r: worker.request_stop() if r.iteration >= 4 else None
        )

        with qtbot.waitSignal(worker.finished, timeout=60000):
            worker.run()

        assert 4 <= case.iteration < 500
        assert len(seen) == case.iteration


class TestPlotHelpers:
    def test_closing_the_seam_repeats_the_first_row(self):
        array = np.arange(12.0).reshape(4, 3)
        closed = close_seam(array)
        assert closed.shape == (5, 3)
        assert np.array_equal(closed[-1], array[0])
