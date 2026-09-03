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
from fluidsolver.gui.pages.numerics import SCHEME_LABELS  # noqa: E402
from fluidsolver.gui.pages.run import ReplayBuffer  # noqa: E402
from fluidsolver.gui.plot_canvas import (  # noqa: E402
    COLOURMAPS,
    FIELDS,
    _cell_centred_grid,
    close_seam,
    field_limits,
    field_style,
    field_values,
)
from fluidsolver.gui.worker import SolverWorker  # noqa: E402
from fluidsolver.solver.simple import Numerics  # noqa: E402


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

    def test_the_scheme_controls_start_on_the_solver_defaults(self, window):
        """A hardcoded combo index would silently disagree with Numerics()."""
        numerics = window.page_widgets[3]
        defaults = Numerics()
        assert SCHEME_LABELS[numerics.scheme.currentText()] == defaults.scheme
        assert (
            SCHEME_LABELS[numerics.turbulence_scheme.currentText()]
            == defaults.turbulence_scheme
        )

    def test_momentum_convection_defaults_to_a_bounded_scheme(self, window):
        """Unbounded central differencing must not be what a new case gets."""
        assert Numerics().scheme == "limited_linear"
        assert "recommended" in window.page_widgets[3].scheme.currentText()


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


class TestCellCentredGrid:
    """The co-located grid gouraud shading needs, built from an annulus."""

    @staticmethod
    def _annulus(n_i=16, n_j=5, inner=1.0, outer=4.0):
        theta = np.linspace(0.0, 2.0 * np.pi, n_i, endpoint=False)
        radius = np.linspace(inner, outer, n_j + 1)
        return np.stack(
            (np.outer(np.cos(theta), radius), np.outer(np.sin(theta), radius)), axis=-1
        )

    def test_coordinates_and_values_are_co_located(self):
        nodes = self._annulus()
        values = np.arange(16 * 5, dtype=float).reshape(16, 5)
        x, y, field = _cell_centred_grid(nodes, values)
        # gouraud needs one coordinate per value, with the seam closed and a row
        # added at each boundary.
        assert x.shape == y.shape == field.shape == (17, 7)

    def test_the_seam_closes(self):
        nodes = self._annulus()
        values = np.random.default_rng(0).random((16, 5))
        x, y, field = _cell_centred_grid(nodes, values)
        for array in (x, y, field):
            assert np.array_equal(array[0], array[-1])

    def test_the_boundaries_are_reached(self):
        """Without the added rows the paint stops half a cell short of each edge."""
        nodes = self._annulus(inner=1.0, outer=4.0)
        values = np.ones((16, 5))
        x, y, _ = _cell_centred_grid(nodes, values)
        radius = np.hypot(x, y)
        # Chord-versus-arc on the polygonal boundary is the only shortfall.
        assert radius.min() == pytest.approx(1.0, rel=0.03)
        assert radius.max() == pytest.approx(4.0, rel=0.03)

    def test_the_boundary_rows_carry_the_adjacent_cell_value(self):
        nodes = self._annulus()
        values = np.arange(16 * 5, dtype=float).reshape(16, 5)
        _, _, field = _cell_centred_grid(nodes, values)
        assert np.array_equal(field[:-1, 0], values[:, 0])
        assert np.array_equal(field[:-1, -1], values[:, -1])
        assert np.array_equal(field[:-1, 1:-1], values)


class TestFieldColouring:
    def test_a_signed_field_defaults_to_a_diverging_map(self):
        colourmap, symmetric = field_style("vorticity")
        assert colourmap == "RdBu_r"
        assert symmetric

    def test_an_unsigned_field_defaults_to_a_sequential_map(self):
        colourmap, symmetric = field_style("speed")
        assert colourmap == "viridis"
        assert not symmetric

    def test_an_explicit_map_wins_but_does_not_change_the_range(self):
        colourmap, symmetric = field_style("speed", COLOURMAPS["Red-blue"])
        assert colourmap == "RdBu_r"
        assert not symmetric

    def test_a_symmetric_range_is_centred_on_zero(self):
        values = np.linspace(-2.0, 8.0, 1000)
        low, high = field_limits(values, symmetric=True)
        assert low == pytest.approx(-high)

    def test_a_field_with_nothing_finite_has_no_range(self):
        assert field_limits(np.full(10, np.nan)) is None


class TestReplayBuffer:
    def _state(self, nbytes=1024):
        return type("Fake", (), {"nbytes": nbytes})()

    def test_frames_are_kept_in_order(self):
        buffer = ReplayBuffer()
        for iteration in (20, 40, 60):
            buffer.append(iteration, self._state())
        assert [it for it, _ in buffer] == [20, 40, 60]

    def test_a_repeated_iteration_replaces_rather_than_duplicates(self):
        buffer = ReplayBuffer()
        buffer.append(20, self._state())
        final = self._state()
        buffer.append(20, final)
        assert len(buffer) == 1
        assert buffer[-1][1] is final

    def test_the_buffer_thins_instead_of_growing_without_bound(self):
        buffer = ReplayBuffer(limit=8)
        for iteration in range(1, 41):
            buffer.append(iteration, self._state())

        assert len(buffer) <= 8
        iterations = [it for it, _ in buffer]
        # Thinning has to keep the whole span of the run, not just its start,
        # and it must never drop the newest frame.
        assert iterations[0] == 1
        assert iterations[-1] == 40
        assert iterations == sorted(iterations)

    def test_an_odd_capacity_still_keeps_the_newest_frame(self):
        for limit in range(5, 12):
            buffer = ReplayBuffer(limit=limit)
            for iteration in range(1, 60):
                buffer.append(iteration, self._state())
            assert buffer[-1][0] == 59, f"lost the newest frame at limit {limit}"

    def test_a_heavy_state_buys_fewer_frames(self):
        buffer = ReplayBuffer(budget=10_000, limit=1000)
        for iteration in range(1, 200):
            buffer.append(iteration, self._state(nbytes=1000))
        assert len(buffer) <= 10


def _small_laminar_case(window):
    """A cheap circle case, built and ready for the run page to draw."""
    window.page_widgets[0].velocity.setValue(1.0)
    window.page_widgets[1].kind.setCurrentIndex(1)  # circle
    window.page_widgets[3].model.setCurrentIndex(1)  # laminar
    session = window.session
    session.fluid = type(session.fluid)(density=1.0, viscosity=0.05, name="test")
    session.mesh_settings.surface_points = 48
    session.mesh_settings.far_field_radius_ratio = 10.0
    return session.build_case()


class TestRunPageView:
    """Panning and zooming, and the view surviving a redraw."""

    @pytest.fixture
    def run_page(self, window):
        _small_laminar_case(window)
        run = window.page_widgets[4]
        run._draw_field()
        return run

    def test_zooming_in_narrows_the_view(self, run_page):
        before = run_page.field_canvas.current_bounds()
        run_page._zoom_by(0.5)
        after = run_page.field_canvas.current_bounds()
        assert (after[1] - after[0]) == pytest.approx(0.5 * (before[1] - before[0]))
        assert (after[3] - after[2]) == pytest.approx(0.5 * (before[3] - before[2]))

    def test_a_redraw_keeps_the_view_the_user_chose(self, run_page):
        run_page._zoom_by(0.25)
        zoomed = run_page.field_canvas.current_bounds()
        run_page._draw_field()
        assert run_page.field_canvas.current_bounds() == pytest.approx(zoomed)

    def test_resetting_returns_to_the_preset(self, run_page):
        preset = run_page.field_canvas.current_bounds()
        run_page._zoom_by(0.25)
        run_page._reset_view()
        assert run_page.field_canvas.current_bounds() == pytest.approx(preset)

    def test_changing_the_preset_moves_the_view(self, run_page):
        run_page.zoom.setCurrentText("Body")
        body = run_page.field_canvas.current_bounds()
        run_page.zoom.setCurrentText("Far field")
        far = run_page.field_canvas.current_bounds()
        assert (far[1] - far[0]) > (body[1] - body[0])

    def test_the_colour_map_choice_reaches_the_plot(self, run_page):
        run_page.field.setCurrentText("Velocity magnitude")
        run_page.colourmap.setCurrentText("Red-blue")
        mesh = run_page.field_canvas.figure.axes[0].collections[0]
        assert mesh.get_cmap().name == "RdBu_r"

        run_page.colourmap.setCurrentText("Automatic")
        mesh = run_page.field_canvas.figure.axes[0].collections[0]
        assert mesh.get_cmap().name == "viridis"

    def test_the_shading_choice_reaches_the_plot(self, run_page):
        """Smooth carries one colour per cell centre, per cell one per cell."""
        case = run_page.session.case
        cells = case.grid.shape[0] * case.grid.shape[1]

        run_page.shading.setCurrentText("Per cell")
        mesh = run_page.field_canvas.figure.axes[0].collections[0]
        assert mesh.get_array().size == cells

        run_page.shading.setCurrentText("Smooth")
        mesh = run_page.field_canvas.figure.axes[0].collections[0]
        # One value per cell centre, plus the closed seam and a row at each
        # boundary: (Ni + 1) x (Nj + 2).
        assert mesh.get_array().size == (case.grid.shape[0] + 1) * (
            case.grid.shape[1] + 2
        )

    def test_smooth_is_the_default(self, run_page):
        assert run_page.shading.currentText() == "Smooth"

    def test_filling_the_window_hides_the_side_panel(self, run_page):
        run_page.fill_window.setChecked(True)
        assert not run_page.side_panel.isVisibleTo(run_page)
        run_page.fill_window.setChecked(False)
        assert run_page.side_panel.isVisibleTo(run_page)


class TestReplayOnTheRunPage:
    @pytest.fixture
    def replayed(self, window):
        case = _small_laminar_case(window)
        run = window.page_widgets[4]
        run.auto_replay.setChecked(False)
        # Stand in for the worker: two snapshots, as it would have emitted them.
        for iteration in (10, 20):
            case.run(max_iterations=2)
            run._snapshot(iteration, case.state.copy())
        run._arm_replay()
        return run, case

    def test_the_snapshots_are_kept_as_frames(self, replayed):
        run, _ = replayed
        assert [it for it, _ in run.replay] == [10, 20]
        assert run.replay_panel.isEnabled()
        assert "(2/2)" in run.frame_label.text()

    def test_drawing_a_snapshot_does_not_disturb_the_solver_state(self, replayed):
        run, case = replayed
        live = case.state
        run._snapshot(30, case.state.copy())
        assert case.state is live

    def test_scrubbing_the_slider_shows_that_frame(self, replayed):
        run, _ = replayed
        run.frame_slider.setValue(0)
        assert "iteration 10" in run.field_canvas.figure.axes[0].get_title()
        assert "(1/2)" in run.frame_label.text()

        run.frame_slider.setValue(1)
        assert "iteration 20" in run.field_canvas.figure.axes[0].get_title()

    def test_the_colour_range_is_pinned_across_frames(self, replayed):
        run, _ = replayed
        run.frame_slider.setValue(0)
        first = run.field_canvas.figure.axes[0].collections[0].get_clim()
        run.frame_slider.setValue(1)
        assert run.field_canvas.figure.axes[0].collections[0].get_clim() == first

    def test_playing_advances_and_stops_at_the_end(self, replayed):
        run, _ = replayed
        run.frame_slider.setValue(0)
        run.play.setChecked(True)
        assert run.play.text() == "Pause"

        run._advance_replay()
        assert run.frame_slider.value() == 1

        run._advance_replay()  # past the end, and looping is off
        assert not run.play.isChecked()
        assert run.play.text() == "Play"

    def test_pressing_play_at_the_end_starts_again(self, replayed):
        run, _ = replayed
        run.frame_slider.setValue(1)  # the last frame
        run.play.setChecked(True)
        assert run.frame_slider.value() == 0
        run.halt()

    def test_looping_wraps_back_to_the_start(self, replayed):
        run, _ = replayed
        run.loop.setChecked(True)
        run.play.setChecked(True)
        run.frame_slider.setValue(1)  # the last frame, while playing

        run._advance_replay()
        assert run.frame_slider.value() == 0
        assert run.play.isChecked()
        run.halt()

    def test_halting_stops_the_replay(self, replayed):
        run, _ = replayed
        run.play.setChecked(True)
        run.halt()
        assert not run.play.isChecked()
        assert not run._replay_timer.isActive()
