"""Run page: solve, watch it converge, then replay how it got there."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fluidsolver.gui import widgets
from fluidsolver.gui.plot_canvas import (
    COLOURMAPS,
    FIELDS,
    Canvas,
    InteractiveCanvas,
    draw_body,
    draw_field,
    draw_streamlines,
    field_limits,
    field_style,
    field_values,
)
from fluidsolver.gui.session import Session
from fluidsolver.gui.worker import SolverWorker

# How often the field plot is redrawn, in solver iterations. Redrawing is far
# slower than an iteration, so doing it every time would leave the GUI thread
# permanently behind the solver.
_REDRAW_EVERY = 20

# Every snapshot kept for the replay is a full copy of the solution, so on a fine
# mesh a frame costs megabytes and a long run would hold gigabytes. The buffer is
# sized by memory rather than by frame count so that a fine mesh gives up frames
# instead of the machine giving up.
_REPLAY_MEMORY_BUDGET = 192 * 1024**2
_MAX_REPLAY_FRAMES = 240

# Playback: the interval between frames at 1x.
_REPLAY_INTERVAL_MS = 90
_SPEEDS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}


class ReplayBuffer:
    """The snapshots kept for the replay, thinned to fit a memory budget.

    When the buffer is full every second frame is dropped and the interval
    between the frames that remain doubles. Dropping the oldest instead would be
    cheaper, but it would leave a replay of the last few hundred iterations of a
    converged run -- which is the part where nothing happens. Thinning keeps the
    frames spread over the whole run, which is what someone watching a replay is
    actually after.

    The capacity is rounded down to an even number so that thinning always
    happens at an odd length, and slicing an odd-length list by two keeps both
    ends. The last frame is the converged solution, and losing it to a rounding
    detail would take the colour range and the field left on screen with it.
    """

    def __init__(
        self,
        budget: int = _REPLAY_MEMORY_BUDGET,
        limit: int = _MAX_REPLAY_FRAMES,
    ):
        self._budget = budget
        self._limit = limit
        self._capacity = limit
        self.frames: list[tuple[int, object]] = []

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def clear(self) -> None:
        self.frames.clear()
        self._capacity = self._limit

    def append(self, iteration: int, state) -> None:
        if not self.frames:
            cost = max(1, getattr(state, "nbytes", 1))
            capacity = int(np.clip(self._budget // cost, 4, self._limit))
            self._capacity = capacity - capacity % 2

        # The solver emits a final snapshot when it stops, which for a run that
        # ended on a snapshot iteration is the frame already held.
        if self.frames and self.frames[-1][0] == iteration:
            self.frames[-1] = (iteration, state)
            return

        self.frames.append((iteration, state))
        if len(self.frames) > self._capacity:
            self.frames = self.frames[::2]


class RunPage(QWidget):
    """Start the solve, watch the residuals and the field, read the forces off."""

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.thread: QThread | None = None
        self.worker: SolverWorker | None = None

        self.replay = ReplayBuffer()
        self._frame_index: int | None = None
        self._replay_timer = QTimer(self)
        self._replay_timer.setSingleShot(True)
        self._replay_timer.timeout.connect(self._advance_replay)

        # The view the field is drawn in, in metres. ``None`` means "take it from
        # the preset"; once the user pans or zooms it holds what they chose, and
        # every redraw honours it.
        self._view: tuple[float, float, float, float] | None = None
        self._limits: dict[tuple[str, bool], tuple[float, float] | None] = {}

        outer, content = widgets.page(
            "Solve",
            "Steady RANS by the SIMPLE algorithm. The field and the residuals "
            "update as it converges, and the run can be replayed afterwards.",
        )
        self.setLayout(outer)

        content.addLayout(self._solver_row())

        self.splitter = QSplitter(Qt.Horizontal)
        content.addWidget(self.splitter, 1)

        self.splitter.addWidget(self._plot_column())
        self.side_panel = self._side_panel()
        self.splitter.addWidget(self.side_panel)

        # The field is the point of the page, so it gets the width. Without the
        # stretch factors the two halves share it evenly and the plot ends up
        # about as big as the residual chart beside it.
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([1100, 420])
        self.splitter.setCollapsible(1, True)

        self.status = widgets.note()
        content.addWidget(self.status)

        self._set_running(False)
        self._set_replay_enabled(False)

    # -- layout ---------------------------------------------------------

    def _solver_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.start = QPushButton("Solve")
        self.start.clicked.connect(self._start)
        self.pause = QPushButton("Pause")
        self.pause.setCheckable(True)
        self.pause.toggled.connect(self._pause_toggled)
        self.stop = QPushButton("Stop")
        self.stop.clicked.connect(self._stop)

        self.field = QComboBox()
        self.field.addItems(FIELDS.keys())
        self.field.currentIndexChanged.connect(self._draw_field)

        self.colourmap = QComboBox()
        self.colourmap.addItems(COLOURMAPS.keys())
        self.colourmap.setToolTip(
            "Automatic uses a red-blue map for the signed fields and viridis for "
            "the rest. Any choice here applies to whatever field is shown."
        )
        self.colourmap.currentIndexChanged.connect(self._draw_field)

        self.streamlines = QCheckBox("Streamlines")
        self.streamlines.toggled.connect(self._draw_field)

        for widget in (self.start, self.pause, self.stop):
            row.addWidget(widget)
        row.addSpacing(24)
        row.addWidget(widgets.readout("Field"))
        row.addWidget(self.field)
        row.addWidget(widgets.readout("Colour map"))
        row.addWidget(self.colourmap)
        row.addWidget(self.streamlines)
        row.addStretch(1)
        return row

    def _view_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.zoom = QComboBox()
        self.zoom.addItems(["Body", "Near field", "Wake", "Far field"])
        self.zoom.currentIndexChanged.connect(self._reset_view)

        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(32)
        zoom_in.setToolTip("Zoom in")
        zoom_in.clicked.connect(lambda: self._zoom_by(1.0 / 1.4))

        zoom_out = QPushButton("-")
        zoom_out.setFixedWidth(32)
        zoom_out.setToolTip("Zoom out")
        zoom_out.clicked.connect(lambda: self._zoom_by(1.4))

        reset = QPushButton("Reset view")
        reset.clicked.connect(self._reset_view)

        self.fill_window = QPushButton("Fill window")
        self.fill_window.setCheckable(True)
        self.fill_window.setToolTip(
            "Hide the residuals and results and give the whole page to the field."
        )
        self.fill_window.toggled.connect(self._fill_window_toggled)

        hint = QLabel("wheel zooms, drag pans, double-click resets")
        hint.setStyleSheet("color: #777777;")

        row.addWidget(widgets.readout("View"))
        row.addWidget(self.zoom)
        row.addWidget(zoom_in)
        row.addWidget(zoom_out)
        row.addWidget(reset)
        row.addSpacing(16)
        row.addWidget(self.fill_window)
        row.addSpacing(16)
        row.addWidget(hint)
        row.addStretch(1)
        return row

    def _replay_row(self) -> QWidget:
        panel = QWidget()
        row = QHBoxLayout(panel)
        row.setContentsMargins(0, 0, 0, 0)

        self.play = QPushButton("Play")
        self.play.setCheckable(True)
        self.play.setFixedWidth(80)
        self.play.toggled.connect(self._play_toggled)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setTracking(True)
        self.frame_slider.setMinimumWidth(200)
        self.frame_slider.valueChanged.connect(self._show_frame)

        self.frame_label = widgets.readout("no frames yet")
        self.frame_label.setMinimumWidth(150)

        self.speed = QComboBox()
        self.speed.addItems(_SPEEDS.keys())
        self.speed.setCurrentText("1x")

        self.loop = QCheckBox("Loop")
        self.auto_replay = QCheckBox("Auto")
        self.auto_replay.setChecked(True)
        self.auto_replay.setToolTip(
            "Play the run back automatically as soon as the solver stops."
        )

        row.addWidget(widgets.readout("Replay"))
        row.addWidget(self.play)
        row.addWidget(self.frame_slider, 1)
        row.addWidget(self.frame_label)
        row.addWidget(widgets.readout("Speed"))
        row.addWidget(self.speed)
        row.addWidget(self.loop)
        row.addWidget(self.auto_replay)
        return panel

    def _plot_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addLayout(self._view_row())

        self.field_canvas = InteractiveCanvas(9.0, 7.5)
        self.field_canvas.setMinimumSize(420, 360)
        self.field_canvas.view_changed.connect(self._view_moved)
        self.field_canvas.home_requested.connect(self._reset_view)
        self.field_canvas.resized.connect(self._canvas_resized)
        layout.addWidget(self.field_canvas, 1)

        self.replay_panel = self._replay_row()
        layout.addWidget(self.replay_panel)
        return column

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.history_canvas = Canvas(4.5, 6.0)
        layout.addWidget(self.history_canvas, 1)

        results_box, results_form = widgets.group("Results")
        self.results = widgets.readout("not run yet")
        self.results.setAlignment(Qt.AlignTop)
        results_form.addRow(self.results)
        layout.addWidget(results_box)

        export_row = QHBoxLayout()
        self.export_csv = QPushButton("Export surface data...")
        self.export_csv.clicked.connect(self._export_surface)
        self.export_png = QPushButton("Save field image...")
        self.export_png.clicked.connect(self._export_image)
        export_row.addWidget(self.export_csv)
        export_row.addWidget(self.export_png)
        layout.addLayout(export_row)
        return panel

    # ------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        self.start.setEnabled(not running)
        self.pause.setEnabled(running)
        self.stop.setEnabled(running)
        if not running:
            self.pause.setChecked(False)

    def _fill_window_toggled(self, filled: bool) -> None:
        self.side_panel.setVisible(not filled)
        self.fill_window.setText("Show results" if filled else "Fill window")

    # ------------------------------------------------------------------

    def _start(self) -> None:
        try:
            case = self.session.build_case()
        except Exception as error:  # noqa: BLE001 - shown to the user verbatim
            QMessageBox.critical(self, "Could not set the case up", str(error))
            return

        for warning in case.quality.warnings:
            self.status.setText(f"mesh warning: {warning}")

        # A new run invalidates the old replay and the colour ranges taken from
        # it. The view is deliberately left alone: it is in metres, and someone
        # who zoomed into a wake to compare two runs wants the same window back.
        self._stop_replay()
        self.replay.clear()
        self._frame_index = None
        self._limits.clear()
        self._set_replay_enabled(False)

        self.worker = SolverWorker(case, snapshot_every=_REDRAW_EVERY)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progressed.connect(self._progressed)
        self.worker.snapshot.connect(self._snapshot)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)

        self._set_running(True)
        self.status.setText(
            f"solving {case.grid.n_cells:,} cells with {case.model.name}, "
            f"Re = {case.reynolds:.3g}"
        )
        self.thread.start()

    def _pause_toggled(self, paused: bool) -> None:
        if self.worker is not None:
            self.worker.set_paused(paused)
        self.pause.setText("Resume" if paused else "Pause")

    def _stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()

    def _shut_down(self) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self._set_running(False)

    def halt(self) -> None:
        """Stop everything that could still fire. Called as the window closes."""
        self._stop_replay()
        self._stop()
        self._shut_down()

    # ------------------------------------------------------------------

    def _progressed(self, residuals) -> None:
        if residuals.iteration % _REDRAW_EVERY == 0:
            self._draw_history()
            self._update_results()

    def _snapshot(self, iteration: int, state) -> None:
        # The case's own state keeps moving; keep and draw the copy handed over.
        if self.session.case is None:
            return
        self.replay.append(iteration, state)
        self._draw_field(state=state, iteration=iteration)

    def _finished(self, reason: str) -> None:
        self.status.setText(reason)
        self._shut_down()
        self._draw_history()
        self._update_results()
        self._arm_replay()

    def _failed(self, message: str) -> None:
        self.status.setText(f"the run stopped: {message}")
        self._shut_down()
        # The frames up to the failure are kept: the iterations before a
        # divergence are where the reason for it is visible.
        self._arm_replay(autoplay=False)
        QMessageBox.warning(self, "The solver stopped", message)

    # -- replay ---------------------------------------------------------

    def _set_replay_enabled(self, enabled: bool) -> None:
        self.replay_panel.setEnabled(enabled)
        if not enabled:
            self.frame_label.setText("no frames yet")

    def _arm_replay(self, autoplay: bool = True) -> None:
        """A run has stopped: point the replay controls at what it produced."""
        count = len(self.replay)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, count - 1))
        self.frame_slider.setValue(max(0, count - 1))
        self.frame_slider.blockSignals(False)

        self._frame_index = count - 1 if count else None
        self._set_replay_enabled(count >= 2)
        self._update_frame_label()
        self._draw_field()

        if count >= 2 and autoplay and self.auto_replay.isChecked():
            self.play.setChecked(True)

    def _stop_replay(self) -> None:
        self._replay_timer.stop()
        if self.play.isChecked():
            self.play.blockSignals(True)
            self.play.setChecked(False)
            self.play.blockSignals(False)
        self.play.setText("Play")

    def _play_toggled(self, playing: bool) -> None:
        self.play.setText("Pause" if playing else "Play")
        if not playing:
            self._replay_timer.stop()
            return
        # Pressing play at the end of the run means "watch it again".
        if self.frame_slider.value() >= len(self.replay) - 1:
            self.frame_slider.setValue(0)
        self._replay_timer.start(self._interval())

    def _interval(self) -> int:
        return max(10, int(_REPLAY_INTERVAL_MS / _SPEEDS[self.speed.currentText()]))

    def _advance_replay(self) -> None:
        """One frame on. The timer is restarted here rather than repeating.

        Drawing a frame can take longer than the interval on a fine mesh. A
        repeating timer would queue the ticks it missed and the replay would run
        away from the redraws; restarting after each frame cannot.
        """
        if not self.play.isChecked() or len(self.replay) < 2:
            return

        index = self.frame_slider.value() + 1
        if index >= len(self.replay):
            if not self.loop.isChecked():
                self.play.setChecked(False)
                return
            index = 0

        self.frame_slider.setValue(index)  # draws, through _show_frame
        self._replay_timer.start(self._interval())

    def _show_frame(self, index: int) -> None:
        if not self.replay:
            return
        self._frame_index = int(np.clip(index, 0, len(self.replay) - 1))
        self._update_frame_label()
        self._draw_field()

    def _update_frame_label(self) -> None:
        if not self.replay or self._frame_index is None:
            self.frame_label.setText("no frames yet")
            return
        iteration, _ = self.replay[self._frame_index]
        self.frame_label.setText(
            f"iteration {iteration}  ({self._frame_index + 1}/{len(self.replay)})"
        )

    def _current_frame(self):
        """The replay frame on show, or ``None`` when the live field is."""
        if self._frame_index is None or not self.replay:
            return None
        return self.replay[min(self._frame_index, len(self.replay) - 1)]

    # -- the view -------------------------------------------------------

    def _canvas_resized(self) -> None:
        """The canvas settled at a new size; refit the view to it."""
        if self.session.case is not None:
            self._draw_field()

    def _view_moved(self) -> None:
        """The user panned or zoomed; hold onto it so a redraw keeps it."""
        bounds = self.field_canvas.current_bounds()
        if bounds is not None:
            self._view = bounds

    def _zoom_by(self, factor: float) -> None:
        self.field_canvas.zoom_by(factor)

    def _reset_view(self, *_args) -> None:
        self._view = None
        self._draw_field()

    def _fit_to_canvas(
        self, bounds: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """``bounds`` grown to the shape of the space the plot has.

        The field is drawn at equal aspect, so a square view in a space that is
        not square leaves the rest of it blank -- and on a wide window that is
        most of the page. Growing the view uses the space to show more of the
        domain instead, with the geometry still undistorted.
        """
        aspect = self.field_canvas.plot_aspect()
        if aspect is None:
            return bounds

        left, right, bottom, top = bounds
        span_x, span_y = right - left, top - bottom
        if span_x <= 0.0 or span_y <= 0.0:
            return bounds

        if span_x / span_y < aspect:
            span_x = span_y * aspect
        else:
            span_y = span_x / aspect

        mid_x, mid_y = 0.5 * (left + right), 0.5 * (bottom + top)
        return (
            mid_x - 0.5 * span_x, mid_x + 0.5 * span_x,
            mid_y - 0.5 * span_y, mid_y + 0.5 * span_y,
        )

    def _preset_bounds(self, case) -> tuple[float, float, float, float]:
        reference = case.reference_length
        centre = case.grid.contour.centroid
        spans = {
            "Body": (0.9, 0.0),
            "Near field": (2.5, 0.5),
            "Wake": (4.0, 2.2),
            "Far field": (None, 0.0),
        }
        span, shift = spans[self.zoom.currentText()]
        if span is None:
            half = np.linalg.norm(case.grid.far_field - centre, axis=1).max()
            origin = centre
        else:
            half = span * reference
            origin = centre + np.array([shift * reference, 0.0])
        return (origin[0] - half, origin[0] + half, origin[1] - half, origin[1] + half)

    # ------------------------------------------------------------------

    def _update_results(self) -> None:
        case = self.session.case
        if case is None:
            return
        try:
            self.results.setText(case.summary())
        except Exception:  # noqa: BLE001 - a partial run may not have forces yet
            pass

    def _draw_history(self) -> None:
        case = self.session.case
        if case is None or not case.history.entries:
            return

        self.history_canvas.clear()
        figure = self.history_canvas.figure
        residual_axes, force_axes = figure.subplots(2, 1, sharex=True)

        iterations = case.history.iterations
        for name, label in (
            ("u", "Ux"), ("v", "Uy"), ("continuity", "mass"),
            ("k", "k"), ("omega", "omega"),
        ):
            series = case.history.series(name)
            if np.any(series > 0.0):
                residual_axes.semilogy(iterations, np.maximum(series, 1e-16), lw=1.0, label=label)
        residual_axes.axhline(
            case.numerics.tolerance, color="grey", ls="--", lw=0.8, label="tolerance"
        )
        residual_axes.set_ylabel("scaled residual")
        residual_axes.legend(fontsize=7, ncols=3)
        residual_axes.grid(alpha=0.3)

        force_axes.plot(
            iterations, case.history.series("lift_coefficient"), lw=1.2, label="Cl"
        )
        force_axes.plot(
            iterations, case.history.series("drag_coefficient"), lw=1.2, label="Cd"
        )
        force_axes.set_xlabel("iteration")
        force_axes.set_ylabel("force coefficient")
        force_axes.legend(fontsize=8)
        force_axes.grid(alpha=0.3)

        self.history_canvas.draw()

    def _colour_limits(self, case, name: str, symmetric: bool):
        """A fixed colour range once a run has finished, otherwise ``None``.

        While the solver is working the range follows the field, because the
        field is still finding its scale. Once there are frames to replay the
        range is pinned to the last of them, so that scrubbing through the run
        shows the solution changing rather than the colour bar changing.
        """
        if self.thread is not None or not self.replay:
            return None

        key = (name, symmetric)
        if key not in self._limits:
            _, final = self.replay[-1]
            try:
                values = field_values(case, name, final)
            except Exception:  # noqa: BLE001 - fall back to per-frame scaling
                return None
            self._limits[key] = field_limits(values, symmetric=symmetric)
        return self._limits[key]

    def _draw_field(self, *_args, state=None, iteration=None) -> None:
        case = self.session.case
        if case is None:
            return

        if state is None:
            frame = self._current_frame()
            if frame is None and self.thread is not None and len(self.replay):
                # Mid-run, with no frame selected: the last snapshot is what is
                # already on screen, and unlike the live state it is not being
                # rewritten by the solver thread as it is read.
                frame = self.replay[-1]
            if frame is not None:
                iteration, state = frame
        if state is None:
            state = case.state
        if iteration is None:
            iteration = case.iteration

        self.field_canvas.clear()
        axes, colourbar_axes = self.field_canvas.field_axes()

        name, label = FIELDS[self.field.currentText()]
        try:
            values = field_values(case, name, state)
        except Exception:  # noqa: BLE001
            return

        colourmap, symmetric = field_style(
            name, COLOURMAPS[self.colourmap.currentText()]
        )
        draw_field(
            axes, case.grid.nodes, values,
            label=label,
            colourmap=colourmap,
            symmetric=symmetric,
            limits=self._colour_limits(case, name, symmetric),
            colourbar_axes=colourbar_axes,
        )

        if self._view is None:
            self._view = self._preset_bounds(case)
        bounds = self._view = self._fit_to_canvas(self._view)

        if self.streamlines.isChecked():
            # Sampled over what is on screen, so zooming in also refines them.
            draw_streamlines(
                axes, case.metrics.centroid, state.u, state.v,
                case.grid.wall, bounds,
            )

        draw_body(axes, case.grid.wall)
        axes.set_xlim(bounds[0], bounds[1])
        axes.set_ylim(bounds[2], bounds[3])
        axes.set_aspect("equal")
        axes.set_title(f"{self.field.currentText()}   -   iteration {iteration}")
        self.field_canvas.draw()

    # ------------------------------------------------------------------

    def _export_surface(self) -> None:
        case = self.session.case
        if case is None:
            QMessageBox.information(self, "Nothing to export", "Run a case first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save surface data", "surface.csv", "CSV (*.csv)"
        )
        if not path:
            return

        surface = case.surface()
        columns = np.column_stack(
            (
                surface.x, surface.y, surface.arclength,
                surface.pressure_coefficient,
                surface.skin_friction_coefficient,
                surface.y_plus,
            )
        )
        np.savetxt(
            path, columns, delimiter=",",
            header="x,y,arclength,Cp,Cf,y_plus", comments="",
        )
        self.status.setText(f"surface data written to {path}")

    def _export_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the field image", "field.png", "PNG (*.png)"
        )
        if path:
            self.field_canvas.figure.savefig(path, dpi=180)
            self.status.setText(f"image written to {path}")
