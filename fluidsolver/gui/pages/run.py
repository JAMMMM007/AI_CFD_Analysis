"""Run page: solve, and watch it converge."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fluidsolver.gui import widgets
from fluidsolver.gui.plot_canvas import (
    FIELDS,
    Canvas,
    draw_body,
    draw_field,
    draw_streamlines,
    field_values,
)
from fluidsolver.gui.session import Session
from fluidsolver.gui.worker import SolverWorker

# How often the field plot is redrawn, in solver iterations. Redrawing is far
# slower than an iteration, so doing it every time would leave the GUI thread
# permanently behind the solver.
_REDRAW_EVERY = 20


class RunPage(QWidget):
    """Start the solve, watch the residuals and the field, read the forces off."""

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.thread: QThread | None = None
        self.worker: SolverWorker | None = None
        self._pending_redraw = False

        outer, content = widgets.page(
            "Solve",
            "Steady RANS by the SIMPLE algorithm. The field and the residuals "
            "update as it converges.",
        )
        self.setLayout(outer)

        content.addLayout(self._toolbar())

        splitter = QSplitter(Qt.Horizontal)
        content.addWidget(splitter, 1)

        self.field_canvas = Canvas(7.0, 6.0)
        splitter.addWidget(self.field_canvas)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.history_canvas = Canvas(4.5, 6.0)
        right_layout.addWidget(self.history_canvas, 1)

        results_box, results_form = widgets.group("Results")
        self.results = widgets.readout("not run yet")
        self.results.setAlignment(Qt.AlignTop)
        results_form.addRow(self.results)
        right_layout.addWidget(results_box)

        export_row = QHBoxLayout()
        self.export_csv = QPushButton("Export surface data...")
        self.export_csv.clicked.connect(self._export_surface)
        self.export_png = QPushButton("Save field image...")
        self.export_png.clicked.connect(self._export_image)
        export_row.addWidget(self.export_csv)
        export_row.addWidget(self.export_png)
        right_layout.addLayout(export_row)

        splitter.addWidget(right)
        splitter.setSizes([700, 460])

        self.status = widgets.note()
        content.addWidget(self.status)

        self._set_running(False)

    # ------------------------------------------------------------------

    def _toolbar(self) -> QHBoxLayout:
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

        self.streamlines = QCheckBox("Streamlines")
        self.streamlines.toggled.connect(self._draw_field)

        self.zoom = QComboBox()
        self.zoom.addItems(["Body", "Near field", "Wake", "Far field"])
        self.zoom.currentIndexChanged.connect(self._draw_field)

        for widget in (self.start, self.pause, self.stop):
            row.addWidget(widget)
        row.addSpacing(24)
        row.addWidget(widgets.readout("Field"))
        row.addWidget(self.field)
        row.addWidget(self.streamlines)
        row.addWidget(widgets.readout("View"))
        row.addWidget(self.zoom)
        row.addStretch(1)
        return row

    def _set_running(self, running: bool) -> None:
        self.start.setEnabled(not running)
        self.pause.setEnabled(running)
        self.stop.setEnabled(running)
        if not running:
            self.pause.setChecked(False)

    # ------------------------------------------------------------------

    def _start(self) -> None:
        try:
            case = self.session.build_case()
        except Exception as error:  # noqa: BLE001 - shown to the user verbatim
            QMessageBox.critical(self, "Could not set the case up", str(error))
            return

        for warning in case.quality.warnings:
            self.status.setText(f"mesh warning: {warning}")

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

    # ------------------------------------------------------------------

    def _progressed(self, residuals) -> None:
        if residuals.iteration % _REDRAW_EVERY == 0:
            self._draw_history()
            self._update_results()

    def _snapshot(self, state) -> None:
        # The case's own state keeps moving; draw the copy that was handed over.
        if self.session.case is not None:
            self._draw_field(state=state)

    def _finished(self, reason: str) -> None:
        self.status.setText(reason)
        self._draw_history()
        self._draw_field()
        self._update_results()
        self._shut_down()

    def _failed(self, message: str) -> None:
        self.status.setText(f"the run stopped: {message}")
        QMessageBox.warning(self, "The solver stopped", message)
        self._shut_down()

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

    def _draw_field(self, *_args, state=None) -> None:
        case = self.session.case
        if case is None:
            return
        if state is not None:
            case.state = state

        self.field_canvas.clear()
        axes = self.field_canvas.figure.add_subplot(111)

        name, label = FIELDS[self.field.currentText()]
        try:
            values = field_values(case, name)
        except Exception:  # noqa: BLE001
            return

        draw_field(
            axes, case.grid.nodes, values,
            label=label,
            colourmap="RdBu_r" if name in ("v", "vorticity") else "viridis",
            symmetric=name in ("v", "vorticity"),
        )

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

        bounds = (origin[0] - half, origin[0] + half, origin[1] - half, origin[1] + half)
        if self.streamlines.isChecked():
            draw_streamlines(
                axes, case.metrics.centroid, case.state.u, case.state.v,
                case.grid.wall, bounds,
            )

        draw_body(axes, case.grid.wall)
        axes.set_xlim(bounds[0], bounds[1])
        axes.set_ylim(bounds[2], bounds[3])
        axes.set_aspect("equal")
        axes.set_title(
            f"{self.field.currentText()}   -   iteration {case.iteration}"
        )
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
