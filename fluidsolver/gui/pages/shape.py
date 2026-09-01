"""Shape page: choose an analytic body or import one from a DXF."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fluidsolver.geometry.contour import ContourError
from fluidsolver.gui import widgets
from fluidsolver.gui.plot_canvas import Canvas, close_seam
from fluidsolver.gui.session import Session

KINDS = [
    ("NACA 4-digit aerofoil", "naca"),
    ("Circle", "circle"),
    ("Square", "square"),
    ("Import DXF", "dxf"),
]


class ShapePage(QWidget):
    """Body selection, with a live preview of the resulting contour."""

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self._imported = []

        outer, content = widgets.page(
            "Body",
            "The shape the flow is solved around. Whatever is chosen here is "
            "scaled so its reference length matches the setup page, and "
            "resampled by the mesher with the point spacing following curvature.",
        )
        self.setLayout(outer)

        columns = QHBoxLayout()
        content.addLayout(columns)

        # The forms are given a fixed share of the width. Left to themselves the
        # group boxes expand to fill whatever is available and the plot -- the
        # part actually worth looking at -- ends up in a corner.
        controls_panel = QWidget()
        controls_panel.setMaximumWidth(540)
        controls = QVBoxLayout(controls_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        columns.addWidget(controls_panel)

        self.kind = QComboBox()
        for label, value in KINDS:
            self.kind.addItem(label, value)
        kind_box, kind_form = widgets.group("Shape")
        kind_form.addRow("Type", self.kind)
        controls.addWidget(kind_box)

        self.parameters = QStackedWidget()
        controls.addWidget(self.parameters)
        self.parameters.addWidget(self._naca_page())
        self.parameters.addWidget(self._circle_page())
        self.parameters.addWidget(self._square_page())
        self.parameters.addWidget(self._dxf_page())

        info_box, info_form = widgets.group("Contour")
        self.points = widgets.readout()
        self.area = widgets.readout()
        self.perimeter = widgets.readout()
        self.extent = widgets.readout()
        info_form.addRow("Points", self.points)
        info_form.addRow("Area", self.area)
        info_form.addRow("Perimeter", self.perimeter)
        info_form.addRow("Extent", self.extent)
        controls.addWidget(info_box)

        self.problem = widgets.note()
        controls.addWidget(self.problem)
        controls.addStretch(1)

        self.canvas = Canvas(7.0, 5.5)
        columns.addWidget(self.canvas, 1)

        self.kind.currentIndexChanged.connect(self._kind_chosen)
        session.changed.connect(self.refresh)
        self._kind_chosen()

    # ------------------------------------------------------------------
    # Parameter pages
    # ------------------------------------------------------------------

    def _naca_page(self) -> QWidget:
        page = QWidget()
        box, form = widgets.group("NACA 4-digit")
        self.naca_code = QLineEdit(self.session.shape.naca_code)
        self.naca_code.setMaxLength(4)
        self.naca_code.setPlaceholderText("e.g. 2412")
        form.addRow("Code", self.naca_code)
        form.addRow(
            "",
            widgets.readout(
                "digits MPXX: M% camber at P/10 chord, XX% thick"
            ),
        )
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        self.naca_code.editingFinished.connect(self.refresh)
        return page

    def _circle_page(self) -> QWidget:
        page = QWidget()
        box, form = widgets.group("Circle")
        form.addRow("", widgets.readout("diameter = the reference length"))
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        return page

    def _square_page(self) -> QWidget:
        page = QWidget()
        box, form = widgets.group("Square")
        self.corner_radius = widgets.number(
            self.session.shape.corner_radius,
            minimum=0.0, maximum=0.49, decimals=4, step=0.01,
        )
        form.addRow("Corner radius / side", self.corner_radius)
        form.addRow(
            "",
            widgets.readout(
                "0 gives a truly sharp corner. A small radius\n"
                "makes the mesh march considerably further."
            ),
        )
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        self.corner_radius.valueChanged.connect(self.refresh)
        return page

    def _dxf_page(self) -> QWidget:
        page = QWidget()
        box, form = widgets.group("DXF import")
        self.dxf_path = QLineEdit()
        self.dxf_path.setReadOnly(True)
        browse = QPushButton("Choose file...")
        browse.clicked.connect(self._browse)
        self.dxf_info = widgets.readout()
        self.loops = QListWidget()
        self.loops.setMaximumHeight(110)
        form.addRow("File", self.dxf_path)
        form.addRow("", browse)
        form.addRow("Detected", self.dxf_info)
        form.addRow("Closed loops", self.loops)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        self.loops.currentRowChanged.connect(self._loop_chosen)
        return page

    # ------------------------------------------------------------------

    def _kind_chosen(self) -> None:
        index = self.kind.currentIndex()
        self.parameters.setCurrentIndex(index)
        self.session.shape.kind = self.kind.currentData()
        self.refresh()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a DXF drawing", "", "DXF drawings (*.dxf);;All files (*)"
        )
        if not path:
            return

        from fluidsolver.geometry.dxf_import import (
            DxfImportError,
            describe_dxf,
            read_contours,
        )

        try:
            described = describe_dxf(path)
            self._imported = read_contours(path)
        except DxfImportError as error:
            QMessageBox.warning(self, "Could not import the drawing", str(error))
            return

        self.dxf_path.setText(path)
        self.session.shape.dxf_path = path
        self.dxf_info.setText(
            f"units: {described['units']}\n"
            + ", ".join(f"{n} x {t}" for t, n in described["entity_counts"].items())
        )

        self.loops.clear()
        for contour in self._imported:
            xmin, ymin, xmax, ymax = contour.bounds
            self.loops.addItem(
                f"{len(contour)} pts, area {contour.area:.4g} m2, "
                f"{xmax - xmin:.4g} x {ymax - ymin:.4g} m"
            )
        if self._imported:
            self.loops.setCurrentRow(0)
        else:
            QMessageBox.warning(
                self,
                "No closed loop",
                "Geometry was found but nothing closed into a loop. The profile is "
                "probably left open by a gap between entity endpoints.",
            )

    def _loop_chosen(self, row: int) -> None:
        if 0 <= row < len(self._imported):
            contour = self._imported[row]
            # Normalise to the reference length so the setup page's Reynolds
            # number means what it says, whatever units the drawing was in.
            self.session.shape.dxf_contour = contour.normalised(
                self.session.shape.reference_length
            )
            self.refresh()

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        shape = self.session.shape
        shape.naca_code = self.naca_code.text().strip() or "0012"
        shape.corner_radius = self.corner_radius.value()

        try:
            contour = self.session.contour()
            contour.validate()
        except (ContourError, ValueError) as error:
            self.problem.setText(str(error))
            self.canvas.clear()
            self.canvas.draw()
            for label in (self.points, self.area, self.perimeter, self.extent):
                label.setText("-")
            return

        self.problem.setText("")
        xmin, ymin, xmax, ymax = contour.bounds
        self.points.setText(f"{len(contour)}")
        self.area.setText(f"{contour.area:.5g} m2")
        self.perimeter.setText(f"{contour.perimeter:.5g} m")
        self.extent.setText(f"{xmax - xmin:.5g} x {ymax - ymin:.5g} m")

        self._draw(contour)

    def _draw(self, contour) -> None:
        self.canvas.clear()
        axes = self.canvas.figure.add_subplot(111)

        incidence = self.session.freestream.angle_of_attack_deg
        rotated = contour
        if incidence:
            pivot = np.array([contour.bounds[0], contour.centroid[1]])
            rotated = contour.rotated(-incidence, about=pivot)

        points = close_seam(rotated.points)
        axes.fill(points[:, 0], points[:, 1], color="#d6e4f0", zorder=1)
        axes.plot(points[:, 0], points[:, 1], lw=1.6, color="#c0392b", zorder=2)

        xmin, ymin, xmax, ymax = rotated.bounds
        span = max(xmax - xmin, ymax - ymin) * 0.75
        centre = rotated.centroid
        axes.set_xlim(centre[0] - span, centre[0] + span)
        axes.set_ylim(centre[1] - span, centre[1] + span)
        axes.set_aspect("equal")
        axes.grid(alpha=0.25)
        axes.set_title(
            f"{contour.name}"
            + (f"   at {incidence:g} deg incidence" if incidence else "")
        )
        # The body is rotated rather than the freestream, so the flow always
        # runs left to right and lift is always the +y force.
        axes.annotate(
            "", xy=(centre[0] - span * 0.55, centre[1] + span * 0.72),
            xytext=(centre[0] - span * 0.85, centre[1] + span * 0.72),
            arrowprops=dict(arrowstyle="->", color="#2c7", lw=1.8),
        )
        axes.text(
            centre[0] - span * 0.83, centre[1] + span * 0.78, "flow",
            color="#2c7", fontsize=9,
        )
        self.canvas.draw()
