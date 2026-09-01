"""Mesh page: build the O-grid and show what it is worth."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fluidsolver.gui import widgets
from fluidsolver.gui.plot_canvas import Canvas, draw_body, draw_mesh
from fluidsolver.gui.session import Session
from fluidsolver.mesh import spacing
from fluidsolver.mesh.metrics import compute_metrics
from fluidsolver.mesh.ogrid import MeshError, build_ogrid
from fluidsolver.mesh.quality import assess

VIEWS = {
    "Far field": None,
    "Near field": 1.5,
    "Body": 0.7,
    "Leading edge": 0.06,
}


class MeshPage(QWidget):
    """Wall spacing, far-field distance, and the quality report."""

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.grid = None
        self.report = None

        outer, content = widgets.page(
            "Mesh",
            "A body-fitted O-grid, marched outward from the surface and blended "
            "into a circular far field. The first cell height is set by the y+ "
            "target, because that is what the turbulence model's wall condition "
            "requires.",
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

        settings = session.mesh_settings
        box, form = widgets.group("Resolution")
        self.surface_points = widgets.integer(
            settings.surface_points, minimum=32, maximum=4000, step=20
        )
        self.y_plus = widgets.number(
            settings.target_y_plus, minimum=0.05, maximum=300.0, decimals=2, step=0.5
        )
        self.far_field = widgets.number(
            settings.far_field_radius_ratio,
            minimum=5.0, maximum=200.0, decimals=1, step=5.0,
            suffix="reference lengths",
        )
        self.growth = widgets.number(
            settings.growth, minimum=1.01, maximum=1.5, decimals=3, step=0.01
        )
        form.addRow("Points around the body", self.surface_points)
        form.addRow("Target y+", self.y_plus)
        form.addRow("Far-field radius", self.far_field)
        form.addRow("Wall-normal growth", self.growth)
        controls.addWidget(box)

        self.advice = widgets.note(
            "y+ of about 1 is required to integrate k-omega SST to the wall. "
            "Larger values leave the viscous sublayer unresolved and the model "
            "is then being asked for something it cannot give."
        )
        controls.addWidget(self.advice)

        self.generate = QPushButton("Generate mesh")
        self.generate.clicked.connect(self.build)
        controls.addWidget(self.generate)

        quality_box, quality_form = widgets.group("Quality")
        self.summary = widgets.readout("no mesh yet")
        self.summary.setAlignment(Qt.AlignTop)
        quality_form.addRow(self.summary)
        controls.addWidget(quality_box)

        self.problem = widgets.note()
        controls.addWidget(self.problem)
        controls.addStretch(1)

        view_row = QHBoxLayout()
        self.view = QComboBox()
        self.view.addItems(VIEWS.keys())
        self.view.setCurrentText("Body")
        self.view.currentTextChanged.connect(self._draw)
        view_row.addWidget(widgets.readout("View"))
        view_row.addWidget(self.view)
        view_row.addStretch(1)

        plot_column = QVBoxLayout()
        plot_column.addLayout(view_row)
        self.canvas = Canvas(6.5, 5.5)
        plot_column.addWidget(self.canvas, 1)
        columns.addLayout(plot_column, 1)

        for control in (self.surface_points, self.y_plus, self.far_field, self.growth):
            control.valueChanged.connect(self._settings_changed)
        session.changed.connect(self._invalidate)

    # ------------------------------------------------------------------

    def _settings_changed(self) -> None:
        settings = self.session.mesh_settings
        settings.surface_points = self.surface_points.value()
        settings.target_y_plus = self.y_plus.value()
        settings.far_field_radius_ratio = self.far_field.value()
        settings.growth = self.growth.value()
        self._invalidate()

    def _invalidate(self) -> None:
        """The mesh no longer matches the settings that produced it."""
        if self.grid is not None:
            self.grid = None
            self.report = None
            self.summary.setText("settings changed - generate the mesh again")
            self.canvas.clear()
            self.canvas.draw()

    def has_mesh(self) -> bool:
        return self.grid is not None and self.report is not None and self.report.is_usable

    # ------------------------------------------------------------------

    def build(self) -> None:
        session = self.session
        self.problem.setText("")
        try:
            contour = session.contour()
            if session.freestream.angle_of_attack_deg:
                pivot = np.array([contour.bounds[0], contour.centroid[1]])
                contour = contour.rotated(
                    -session.freestream.angle_of_attack_deg, about=pivot
                )
            surface = contour.resample(session.mesh_settings.surface_points)

            first_layer = session.mesh_settings.resolve_first_layer(
                session.fluid,
                session.freestream,
                session.shape.reference_length,
                laminar=session.model_name == "laminar",
            )
            self.grid = build_ogrid(
                surface,
                first_layer=first_layer,
                far_field_radius=session.mesh_settings.far_field_radius(
                    session.shape.reference_length
                ),
                growth=session.mesh_settings.growth,
            )
        except (MeshError, ValueError) as error:
            self.grid = None
            self.report = None
            self.problem.setText(str(error))
            self.summary.setText("mesh generation failed")
            self.canvas.clear()
            self.canvas.draw()
            return

        metrics = compute_metrics(self.grid.nodes)
        self.report = assess(metrics, self.grid.nodes)

        achieved = spacing.y_plus_of(
            first_layer,
            session.freestream.velocity,
            session.shape.reference_length,
            session.fluid.density,
            session.fluid.viscosity,
        )
        lines = [
            f"first cell        {first_layer:.4g} m  (y+ approx {achieved:.2f})",
            self.report.summary(),
        ]
        lines.extend(self.grid.notes)
        self.summary.setText("\n".join(lines))
        self._draw()

    def _draw(self) -> None:
        if self.grid is None:
            return

        self.canvas.clear()
        axes = self.canvas.figure.add_subplot(111)

        nodes = self.grid.nodes
        span = VIEWS[self.view.currentText()]
        centre = self.grid.contour.centroid

        if span is None:
            half = np.linalg.norm(self.grid.far_field - centre, axis=1).max() * 1.02
            stride_i, stride_j = max(1, nodes.shape[0] // 40), max(1, nodes.shape[1] // 30)
        else:
            half = span * self.session.shape.reference_length
            if span < 0.1:
                centre = nodes[int(np.argmin(nodes[:, 0, 0])), 0]
            stride_i = stride_j = 1

        draw_mesh(axes, nodes, stride_i=stride_i, stride_j=stride_j)
        draw_body(axes, self.grid.wall)

        axes.set_xlim(centre[0] - half, centre[0] + half)
        axes.set_ylim(centre[1] - half, centre[1] + half)
        axes.set_aspect("equal")
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(
            f"{self.grid.shape[0]} x {self.grid.shape[1]} = "
            f"{self.grid.n_cells:,} cells"
        )
        self.canvas.draw()
