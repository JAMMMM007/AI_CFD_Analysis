"""Numerics page: turbulence model, discretisation and iteration control."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from fluidsolver.gui import widgets
from fluidsolver.gui.session import Session
from fluidsolver.solver import operators as ops

MODEL_LABELS = {
    "k-omega SST (Menter 2003)": "k-omega-sst",
    "Laminar - no turbulence model (validation)": "laminar",
}

SCHEME_LABELS = {
    "linear (second order, recommended)": "linear",
    "linear upwind (second order)": "linear_upwind",
    "limited linear (bounded second order)": "limited_linear",
    "upwind (first order, most robust)": "upwind",
}


class NumericsPage(QWidget):
    """Model choice and the knobs that govern how the iteration behaves."""

    def __init__(self, session: Session):
        super().__init__()
        self.session = session

        outer, content = widgets.page(
            "Model and numerics",
            "The turbulence closure, the convection scheme, and how hard each "
            "iteration is allowed to move the solution.",
        )
        self.setLayout(outer)

        columns = QHBoxLayout()
        content.addLayout(columns)
        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left)
        columns.addLayout(right)
        columns.addStretch(1)

        model_box, model_form = widgets.group("Turbulence")
        self.model = QComboBox()
        self.model.addItems(MODEL_LABELS.keys())
        model_form.addRow("Model", self.model)
        left.addWidget(model_box)

        self.model_note = widgets.readout()
        left.addWidget(self.model_note)

        scheme_box, scheme_form = widgets.group("Discretisation")
        self.scheme = QComboBox()
        self.scheme.addItems(SCHEME_LABELS.keys())
        self.scheme.setCurrentIndex(0)
        self.turbulence_scheme = QComboBox()
        self.turbulence_scheme.addItems(SCHEME_LABELS.keys())
        self.turbulence_scheme.setCurrentIndex(3)  # upwind
        scheme_form.addRow("Momentum convection", self.scheme)
        scheme_form.addRow("Turbulence convection", self.turbulence_scheme)
        left.addWidget(scheme_box)

        left.addWidget(
            widgets.note(
                "Turbulence transport defaults to upwind on purpose. Near a wall "
                "omega spans six orders of magnitude across a few cells, and a "
                "higher-order correction there becomes an explicit source far "
                "larger than the physical terms it sits beside."
            )
        )
        left.addStretch(1)

        relax_box, relax_form = widgets.group("Under-relaxation")
        numerics = session.numerics
        self.relax_velocity = widgets.number(
            numerics.relax_velocity, minimum=0.05, maximum=1.0, decimals=2, step=0.05
        )
        self.relax_pressure = widgets.number(
            numerics.relax_pressure, minimum=0.02, maximum=1.0, decimals=2, step=0.05
        )
        self.relax_turbulence = widgets.number(
            numerics.relax_turbulence, minimum=0.05, maximum=1.0, decimals=2, step=0.05
        )
        self.relax_eddy = widgets.number(
            numerics.relax_eddy_viscosity,
            minimum=0.05, maximum=1.0, decimals=2, step=0.05,
        )
        relax_form.addRow("Velocity", self.relax_velocity)
        relax_form.addRow("Pressure", self.relax_pressure)
        relax_form.addRow("k and omega", self.relax_turbulence)
        relax_form.addRow("Eddy viscosity", self.relax_eddy)
        right.addWidget(relax_box)

        right.addWidget(
            widgets.note(
                "Velocity and pressure relaxation should roughly sum to one. "
                "Relaxation is applied implicitly, so it changes how the run gets "
                "there and never where it ends up."
            )
        )

        stop_box, stop_form = widgets.group("Stopping")
        self.max_iterations = widgets.integer(
            numerics.max_iterations, minimum=10, maximum=100000, step=100
        )
        self.tolerance = widgets.number(
            numerics.tolerance, minimum=1e-12, maximum=1e-1, decimals=10, step=1e-6
        )
        stop_form.addRow("Maximum iterations", self.max_iterations)
        stop_form.addRow("Residual tolerance", self.tolerance)
        right.addWidget(stop_box)
        right.addStretch(1)

        self.model.currentIndexChanged.connect(self._apply)
        self.scheme.currentIndexChanged.connect(self._apply)
        self.turbulence_scheme.currentIndexChanged.connect(self._apply)
        for control in (
            self.relax_velocity, self.relax_pressure, self.relax_turbulence,
            self.relax_eddy, self.max_iterations, self.tolerance,
        ):
            control.valueChanged.connect(self._apply)

        self._apply()

    # ------------------------------------------------------------------

    def _apply(self) -> None:
        session = self.session
        session.model_name = MODEL_LABELS[self.model.currentText()]

        numerics = session.numerics
        numerics.scheme = SCHEME_LABELS[self.scheme.currentText()]
        numerics.turbulence_scheme = SCHEME_LABELS[self.turbulence_scheme.currentText()]
        numerics.relax_velocity = self.relax_velocity.value()
        numerics.relax_pressure = self.relax_pressure.value()
        numerics.relax_turbulence = self.relax_turbulence.value()
        numerics.relax_eddy_viscosity = self.relax_eddy.value()
        numerics.max_iterations = self.max_iterations.value()
        numerics.tolerance = self.tolerance.value()

        turbulent = session.model_name != "laminar"
        for control in (self.relax_turbulence, self.relax_eddy, self.turbulence_scheme):
            control.setEnabled(turbulent)

        self.model_note.setText(
            "Integrates to the wall; needs y+ of about 1.\nMesh spacing is sized "
            "from the y+ target on the mesh page."
            if turbulent
            else "No turbulence model: the Navier-Stokes equations directly.\n"
            "Mesh spacing is sized to resolve the laminar boundary layer instead."
        )
        assert numerics.scheme in ops.SCHEMES
        session.notify()
