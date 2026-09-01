"""Setup page: the fluid and the oncoming flow."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from fluidsolver.gui import widgets
from fluidsolver.gui.session import Session
from fluidsolver.solver.fluid import PRESETS, Fluid


class SetupPage(QWidget):
    """Fluid properties, freestream condition and inlet turbulence."""

    def __init__(self, session: Session):
        super().__init__()
        self.session = session

        outer, content = widgets.page(
            "Fluid and flow",
            "Everything the Reynolds number depends on. The mesh is sized from "
            "these values, so changing them invalidates any mesh already built.",
        )
        self.setLayout(outer)

        columns = QHBoxLayout()
        content.addLayout(columns)

        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left)
        columns.addLayout(right)
        columns.addStretch(1)

        # -- fluid ------------------------------------------------------
        fluid_box, fluid_form = widgets.group("Fluid")
        self.preset = QComboBox()
        for preset in PRESETS:
            self.preset.addItem(preset.name, preset)
        self.preset.addItem("custom", None)
        self.density = widgets.number(
            session.fluid.density, minimum=1e-6, decimals=4, step=0.1, suffix="kg/m3"
        )
        self.viscosity = widgets.scientific(session.fluid.viscosity, suffix="Pa s")
        fluid_form.addRow("Preset", self.preset)
        fluid_form.addRow("Density", self.density)
        fluid_form.addRow("Dynamic viscosity", self.viscosity)
        left.addWidget(fluid_box)

        # -- freestream -------------------------------------------------
        flow_box, flow_form = widgets.group("Freestream")
        self.velocity = widgets.number(
            session.freestream.velocity, minimum=1e-6, decimals=3, step=1.0, suffix="m/s"
        )
        self.incidence = widgets.number(
            session.freestream.angle_of_attack_deg,
            minimum=-90.0, maximum=90.0, decimals=2, step=0.5, suffix="deg",
        )
        self.reference_length = widgets.number(
            session.shape.reference_length, minimum=1e-9, decimals=5, step=0.1, suffix="m"
        )
        flow_form.addRow("Velocity", self.velocity)
        flow_form.addRow("Angle of attack", self.incidence)
        flow_form.addRow("Reference length", self.reference_length)
        left.addWidget(flow_box)

        # -- turbulence inlet -------------------------------------------
        turbulence_box, turbulence_form = widgets.group("Freestream turbulence")
        self.intensity = widgets.number(
            session.freestream.turbulence_intensity * 100.0,
            minimum=1e-4, maximum=50.0, decimals=3, step=0.1, suffix="%",
        )
        self.viscosity_ratio = widgets.number(
            session.freestream.eddy_viscosity_ratio,
            minimum=1e-3, maximum=1e4, decimals=3, step=1.0,
        )
        turbulence_form.addRow("Turbulence intensity", self.intensity)
        turbulence_form.addRow("Eddy viscosity ratio mu_t/mu", self.viscosity_ratio)
        left.addWidget(turbulence_box)
        left.addStretch(1)

        # -- readouts ---------------------------------------------------
        summary_box, summary_form = widgets.group("Resulting condition")
        self.reynolds = widgets.readout()
        self.mach = widgets.readout()
        self.dynamic_pressure = widgets.readout()
        self.turbulence_values = widgets.readout()
        self.first_cell = widgets.readout()
        summary_form.addRow("Reynolds number", self.reynolds)
        summary_form.addRow("Mach number", self.mach)
        summary_form.addRow("Dynamic pressure", self.dynamic_pressure)
        summary_form.addRow("Inlet k, omega", self.turbulence_values)
        summary_form.addRow("First cell for y+ = 1", self.first_cell)
        right.addWidget(summary_box)

        self.warning = widgets.note()
        right.addWidget(self.warning)
        right.addStretch(1)

        for control in (
            self.density, self.viscosity, self.velocity, self.incidence,
            self.reference_length, self.intensity, self.viscosity_ratio,
        ):
            control.valueChanged.connect(self._apply)
        self.preset.currentIndexChanged.connect(self._preset_chosen)

        self._preset_chosen()

    # ------------------------------------------------------------------

    def _preset_chosen(self) -> None:
        preset = self.preset.currentData()
        custom = preset is None
        self.density.setEnabled(custom)
        self.viscosity.setEnabled(custom)
        if preset is not None:
            self.density.blockSignals(True)
            self.viscosity.blockSignals(True)
            self.density.setValue(preset.density)
            self.viscosity.setValue(preset.viscosity)
            self.density.blockSignals(False)
            self.viscosity.blockSignals(False)
        self._apply()

    def _apply(self) -> None:
        from dataclasses import replace

        from fluidsolver.mesh import spacing

        session = self.session
        name = self.preset.currentText()
        session.fluid = Fluid(
            density=self.density.value(),
            viscosity=self.viscosity.value(),
            name=name,
        )
        session.freestream = replace(
            session.freestream,
            velocity=self.velocity.value(),
            angle_of_attack_deg=self.incidence.value(),
            turbulence_intensity=max(self.intensity.value() / 100.0, 1e-6),
            eddy_viscosity_ratio=self.viscosity_ratio.value(),
        )
        session.shape.reference_length = self.reference_length.value()

        self.reynolds.setText(f"{session.reynolds:.4g}")
        mach = session.freestream.mach()
        self.mach.setText(f"{mach:.4f}   (air at 15 C)")
        self.dynamic_pressure.setText(
            f"{session.freestream.dynamic_pressure(session.fluid):.4g} Pa"
        )
        self.turbulence_values.setText(
            f"{session.freestream.turbulent_kinetic_energy():.4g} m2/s2,  "
            f"{session.freestream.specific_dissipation(session.fluid):.4g} 1/s"
        )
        try:
            thickness = spacing.first_layer_thickness(
                1.0, session.freestream.velocity, session.shape.reference_length,
                session.fluid.density, session.fluid.viscosity,
            )
            self.first_cell.setText(
                f"{thickness:.4g} m   ({thickness / session.shape.reference_length:.3g} "
                f"of the reference length)"
            )
        except Exception:
            self.first_cell.setText("-")

        self.warning.setText(session.freestream.compressibility_warning() or "")
        session.notify()
