"""The choices a user has made, shared between pages.

The pages are deliberately dumb: they read and write this object and emit a
signal when something changes. Nothing in the solver knows the GUI exists, and
nothing in the GUI holds solver state except through :attr:`Session.case`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from fluidsolver.geometry.contour import Contour
from fluidsolver.geometry.naca import naca4
from fluidsolver.geometry.primitives import circle, rectangle
from fluidsolver.solver.case import Case, MeshSettings, build_case
from fluidsolver.solver.fluid import AIR_15C, Fluid, Freestream
from fluidsolver.solver.simple import Numerics

# Resolution of the reference contour the generators produce. The mesher
# resamples this down to the requested surface point count, so it only has to be
# fine enough not to lose the shape.
_REFERENCE_POINTS = 1201


@dataclass
class ShapeChoice:
    """Which body, and its parameters."""

    kind: str = "naca"  # naca | circle | square | dxf
    naca_code: str = "0012"
    diameter: float = 1.0
    side: float = 1.0
    corner_radius: float = 0.0
    dxf_path: str = ""
    dxf_scale: float | None = None
    dxf_layers: list[str] = field(default_factory=list)
    dxf_contour: Contour | None = None
    reference_length: float = 1.0

    def build(self) -> Contour:
        """The body, normalised so its reference length is what the user asked for."""
        if self.kind == "naca":
            return naca4(self.naca_code, _REFERENCE_POINTS, chord=self.reference_length)
        if self.kind == "circle":
            return circle(self.reference_length, 512)
        if self.kind == "square":
            return rectangle(
                self.reference_length,
                self.reference_length,
                512,
                corner_radius=self.corner_radius * self.reference_length,
            )
        if self.kind == "dxf":
            if self.dxf_contour is None:
                raise ValueError("no DXF has been imported yet")
            return self.dxf_contour
        raise ValueError(f"unknown shape kind {self.kind!r}")


class Session(QObject):
    """Everything the user has configured, plus the case once it exists."""

    changed = Signal()
    case_changed = Signal()

    def __init__(self):
        super().__init__()
        self.fluid: Fluid = AIR_15C
        self.freestream = Freestream(velocity=30.0)
        self.shape = ShapeChoice()
        self.mesh_settings = MeshSettings()
        self.numerics = Numerics()
        self.model_name = "k-omega-sst"
        self.case: Case | None = None

    # ------------------------------------------------------------------

    @property
    def reynolds(self) -> float:
        return self.fluid.reynolds(
            self.freestream.velocity, self.shape.reference_length
        )

    def contour(self) -> Contour:
        return self.shape.build()

    def notify(self) -> None:
        """Something the user set has changed; any preview is now stale."""
        self.changed.emit()

    def build_case(self) -> Case:
        """Mesh the body and assemble the case. Invalidates any previous run."""
        self.case = build_case(
            self.contour(),
            self.fluid,
            self.freestream,
            mesh_settings=self.mesh_settings,
            numerics=self.numerics,
            model_name=self.model_name,
        )
        self.case_changed.emit()
        return self.case

    def description(self) -> str:
        """One-line summary for the window title bar."""
        return (
            f"{self.shape.kind}  |  {self.fluid.name}  |  "
            f"U = {self.freestream.velocity:g} m/s  |  "
            f"alpha = {self.freestream.angle_of_attack_deg:g} deg  |  "
            f"Re = {self.reynolds:.3g}"
        )
