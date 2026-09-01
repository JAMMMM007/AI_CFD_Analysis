"""Small Qt helpers, so the pages read as forms rather than as widget plumbing."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


def number(
    value: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1e12,
    decimals: int = 4,
    step: float = 0.1,
    suffix: str = "",
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(f"  {suffix}")
    box.setKeyboardTracking(False)
    box.setMinimumWidth(150)
    return box


def scientific(value: float, *, suffix: str = "") -> QDoubleSpinBox:
    """A spin box for quantities like viscosity that span many decades."""
    box = number(value, minimum=1e-12, maximum=1e6, decimals=10, step=1e-6, suffix=suffix)
    box.setMinimumWidth(180)
    return box


def integer(
    value: int, *, minimum: int = 1, maximum: int = 1_000_000, step: int = 1
) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setMinimumWidth(150)
    return box


def group(title: str) -> tuple[QGroupBox, QFormLayout]:
    box = QGroupBox(title)
    form = QFormLayout(box)
    form.setLabelAlignment(Qt.AlignRight)
    form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
    return box, form


def readout(text: str = "") -> QLabel:
    """A monospaced label for computed quantities, so columns line up."""
    label = QLabel(text)
    label.setFont(QFont("Consolas", 9))
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def note(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #8a6d00;")
    return label


def heading(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setPointSize(font.pointSize() + 4)
    font.setBold(True)
    label.setFont(font)
    return label


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def page(title: str, subtitle: str = "") -> tuple[QVBoxLayout, QVBoxLayout]:
    """Standard page scaffolding: a title, an optional subtitle, then content.

    Returns the outer layout (to add the page to) and the content layout.
    """
    outer = QVBoxLayout()
    outer.addWidget(heading(title))
    if subtitle:
        caption = QLabel(subtitle)
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #555555;")
        outer.addWidget(caption)
    outer.addWidget(separator())

    content = QVBoxLayout()
    outer.addLayout(content, 1)
    return outer, content
