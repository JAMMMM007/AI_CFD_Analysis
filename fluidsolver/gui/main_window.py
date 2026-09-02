"""The main window: a step rail down the side, one page at a time beside it."""

from __future__ import annotations

from PySide6.QtCore import QMargins
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from fluidsolver import __version__
from fluidsolver.gui.pages.mesh import MeshPage
from fluidsolver.gui.pages.numerics import NumericsPage
from fluidsolver.gui.pages.run import RunPage
from fluidsolver.gui.pages.setup import SetupPage
from fluidsolver.gui.pages.shape import ShapePage
from fluidsolver.gui.session import Session

STEPS = [
    ("1  Fluid and flow", SetupPage),
    ("2  Body", ShapePage),
    ("3  Mesh", MeshPage),
    ("4  Model and numerics", NumericsPage),
    ("5  Solve", RunPage),
]


class MainWindow(QMainWindow):
    """Holds the session and the five pages that edit it."""

    def __init__(self):
        super().__init__()
        self.session = Session()

        self.setWindowTitle(f"fluidsolver {__version__}  -  2-D RANS")
        self._size_to_the_screen()

        central = QWidget()
        layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # -- step rail --------------------------------------------------
        rail = QVBoxLayout()
        self.steps = QListWidget()
        self.steps.setFixedWidth(220)
        self.steps.setSpacing(2)
        for label, _ in STEPS:
            item = QListWidgetItem(label)
            item.setSizeHint(item.sizeHint().grownBy(QMargins(0, 8, 0, 8)))
            self.steps.addItem(item)
        rail.addWidget(self.steps, 1)

        self.back = QPushButton("< Back")
        self.next = QPushButton("Next >")
        self.back.clicked.connect(lambda: self._go(self.steps.currentRow() - 1))
        self.next.clicked.connect(lambda: self._go(self.steps.currentRow() + 1))
        buttons = QHBoxLayout()
        buttons.addWidget(self.back)
        buttons.addWidget(self.next)
        rail.addLayout(buttons)
        layout.addLayout(rail)

        # -- pages ------------------------------------------------------
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)

        self.page_widgets = []
        for _, factory in STEPS:
            widget = factory(self.session)
            self.page_widgets.append(widget)
            self.pages.addWidget(widget)

        self.steps.currentRowChanged.connect(self._step_selected)
        self.steps.setCurrentRow(0)

        self.setStatusBar(QStatusBar())
        self.session.changed.connect(self._refresh_status)
        self._refresh_status()

    # ------------------------------------------------------------------

    def _size_to_the_screen(self) -> None:
        """Open nearly full screen, and centred.

        The field plot is what this application is for, and a fixed default size
        leaves most of a modern monitor empty around it. The available geometry
        is used rather than the whole screen so the window does not open under
        the task bar, and a floor keeps it usable if no screen is reported --
        which is the case under a headless test run.
        """
        self.setMinimumSize(1100, 720)

        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1600, 1000)
            return

        available = screen.availableGeometry()
        width = max(1100, int(available.width() * 0.94))
        height = max(720, int(available.height() * 0.94))
        self.resize(width, height)
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )

    def _go(self, row: int) -> None:
        if 0 <= row < self.steps.count():
            self.steps.setCurrentRow(row)

    def _step_selected(self, row: int) -> None:
        self.pages.setCurrentIndex(row)
        self.back.setEnabled(row > 0)
        self.next.setEnabled(row < self.steps.count() - 1)

        # The shape preview depends on the incidence set on the setup page, so it
        # is refreshed on arrival rather than only when the shape itself changes.
        widget = self.page_widgets[row]
        if hasattr(widget, "refresh"):
            widget.refresh()

    def _refresh_status(self) -> None:
        self.statusBar().showMessage(self.session.description())

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        """Stop a running solve, and the replay, before the window goes away.

        A QThread still running when its Python objects are collected takes the
        interpreter down with it, and the crash looks nothing like its cause. A
        replay timer left armed is the milder version of the same problem: it
        fires into a half-deleted page.
        """
        run_page = self.page_widgets[-1]
        if hasattr(run_page, "halt"):
            run_page.halt()
        event.accept()
