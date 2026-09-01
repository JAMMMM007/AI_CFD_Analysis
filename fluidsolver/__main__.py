"""Launch the application: ``py -m fluidsolver``."""

from __future__ import annotations

import sys


def main() -> int:
    import matplotlib

    # Qt must be the matplotlib backend before any figure is created, or the
    # first canvas quietly opens its own window instead of embedding.
    matplotlib.use("QtAgg")

    from PySide6.QtWidgets import QApplication

    from fluidsolver.gui.main_window import MainWindow

    application = QApplication(sys.argv)
    application.setApplicationName("fluidsolver")

    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
