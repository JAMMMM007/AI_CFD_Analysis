"""Running the solver off the GUI thread.

A converged RANS solve takes minutes. Running it on the GUI thread would freeze
the window for the duration, so it runs in a :class:`QThread` and reports back by
signal.

The two rules that make this safe are worth stating, because breaking either
produces intermittent crashes rather than honest errors:

* The worker never touches a Qt widget, and never touches matplotlib. It emits
  signals; the GUI thread does the drawing.
* Field snapshots are *copied* before they are emitted. The solver keeps mutating
  its state array in place, and handing the GUI a live reference means it would
  be drawing a field that is being overwritten underneath it.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from fluidsolver.solver.case import Case
from fluidsolver.solver.fields import Residuals


class SolverWorker(QObject):
    """Drives a :class:`~fluidsolver.solver.case.Case` and reports progress."""

    progressed = Signal(object)  # Residuals
    snapshot = Signal(int, object)  # iteration, State (a copy)
    finished = Signal(str)  # why it stopped
    failed = Signal(str)

    def __init__(self, case: Case, snapshot_every: int = 20):
        super().__init__()
        self.case = case
        self.snapshot_every = snapshot_every
        self._stop = False
        self._paused = False

    # -- called from the GUI thread ------------------------------------

    def request_stop(self) -> None:
        self._stop = True

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    # -- runs on the worker thread -------------------------------------

    def run(self) -> None:
        try:
            self.case.run(callback=self._report, report_every=1)
        except FloatingPointError as error:
            self.failed.emit(str(error))
            return
        except Exception as error:  # noqa: BLE001 - surfaced to the user verbatim
            self.failed.emit(f"{type(error).__name__}: {error}")
            return

        self._emit_snapshot(self.case.iteration)
        self.finished.emit(self._reason())

    def _report(self, residuals: Residuals) -> bool:
        while self._paused and not self._stop:
            QThread.msleep(80)

        self.progressed.emit(residuals)
        if residuals.iteration % self.snapshot_every == 0:
            self._emit_snapshot(residuals.iteration)

        return not self._stop

    def _emit_snapshot(self, iteration: int) -> None:
        """Hand the GUI a copy of the field, tagged with the iteration it is.

        The iteration travels with the snapshot rather than being read off the
        case afterwards. The solver keeps iterating, so by the time the GUI
        handles the signal ``case.iteration`` is a different number and the
        replay would be labelled with iterations its frames do not belong to.
        """
        self.snapshot.emit(iteration, self.case.state.copy())

    def _reason(self) -> str:
        history = self.case.history
        if not history.entries:
            return "stopped before the first iteration"

        last = history.entries[-1]
        if self._stop:
            return f"stopped at iteration {last.iteration}"
        if last.has_converged(self.case.numerics.tolerance):
            return (
                f"converged in {last.iteration} iterations "
                f"(residual {last.worst:.2e})"
            )
        return (
            f"reached the {self.case.numerics.max_iterations}-iteration limit "
            f"at residual {last.worst:.2e}"
        )
