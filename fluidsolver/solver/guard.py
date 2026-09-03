"""Bounding a run that is going wrong, and stopping it with something to read.

:mod:`fluidsolver.solver.health` refuses cases that cannot work before they
start. This module is for the ones that pass that check and go wrong anyway --
because a mesh is locally bad, or a model is being pushed past where it holds, or
the flow simply has no steady state to find.

Two mechanisms, doing different jobs.

**Limits** keep a transient excursion from becoming a permanent one. A cell that
briefly reaches ten times the freestream speed during start-up is recoverable; a
cell that reaches a thousand times is not, and left alone it will take its
neighbours with it through the next assembly. The bounds here are deliberately
generous -- they are not there to shape the answer, and if they are shaping the
answer something else is already wrong. That is why the report counts them: a
limiter firing on a handful of cells for a few iterations is start-up, and one
firing on five per cent of the mesh continuously means the run is worthless and
the numbers coming out of it should not be believed.

**Divergence detection** stops the run while there is still something to look at.
Before this, the first sign of trouble was a ``FloatingPointError`` from inside
the linear solver, hundreds of iterations after the solution stopped being
physical, with no indication of where or why. On the case that prompted all of
this, the run reached a lift coefficient of 37525 before anything objected. The
monitor watches for a sustained rise rather than a spike -- residuals rattle by a
factor of two on a perfectly healthy run -- and when it fires it says which
equation lost it and whereabouts in the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fluidsolver.solver.fields import Residuals, State
from fluidsolver.solver.fluid import Fluid, Freestream

# --- Limits ----------------------------------------------------------------
#
# Multiples of the freestream, chosen to sit well clear of anything physical.
# Inviscid flow reaches twice the freestream over a cylinder and three or four
# times at an aerofoil suction peak; a sharp corner is formally singular and
# viscosity is what bounds it in practice. Measured over the opening iterations,
# the worst a healthy case here reaches is 3.41 times freestream, on a NACA 2412
# at 15 degrees. Ten leaves threefold headroom above that, so reaching it means
# the solution is running away rather than being interesting.
_SPEED_LIMIT = 10.0
# A thousand dynamic pressures, which is far past anything a converging run does
# and far short of a diverging one.
#
# Sized by measurement, after a first attempt at a hundred turned out to sit
# *inside* the healthy band. Starting from a uniform field against a no-slip wall
# is a discontinuity, and the first pressure correction to it is enormous before
# decaying monotonically away: the peak |Cp| over the opening iterations is 123.5
# on the Re = 40 cylinder, 50.4 on the cylinder at Re = 2e6, and 30.3 on a NACA
# 2412 at 15 degrees. A cap of 100 clipped 126 cells on the first iteration of
# the Re 40 benchmark -- harmless there, but a backstop that fires on a case
# which was always going to converge is a backstop shaping the answer, which is
# the one thing it must not do. The run that reached Cl = 37525 was carrying
# pressures of order eighteen thousand dynamic heads, so the two bands are
# separated by two orders of magnitude and the threshold belongs between them.
_PRESSURE_LIMIT = 1000.0
# Turbulent kinetic energy against the freestream kinetic energy. A violent shear
# layer runs at a few per cent of U^2 and the worst measured here was 0.14; one
# whole U^2 is not a physical state.
_ENERGY_LIMIT = 1.0

# --- Divergence detection --------------------------------------------------
#
# Long enough that a genuine trend is needed rather than a run of bad luck.
_MONITOR_WINDOW = 50
# How far the trailing window median must sit above the previous one.
#
# Modest on purpose, and this is the number that matters. An earlier version
# demanded a tenfold climb inside one window and was useless: the divergence this
# exists to catch is slow and steady, about 1.3% per iteration, which is only
# 1.9x over fifty. Tested end to end on the laminar cylinder at Re = 2e6 it ran
# 900 iterations to a residual of 7.7 without ever objecting. Real divergences
# grind; they do not leap.
_MONITOR_RISE = 1.5
# Below this the run is converged enough that a rise does not matter, whatever
# its ratio. Without it a case settling at 1e-9 and drifting to 1e-8 would be
# killed for a rise of ten.
_MONITOR_FLOOR = 1.0e-3
# And it must have genuinely lost ground against its own best, not merely be
# noisy about a plateau it never left.
_MONITOR_LOST = 100.0


class SolverDiverged(RuntimeError):
    """Raised when a run is stopped because it is not going to recover."""


@dataclass(frozen=True)
class LimiterReport:
    """How much of the mesh had to be held back, and for how long."""

    speed: int = 0
    pressure: int = 0
    energy: int = 0
    cells: int = 1
    iterations_active: int = 0

    @property
    def total(self) -> int:
        return self.speed + self.pressure + self.energy

    @property
    def fraction(self) -> float:
        """Share of the mesh clipped on the most recent iteration."""
        return self.total / self.cells

    @property
    def is_quiet(self) -> bool:
        return self.total == 0

    def summary(self) -> str:
        if self.is_quiet and not self.iterations_active:
            return "limiter          never active"
        parts = []
        if self.speed:
            parts.append(f"{self.speed} on speed")
        if self.pressure:
            parts.append(f"{self.pressure} on pressure")
        if self.energy:
            parts.append(f"{self.energy} on k")
        current = ", ".join(parts) if parts else "quiet now"
        return (
            f"limiter          {current}; active on "
            f"{self.iterations_active} iterations"
        )


@dataclass
class SolutionLimits:
    """Generous physical bounds, applied after each outer iteration."""

    fluid: Fluid
    freestream: Freestream
    cells: int
    iterations_active: int = field(default=0, init=False)

    def apply(self, state: State) -> LimiterReport:
        """Clip the state in place and report what had to be held."""
        speed_cap = _SPEED_LIMIT * self.freestream.velocity
        pressure_cap = _PRESSURE_LIMIT * self.freestream.dynamic_pressure(self.fluid)
        energy_cap = _ENERGY_LIMIT * self.freestream.velocity**2

        speed = np.hypot(state.u, state.v)
        over_speed = speed > speed_cap
        if over_speed.any():
            # Scale rather than clip each component, so the direction the flow
            # was heading in survives and only its magnitude is held back.
            scale = np.where(over_speed, speed_cap / np.maximum(speed, 1e-300), 1.0)
            state.u = state.u * scale
            state.v = state.v * scale

        over_pressure = np.abs(state.pressure) > pressure_cap
        if over_pressure.any():
            state.pressure = np.clip(state.pressure, -pressure_cap, pressure_cap)

        over_energy = state.k > energy_cap
        if over_energy.any():
            state.k = np.minimum(state.k, energy_cap)

        report = LimiterReport(
            speed=int(over_speed.sum()),
            pressure=int(over_pressure.sum()),
            energy=int(over_energy.sum()),
            cells=self.cells,
            iterations_active=self.iterations_active,
        )
        if not report.is_quiet:
            self.iterations_active += 1
        return LimiterReport(
            speed=report.speed,
            pressure=report.pressure,
            energy=report.energy,
            cells=self.cells,
            iterations_active=self.iterations_active,
        )


class DivergenceMonitor:
    """Watches the residual history for a rise that is not going to turn around.

    Three conditions, all required, because stopping a run that would have
    recovered is its own kind of failure:

    * the residual is above :data:`_MONITOR_FLOOR`, so a converged run drifting
      in the ninth decimal place is left alone whatever its ratios say;
    * it is :data:`_MONITOR_LOST` times worse than the best the run ever managed,
      so a case that has merely plateaued noisily is left alone too -- that one
      does most of the work, since a plateau is never far from its own best;
    * and the trailing window *median* sits :data:`_MONITOR_RISE` above the
      previous window's, so it is still losing ground rather than recovering.

    The median rather than the mean is what makes the last condition safe. One
    spike in fifty samples moves a log-mean by twenty per cent and a median not
    at all, and residuals do spike on healthy runs.
    """

    def __init__(self):
        self.best = float("inf")
        self._history: list[float] = []

    def update(self, residual: float) -> bool:
        """Fold in one iteration. Returns whether the run should be stopped."""
        if not np.isfinite(residual):
            return True
        if residual <= 0.0:
            return False
        self.best = min(self.best, residual)
        self._history.append(residual)
        if len(self._history) > 2 * _MONITOR_WINDOW:
            self._history.pop(0)

        if residual < _MONITOR_FLOOR:
            return False
        if residual < _MONITOR_LOST * self.best:
            return False
        if len(self._history) < 2 * _MONITOR_WINDOW:
            return False
        recent = np.median(self._history[-_MONITOR_WINDOW:])
        earlier = np.median(self._history[:_MONITOR_WINDOW])
        return bool(recent > _MONITOR_RISE * earlier)


def diagnose(
    state: State,
    residuals: Residuals,
    centroid: np.ndarray,
    fluid: Fluid,
    freestream: Freestream,
    limiter: LimiterReport,
) -> str:
    """Say which equation lost it, and whereabouts in the mesh.

    A residual on its own tells nobody anything actionable. What is wanted is the
    same thing one would go looking for by hand: which field is worst, where it
    is worst, and whether the limiter had been papering over it.
    """
    worst_name, worst_value = max(
        (
            ("Ux", residuals.u),
            ("Uy", residuals.v),
            ("continuity", residuals.continuity),
            ("k", residuals.k),
            ("omega", residuals.omega),
        ),
        key=lambda pair: pair[1],
    )

    speed = np.hypot(state.u, state.v)
    fast = np.unravel_index(int(speed.argmax()), speed.shape)
    high = np.unravel_index(int(np.abs(state.pressure).argmax()), state.pressure.shape)
    q = freestream.dynamic_pressure(fluid)

    return "\n".join(
        [
            f"the run diverged at iteration {residuals.iteration}.",
            "",
            f"  worst equation   {worst_name}, at {worst_value:.3e}",
            f"  fastest cell     {speed[fast]:.4g} m/s "
            f"({speed[fast] / freestream.velocity:.1f} x freestream) "
            f"at (i={fast[0]}, j={fast[1]}), "
            f"x = {centroid[fast][0]:.4g}, y = {centroid[fast][1]:.4g}",
            f"  highest pressure {state.pressure[high]:+.4g} Pa "
            f"(Cp = {state.pressure[high] / q:+.1f}) "
            f"at (i={high[0]}, j={high[1]}), "
            f"x = {centroid[high][0]:.4g}, y = {centroid[high][1]:.4g}",
            f"  {limiter.summary()}",
            "",
            "Look at the mesh around those cells first. If the limiter was "
            "active on most iterations the solution was already unphysical long "
            "before this point.",
        ]
    )
