"""Whether a case can be solved at all, asked before any time is spent on it.

:mod:`fluidsolver.mesh.quality` judges the mesh on its own terms -- inverted
cells, non-orthogonality, skewness -- and it cannot see the flow that will be run
on it. This module asks the question that needs both: given *this* mesh, *this*
fluid and *this* model, is the answer going to mean anything?

The failure it exists to prevent was a real one. A cylinder at Re = 2.03e6 with
the laminar model ran for 459 iterations, passed through a lift coefficient of
37525, and died in the linear solver. Nothing in the program had anything to say
beforehand, and nothing said anything useful afterwards either. Both halves of
that case were hopeless before the first iteration:

* **Numerically.** The cell Peclet number -- how far convection outruns diffusion
  across one cell -- had a median of 3.2e4 and a peak of 2.6e7. Central
  differencing is unbounded above 2, and no relaxation factor rescues a
  discretisation being asked to resolve a length scale four orders below the
  mesh.
* **Physically.** Flow past a cylinder at Re = 2e6 is turbulent. A laminar model
  there is not an approximation, it is a different problem, and resolving it
  honestly would need on the order of ``Re^(9/4)`` cells.

The thresholds below are measured rather than assumed. Holding the mesh and the
solver fixed and sweeping only the Reynolds number, the laminar cylinder gives:

    Re          median Pe    outcome
    40                8.8    converged, 4.1e-04
    200                29    converged, 1.4e-03
    1 000              98    converged, 8.8e-04
    5 000             328    symmetry lost, 4.8e-03
    2 030 000      32 000    diverged

which brackets the two limits used here. Note the Re = 5000 row: losing symmetry
there is the cylinder genuinely shedding, not the discretisation failing, and a
threshold that refused it would be refusing a correct answer. That is why 328 is
allowed and 32 000 is not.

The Peclet test applies to the laminar model only. A turbulent case runs at a
high *molecular* Peclet number by design -- the eddy viscosity supplies the
diffusion, often thousands of times the molecular value -- so the same number
means something entirely different there, and judging SST by it would reject
every case it is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluidsolver.mesh import spacing
from fluidsolver.mesh.metrics import Metrics
from fluidsolver.solver.fluid import Fluid, Freestream

#: Median cell Peclet number above which a laminar run is refused. The laminar
#: cylinder converged at 98 and died at 3.2e4; this sits between them, nearer the
#: failure, because the cost of refusing a case that would have worked is an
#: annoyed user and the cost of accepting one that cannot is a plausible wrong
#: number in a report.
_PECLET_BLOCK = 1.0e4
#: And where it is worth saying something without refusing.
_PECLET_WARN = 1.0e3

#: Above this the flow being modelled is turbulent whatever the mesh does. Not a
#: blocker: a laminar reference solution at high Reynolds number is a legitimate
#: thing to want, as long as nobody mistakes it for the real flow.
_LAMINAR_REYNOLDS_WARN = 1.0e5

#: What integrating k-omega SST to the wall needs, and where it stops being met.
#: Menter's automatic wall treatment removes this constraint; until it lands, a
#: first cell outside the viscous sublayer is being asked for something the model
#: cannot give.
_Y_PLUS_TARGET = 1.0
_Y_PLUS_WARN = 5.0


class UnsolvableCase(ValueError):
    """Raised when a case is refused before it is run.

    A distinct type because it means something a caller may want to act on: not
    that the request was malformed, but that it was well formed and hopeless. The
    GUI shows it as advice rather than as a crash.
    """


@dataclass(frozen=True)
class HealthReport:
    """What the mesh, the fluid and the model say about each other."""

    model_name: str
    reynolds: float
    median_peclet: float
    peak_peclet: float
    estimated_y_plus: float
    cells: int

    @property
    def laminar(self) -> bool:
        return self.model_name == "laminar"

    @property
    def blockers(self) -> list[str]:
        """Reasons the run cannot produce a meaningful answer. Empty means go."""
        issues = []
        if self.laminar and self.median_peclet > _PECLET_BLOCK:
            issues.append(
                f"the median cell Peclet number is {self.median_peclet:.3g} "
                f"(peak {self.peak_peclet:.3g}) with the laminar model. Convection "
                f"outruns diffusion by that factor across a single cell, so the "
                f"discretisation cannot represent this flow at any relaxation "
                f"factor -- the run would not converge, and if it appeared to, the "
                f"answer would be an artefact of the scheme. Refine the mesh, lower "
                f"the Reynolds number, or use a turbulence model."
            )
        return issues

    @property
    def warnings(self) -> list[str]:
        """Concerns worth raising that do not make the answer meaningless."""
        issues = []
        if self.laminar and _PECLET_WARN < self.median_peclet <= _PECLET_BLOCK:
            issues.append(
                f"median cell Peclet number {self.median_peclet:.3g} with the "
                f"laminar model. High enough that the convection scheme is doing "
                f"most of the work; treat the answer as indicative and check it "
                f"against a finer mesh."
            )
        if self.laminar and self.reynolds > _LAMINAR_REYNOLDS_WARN:
            issues.append(
                f"Reynolds number {self.reynolds:.3g} with the laminar model. The "
                f"real flow is turbulent well below this, so what comes out is a "
                f"laminar solution to a turbulent problem, not an approximation of "
                f"one. Use k-omega SST unless a laminar reference is what you want."
            )
        if not self.laminar and self.estimated_y_plus > _Y_PLUS_WARN:
            issues.append(
                f"first cell at y+ of about {self.estimated_y_plus:.2f}, against "
                f"the {_Y_PLUS_TARGET:.0f} that integrating k-omega SST to the wall "
                f"needs. The viscous sublayer is unresolved, so the wall shear and "
                f"the separation point will both be wrong. Lower the y+ target on "
                f"the mesh page."
            )
        return issues

    @property
    def is_runnable(self) -> bool:
        return not self.blockers

    def summary(self) -> str:
        """Multi-line report, in the same shape as the mesh quality one."""
        lines = [
            f"model                 {self.model_name}",
            f"Reynolds number       {self.reynolds:.4g}",
            f"cell Peclet           {self.median_peclet:.4g} median, "
            f"{self.peak_peclet:.4g} peak",
            f"estimated y+          {self.estimated_y_plus:.3f}",
        ]
        lines.extend(f"BLOCKED: {b}" for b in self.blockers)
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def cell_peclet(metrics: Metrics, nodes: np.ndarray, fluid: Fluid, velocity: float):
    """``rho U h / mu`` per cell, with ``h`` the longer of the two cell edges.

    The longer edge is the right one to use. A boundary-layer cell is thin across
    the layer and long along it, the flow runs along it, and it is the streamwise
    spacing that decides whether convection can be resolved -- taking the short
    edge would report every such cell as comfortable when the opposite is true.
    """
    along_i = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=-1)
    along_j = np.linalg.norm(nodes[:, 1:] - nodes[:, :-1], axis=-1)
    size = np.maximum(
        0.5 * (along_i[:, :-1] + along_i[:, 1:]),
        0.5 * (along_j + np.roll(along_j, -1, axis=0)),
    )
    return fluid.density * velocity * size / fluid.viscosity


def assess(
    metrics: Metrics,
    nodes: np.ndarray,
    fluid: Fluid,
    freestream: Freestream,
    model_name: str,
    reference_length: float,
) -> HealthReport:
    """Judge a case before it is run."""
    peclet = cell_peclet(metrics, nodes, fluid, freestream.velocity)
    first_layer = float(
        np.linalg.norm(nodes[:, 1] - nodes[:, 0], axis=-1).min()
    )
    return HealthReport(
        model_name=model_name,
        reynolds=fluid.reynolds(freestream.velocity, reference_length),
        median_peclet=float(np.median(peclet)),
        peak_peclet=float(peclet.max()),
        estimated_y_plus=spacing.y_plus_of(
            first_layer,
            freestream.velocity,
            reference_length,
            fluid.density,
            fluid.viscosity,
        ),
        cells=int(metrics.volume.size),
    )
