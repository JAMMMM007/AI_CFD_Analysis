"""No turbulence model: the Navier-Stokes equations, solved directly.

This exists to make the rest of the solver testable. The discretisation, the
boundary conditions, the pressure-velocity coupling and the force integration all
have to be right before a turbulence model can be judged, and only laminar flow
has benchmarks precise enough to prove them: a cylinder at Re = 40 has a drag
coefficient, a wake length and a separation angle all agreed to within a percent
across dozens of published studies.

If those numbers come out, the machinery underneath k-omega SST is sound. If they
do not, no amount of work on the turbulence model will help.
"""

from __future__ import annotations

import numpy as np

from fluidsolver.solver.fields import State
from fluidsolver.solver.turbulence.base import TurbulenceModel


class Laminar(TurbulenceModel):
    """Zero eddy viscosity."""

    name = "laminar"

    def update(self, state: State) -> tuple[float, float]:
        state.eddy_viscosity = np.zeros(self.faces.shape)
        return 0.0, 0.0
