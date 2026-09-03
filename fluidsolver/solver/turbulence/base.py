"""The interface every turbulence closure implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fluidsolver.solver import operators as ops
from fluidsolver.solver.bc import Boundaries
from fluidsolver.solver.faces import FaceGeometry
from fluidsolver.solver.fields import State
from fluidsolver.solver.fluid import Fluid


class TurbulenceModel(ABC):
    """A closure supplying the eddy viscosity to the momentum equations.

    The contract is narrow on purpose. A model is handed the current state and
    must leave ``state.eddy_viscosity`` consistent with it; whether it does that
    by solving transport equations or by returning zero is its own business.
    """

    name = "turbulence"

    def __init__(
        self,
        faces: FaceGeometry,
        fluid: Fluid,
        boundaries: Boundaries,
        numerics,
        reference_length: float = 1.0,
    ):
        self.faces = faces
        self.fluid = fluid
        self.boundaries = boundaries
        self.numerics = numerics
        self.reference_length = reference_length
        #: Current CFL number, kept in step with the pressure-velocity coupling's
        #: by whoever owns the run. See
        #: :func:`fluidsolver.solver.operators.pseudo_time_diagonal`.
        self.cfl = numerics.cfl

    def pseudo_time_diagonal(self, state: State) -> np.ndarray | None:
        """``rho V / dtau`` for the transported turbulence quantities.

        The same local step the momentum equations use, so that k and omega are
        damped consistently with the velocity field driving them. ``None`` when
        pseudo-transient continuation is switched off.
        """
        if not self.numerics.pseudo_transient:
            return None
        return ops.pseudo_time_diagonal(
            state.flux_i,
            state.flux_j,
            self.faces.metrics.volume,
            density=self.fluid.density,
            velocity=self.boundaries.freestream.velocity,
            reference_length=self.reference_length,
            cfl=self.cfl,
        )

    @abstractmethod
    def update(self, state: State) -> tuple[float, float]:
        """Advance the model one outer iteration and refresh the eddy viscosity.

        Returns the residuals of whatever equations were solved, as a
        ``(k, omega)`` pair; a model with no transport equations returns zeros.
        """

    def strain_rate(self, state: State, gradient) -> np.ndarray:
        """``S = sqrt(2 S_ij S_ij)``, the invariant both production terms use.

        In two dimensions

            2 S_ij S_ij = 2[(du/dx)^2 + (dv/dy)^2] + (du/dy + dv/dx)^2
        """
        far_flux = state.flux_j[:, -1]
        wall_u, wall_v = self.boundaries.wall_velocity()
        far_u, far_v = self.boundaries.far_velocity(state.u, state.v, far_flux)

        grad_u = gradient(state.u, wall_u, far_u)
        grad_v = gradient(state.v, wall_v, far_v)

        return np.sqrt(
            2.0 * (grad_u[..., 0] ** 2 + grad_v[..., 1] ** 2)
            + (grad_u[..., 1] + grad_v[..., 0]) ** 2
        )

    def vorticity(self, state: State, gradient) -> np.ndarray:
        """``Omega = sqrt(2 W_ij W_ij)``, which in two dimensions is |dv/dx - du/dy|.

        No closure here uses it: SST takes its production from the strain rate,
        as Menter's equation (5) specifies. It is kept because it is the other
        invariant of the velocity gradient and a rotation-curvature correction or
        a vorticity-based limiter would need it, and because it is measured with
        the same boundary treatment as :meth:`strain_rate`, which is the part
        that is easy to get wrong. :func:`fluidsolver.solver.post.vorticity` is
        the *signed* version, for plotting.
        """
        far_flux = state.flux_j[:, -1]
        wall_u, wall_v = self.boundaries.wall_velocity()
        far_u, far_v = self.boundaries.far_velocity(state.u, state.v, far_flux)

        grad_u = gradient(state.u, wall_u, far_u)
        grad_v = gradient(state.v, wall_v, far_v)
        return np.abs(grad_v[..., 0] - grad_u[..., 1])
