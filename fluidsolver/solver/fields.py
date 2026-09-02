"""The solution state, and the residuals that measure its convergence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fluidsolver.solver.bc import Boundaries
from fluidsolver.solver.faces import FaceGeometry
from fluidsolver.solver.fluid import Fluid, Freestream


@dataclass
class State:
    """Every field the solver carries between iterations.

    Cell fields are ``(Ni, Nj)``. The two flux arrays hold *face* mass fluxes
    ``rho u . S``, signed towards increasing index, and are part of the state
    rather than derived quantities: SIMPLE corrects them directly, and
    reconstructing them from the cell velocities each iteration would discard the
    Rhie-Chow term that keeps pressure and velocity coupled.
    """

    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    k: np.ndarray
    omega: np.ndarray
    eddy_viscosity: np.ndarray
    flux_i: np.ndarray
    flux_j: np.ndarray

    @classmethod
    def uniform(
        cls, faces: FaceGeometry, fluid: Fluid, freestream: Freestream
    ) -> "State":
        """Freestream velocity everywhere, with ``omega`` given its wall profile.

        The velocity may safely start uniform. It violates no-slip, but that is a
        smooth thing to correct and the momentum equations resolve it within a few
        iterations.

        ``omega`` may not. Its wall condition is ``6 nu / (beta1 d1^2)``, which on
        a ``y+ = 1`` mesh is of order ``1e8``, while its freestream value is of
        order ``1e2``. Starting it uniform therefore opens the run with a
        six-order discontinuity across the first cell -- and ``omega`` is the one
        field where that is not survivable. The enormous ``grad omega`` inflates
        the cross-diffusion measure ``CD_k-omega``, which is the denominator of
        the third argument of the blending function ``F1``. ``F1`` collapses to
        zero at the wall, which tells the model to behave as ``k-epsilon``
        *precisely where it must behave as ``k-omega``*, switching the
        cross-diffusion on in the one region it is meant to be switched off. The
        run then diverges within a hundred iterations, and does so in the momentum
        equations, where the actual cause is invisible.

        Seeding ``omega`` with the near-wall asymptote it is heading for anyway
        removes the discontinuity and costs one line.
        """
        shape = faces.shape
        stream = freestream.vector
        boundaries = Boundaries(faces, fluid, freestream)

        k = np.full(shape, freestream.turbulent_kinetic_energy())
        # omega -> 6 nu / (beta1 y^2) approaching a wall; far away the freestream
        # value dominates and this term is negligible.
        distance = np.maximum(faces.metrics.wall_distance, 1e-300)
        omega = freestream.specific_dissipation(fluid) + 6.0 * fluid.kinematic_viscosity / (
            0.075 * distance**2
        )

        state = cls(
            u=np.full(shape, stream[0]),
            v=np.full(shape, stream[1]),
            pressure=np.zeros(shape),
            k=k,
            omega=omega,
            eddy_viscosity=fluid.density * k / omega,
            flux_i=fluid.density * np.sum(faces.metrics.face_i_area * stream, axis=-1),
            flux_j=fluid.density * np.sum(faces.metrics.face_j_area * stream, axis=-1),
        )
        # No mass crosses the wall, ever.
        state.flux_j[:, 0] = 0.0
        state.flux_j[:, -1] = boundaries.enforce_global_mass_balance(
            boundaries.far_flux_from_freestream()
        )
        return state

    @property
    def velocity(self) -> np.ndarray:
        """``(Ni, Nj, 2)`` velocity vector field."""
        return np.stack((self.u, self.v), axis=-1)

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.u, self.v)

    #: The arrays that make up the state, in one place, so that copying it and
    #: measuring it cannot drift apart from each other.
    ARRAYS = (
        "u", "v", "pressure", "k", "omega", "eddy_viscosity", "flux_i", "flux_j",
    )

    def copy(self) -> "State":
        return State(**{name: getattr(self, name).copy() for name in self.ARRAYS})

    @property
    def nbytes(self) -> int:
        """What one copy of this state costs, for callers that keep several."""
        return sum(getattr(self, name).nbytes for name in self.ARRAYS)

    def is_finite(self) -> bool:
        return all(
            np.all(np.isfinite(getattr(self, name)))
            for name in ("u", "v", "pressure", "k", "omega", "eddy_viscosity")
        )


@dataclass
class Residuals:
    """Scaled residuals for one outer iteration, plus the derived quantities.

    ``continuity`` is the one to watch. The momentum residuals measure how well
    each equation was solved with the *current* pressure, but SIMPLE guarantees
    that anyway; it is the mass imbalance that says whether pressure and velocity
    have actually agreed with each other.
    """

    iteration: int
    u: float
    v: float
    pressure: float
    continuity: float
    k: float = 0.0
    omega: float = 0.0
    lift_coefficient: float = 0.0
    drag_coefficient: float = 0.0

    @property
    def worst(self) -> float:
        return max(self.u, self.v, self.continuity, self.k, self.omega)

    def has_converged(self, tolerance: float) -> bool:
        return self.worst < tolerance

    def __str__(self) -> str:
        return (
            f"{self.iteration:6d}  Ux {self.u:.3e}  Uy {self.v:.3e}  "
            f"p {self.pressure:.3e}  mass {self.continuity:.3e}  "
            f"k {self.k:.3e}  w {self.omega:.3e}  "
            f"Cl {self.lift_coefficient:+.4f}  Cd {self.drag_coefficient:+.5f}"
        )


@dataclass
class History:
    """Residual and force history, for the live plots and the convergence check."""

    entries: list[Residuals] = field(default_factory=list)

    def append(self, residuals: Residuals) -> None:
        self.entries.append(residuals)

    def __len__(self) -> int:
        return len(self.entries)

    def series(self, name: str) -> np.ndarray:
        return np.array([getattr(entry, name) for entry in self.entries])

    @property
    def iterations(self) -> np.ndarray:
        return self.series("iteration")

    def forces_are_steady(self, window: int = 50, tolerance: float = 1e-4) -> bool:
        """Whether the force coefficients have stopped moving.

        Residuals alone can mislead. A case can sit at 1e-4 for hundreds of
        iterations with the lift still drifting, and it can also stall at a
        residual floor set by the deferred correction while the forces are
        perfectly settled. Checking the quantity actually wanted catches both.
        """
        if len(self.entries) < 2 * window:
            return False
        recent = self.series("lift_coefficient")[-window:]
        earlier = self.series("lift_coefficient")[-2 * window : -window]
        scale = max(abs(recent.mean()), 1e-3)
        return abs(recent.mean() - earlier.mean()) / scale < tolerance
