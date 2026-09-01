"""Boundary conditions on the two boundaries an O-grid has.

**The wall** is straightforward: no slip, no flux, and the turbulence conditions
that come with integrating k-omega SST to the wall.

**The far field** is not, and the treatment here matters more than it looks. The
outer boundary is a single closed circle, so the same boundary carries the
oncoming flow, the wake leaving, and everything in between. Nominating parts of
it "inlet" and "outlet" in advance would be a guess about where the flow goes.

Instead each face decides for itself, on the sign of ``u . n``:

* **Inflow** faces fix the velocity to the freestream and let the pressure float.
  The flow arriving is known; the pressure it arrives at is not.
* **Outflow** faces fix the pressure and let the velocity float. What leaves is
  determined by the interior; imposing a velocity there would over-specify the
  problem and reflect disturbances back inside.

This is the standard characteristic argument -- information travels in along
incoming characteristics and out along outgoing ones, and a boundary condition
may only be imposed where information enters. It also means the far field needs
no user input beyond the freestream itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluidsolver.solver.faces import FaceGeometry
from fluidsolver.solver.fluid import Fluid, Freestream

# Wilcox's exact near-wall solution, omega -> 6 nu / (beta1 y^2) as y -> 0.
#
# Menter quotes this with a factor of ten, as 60 nu / (beta1 dy^2), and that form
# is widely copied -- but it is for codes that need a value *on the wall face*,
# where omega is formally infinite, and the ten is deliberate over-specification
# to force the right asymptotic behaviour. Here the value is prescribed in the
# first cell instead, at a point where the asymptote is simply valid, so the
# factor does not belong: including it puts omega ten times too high in the
# stiffest cell of the mesh.
_OMEGA_WALL_FACTOR = 6.0
_BETA_1 = 0.075


@dataclass
class Boundaries:
    """Boundary values for every transported field, given the current state.

    Held together in one place because they are all coupled to the same
    inflow/outflow split, which changes as the solution develops: a far-field
    face can start as outflow and become inflow while the wake settles.
    """

    faces: FaceGeometry
    fluid: Fluid
    freestream: Freestream

    # ------------------------------------------------------------------
    # Wall
    # ------------------------------------------------------------------

    def wall_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """No slip: both components zero on the surface."""
        zero = np.zeros(self.faces.shape[0])
        return zero, zero.copy()

    def wall_turbulence(self) -> tuple[np.ndarray, np.ndarray]:
        """``k = 0`` and Menter's ``omega_w = 60 nu / (beta1 d1^2)``.

        Turbulence cannot survive at a solid surface, so ``k`` vanishes there.
        ``omega`` does the opposite -- it is singular at the wall, growing like
        ``1/y^2`` -- so it cannot be imposed as a value *on* the surface at all.
        What is imposed instead is the asymptotic solution evaluated at the first
        cell centre, which is why this condition is only valid when that cell
        sits at ``y+`` of order one, and why the mesher sizes it that way.

        ``d1`` is the true perpendicular distance to the wall, not the marching
        distance, and it enters as an inverse square: a 10% error here is a 20%
        error in the wall value of omega.
        """
        distance = self.faces.wall.wall_normal_distance
        omega = (
            _OMEGA_WALL_FACTOR
            * self.fluid.kinematic_viscosity
            / (_BETA_1 * distance**2)
        )
        return np.zeros_like(omega), omega

    # ------------------------------------------------------------------
    # Far field
    # ------------------------------------------------------------------

    def inflow_mask(self, far_flux: np.ndarray) -> np.ndarray:
        """True on faces where fluid is entering the domain."""
        return far_flux < 0.0

    def far_velocity(
        self, u: np.ndarray, v: np.ndarray, far_flux: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Freestream where flow enters, extrapolated where it leaves."""
        entering = self.inflow_mask(far_flux)
        stream = self.freestream.vector
        return (
            np.where(entering, stream[0], u[:, -1]),
            np.where(entering, stream[1], v[:, -1]),
        )

    def far_pressure(self, p: np.ndarray, far_flux: np.ndarray) -> np.ndarray:
        """Zero where flow leaves, extrapolated where it enters.

        Fixing the pressure on the outflow is also what makes the pressure
        equation solvable at all: with a pure Neumann condition everywhere the
        pressure would be determined only up to a constant, and the matrix would
        be singular.
        """
        return np.where(self.inflow_mask(far_flux), p[:, -1], 0.0)

    def far_pressure_is_fixed(self, far_flux: np.ndarray) -> np.ndarray:
        """Faces where the pressure correction is pinned to zero."""
        return ~self.inflow_mask(far_flux)

    def far_turbulence(
        self, k: np.ndarray, omega: np.ndarray, far_flux: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Freestream turbulence entering, interior values leaving."""
        entering = self.inflow_mask(far_flux)
        return (
            np.where(entering, self.freestream.turbulent_kinetic_energy(), k[:, -1]),
            np.where(
                entering, self.freestream.specific_dissipation(self.fluid), omega[:, -1]
            ),
        )

    # ------------------------------------------------------------------
    # Fluxes
    # ------------------------------------------------------------------

    def far_flux_from_freestream(self) -> np.ndarray:
        """Mass flux through the far field for a uniform freestream.

        Used to initialise, before there is a velocity field to take the sign of.
        """
        return self.fluid.density * np.sum(
            self.faces.far_field.area * self.freestream.vector, axis=-1
        )

    def enforce_global_mass_balance(self, far_flux: np.ndarray) -> np.ndarray:
        """Scale the outflow so that what leaves equals what enters.

        The pressure-correction equation is a discrete Poisson problem, and it has
        a solution only if its source integrates to zero -- that is, only if the
        boundary fluxes balance. Extrapolating velocity onto the outflow gives no
        guarantee of that, and even a small imbalance makes the pressure solve
        drift or fail. Rescaling the outflow to match the inflow restores
        solvability, and the correction vanishes as the solution converges.
        """
        entering = far_flux < 0.0
        inflow = -far_flux[entering].sum()
        outflow = far_flux[~entering].sum()

        if outflow <= 0.0 or inflow <= 0.0:
            return far_flux

        balanced = far_flux.copy()
        balanced[~entering] *= inflow / outflow
        return balanced
