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
_BETA_STAR = 0.09

# von Karman's constant and the additive constant of the smooth-wall log law,
# U+ = (1/kappa) ln(y+) + C. The pair must be quoted together -- 0.41 with 5.2 is
# the CFX pairing and the one Esch and Menter's formulation assumes; 0.4187 with
# 5.45 is Fluent's. Mixing one from each shifts the log layer by a percent or so
# for no reason.
_KAPPA = 0.41
_LOG_LAW_CONSTANT = 5.2

# Below y+ of about one the logarithmic branch is not merely inaccurate, it is
# singular: ln(y+) passes through zero and then negative, and the branch changes
# sign. It contributes nothing there anyway -- the viscous branch is larger by
# orders of magnitude and the fourth-power blend ignores it -- so it is simply
# held at a value where it stays finite and positive.
_Y_PLUS_FLOOR = 1.0

# Fixed-point passes for the friction velocity. y+ enters only through a
# logarithm, so the map contracts hard and five is generous; the cost is five
# cheap array operations over one row of cells, once per outer iteration.
_FRICTION_VELOCITY_PASSES = 5

# A stagnation point has no tangential velocity and therefore no wall shear and
# no effective wall viscosity to compute. The floor keeps the division finite;
# the shear there is genuinely zero and the molecular floor on the viscosity is
# what the cell ends up with, which is correct.
_SPEED_FLOOR = 1.0e-10


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

    def wall_tangential_velocity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Speed of the first cell centre along the surface."""
        normal = self.faces.wall.normal
        velocity = np.stack((u[:, 0], v[:, 0]), axis=-1)
        tangential = velocity - np.sum(velocity * normal, axis=-1)[:, None] * normal
        return np.linalg.norm(tangential, axis=-1)

    def friction_velocity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``u_tau``, blended between the viscous and logarithmic branches.

        Esch and Menter's automatic wall treatment, equations (17) and (18):

            u_tau_vis = U1 / y+          u_tau_log = U1 / ((1/kappa) ln y+ + C)

            u_tau = (u_tau_vis^4 + u_tau_log^4)^(1/4)

        The viscous branch is written here as ``sqrt(nu U1 / y1)`` rather than as
        ``U1 / y+``. They are the same statement: substituting ``y+ = u_tau y1 /
        nu`` into the first and solving for ``u_tau`` gives the second, and the
        explicit form avoids an equation that defines ``u_tau`` in terms of
        itself. The logarithmic branch has no such escape, so it takes ``y+``
        from the previous outer iteration -- which is what makes this a lagged
        boundary condition rather than a nonlinear solve in every wall cell.

        The fourth-power blend is what makes the whole thing work. Each branch is
        only valid at one end, and a fourth power is sharp enough that whichever
        is larger dominates almost completely while still being differentiable
        through the buffer layer, where neither is right and no formulation can
        be. That is the honest position: the buffer layer is interpolated, not
        resolved, and the blend keeps the interpolation smooth and bounded.
        """
        distance = self.faces.wall.wall_normal_distance
        speed = self.wall_tangential_velocity(u, v)
        viscosity = self.fluid.kinematic_viscosity

        viscous = np.sqrt(viscosity * speed / distance)

        # y+ has to come from the *blended* friction velocity, not from the
        # viscous branch, and the difference is not a refinement.
        #
        # Seeding y+ from the viscous branch alone underestimates u_tau wherever
        # the logarithmic branch matters. A smaller y+ means a smaller ln(y+),
        # a smaller denominator in the logarithmic branch, and therefore a
        # *larger* u_tau -- a systematic overestimate of the wall shear in
        # exactly the regime this treatment exists for. Measured on a NACA 2412
        # with a first cell at y+ ~ 5, that put friction drag 32% above the
        # y+ ~ 1 answer, having started 18% below it.
        #
        # So iterate instead. The map contracts quickly because y+ enters only
        # through a logarithm; five passes take it well inside the convergence
        # of the outer SIMPLE loop, and starting from the viscous branch means
        # the fine-mesh limit is exact on the first pass and the iteration is a
        # no-op there.
        friction = viscous
        for _ in range(_FRICTION_VELOCITY_PASSES):
            y_plus = np.maximum(friction * distance / viscosity, _Y_PLUS_FLOOR)
            logarithmic = speed / (np.log(y_plus) / _KAPPA + _LOG_LAW_CONSTANT)
            friction = (viscous**4 + np.maximum(logarithmic, 0.0) ** 4) ** 0.25
        return friction

    def wall_shear(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``tau_w = rho u_tau^2``, the traction the surface exerts on the flow."""
        return self.fluid.density * self.friction_velocity(u, v) ** 2

    def wall_velocity_gradient(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``dU/dy`` at the first cell centre, from the two-layer profile.

        The discrete strain rate in a wall cell is built from ``U1 / y1``, which
        is the *average* gradient between the wall and the cell centre. In the
        viscous sublayer the profile is linear and the two coincide; in the log
        layer the profile is concave and they do not. Their ratio is ``kappa U+``,
        which at ``y+`` of 30 is 5.5 -- and since production goes as the square,
        the resolved value overstates it by a factor of thirty.

        Left uncorrected that is not a small error. Measured on a NACA 2412 with
        a first cell at ``y+`` 30, momentum and continuity converged by two
        orders while ``k`` climbed from 4.4e-03 to 7.0e-03 and stuck: it had run
        into the solution limiter, which was clipping 165 cells on 554 of 600
        iterations. Everything else in the run was healthy.

            dU/dy = min( u_tau^2 / nu ,  u_tau / (kappa y1) )

        The minimum picks the viscous branch below ``y+`` of about 2.4 and the
        logarithmic one above, which is where they cross. Nothing is invented by
        this: in the sublayer ``u_tau^2 / nu`` *is* ``U1 / y1``, so the resolved
        value is recovered identically and a wall-resolved mesh sees no change at
        all.
        """
        distance = self.faces.wall.wall_normal_distance
        friction = self.friction_velocity(u, v)
        viscous = friction**2 / self.fluid.kinematic_viscosity
        logarithmic = friction / (_KAPPA * distance)
        return np.minimum(viscous, logarithmic)

    def wall_viscosity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Effective viscosity on the wall face that reproduces ``tau_w``.

        The momentum equation imposes no-slip through a diffusive flux
        ``mu_eff (U1 - 0) / y1``, which assumes the profile through the first
        cell is linear. That is the truth inside the viscous sublayer and an
        *under*-estimate anywhere above it: the log law bends below the linear
        profile, so ``U+ < y+``, and the ratio of the true shear to the linear
        one is ``y+ / U+``, which exceeds one. At ``y+`` of 15 that is a factor
        of 1.27, and the measured effect is friction drag falling as the mesh
        coarsens -- 0.00746 at ``y+ ~ 1`` against 0.00608 at ``y+ ~ 5`` -- when
        the physics says it should hold steady.

        Rather than special-casing the momentum assembly, the same flux is kept
        and the viscosity carrying it is replaced by whatever value delivers the
        blended shear:

            mu_wall = tau_w y1 / U1

        In the fine-mesh limit ``u_tau -> sqrt(nu U1 / y1)``, so ``tau_w ->
        mu U1 / y1`` and ``mu_wall -> mu``: the low-Reynolds treatment is
        recovered exactly, not approximately. Floored at the molecular value,
        since a wall cannot be less viscous than the fluid.
        """
        distance = self.faces.wall.wall_normal_distance
        speed = np.maximum(self.wall_tangential_velocity(u, v), _SPEED_FLOOR)
        shear = self.wall_shear(u, v)
        return np.maximum(shear * distance / speed, self.fluid.viscosity)

    def wall_turbulence(
        self, u: np.ndarray | None = None, v: np.ndarray | None = None
    ) -> tuple[None, np.ndarray]:
        """Zero flux for ``k``, and a blended wall value for ``omega``.

        **k takes a zero-flux condition, not k = 0.** Esch and Menter are
        explicit that this is what is correct in *both* the low-Reynolds and the
        logarithmic limit. Setting ``k = 0`` on the face is right only in the
        first of those: once the near-wall cell sits in the log layer its centre
        carries a substantial turbulent kinetic energy, and driving it to zero
        across that cell removes energy the flow actually has.

        **omega** is blended between the two analytic near-wall solutions,
        equations (15) and (16):

            omega_vis = 6 nu / (beta1 y1^2)      omega_log = u_tau / (0.3 kappa y1)

            omega_wall = sqrt(omega_vis^2 + omega_log^2)

        The blend needs no switch because the two branches separate themselves:
        the viscous one goes as ``1/y^2`` and the logarithmic as ``1/y``, so on a
        fine mesh the first dominates by orders of magnitude and on a coarse one
        the second does. What was there before was ``omega_vis`` alone, which is
        why a first cell at y+ 30 did not merely lose accuracy but diverged --
        the asymptote was being asserted a factor of thirty outside its range.

        ``u`` and ``v`` may be omitted, in which case only the viscous branch is
        available; that is the right answer before there is a velocity field to
        take a friction velocity from.
        """
        distance = self.faces.wall.wall_normal_distance
        viscous = (
            _OMEGA_WALL_FACTOR
            * self.fluid.kinematic_viscosity
            / (_BETA_1 * distance**2)
        )
        if u is None or v is None:
            return None, viscous

        logarithmic = self.friction_velocity(u, v) / (
            np.sqrt(_BETA_STAR) * _KAPPA * distance
        )
        return None, np.sqrt(viscous**2 + logarithmic**2)

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
