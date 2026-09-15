"""Boundary conditions on the boundaries a structured mesh has.

**The wall** is straightforward: no slip, no flux, and the turbulence conditions
that come with integrating k-omega SST to the wall.

**Part of it may be a symmetry plane instead**, which is what a flat-plate case
needs ahead of its leading edge and what ``Boundaries.wall_mask`` selects.
Symmetry is not a fourth kind of condition, it is the wall's condition with the
tangential target changed: the mirrored face value

    u_face = u_cell - (u_cell . n) n

carried by the interior viscosity through the same diffusive coupling. The
tangential flux is then identically zero -- no shear -- and the normal part is
``-mu g (u_cell . n)``, which drives the face-normal velocity to zero. Both of
those are what a symmetry plane means, and neither needs new machinery.

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
#: Public, because ``fields.State.uniform`` seeds ``omega`` with the same
#: asymptote and used to carry its own copy of both numbers. One definition,
#: in the module that owns the boundary condition.
OMEGA_WALL_FACTOR = 6.0
BETA_1 = 0.075
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

    #: Which ``j = 0`` faces are solid wall, ``True`` where they are. ``None``
    #: means all of them, which is what a mesh wrapped around a body has and is
    #: the only thing an O-grid case ever passes.
    wall_mask: np.ndarray | None = None

    #: Reference length, for the bound circulation. Only the far-field vortex
    #: correction uses it, and only through ``Gamma = Cl U c / 2``.
    reference_length: float = 1.0
    #: Bound circulation, positive for positive lift, updated once per outer
    #: iteration from the lift the solution currently carries. Zero disables the
    #: far-field vortex correction entirely, which is what a non-lifting case
    #: settles at on its own and what every case starts from.
    circulation: float = 0.0
    #: Where the bound vortex is placed. Defaults to the centroid of the wall
    #: face centres. The quarter chord is the textbook choice; at forty chords
    #: the difference between the two is ``O(c/R)`` of a term that is itself
    #: ``O(c/R)``, so it is second order in exactly the quantity being corrected.
    vortex_centre: np.ndarray | None = None

    def __post_init__(self):
        if self.vortex_centre is None:
            self.vortex_centre = self.faces.wall.centre.mean(axis=0)

    @property
    def solid_wall(self) -> np.ndarray:
        """The mask as an array, ``True`` everywhere when there is none.

        Callers that only need to *weight* something by it can use this without
        branching, and get an all-``True`` array that leaves their arithmetic
        exactly where it was.
        """
        if self.wall_mask is None:
            return np.ones(self.faces.shape[0], dtype=bool)
        return self.wall_mask

    def _mirrored_velocity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """The first cell's velocity with its wall-normal component removed."""
        normal = self.faces.wall.normal
        velocity = np.stack((u[:, 0], v[:, 0]), axis=-1)
        return velocity - np.sum(velocity * normal, axis=-1)[:, None] * normal

    def set_circulation(self, lift_coefficient: float) -> None:
        """``Gamma = Cl U c / 2``, from Kutta-Joukowski.

        ``L = rho U Gamma`` and ``Cl = L / (rho U^2 c / 2)`` together give
        ``Gamma = Cl U c / 2``. Called once per outer iteration with the lift the
        solution currently has, so the correction is lagged by one iteration and
        exact at the fixed point. A cold start has ``Cl = 0`` and the correction
        switches itself on as the circulation develops.
        """
        self.circulation = 0.5 * lift_coefficient * self.freestream.velocity * (
            self.reference_length
        )

    def far_vortex_velocity(self, centres: np.ndarray | None = None) -> np.ndarray:
        """Velocity the bound vortex induces at each far-field face centre.

        **The sign is derived, not copied, because the obvious source has it the
        other way round.** A point vortex of counter-clockwise strength ``G`` at
        ``r_0`` induces ``(G / 2 pi) (-(y - y_0), (x - x_0)) / |r - r_0|^2``. Put
        ``G = +Gamma`` with ``Gamma = Cl U c / 2`` -- which is how the physics
        audit writes it -- and evaluate above a lifting body: the induced velocity
        comes out along ``-x``, so the flow is *slower* over the suction side.
        That is the wrong way round, and two independent checks say so:

            above the body   u must be > 0   (faster where the pressure is lower)
            ahead of it      v must be > 0   (upwash)
            behind it        v must be < 0   (downwash)

        All three fail together with ``G = +Gamma`` and hold together with
        ``G = -Gamma``. The bound vortex of a body lifting along ``+y`` in a
        freestream along ``+x`` is *clockwise*. Written out with ``Gamma``
        positive for positive lift, that is

            u_induced = (Gamma / 2 pi) ( (y - y_0), -(x - x_0) ) / |r - r_0|^2

        which is what this returns.
        """
        if centres is None:
            centres = self.faces.far_field.centre
        if self.circulation == 0.0:
            return np.zeros_like(centres)

        offset = centres - self.vortex_centre
        radius_squared = np.sum(offset * offset, axis=-1)
        rotated = np.stack((offset[:, 1], -offset[:, 0]), axis=-1)
        return (self.circulation / (2.0 * np.pi)) * rotated / np.maximum(
            radius_squared, 1e-300
        )[:, None]

    # ------------------------------------------------------------------
    # Wall
    # ------------------------------------------------------------------

    def wall_velocity(
        self, u: np.ndarray | None = None, v: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """No slip on a solid face; the mirrored velocity on a symmetry one.

        ``u`` and ``v`` are needed only where :attr:`wall_mask` says a face is
        not solid, because that is the only case whose answer depends on the
        flow. A wall-only boundary returns zeros without looking at them, which
        is why they are optional and why nothing on the O-grid path moved.
        """
        zero = np.zeros(self.faces.shape[0])
        if self.wall_mask is None:
            return zero, zero.copy()
        if u is None or v is None:
            raise ValueError(
                "a boundary with a symmetry plane in it needs the velocity "
                "field: the face value there is the flow's own tangential part"
            )
        mirrored = self._mirrored_velocity(u, v)
        return (
            np.where(self.wall_mask, 0.0, mirrored[:, 0]),
            np.where(self.wall_mask, 0.0, mirrored[:, 1]),
        )

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
        """``tau_w = rho u_tau^2``, the traction the surface exerts on the flow.

        Zero on a symmetry face, which is what symmetry *means*: the normal
        derivative of the tangential velocity vanishes there, so there is no
        shear. A friction velocity computed from the tangential cell speed would
        be perfectly finite and entirely fictitious, and it would be integrated
        into a drag, so it is masked out rather than left to be.
        """
        shear = self.fluid.density * self.friction_velocity(u, v) ** 2
        if self.wall_mask is None:
            return shear
        return np.where(self.wall_mask, shear, 0.0)

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

    def wall_viscosity(
        self, u: np.ndarray, v: np.ndarray, interior: np.ndarray | None = None
    ) -> np.ndarray:
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
        blended = np.maximum(shear * distance / speed, self.fluid.viscosity)
        if self.wall_mask is None:
            return blended
        # A symmetry face takes the interior viscosity instead. The wall function
        # has nothing to say there -- no boundary layer, no log law -- and the
        # flux that face does carry is the normal-velocity penalty, which is an
        # ordinary viscous term and wants the ordinary viscosity.
        interior = self.fluid.viscosity if interior is None else interior
        return np.where(self.wall_mask, blended, interior)

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
            OMEGA_WALL_FACTOR
            * self.fluid.kinematic_viscosity
            / (BETA_1 * distance**2)
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
        """Freestream plus the bound vortex where flow enters; extrapolated where
        it leaves.

        **Why the freestream alone is not good enough.** A lifting body carries a
        bound circulation, and outside the viscous region the flow it sets up is

            u = U_inf + Gamma/(2 pi r) e_theta + O(r^-2),

        so imposing ``u = U_inf`` on the boundary imposes an error of
        ``-u_vortex`` there. Its size is ``Cl c / (4 pi R)`` -- for ``Cl = 0.75``
        at forty chords, 1.5 parts in a thousand of the freestream, which sounds
        negligible. It is not, because it is imposed as a velocity over an arc of
        length ``2 pi R``, so the spurious volume flux it removes is ``O(Gamma)``:
        the whole circulation, however far away the boundary is put.

        The error is therefore ``O(1/R)`` in every integrated coefficient, and
        that is the prediction to test rather than assume -- halving ``R`` should
        double it. Measured on this solver, the NACA 2412 at 5 degrees between 20
        and 40 chords: see ``docs/audit-response-plan.md``.

        With the vortex superposed the leading term is gone and the error falls to
        ``O(1/R^2)``. This is the standard treatment since Thomas and Salas
        (1986); Vassberg and Jameson (2010) give the grid-convergence evidence for
        how far out one has to go without it, which is several hundred chords for
        0.1% in ``Cl``.

        Only inflow faces get it. Outflow faces extrapolate the interior velocity
        and always did; imposing anything there would over-specify the problem.
        """
        entering = self.inflow_mask(far_flux)
        stream = self.freestream.vector
        induced = self.far_vortex_velocity()
        return (
            np.where(entering, stream[0] + induced[:, 0], u[:, -1]),
            np.where(entering, stream[1] + induced[:, 1], v[:, -1]),
        )

    def far_pressure(self, p: np.ndarray, far_flux: np.ndarray) -> np.ndarray:
        """The vortex's Bernoulli pressure where flow leaves, extrapolated where
        it enters.

        Fixing the pressure on the outflow is what makes the pressure equation
        solvable at all: with a pure Neumann condition everywhere the pressure
        would be determined only up to a constant and the matrix would be
        singular.

        The value fixed there used to be zero, which is right only if the
        far-field velocity is the freestream. Once the bound vortex is superposed
        it is not, and the matching pressure follows from Bernoulli along a
        streamline from infinity:

            p_far = (rho / 2) ( U_inf^2 - |u_far|^2 ).

        Imposing the corrected velocity while leaving the pressure pinned at zero
        would assert a boundary state that does not satisfy the outer flow's own
        momentum equation, which is a worse inconsistency than the one being
        removed. With ``Gamma = 0`` this reduces to zero exactly, so a non-lifting
        case is untouched.
        """
        stream = self.freestream.vector
        far = stream + self.far_vortex_velocity()
        bernoulli = 0.5 * self.fluid.density * (
            self.freestream.velocity**2 - np.sum(far * far, axis=-1)
        )
        return np.where(self.inflow_mask(far_flux), p[:, -1], bernoulli)

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
    # The i ends
    # ------------------------------------------------------------------
    #
    # An open mesh has two more boundaries, and they take the far field's
    # condition: each face decides for itself on the sign of u . n. An inlet is
    # that condition on a boundary that happens to be inflow everywhere, and an
    # outlet one that happens to be outflow, so nothing here is told which end is
    # which -- a flat plate's left end selects inflow for itself and its right
    # end outflow.
    #
    # Every method returns a ``(start, end)`` pair, and ``(None, None)`` on a
    # mesh whose i direction wraps, which the operators read as "no such
    # boundary". The one thing that differs from the far field is the sign of the
    # stored flux: ``flux_i`` is signed towards increasing i, which is outward at
    # the high end and *inward* at the low one. It is turned into an outward flux
    # here, once, so the condition itself never sees the difference.

    def i_outward_flux(self, flux_i: np.ndarray):
        """Mass flux leaving the domain through each i end, per face."""
        if self.faces.periodic_i:
            return None, None
        return -flux_i[0], flux_i[-1]

    def i_inflow_mask(self, flux_i: np.ndarray):
        """True on i-end faces where fluid is entering."""
        start, end = self.i_outward_flux(flux_i)
        if start is None:
            return None, None
        return self.inflow_mask(start), self.inflow_mask(end)

    def _i_end_faces(self):
        return (self.faces.i_start, 0), (self.faces.i_end, -1)

    def i_velocity(self, u: np.ndarray, v: np.ndarray, flux_i: np.ndarray):
        """``((u_start, u_end), (v_start, v_end))``: freestream in, interior out.

        The bound vortex is superposed on inflow faces exactly as it is on the far
        field, and for the same reason; it is zero on a non-lifting case.
        """
        if self.faces.periodic_i:
            return (None, None), (None, None)
        stream = self.freestream.vector
        entering = self.i_inflow_mask(flux_i)
        values_u, values_v = [], []
        for (boundary, index), inflow in zip(self._i_end_faces(), entering):
            induced = self.far_vortex_velocity(boundary.centre)
            values_u.append(np.where(inflow, stream[0] + induced[:, 0], u[index]))
            values_v.append(np.where(inflow, stream[1] + induced[:, 1], v[index]))
        return tuple(values_u), tuple(values_v)

    def i_pressure(self, p: np.ndarray, flux_i: np.ndarray):
        """Bernoulli's pressure where flow leaves, extrapolated where it enters."""
        if self.faces.periodic_i:
            return None, None
        stream = self.freestream.vector
        entering = self.i_inflow_mask(flux_i)
        values = []
        for (boundary, index), inflow in zip(self._i_end_faces(), entering):
            outer = stream + self.far_vortex_velocity(boundary.centre)
            bernoulli = 0.5 * self.fluid.density * (
                self.freestream.velocity**2 - np.sum(outer * outer, axis=-1)
            )
            values.append(np.where(inflow, p[index], bernoulli))
        return tuple(values)

    def i_pressure_is_fixed(self, flux_i: np.ndarray):
        """i-end faces where the pressure correction is pinned to zero."""
        start, end = self.i_inflow_mask(flux_i)
        if start is None:
            return None, None
        return ~start, ~end

    def i_turbulence(self, k: np.ndarray, omega: np.ndarray, flux_i: np.ndarray):
        """``((k_start, k_end), (omega_start, omega_end))``."""
        if self.faces.periodic_i:
            return (None, None), (None, None)
        entering = self.i_inflow_mask(flux_i)
        k_inflow = self.freestream.turbulent_kinetic_energy()
        omega_inflow = self.freestream.specific_dissipation(self.fluid)
        values_k, values_omega = [], []
        for (_, index), inflow in zip(self._i_end_faces(), entering):
            values_k.append(np.where(inflow, k_inflow, k[index]))
            values_omega.append(np.where(inflow, omega_inflow, omega[index]))
        return tuple(values_k), tuple(values_omega)

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

    def far_flux_is_solvable(
        self, far_flux: np.ndarray, flux_i: np.ndarray | None = None
    ) -> bool:
        """Whether the pressure equation needs its source projecting to zero mean.

        It does only when *no* far-field face holds the pressure -- that is, when
        the boundary is inflow everywhere, so the pressure correction sees a pure
        Neumann problem with a singular matrix and a compatibility condition.

        This replaces ``enforce_global_mass_balance``, which rescaled every
        outflow face by ``M_in / M_out`` on the stated grounds that "the
        pressure-correction equation is a discrete Poisson problem, and it has a
        solution only if its source integrates to zero". That is the compatibility
        condition of a *pure Neumann* problem, and this one is not pure Neumann:
        ``PressureVelocityCoupling._far_field_coupling`` adds a Dirichlet coupling
        to ``p' = 0`` on the diagonal of every cell behind an outflow face, and a
        matrix with any Dirichlet row is non-singular. The stated reason did not
        apply.

        What the rescaling did instead was impose global conservation on an
        extrapolated outflow, and it did so **at convergence**. Nothing forces the
        raw extrapolated flux to balance the inflow, so the factor settles at some
        ``c != 1`` and stays: measured on the converged Re 40 cylinder,
        ``0.999747534``, rescaling every outflow face by -0.0252% for ever. By
        this project's own standard -- anything active at convergence is part of
        the model, whatever it is labelled -- that made it an undocumented
        boundary condition.

        It also declined to act in exactly the situation it was written for. Its
        guard returned the flux untouched when either total was non-positive,
        which is the start-up transient or a boundary that has gone almost
        entirely inflow -- and it said nothing when it did. The factor was
        unbounded as the outflow went to zero.
        """
        if bool(np.any(self.far_pressure_is_fixed(far_flux))):
            return True
        # On an open mesh the i ends can hold the pressure too, and on a flat
        # plate they are where it is held: the top boundary carries almost no
        # flux, and the outlet carries all of it.
        if flux_i is None:
            return False
        return any(
            fixed is not None and bool(np.any(fixed))
            for fixed in self.i_pressure_is_fixed(flux_i)
        )
