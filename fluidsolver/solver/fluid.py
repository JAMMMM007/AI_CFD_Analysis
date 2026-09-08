"""Fluid properties and the freestream condition.

Density is deliberately carried as a *field* rather than a scalar even though the
solver is incompressible. Nothing here needs that today; it is the seam along
which a compressible path is added later, when density stops being uniform and
the pressure equation picks up a ``d(rho)/dp`` term. Keeping the shape right from
the start costs nothing and avoids a rewrite of every flux expression.

See ``docs/compressible.md`` for what else would have to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Above roughly this Mach number the incompressible assumption starts to cost
# more than a percent in density, and the user should be told.
_INCOMPRESSIBLE_MACH_LIMIT = 0.3

# Speed of sound in air at 15 C, used only to warn about the Mach number.
_AIR_SPEED_OF_SOUND = 340.3


@dataclass(frozen=True)
class Fluid:
    """A constant-property Newtonian fluid.

    Attributes
    ----------
    density
        Mass density, kg/m^3.
    viscosity
        Dynamic viscosity mu, Pa s. The solver mostly works in terms of the
        kinematic viscosity, exposed as :attr:`kinematic_viscosity`.
    name
        Label for the UI.
    """

    density: float
    viscosity: float
    name: str = "fluid"

    def __post_init__(self):
        if self.density <= 0.0:
            raise ValueError(f"density must be positive, got {self.density}")
        if self.viscosity <= 0.0:
            raise ValueError(f"viscosity must be positive, got {self.viscosity}")

    @property
    def kinematic_viscosity(self) -> float:
        """nu = mu / rho."""
        return self.viscosity / self.density

    def reynolds(self, velocity: float, length: float) -> float:
        return self.density * velocity * length / self.viscosity

    def density_field(self, shape: tuple[int, ...]) -> np.ndarray:
        """Density as an array over the mesh.

        Constant for now. This is the compressible extension point: a
        variable-density model returns a field varying with the local state, and
        every flux in the solver already multiplies by it.
        """
        return np.full(shape, self.density)


# Standard fluids, so the setup page does not start from a blank form.
AIR_15C = Fluid(density=1.225, viscosity=1.81e-5, name="air, 15 C")
AIR_20C = Fluid(density=1.204, viscosity=1.82e-5, name="air, 20 C")
WATER_20C = Fluid(density=998.2, viscosity=1.002e-3, name="water, 20 C")

PRESETS = (AIR_15C, AIR_20C, WATER_20C)


@dataclass(frozen=True)
class Freestream:
    """The oncoming flow.

    Attributes
    ----------
    velocity
        Speed far from the body, m/s.
    angle_of_attack_deg
        Incidence, degrees. Applied by rotating the *body* rather than tilting
        the freestream, so the circular far-field boundary stays aligned with the
        mesh and plots stay upright.
    turbulence_intensity
        ``u'/U`` in the freestream, as a fraction. Sets the inlet ``k``.
    eddy_viscosity_ratio
        Freestream ``mu_t / mu``. Together with the intensity this fixes the
        inlet ``omega``.

    **The defaults are the NASA TMR's ambient values, and the ones they replaced
    were two orders outside them.** The Turbulence Modeling Resource specifies,
    for external aerodynamics:

        U/L < omega_far < 10 U/L
        1e-5 U^2/Re_L < k_far < 0.1 U^2/Re_L
        1e-5 < mu_t_far / mu < 1e-2

    The old defaults, 0.1% intensity with a viscosity ratio of 1, put ``k`` 30
    times and ``mu_t/mu`` a hundred times above the top of those bands. Measured
    on the primary case -- 30 m/s, unit chord, air at 15 C, so ``Re_L = 2.03e6``:

        I 1e-3, ratio 1.0     k 1.3500e-03  OUT   omega 91.4  in   mu_t/mu 1.0    OUT
        I 5e-5, ratio 1e-3    k 3.3750e-06  in    omega 228   in   mu_t/mu 1e-3   in

    The second pair is inside all three with margin rather than at a band edge,
    and is what these defaults are.

    The physics audit's suggested replacement -- ``k_amb = 1e-6 U^2`` with
    ``omega_amb = 5 U/L`` -- was measured against the same bands and is outside
    two of them: ``k`` comes to 9.0e-04, twenty times the top of the ``k`` band,
    and ``mu_t/mu`` to 4.06e-01 where the audit states 7.4e-3. It would not have
    fixed what it diagnoses.

    **A fixed default cannot satisfy these bands at every operating point, and
    that is a property of the parameterisation rather than of the numbers.** The
    ``k`` band scales as ``U mu / (rho L)`` and the ``omega`` band as ``U / L``,
    while ``k = 3/2 (I U)^2`` and ``omega = rho k / (mu r)`` both scale as ``U^2``
    at fixed ``I`` and ``r``. So any pair chosen here is right at one Reynolds
    number and drifts from the bands away from it. Rather than pretend otherwise,
    :func:`fluidsolver.solver.health.assess` measures the actual values against
    the actual bands for the case being run and says so when they fall outside.
    """

    velocity: float
    angle_of_attack_deg: float = 0.0
    turbulence_intensity: float = 5.0e-5
    eddy_viscosity_ratio: float = 1.0e-3

    def __post_init__(self):
        if self.velocity <= 0.0:
            raise ValueError(f"velocity must be positive, got {self.velocity}")
        if not 0.0 < self.turbulence_intensity < 1.0:
            raise ValueError(
                f"turbulence_intensity is a fraction, not a percentage; "
                f"got {self.turbulence_intensity}"
            )
        if self.eddy_viscosity_ratio <= 0.0:
            raise ValueError(
                f"eddy_viscosity_ratio must be positive, got {self.eddy_viscosity_ratio}"
            )

    @property
    def direction(self) -> np.ndarray:
        """Unit vector along the freestream, in mesh coordinates.

        Always ``(1, 0)``: incidence lives in the body's orientation, not here.
        """
        return np.array([1.0, 0.0])

    @property
    def vector(self) -> np.ndarray:
        return self.velocity * self.direction

    def dynamic_pressure(self, fluid: Fluid) -> float:
        """``q = rho U^2 / 2``, the normaliser for Cp, Cl, Cd and Cm."""
        return 0.5 * fluid.density * self.velocity**2

    def turbulent_kinetic_energy(self) -> float:
        """``k = 3/2 (I U)^2``, assuming isotropic freestream turbulence."""
        return 1.5 * (self.turbulence_intensity * self.velocity) ** 2

    def specific_dissipation(self, fluid: Fluid) -> float:
        """``omega = rho k / (mu * (mu_t/mu))``, from ``mu_t = rho k / omega``."""
        return (
            fluid.density
            * self.turbulent_kinetic_energy()
            / (fluid.viscosity * self.eddy_viscosity_ratio)
        )

    def mach(self, speed_of_sound: float = _AIR_SPEED_OF_SOUND) -> float:
        return self.velocity / speed_of_sound

    def compressibility_warning(self, speed_of_sound: float = _AIR_SPEED_OF_SOUND) -> str | None:
        """Warn if the incompressible assumption is being stretched.

        The density error from assuming incompressible flow is about
        ``M^2 / 2``, so 0.3 costs roughly 4.5% at a stagnation point. Past that
        the answer is no longer just approximate, it is wrong in kind.
        """
        mach = self.mach(speed_of_sound)
        if mach <= _INCOMPRESSIBLE_MACH_LIMIT:
            return None
        return (
            f"freestream Mach number is {mach:.2f}, above the {_INCOMPRESSIBLE_MACH_LIMIT} "
            f"limit for the incompressible assumption. Density would vary by roughly "
            f"{50 * mach**2:.0f}% near a stagnation point, which this solver does not model."
        )


#: NASA Turbulence Modeling Resource ambient bands for external aerodynamics,
#: as (low, high) multipliers on the quantities named in Freestream's docstring.
TMR_OMEGA_BAND = (1.0, 10.0)
TMR_K_BAND = (1.0e-5, 0.1)
TMR_VISCOSITY_RATIO_BAND = (1.0e-5, 1.0e-2)


def ambient_turbulence_bands(
    freestream: "Freestream", fluid: Fluid, reference_length: float
) -> dict[str, tuple[float, float, float]]:
    """The three ambient quantities, each with the band it should lie in.

    Returns ``{name: (value, low, high)}``. Separate from the warning that uses
    it so that a caller can report the numbers rather than only the verdict --
    "outside the band" is much less useful than "1.35e-03 against a top of
    4.43e-05".
    """
    velocity = freestream.velocity
    reynolds = fluid.reynolds(velocity, reference_length)
    k = freestream.turbulent_kinetic_energy()
    omega = freestream.specific_dissipation(fluid)

    scale_k = velocity**2 / reynolds
    scale_omega = velocity / reference_length
    return {
        "k": (k, TMR_K_BAND[0] * scale_k, TMR_K_BAND[1] * scale_k),
        "omega": (
            omega,
            TMR_OMEGA_BAND[0] * scale_omega,
            TMR_OMEGA_BAND[1] * scale_omega,
        ),
        "mu_t/mu": (
            freestream.eddy_viscosity_ratio,
            TMR_VISCOSITY_RATIO_BAND[0],
            TMR_VISCOSITY_RATIO_BAND[1],
        ),
    }
