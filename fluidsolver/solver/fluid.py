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
        ``u'/U`` in the freestream, as a fraction. 0.001 (0.1%) is representative
        of a clean wind tunnel or free air; 0.05 of a noisy one. Sets the inlet
        ``k``.
    eddy_viscosity_ratio
        Freestream ``mu_t / mu``. Together with the intensity this fixes the
        inlet ``omega``. Values of 1 to 10 are usual for external aerodynamics;
        much larger and the freestream turbulence contaminates the boundary
        layer.
    """

    velocity: float
    angle_of_attack_deg: float = 0.0
    turbulence_intensity: float = 0.001
    eddy_viscosity_ratio: float = 1.0

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
