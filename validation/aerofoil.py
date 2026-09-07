"""NACA 2412 at 5 degrees -- the change-detection gate for the turbulent path.

**This is not a validation case and it does not certify accuracy.** Its job is to
notice when a change moves an answer, and it exists because the cylinder gate
structurally cannot.

The cylinder is orthogonal, non-lifting, laminar and run at one mesh density.
Measured, its mesh has a non-orthogonality of 0.0000 degrees, mean and peak,
which is a genuine property of a circle marched into a circular far field and the
reason to keep it. But it means every error term proportional to the
non-orthogonal remainder ``T = S - g d`` is identically zero there, and every
term proportional to the bound circulation is identically zero as well. Seven of
the seventeen findings of the 2026-09-07 physics audit are invisible to it for
that reason alone: the Rhie-Chow non-orthogonal correction, the far-field vortex,
the ``omega`` wall constant, the ``omega`` production form, the freestream
turbulence levels, the zero-gradient condition imposed along ``d``, and the two
disagreeing wall distances.

This case sees all seven. Measured on this mesh: non-orthogonality mean 3.9
degrees and peak 60.1, aspect ratio 453, expansion ratio 4.77, `Cl` about 0.75.

**The specification is here, in full, and that is deliberate.** The hardening
plan, the README and the handover between them directed the next engineer at a
NACA 0012 case that no longer reproduces, and none of the three recorded its
angle of attack, surface point count, ``y+`` target or far-field ratio -- all
four of which change the answer. Every number below is fixed in this module so
that a run of this gate today and a run in two years are the same run.

    geometry              NACA 2412, 400 points before resampling
    incidence             5 degrees, applied by rotating the body
    fluid                 air at 15 C: rho 1.225, mu 1.81e-5
    freestream            30 m/s, so Re = 2.03e6 on a unit chord
    turbulence intensity  0.1%, eddy viscosity ratio 1.0 (the defaults)
    surface points        240
    target y+             1
    far-field radius      40 chords
    model                 k-omega SST
    scheme                limited_linear (the default)
    relaxation            0.7 velocity, 0.3 pressure, 0.7 turbulence
    tolerance             1e-6
    max iterations        1500

The bands are wide on purpose. A tight band on an unvalidated case would be a
claim to accuracy this case cannot support; a wide one still catches the size of
change every finding in the audit predicts, which is what it is for. Where a
published number exists it is quoted in the comment beside the band, as context
and not as a target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluidsolver.geometry.naca import naca4
from fluidsolver.solver.case import Case, MeshSettings, build_case
from fluidsolver.solver.fluid import AIR_15C, Freestream
from fluidsolver.solver.simple import Numerics

CHORD = 1.0
VELOCITY = 30.0
INCIDENCE_DEG = 5.0

#: Bands, wide, for change detection. The centres are the values measured on
#: `main` at 37fabd2 and recorded by ``BASELINE`` below; the widths are roughly
#: +/- 10% on the force coefficients, which is larger than any single finding in
#: the audit predicts except F1's 8.94% in `Cd` -- deliberately, so that landing
#: one finding at a time does not require re-baselining on every commit.
#:
#: For orientation only, not as targets: Abbott and von Doenhoff give the
#: NACA 2412 a section `Cl` near 0.75 at 5 degrees and `Cm_ac` near -0.05.
REFERENCE = {
    "Cl": (0.68, 0.83),
    "Cd": (0.0112, 0.0138),
    "Cd_friction": (0.0067, 0.0082),
}

#: What this specification read on `main` at 37fabd2, before any audit finding
#: was implemented. Printed alongside every run so that a movement is visible
#: without going to the git history for it. These are the digits to update,
#: deliberately and in the same commit, whenever a change is meant to move them.
#:
#: The figures are the physics audit's, which were carried to more digits than a
#: run of this module prints. They were reproduced here by an independent run of
#: exactly the specification above, and the two agree:
#:
#:     audit   1007 it  residual 9.9385e-07  Cl 0.7535096  Cd 0.01250649
#:                      Cdp 0.00505258  Cdf 0.00745391  Cm -0.0825130
#:     here    1008 it  residual 9.97e-07    Cl 0.75354    Cd 0.012506
#:                      Cdp 0.005051    Cdf 0.007455    Cm -0.08251
#:
#: Every force coefficient agrees to better than 0.03%, on a run one iteration
#: longer. The one figure that does not is the top of the `y+` range, 2.301 here
#: against 2.357 there -- 2.4% apart, on a quantity read off the surface
#: distribution rather than integrated, and not currently explained.
#:
#: **This run also settles the hardening plan's first open item.** It was made on
#: factory `Numerics()` with the divergence monitor *armed*, and it converged to
#: 9.97e-07 in 1008 iterations without raising anything. The plan, the README and
#: the handover all describe a NACA 0012 at Re 2e6 that `SolverDiverged` stops at
#: iteration 367, and all three call it the first thing to fix. The audit could
#: not reproduce it and neither can this. See `docs/audit-response-plan.md`.
BASELINE = {
    "iterations": 1007,
    "residual": 9.9385e-07,
    "Cl": 0.7535096,
    "Cd": 0.01250649,
    "Cd_pressure": 0.00505258,
    "Cd_friction": 0.00745391,
    "Cm": -0.0825130,
    "y_plus_min": 0.301,
    "y_plus_max": 2.357,
}


@dataclass
class AerofoilResult:
    lift_coefficient: float
    drag_coefficient: float
    pressure_drag: float
    friction_drag: float
    moment_coefficient: float
    y_plus_min: float
    y_plus_max: float
    iterations: int
    residual: float
    converged: bool

    def compare(self) -> str:
        lines = [
            f"NACA 2412, {INCIDENCE_DEG:g} deg, Re {VELOCITY * CHORD * AIR_15C.density / AIR_15C.viscosity:.3e}"
            f"   ({self.iterations} iterations, residual {self.residual:.2e}"
            f"{'' if self.converged else ', NOT CONVERGED'})",
            _line("Cl", self.lift_coefficient, REFERENCE["Cl"], BASELINE["Cl"]),
            _line("Cd", self.drag_coefficient, REFERENCE["Cd"], BASELINE["Cd"], 6),
            _line(
                "Cd friction",
                self.friction_drag,
                REFERENCE["Cd_friction"],
                BASELINE["Cd_friction"],
                6,
            ),
            f"  {'Cd pressure':<20} {self.pressure_drag:9.6f}"
            f"        was {BASELINE['Cd_pressure']:.6f}"
            f"   ({_delta(self.pressure_drag, BASELINE['Cd_pressure'])})",
            f"  {'Cm':<20} {self.moment_coefficient:+9.5f}"
            f"        was {BASELINE['Cm']:+.5f}"
            f"   ({_delta(self.moment_coefficient, BASELINE['Cm'])})",
            f"  {'y+ range':<20} {self.y_plus_min:.3f} .. {self.y_plus_max:.3f}"
            f"     was {BASELINE['y_plus_min']:.3f} .. {BASELINE['y_plus_max']:.3f}",
        ]
        return "\n".join(lines)

    def passes(self) -> bool:
        """Inside every band, and actually converged.

        Convergence is part of the criterion rather than a footnote. A case that
        stopped on ``max_iterations`` may sit inside every band and still be
        reporting a number that is still moving, and the audit's K1 found exactly
        that on the NACA 0012 -- forces steady in the sixth decimal while the
        residual plateaued two orders above tolerance.
        """
        return (
            self.converged
            and _within(self.lift_coefficient, REFERENCE["Cl"])
            and _within(self.drag_coefficient, REFERENCE["Cd"])
            and _within(self.friction_drag, REFERENCE["Cd_friction"])
        )


def _within(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _delta(value: float, was: float) -> str:
    if was == 0.0:
        return "n/a"
    return f"{100.0 * (value - was) / abs(was):+.3f}%"


def _line(name, value, bounds, was, digits: int = 5) -> str:
    mark = "ok " if _within(value, bounds) else "OFF"
    return (
        f"  {name:<20} {value:9.{digits}f}  [{mark}]  was {was:.{digits}f}"
        f"   ({_delta(value, was)})"
    )


def build(
    *,
    surface_points: int = 240,
    target_y_plus: float = 1.0,
    far_field_ratio: float = 40.0,
    max_iterations: int = 1500,
    tolerance: float = 1e-6,
) -> Case:
    """The case, assembled but not run.

    Exposed separately so that a study can take the same mesh and settings and
    change exactly one thing -- which is what separates a measurement of a
    boundary condition from a measurement of a mesh, and is the distinction the
    audit's far-field sweep could not make.
    """
    return build_case(
        naca4("2412", 400),
        AIR_15C,
        Freestream(velocity=VELOCITY, angle_of_attack_deg=INCIDENCE_DEG),
        mesh_settings=MeshSettings(
            surface_points=surface_points,
            target_y_plus=target_y_plus,
            far_field_radius_ratio=far_field_ratio,
        ),
        numerics=Numerics(max_iterations=max_iterations, tolerance=tolerance),
        model_name="k-omega-sst",
    )


def run(*, progress: bool = False, **settings) -> AerofoilResult:
    """Solve the case and read the quantities the gate watches."""
    case = build(**settings)

    def report(residuals):
        if progress and residuals.iteration % 100 == 0:
            print(f"  {residuals}")

    case.run(callback=report if progress else None)

    forces = case.forces()
    surface = case.surface()
    last = case.history.entries[-1]
    return AerofoilResult(
        lift_coefficient=forces.lift_coefficient,
        drag_coefficient=forces.drag_coefficient,
        pressure_drag=forces.pressure_drag_coefficient,
        friction_drag=forces.friction_drag_coefficient,
        moment_coefficient=forces.moment_coefficient,
        y_plus_min=float(surface.y_plus.min()),
        y_plus_max=float(surface.y_plus.max()),
        iterations=case.iteration,
        residual=last.worst,
        converged=last.has_converged(case.numerics.tolerance),
    )


def main() -> int:
    result = run(progress=True)
    print()
    print(result.compare())
    print()
    ok = result.passes()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
