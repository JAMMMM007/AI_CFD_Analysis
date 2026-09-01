"""Laminar flow over a circular cylinder -- the validation gate for the solver core.

Steady flow past a cylinder below the vortex-shedding threshold (Re < ~47) is the
best-documented benchmark in incompressible CFD. Drag, wake length and separation
angle have all been measured and computed repeatedly, and independent studies
agree on them to within a percent or two. That makes it a genuine test rather
than a plausibility check: nothing here is tuned to hit these numbers.

Reference values, from the body of literature (Tritton's experiments,
Fornberg 1980, Dennis & Chang 1970, Coutanceau & Bouard 1977, and the many
later computations that reproduce them):

    Re = 20   Cd = 2.00 - 2.09    L/D = 0.91 - 0.94    separation 43 - 45 deg
    Re = 40   Cd = 1.50 - 1.58    L/D = 2.10 - 2.35    separation 52 - 54 deg

The Re = 40 wake length band is wide because the sources genuinely disagree:
Coutanceau and Bouard measured 2.13, while the computations cluster higher --
Nieuwstadt and Keller 2.21, Fornberg 2.24, Dennis and Chang 2.35. Quoting only
the computational range would be quietly choosing which evidence to be judged
against.

Note the convention: separation angle is quoted here measured from the *rear*
stagnation point, which is how the experimental literature reports it. Measured
from the front it is 180 minus that, so Re = 40 separates at about 126 degrees
from the front.

What each number tests is different, which is the point of checking all three:
drag integrates pressure and shear over the whole surface, the wake length tests
the momentum balance well away from the wall, and the separation angle tests the
near-wall shear directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluidsolver.geometry.primitives import circle
from fluidsolver.solver.case import Case, MeshSettings, build_case
from fluidsolver.solver.fluid import Fluid, Freestream
from fluidsolver.solver.simple import Numerics

DIAMETER = 1.0
VELOCITY = 1.0

# (Cd range, wake length L/D range, separation angle from the rear, in degrees)
REFERENCE = {
    20: ((2.00, 2.09), (0.91, 0.94), (43.0, 45.0)),
    40: ((1.50, 1.58), (2.10, 2.35), (52.0, 54.0)),
}


@dataclass
class CylinderResult:
    reynolds: float
    drag_coefficient: float
    pressure_drag: float
    friction_drag: float
    lift_coefficient: float
    wake_length: float
    separation_angle_deg: float
    iterations: int
    residual: float

    def compare(self) -> str:
        drag, wake, angle = REFERENCE[int(round(self.reynolds))]
        return "\n".join(
            [
                f"Re = {self.reynolds:.0f}   ({self.iterations} iterations, "
                f"residual {self.residual:.2e})",
                _line("Cd", self.drag_coefficient, drag),
                _line("wake L/D", self.wake_length, wake),
                _line("separation (from rear)", self.separation_angle_deg, angle, "deg"),
                f"  {'Cl (symmetry)':<24} {self.lift_coefficient:+9.5f}   "
                f"expect 0",
                f"  {'Cd split':<24} pressure {self.pressure_drag:.4f}, "
                f"friction {self.friction_drag:.4f}",
            ]
        )

    def passes(self) -> bool:
        drag, wake, angle = REFERENCE[int(round(self.reynolds))]
        return (
            _within(self.drag_coefficient, drag)
            and _within(self.wake_length, wake)
            and _within(self.separation_angle_deg, angle)
            and abs(self.lift_coefficient) < 1e-3
        )


def _within(value: float, bounds: tuple[float, float], slack: float = 0.05) -> bool:
    """Inside the published range, with a little slack for mesh resolution."""
    low, high = bounds
    margin = slack * (high - low)
    return low - margin <= value <= high + margin


def _line(name: str, value: float, bounds: tuple[float, float], unit: str = "") -> str:
    mark = "ok " if _within(value, bounds) else "OFF"
    return (
        f"  {name:<24} {value:9.4f}{unit:<4}  expect {bounds[0]}-{bounds[1]}  [{mark}]"
    )


def wake_length(case: Case) -> float:
    """Length of the recirculation bubble behind the cylinder, in diameters.

    Measured along the centreline from the rear of the body to where the
    streamwise velocity changes back to positive. The mesh is polar, so the
    centreline is not a grid line; the velocity is sampled along it and the sign
    change interpolated.
    """
    radius = 0.5 * DIAMETER
    stations = radius + np.linspace(1e-3, 4.0 * DIAMETER, 800)
    samples = np.stack((stations, np.zeros_like(stations)), axis=-1)

    velocity = _sample(case, case.state.u, samples)
    reversed_flow = velocity < 0.0
    if not reversed_flow.any():
        return 0.0

    # The bubble closes at the last reversal, not the first: sample noise near
    # the surface can flip the sign momentarily.
    last = int(np.flatnonzero(reversed_flow).max())
    if last + 1 >= len(stations):
        return float("inf")

    before, after = velocity[last], velocity[last + 1]
    crossing = stations[last] + (stations[last + 1] - stations[last]) * before / (
        before - after
    )
    return float((crossing - radius) / DIAMETER)


def _sample(case: Case, field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-cell sampling of a cell field at arbitrary points."""
    from scipy.spatial import cKDTree

    tree = cKDTree(case.metrics.centroid.reshape(-1, 2))
    _, index = tree.query(points)
    return field.ravel()[index]


def separation_angle(case: Case) -> float:
    """Angle from the rear stagnation point to the separation line, in degrees.

    Zero if the flow stays attached.
    """
    points = case.separation_points()
    if len(points) == 0:
        return 0.0

    # The rear stagnation point sits at theta = 0, so the polar angle of a
    # separation point *is* its angle from the rear. Measured from the front it
    # would be 180 minus this, which is the other convention in circulation and
    # the reason to be explicit about which one is meant.
    angles = np.abs(np.degrees(np.arctan2(points[:, 1], points[:, 0])))

    # Discard crossings at the stagnation points themselves, where the shear
    # vanishes for reasons of symmetry rather than separation.
    genuine = angles[(angles > 1.0) & (angles < 179.0)]
    return float(genuine.max()) if len(genuine) else 0.0


def run(
    reynolds: float,
    *,
    surface_points: int = 180,
    far_field_ratio: float = 40.0,
    max_iterations: int = 3000,
    tolerance: float = 1e-7,
    progress: bool = False,
) -> CylinderResult:
    """Solve the cylinder at one Reynolds number and measure the benchmarks."""
    fluid = Fluid(density=1.0, viscosity=VELOCITY * DIAMETER / reynolds, name="test")
    freestream = Freestream(velocity=VELOCITY)

    case = build_case(
        circle(DIAMETER, surface_points),
        fluid,
        freestream,
        mesh_settings=MeshSettings(
            surface_points=surface_points, far_field_radius_ratio=far_field_ratio
        ),
        numerics=Numerics(
            scheme="linear", max_iterations=max_iterations, tolerance=tolerance
        ),
        model_name="laminar",
    )

    def report(residuals):
        if progress and residuals.iteration % 100 == 0:
            print(f"  {residuals}")

    case.run(callback=report if progress else None)

    forces = case.forces()
    return CylinderResult(
        reynolds=reynolds,
        drag_coefficient=forces.drag_coefficient,
        pressure_drag=forces.pressure_drag_coefficient,
        friction_drag=forces.friction_drag_coefficient,
        lift_coefficient=forces.lift_coefficient,
        wake_length=wake_length(case),
        separation_angle_deg=separation_angle(case),
        iterations=case.iteration,
        residual=case.history.entries[-1].worst,
    )


def main() -> int:
    results = [run(re, progress=True) for re in (20, 40)]
    print()
    for result in results:
        print(result.compare())
        print()
    ok = all(result.passes() for result in results)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
