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

**Each is printed with the discretisation error on this mesh.** The band decides
pass or fail, as it always has; the extra column says how far the value sits from
the answer the same solver gives as the mesh is refined to nothing. That
distinction is the whole reason for it. ``Cd`` lands 0.12% below its
grid-converged value and the wake length 9.7% below its own, so one of the three
agrees with the literature because it is right and another agrees because the
mesh is coarse -- and without the column they look alike. ASME V&V 20 validates a
result at a stated uncertainty rather than against a fixed percentage, and this
is the stated uncertainty.

The figures come from ``python -m validation.convergence`` and are recorded per
Reynolds number, because a study belongs to the case it was run on. Only Re 40 has
one; Re 20 prints the bands alone. See :data:`GRID_CONVERGENCE`.
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

#: What a three-mesh study says this gate's own mesh is worth.
#:
#: ``{quantity: (observed order, Richardson limit, GCI on the finest mesh)}``,
#: from ``python -m validation.convergence`` -- three meshes refined by 1.5 in
#: both directions at once, 180x53, 270x80 and 405x119, each converged to 1e-8.
#: The measured refinement ratios in Celik's ``h`` are 1.4937 and 1.5046, so the
#: family really is refined by what it claims.
#:
#: The gate runs the *coarsest* of the three, so what it reports is not the
#: fine-grid GCI but the distance from its own answer to the extrapolated one.
#: That is a measured discretisation error rather than an estimate of one, and it
#: is the number that belongs beside a value being compared with experiment: ASME
#: V&V 20 validates a result at a stated uncertainty, not against a fixed
#: percentage.
#:
#: Two of these deserve reading before the drag does.
#:
#: ``Cd`` comes out at an observed order of **2.006**. The scheme is formally
#: second order and, on a solved field with the boundary conditions, the
#: pressure-velocity coupling and the force integral all in the loop, it now
#: delivers that.
#:
#: **That it is the code and not the family is measured, not assumed.** The audit
#: reported 1.261 for the same quantity, on a family built differently -- surface
#: points and first layer scaled by 1.5 with ``growth`` left alone, which adds
#: only three layers a level and gives 53, 56, 59. Running *that* construction on
#: today's code gives an order of **2.047** by the audit's own arithmetic. Same
#: family, same assumed ratio, 1.261 before and 2.047 after: the first-order terms
#: coming out of the flux definition and the force integral are what moved it.
#:
#: The same control says something about the family as well. Its measured
#: refinement ratios in ``h`` are 1.2571 and 1.2589, not the 1.5 it was assumed to
#: have, and Celik's procedure requires at least 1.3 -- so ``validation.convergence``
#: refuses to extrapolate on it at all, which is the refusal working. And the wake
#: length comes out at 0.735 there, reproducing the audit's 0.735 exactly, which is
#: as good a cross-check between two independent implementations as this project
#: has.
#:
#: The **wake length still does not converge**: observed order 1.124 and a GCI of
#: 5.7%, against 0.03% for the drag. Its extrapolated value is 2.349, and the
#: gate reports 2.12 on the coarsest mesh -- so the agreement with Coutanceau and
#: Bouard's measured 2.13 is a coincidence of resolution, and the solver's own
#: converged answer sits with the computations at 2.21 to 2.35. That was true
#: when the audit found it and it is still true.
#: Keyed by Reynolds number, because a study is a property of the case it was
#: run on and not of the geometry. Only Re 40 has one; Re 20 prints no column
#: rather than borrowing its neighbour's, which the first version of this did --
#: it showed the Re 20 drag of 2.027 as being 33% from a "grid-converged" 1.518
#: that belongs to a different flow.
GRID_CONVERGENCE = {
    40: {
        "Cd": (2.006, 1.517935, 0.00031),
        "wake": (1.124, 2.348867, 0.05745),
        "separation": (5.661, 53.830074, 0.00024),
    }
}

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
        study = GRID_CONVERGENCE.get(int(round(self.reynolds)), {})
        return "\n".join(
            [
                f"Re = {self.reynolds:.0f}   ({self.iterations} iterations, "
                f"residual {self.residual:.2e})",
                _line("Cd", self.drag_coefficient, drag,
                      study=study, quantity="Cd"),
                _line("wake L/D", self.wake_length, wake,
                      study=study, quantity="wake"),
                _line("separation (from rear)", self.separation_angle_deg, angle,
                      "deg", study=study, quantity="separation"),
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


def _line(
    name: str,
    value: float,
    bounds: tuple[float, float],
    unit: str = "",
    study: dict | None = None,
    quantity: str | None = None,
) -> str:
    """One reported quantity, with the discretisation error on this mesh beside it.

    The band still decides pass or fail. The extrapolated column is there so that
    a reader can tell a value that sits inside the band because it is right from
    one that sits inside because the mesh is coarse -- which is exactly the
    distinction the wake length fails.

    ``study`` is the grid-convergence result *for this Reynolds number*, and is
    empty for a case that has not had one. Nothing is printed then. The first
    version of this looked the quantity up by name alone and so showed the Re 20
    drag against the Re 40 limit -- 2.027 reported as 33% from a converged 1.518
    belonging to a different flow. An uncertainty attached to the wrong case is
    worse than none.
    """
    mark = "ok " if _within(value, bounds) else "OFF"
    line = (
        f"  {name:<24} {value:9.4f}{unit:<4}  expect {bounds[0]}-{bounds[1]}  [{mark}]"
    )
    if not study or quantity not in study:
        return line
    order, limit, _ = study[quantity]
    error = 100.0 * (value - limit) / abs(limit)
    return line + f"   grid-converged {limit:.4f} ({error:+.3f}%, order {order:.2f})"


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
