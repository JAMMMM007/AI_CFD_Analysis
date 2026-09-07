"""Grid convergence: observed order, Richardson limit and the ASME GCI.

The project has committed to reporting discretisation uncertainty by the ASME
Journal of Fluids Engineering procedure (Celik et al. 2008), and until now has
had no way to compute one. This module is that machinery, plus the mesh families
to feed it.

**Why the observed order and not the formal one.** A code whose observed order is
far from its formal order fails Roache's own precondition for Richardson
extrapolation to mean anything, so the order is measured from the solutions and
never assumed. The difference is not academic here: measured on the cylinder, the
observed order of `Cd` is about 1.26 against a formal 2, and a GCI computed at
`p = 2` is 1.87 times narrower than the honest one.

**Why the cell size is measured and not assumed.** Celik's `r` is a ratio of
representative cell sizes, which in two dimensions is

    h = sqrt( (1/N) sum_i A_i )

-- the root-mean cell area, computed from the mesh rather than inferred from the
mesh generator's inputs. That distinction matters for a family built by an
O-grid: asking for 1.5 times the surface points and 1.5 times finer a first layer
does *not* give 1.5 times finer a mesh, because the layer count is set by
geometric growth and rises only logarithmically. See :func:`cylinder_family` for
what that does and how it is avoided.

References
----------
Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J., Coleman, H. & Raad, P. E.
(2008), "Procedure for Estimation and Reporting of Uncertainty Due to
Discretization in CFD Applications", *Journal of Fluids Engineering* 130(7),
078001.

Roache, P. J. (1998), *Verification and Validation in Computational Science and
Engineering*, Hermosa, chapters 3 and 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Celik's safety factor for a three-grid study. 3.0 is the two-grid value and
#: is not used here: a two-grid GCI has no measured order behind it, so it is an
#: assumption dressed as an uncertainty.
SAFETY_FACTOR = 1.25


def representative_size(metrics) -> float:
    """Celik's ``h`` for a two-dimensional mesh: the root-mean cell area."""
    volume = np.asarray(metrics.volume)
    return float(math.sqrt(volume.sum() / volume.size))


@dataclass(frozen=True)
class GridConvergence:
    """One quantity's convergence over three systematically refined meshes.

    ``fine``, ``medium`` and ``coarse`` are the solution values, and ``h_*`` the
    matching representative cell sizes -- fine first throughout, which is Celik's
    ordering and the opposite of the order the runs happen in.
    """

    name: str
    fine: float
    medium: float
    coarse: float
    h_fine: float
    h_medium: float
    h_coarse: float
    order: float
    extrapolated: float
    gci_fine: float
    monotone: bool

    @property
    def uncertainty(self) -> float:
        """The discretisation uncertainty on the fine-grid value, absolute."""
        return abs(self.gci_fine * self.fine)

    def __str__(self) -> str:
        flag = "" if self.monotone else "   [OSCILLATORY -- see note]"
        return (
            f"  {self.name:<22} {self.fine:.6f} +/- {self.uncertainty:.6f}"
            f"   (GCI {100.0 * self.gci_fine:.3f}%, observed order {self.order:.3f},"
            f" limit {self.extrapolated:.6f}){flag}"
        )


def analyse(name: str, values, sizes) -> GridConvergence:
    """Celik's procedure on one quantity.

    ``values`` and ``sizes`` are ``(fine, medium, coarse)``.

    The order comes from

        p = |ln|eps32 / eps21| + q(p)| / ln(r21),
        q(p) = ln( (r21^p - s) / (r32^p - s) ),   s = sign(eps32 / eps21),

    solved by fixed-point iteration. ``q`` vanishes identically when the two
    refinement ratios are equal, which is the form the audit used and the form
    most of the literature quotes; it is kept in the general shape here because a
    real mesh family rarely lands on exactly equal ratios and silently assuming
    it does is how a measured order becomes an assumed one.

    ``s = -1`` means the three values do not fall monotonically -- ``eps32`` and
    ``eps21`` have opposite signs. Celik's procedure still returns a number in
    that case and the number should not be trusted; it is flagged rather than
    suppressed, because a non-monotone sequence usually means the meshes are not
    in the asymptotic range and that is the finding, not an inconvenience.
    """
    f1, f2, f3 = (float(v) for v in values)
    h1, h2, h3 = (float(h) for h in sizes)

    if not (h1 < h2 < h3):
        raise ValueError(
            f"meshes must be given fine to coarse; got h = {h1:g}, {h2:g}, {h3:g}"
        )

    r21, r32 = h2 / h1, h3 / h2
    if min(r21, r32) < 1.3:
        # Celik's own precondition. Below it the difference between the solutions
        # is comparable with the noise in them and the extrapolation is fitting
        # iteration error.
        raise ValueError(
            f"refinement ratios {r21:.3f} and {r32:.3f} -- Celik requires at "
            f"least 1.3. This family is not systematically refined enough to "
            f"support an extrapolation."
        )

    eps21, eps32 = f2 - f1, f3 - f2
    if eps21 == 0.0:
        raise ValueError(f"{name}: the two finest meshes agree exactly")

    ratio = eps32 / eps21
    sign = 1.0 if ratio > 0.0 else -1.0

    order = abs(math.log(abs(ratio))) / math.log(r21)
    for _ in range(50):
        q = math.log((r21**order - sign) / (r32**order - sign))
        updated = abs(math.log(abs(ratio)) + q) / math.log(r21)
        if abs(updated - order) < 1e-12:
            order = updated
            break
        order = updated

    extrapolated = (r21**order * f1 - f2) / (r21**order - 1.0)
    relative_error = abs((f1 - f2) / f1) if f1 != 0.0 else abs(f1 - f2)
    gci = SAFETY_FACTOR * relative_error / (r21**order - 1.0)

    return GridConvergence(
        name=name,
        fine=f1,
        medium=f2,
        coarse=f3,
        h_fine=h1,
        h_medium=h2,
        h_coarse=h3,
        order=order,
        extrapolated=extrapolated,
        gci_fine=gci,
        monotone=sign > 0.0,
    )


# ----------------------------------------------------------------------
# Mesh families
# ----------------------------------------------------------------------


def cylinder_family(
    reynolds: float = 40.0,
    *,
    base_points: int = 180,
    base_layers: int = 53,
    ratio: float = 1.5,
    levels: int = 3,
    tolerance: float = 1e-8,
):
    """Three systematically refined cylinder meshes, coarse to fine.

    Yields ``(settings, case)`` so that a caller can run each and read whatever
    it wants off it.

    **How the family is refined, and why not the obvious way.** Scaling
    ``surface_points`` and ``first_layer`` by ``ratio`` and leaving ``growth``
    alone is the obvious construction and it does not produce a family. The layer
    count is set by geometric growth spanning a fixed total, so halving the first
    layer adds only ``ln(ratio) / ln(growth)`` layers: measured on this geometry,
    180, 270 and 405 surface points give 53, 56 and 59 layers. The surface
    direction refines by 1.5 per level and the wall-normal direction by about
    1.06, so ``h`` -- which the far-field cells dominate -- barely moves, and an
    order read off the sequence against an assumed ``r = 1.5`` is not an order in
    ``h`` at all.

    Here the layer count is scaled by ``ratio`` as well and ``growth`` taken to
    the matching root, so that the same total thickness is spanned by ``ratio``
    times the layers with the same distribution. Both directions then refine by
    ``ratio``, and :func:`representative_size` confirms it rather than assuming
    it -- which is the point.

    The cost is real and worth stating: the finest mesh of a properly refined
    three-level family is ``ratio^2`` times the cells of the naive one, and the
    laminar cylinder needs more iterations on a finer mesh, so a level costs
    considerably more than the cell count suggests.
    """
    from fluidsolver.geometry.primitives import circle
    from fluidsolver.solver.case import MeshSettings, build_case
    from fluidsolver.solver.fluid import Fluid, Freestream
    from fluidsolver.solver.simple import Numerics

    diameter, velocity = 1.0, 1.0
    fluid = Fluid(density=1.0, viscosity=velocity * diameter / reynolds, name="test")
    freestream = Freestream(velocity=velocity)

    # The base mesh is the gate's, so the coarsest level of the family *is* the
    # mesh every number in the project has been quoted on.
    total = 40.0 * diameter - 0.5 * diameter
    base_growth = 1.15

    for level in range(levels):
        scale = ratio**level
        points = int(round(base_points * scale))
        layers = int(round(base_layers * scale))
        growth = base_growth ** (1.0 / scale)
        first_layer = total * (growth - 1.0) / (growth**layers - 1.0)

        settings = MeshSettings(
            surface_points=points,
            far_field_radius_ratio=40.0,
            growth=growth,
            first_layer=first_layer,
        )
        case = build_case(
            circle(diameter, points),
            fluid,
            freestream,
            mesh_settings=settings,
            numerics=Numerics(
                scheme="linear", max_iterations=6000, tolerance=tolerance
            ),
            model_name="laminar",
        )
        yield settings, case


def main() -> int:
    """Run the cylinder family and report every quantity the gate reports."""
    import time

    from validation.cylinder import separation_angle, wake_length

    print("Cylinder at Re 40, three systematically refined meshes.\n")
    rows = []
    for settings, case in cylinder_family():
        started = time.perf_counter()
        case.run()
        forces = case.forces()
        last = case.history.entries[-1]
        h = representative_size(case.metrics)
        rows.append(
            {
                "shape": case.metrics.shape,
                "h": h,
                "Cd": forces.drag_coefficient,
                "Cd_pressure": forces.pressure_drag_coefficient,
                "Cd_friction": forces.friction_drag_coefficient,
                "wake": wake_length(case),
                "separation": separation_angle(case),
            }
        )
        print(
            f"  {case.metrics.shape[0]:4d} x {case.metrics.shape[1]:3d}"
            f"   h {h:.6e}   {case.iteration:5d} it   residual {last.worst:.2e}"
            f"   Cd {forces.drag_coefficient:.9f}"
            f"   ({time.perf_counter() - started:.0f}s)"
        )

    rows.reverse()  # Celik orders fine first
    sizes = [row["h"] for row in rows]
    print(
        f"\n  refinement ratios in h: "
        f"{sizes[1] / sizes[0]:.4f}, {sizes[2] / sizes[1]:.4f}\n"
    )
    for name in ("Cd", "Cd_pressure", "Cd_friction", "wake", "separation"):
        try:
            print(analyse(name, [row[name] for row in rows], sizes))
        except ValueError as error:
            print(f"  {name:<22} no extrapolation: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
