"""Assembling the complete O-grid: marched near field, analytic far field.

The two regions of an external-aerodynamics mesh want opposite things.

Near the wall the grid has to follow the body exactly, meet it at right angles,
and hold a first-cell height fixed by the turbulence model. That is what the
hyperbolic marcher in :mod:`fluidsolver.mesh.hyperbolic` is for, and on a smooth
body it reproduces the exact offset surface to machine precision.

Tens of chords away none of that applies. The body is a point, the flow is
uniform, and the only thing the mesh owes the solver is a smooth outer boundary
of known shape -- ideally a circle, since the characteristic freestream condition
splits each outer face on the sign of ``U . n`` and a circle keeps that clean.

Marching all the way out serves neither end well. The march is at its most
fragile precisely where it matters least: cells are enormous, curvature
information from the body has long since decayed into noise, and the surviving
grids carry outer cells varying several hundred to one in size. So the march is
stopped once it has done its job and the remainder is built directly.

The far field is interpolated in *polar* coordinates about the body centroid,
between the last marched layer and the target circle. That choice is what makes
it safe: the construction blends two monotone angle sequences and a monotone
radius, so grid lines cannot cross and every cell is positive by construction,
with no smoothing parameter to tune. The angular distribution relaxes from
whatever the march produced towards uniform, so the outer boundary is an evenly
spaced circle of exactly the radius requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fluidsolver.geometry.contour import Contour
from fluidsolver.mesh import spacing
from fluidsolver.mesh.hyperbolic import MeshError, hyperbolic_grid


@dataclass
class OGrid:
    """A structured O-grid and a record of how it was built.

    Attributes
    ----------
    nodes
        ``(Ni, Nj + 1, 2)`` node coordinates. ``i`` wraps the body and is
        periodic, ``j`` runs from the wall outward.
    contour
        The body the grid was built around.
    thicknesses
        The wall-normal layer distribution that was requested.
    marched_layers
        How many layers the hyperbolic marcher produced before the analytic far
        field took over.
    notes
        Anything the caller should know -- in particular whether the march was cut
        short by a fold rather than reaching the intended transition.
    """

    nodes: np.ndarray
    contour: Contour
    thicknesses: np.ndarray
    marched_layers: int
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        """``(Ni, Nj)`` -- cells around the body, cells in the wall-normal direction."""
        return self.nodes.shape[0], self.nodes.shape[1] - 1

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def wall(self) -> np.ndarray:
        """``(Ni, 2)`` nodes on the body surface."""
        return self.nodes[:, 0]

    @property
    def far_field(self) -> np.ndarray:
        """``(Ni, 2)`` nodes on the outer boundary."""
        return self.nodes[:, -1]

    def __repr__(self) -> str:
        n_i, n_j = self.shape
        return f"OGrid({self.contour.name!r}, {n_i}x{n_j} = {self.n_cells} cells)"


def build_ogrid(
    contour: Contour,
    *,
    first_layer: float,
    far_field_radius: float,
    growth: float = 1.15,
    transition_distance: float | None = None,
    dissipation: float = 0.003,
    second_difference: float = 0.0,
) -> OGrid:
    """Build a complete O-grid around ``contour``.

    Parameters
    ----------
    first_layer
        Thickness of the wall-adjacent cell, normally from
        :func:`fluidsolver.mesh.spacing.first_layer_thickness` so that the first
        cell centre lands at the ``y+`` the turbulence model needs.
    far_field_radius
        Radius of the outer boundary, measured from the body centroid. Thirty to
        fifty reference lengths is usual for a lifting case; less than about
        twenty and the boundary starts to interfere with the circulation.
    growth
        Geometric growth ratio between successive wall-normal layers.
    transition_distance
        Wall distance at which the marched near field hands over to the analytic
        far field. Defaults to one reference length, by which point the grid has
        stopped resolving anything the body is responsible for. The march is
        allowed to stop earlier if it runs into trouble; the handover simply
        happens sooner and the fact is recorded in :attr:`OGrid.notes`.
    dissipation
        Fourth-difference smoothing in the marcher. The default is deliberately
        small: the stencil carries a diagonal weight of six, so anything above
        roughly 0.03 stops being a correction and starts being the equation.
    second_difference
        Second-difference smoothing, off by default. Useful only for stubborn
        re-entrant geometry, and it blunts sharp corners when used heavily.
    """
    contour.validate()
    if far_field_radius <= 0.0:
        raise MeshError(f"far_field_radius must be positive, got {far_field_radius}")

    centre = contour.centroid
    body_reach = float(np.linalg.norm(contour.points - centre, axis=1).max())
    if far_field_radius <= 2.0 * body_reach:
        raise MeshError(
            f"far_field_radius {far_field_radius:g} is not clear of the body, which "
            f"reaches {body_reach:g} from its centroid. Use at least a few times that."
        )

    total = far_field_radius - body_reach
    thicknesses = spacing.geometric_layers(first_layer, total, growth)

    if transition_distance is None:
        transition_distance = contour.reference_length
    transition_distance = float(np.clip(transition_distance, 0.0, 0.5 * total))

    distances = np.cumsum(thicknesses)
    n_near = int(np.searchsorted(distances, transition_distance) + 1)
    n_near = int(np.clip(n_near, 1, len(thicknesses) - 1))

    notes: list[str] = []
    nodes = np.empty((len(contour), len(thicknesses) + 1, 2))
    marched, completed = hyperbolic_grid(
        contour,
        thicknesses[:n_near],
        dissipation=dissipation,
        second_difference=second_difference,
        allow_partial=True,
    )
    if completed == 0:
        raise MeshError(
            "the hyperbolic march folded on its very first layer. The wall "
            "distribution is probably irregular; resample the contour before "
            "meshing, or reduce the first-layer thickness."
        )
    if completed < n_near:
        notes.append(
            f"march stopped after {completed} of {n_near} near-field layers "
            f"(wall distance {distances[completed - 1]:.4g} rather than "
            f"{transition_distance:.4g}); the analytic far field took over there."
        )

    nodes[:, : completed + 1] = marched[:, : completed + 1]
    nodes[:, completed:] = _blend_to_circle(
        marched[:, completed],
        centre,
        far_field_radius,
        thicknesses[completed:],
    )

    return OGrid(nodes, contour, thicknesses, completed, notes)


def _blend_to_circle(
    inner: np.ndarray,
    centre: np.ndarray,
    radius: float,
    thicknesses: np.ndarray,
) -> np.ndarray:
    """Interpolate from a marched layer out to a circle, in polar coordinates.

    Working in ``(r, theta)`` rather than ``(x, y)`` is what makes this
    unconditionally safe. Each grid line keeps its own monotone sweep of angle
    and its own monotonically increasing radius, so no two lines can cross and
    every cell comes out positive without a single tuning parameter.

    The angular distribution relaxes from the marched layer's towards a uniform
    one, using a smoothstep in *log* radius -- the variable that advances evenly
    per layer on a geometrically graded mesh. Its zero derivative at both ends
    means the far field joins the marched grid without a kink in cell aspect
    ratio, and arrives at the outer boundary evenly spaced.

    What this construction cannot do is match the *direction* the march arrived
    on. It leaves the seam along a ray from the body centroid; the march arrives
    along the surface normal. On a circle those coincide and the join is exact,
    which is why a circle mesh comes out perfectly orthogonal and an aerofoil
    does not. Some non-orthogonality at the transition layer is therefore
    inherent to a polar far field, and the way to keep it small is to hand over
    where the marched layer is already close to circular -- which is what the
    marcher's ``max_width_ratio`` governs, and why it is set loosely.
    """
    offset = inner - centre
    r_inner = np.hypot(offset[:, 0], offset[:, 1])
    if radius <= r_inner.max():
        raise MeshError(
            f"far-field radius {radius:g} is inside the marched grid, which already "
            f"reaches {r_inner.max():g}. Move the boundary out or march less far."
        )

    angle = _monotone_angles(offset)
    # Uniform angles anchored on the same point, so the far field does not rotate
    # relative to the body. The sweep is exactly one full turn, by definition of a
    # closed loop -- estimating it from the last marched step instead leaves the
    # wrap-around gap disagreeing with all the others, and the seam shows up as a
    # single stretched cell on the freestream boundary.
    sweep = 2.0 * np.pi * np.sign(angle[-1] - angle[0])
    uniform = angle[0] + sweep * np.arange(len(angle)) / len(angle)

    # Fraction of the remaining radial gap covered by each layer. This has to
    # place the radii, because it is what carries the requested thicknesses.
    fraction = np.concatenate(([0.0], np.cumsum(thicknesses) / thicknesses.sum()))
    r = r_inner[:, None] + (radius - r_inner[:, None]) * fraction[None, :]

    # The angular relaxation is driven by log radius instead, which is the
    # variable that advances evenly from layer to layer once the thicknesses grow
    # geometrically. Driving it by ``fraction`` -- as this did -- concentrates the
    # whole relaxation into the outermost handful of layers, because with a growth
    # ratio of 1.15 the cumulative thickness is still under a tenth of the total
    # two thirds of the way out. Everything inside that keeps the marched angular
    # distribution while its radii have already been forced towards a circle, and
    # the two disagree: the grid line is no longer perpendicular to the layer it
    # crosses. On a NACA 0012 that left a mean non-orthogonality of 15 degrees
    # across the far field, decaying only in the last few layers; spreading the
    # relaxation evenly brings it to 4.5 and confines what remains to the seam.
    progress = np.log(r / r_inner[:, None]) / np.log(radius / r_inner[:, None])
    weight = progress**2 * (3.0 - 2.0 * progress)  # smoothstep
    theta = angle[:, None] + (uniform - angle)[:, None] * weight

    return centre + np.stack((r * np.cos(theta), r * np.sin(theta)), axis=-1)


def _monotone_angles(offset: np.ndarray, max_correction: float = 0.5) -> np.ndarray:
    """Polar angles of a closed loop, unwrapped into a single monotone sweep.

    The blend needs the angles to advance steadily around the loop. They do so if
    the layer is star-shaped about the centre -- every ray from the centre
    crossing it exactly once.

    A marched layer is usually star-shaped but not always *exactly* so: around a
    sharp corner the march leaves a slight wobble, and a single step of a few per
    cent of the mean can run backwards. Refusing outright over that would reject
    a perfectly serviceable grid, so instead the angles are nudged towards a
    uniform sweep by the smallest fraction that restores monotonicity. Uniform
    angles advance by a constant, so a small nudge repairs a small violation
    while leaving the distribution essentially as the march made it.

    A layer that needs more than ``max_correction`` of the way to uniform is
    genuinely tangled, and that is worth failing on.
    """
    angle = np.unwrap(np.arctan2(offset[:, 1], offset[:, 0]))
    n = len(angle)

    # Work in the direction of travel so the test is always "must increase".
    direction = 1.0 if angle[-1] > angle[0] else -1.0
    step = np.diff(angle) * direction
    if np.all(step > 0.0):
        return angle

    uniform_step = (angle[-1] - angle[0]) * direction / (n - 1)
    if uniform_step <= 0.0:
        raise MeshError(
            "the layer handed to the far-field blend does not sweep once around "
            "the body centroid, so no far field can be built from it."
        )

    # Blending fraction that lifts every backwards step to exactly zero;
    # (1 - a) * step + a * uniform_step > 0.
    violating = step <= 0.0
    needed = (-step[violating] / (uniform_step - step[violating])).max()
    if needed > max_correction:
        raise MeshError(
            f"the layer handed to the far-field blend is not star-shaped about the "
            f"body centroid -- straightening it would take {needed:.0%} of the way "
            "to a uniform sweep, so its grid lines would cross. March further out "
            "before the transition, or simplify the geometry."
        )

    blend = min(1.0, 1.5 * needed + 1e-3)
    uniform = angle[0] + direction * uniform_step * np.arange(n)
    return (1.0 - blend) * angle + blend * uniform
