"""Analytic bluff bodies: the circle and the square.

Both exist mainly as validation geometry. The circle at low Reynolds number is
the standard benchmark for a laminar incompressible solver, and the square with
its fixed separation points at the leading corners is the cleanest test that the
mesher survives a sharp convex vertex.
"""

from __future__ import annotations

import numpy as np

from fluidsolver.geometry.contour import Contour, ContourError


def circle(diameter: float = 1.0, n_points: int = 256, *, centre=(0.0, 0.0)) -> Contour:
    """A circle discretised with uniform angular spacing.

    Uniform angles give uniform arclength here, which is what you want: the
    curvature is constant, so no part of the surface deserves more points than
    any other.

    The reference length is the diameter, matching the convention for the
    published cylinder drag coefficients used in the validation cases.
    """
    if diameter <= 0.0:
        raise ContourError(f"diameter must be positive, got {diameter}")
    if n_points < 8:
        raise ContourError(f"a circle needs at least 8 points, got {n_points}")

    # endpoint=False leaves the closing point implicit, as Contour expects.
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    radius = 0.5 * diameter
    points = np.column_stack(
        (centre[0] + radius * np.cos(theta), centre[1] + radius * np.sin(theta))
    )
    return Contour(points, name=f"circle d={diameter:g}", reference_length=diameter)


def square(side: float = 1.0, n_points: int = 256, **kwargs) -> Contour:
    """A square. Thin wrapper over :func:`rectangle`."""
    return rectangle(side, side, n_points, **kwargs)


def rectangle(
    width: float = 1.0,
    height: float = 1.0,
    n_points: int = 256,
    *,
    centre=(0.0, 0.0),
    corner_radius: float = 0.0,
    corner_clustering: float | None = None,
) -> Contour:
    """An axis-aligned rectangle, optionally with filleted corners.

    Points are cosine-clustered towards the ends of each edge. A sharp corner is
    a singular point of the potential flow -- the inviscid velocity there is
    unbounded -- and in the viscous problem it fixes the separation point, so the
    cells approaching it need to be small.

    Parameters
    ----------
    corner_radius
        Fillet radius. Zero gives a true sharp corner, which is what the square
        benchmark wants. A small non-zero radius is worth trying if the mesher
        reports poor cell quality at the vertices, since it removes the
        discontinuity in surface normal that the marching scheme has to absorb.
    corner_clustering
        Strength of the edge-end clustering: 0 is uniform spacing, 1 is full
        cosine. Defaults to full cosine for a sharp corner and to uniform when a
        fillet is present. Clustering into a fillet is actively harmful: the arc
        already carries its own points at roughly uniform arclength, so driving
        the neighbouring edge cells towards zero leaves a large spacing jump
        exactly where the two meet.

    The reference length is the width, so a square's Reynolds number is built on
    its side length.
    """
    if width <= 0.0 or height <= 0.0:
        raise ContourError(f"width and height must be positive, got {width} x {height}")
    if n_points < 12:
        raise ContourError(f"a rectangle needs at least 12 points, got {n_points}")

    max_radius = 0.5 * min(width, height)
    if not 0.0 <= corner_radius < max_radius:
        raise ContourError(
            f"corner_radius must lie in [0, {max_radius:g}) for a "
            f"{width:g} x {height:g} rectangle, got {corner_radius}"
        )

    cx, cy = centre
    hw, hh = 0.5 * width, 0.5 * height
    r = corner_radius
    if corner_clustering is None:
        corner_clustering = 1.0 if r == 0.0 else 0.0

    # Corner centres, counter-clockwise from the bottom right. With r = 0 these
    # collapse onto the rectangle's vertices and the arcs degenerate to points,
    # which is exactly the sharp-cornered case.
    corners = np.array(
        [
            [cx + hw - r, cy - hh + r],
            [cx + hw - r, cy + hh - r],
            [cx - hw + r, cy + hh - r],
            [cx - hw + r, cy - hh + r],
        ]
    )
    # Outward direction of the straight edge leaving each corner, and the arc's
    # start angle there.
    edge_dirs = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
    arc_starts = np.array([-0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    edge_lengths = np.array(
        [height - 2 * r, width - 2 * r, height - 2 * r, width - 2 * r]
    )

    straight_total = float(np.sum(edge_lengths))
    arc_total = 2.0 * np.pi * r
    # Share the point budget between straight edges and fillets by arclength, so
    # the spacing is roughly continuous where one meets the other.
    n_arc_total = int(round(n_points * arc_total / (straight_total + arc_total)))
    n_arc = max(3, n_arc_total // 4) if r > 0.0 else 0
    n_straight_total = n_points - 4 * n_arc

    pieces: list[np.ndarray] = []
    for k in range(4):
        if r > 0.0:
            angle = arc_starts[k] + np.linspace(0.0, 0.5 * np.pi, n_arc + 1)[:-1]
            pieces.append(
                corners[k] + r * np.column_stack((np.cos(angle), np.sin(angle)))
            )
        else:
            pieces.append(corners[k][None, :])

        # Straight run from the end of this corner to the start of the next.
        share = edge_lengths[k] / straight_total
        n_edge = max(2, int(round(n_straight_total * share)))
        start = corners[k] + _arc_end_offset(arc_starts[k], r)
        t = _clustered_unit_interval(n_edge + 1, corner_clustering)[:-1]
        if r == 0.0:
            t = t[1:]  # the corner itself was already emitted above
        pieces.append(start + np.outer(t * edge_lengths[k], edge_dirs[k]))

    name = "square" if np.isclose(width, height) else "rectangle"
    return Contour(
        np.vstack(pieces),
        name=f"{name} {width:g}x{height:g}",
        reference_length=width,
    )


def _arc_end_offset(arc_start: float, r: float) -> np.ndarray:
    """Offset from a corner centre to where its fillet ends and the straight edge begins."""
    end = arc_start + 0.5 * np.pi
    return r * np.array([np.cos(end), np.sin(end)])


def _clustered_unit_interval(n: int, strength: float) -> np.ndarray:
    """``n`` points on [0, 1], cosine-clustered towards both ends.

    ``strength`` blends between uniform spacing (0) and full cosine (1), so the
    caller can dial the clustering back if it produces cells that are too
    stretched next to the corner.
    """
    uniform = np.linspace(0.0, 1.0, n)
    cosine = 0.5 * (1.0 - np.cos(np.pi * uniform))
    return (1.0 - strength) * uniform + strength * cosine
