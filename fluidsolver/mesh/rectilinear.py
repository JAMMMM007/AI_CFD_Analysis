"""Structured grids on a rectangle, for geometry an O-grid cannot wrap.

The O-grid exists because external aerodynamics is mostly flow around a closed
body, and around a closed body it is the right construction: it marches from the
surface, it is orthogonal at the wall by definition, and on a circle it is exact.
What it cannot do is a domain with two open ends -- a flat plate, a channel, a
bump -- because it closes the ``i`` direction into a loop.

Those are exactly the shapes the NASA Turbulence Modeling Resource's verification
cases have. ``2DZP`` is a zero-pressure-gradient flat plate with a symmetry plane
upstream of the leading edge, inflow, outflow and a far field; ``2DB`` is a bump
in a channel. Neither can be meshed here without an open ``i``, which is why
``Metrics`` and the operators learned about topology first and this module comes
after them.

**Nothing here marches.** The grid is a tensor product of two node
distributions, so it cannot fold, its metrics are analytic, and every cell is a
rectangle wherever the distributions are uniform. That is a virtue for a
verification case rather than a limitation: the mesh should not be a variable
when what is being verified is the discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fluidsolver.mesh.hyperbolic import MeshError
from fluidsolver.mesh.spacing import geometric_layers


@dataclass
class RectilinearGrid:
    """A structured grid on a rectangle, open in ``i``.

    Carries the same surface ``OGrid`` does -- ``nodes``, ``shape``, ``n_cells``,
    a name, a reference length and a moment reference -- so that ``Case`` can
    take either without knowing which. What it adds is ``periodic_i``, which is
    False here and True there, and which is what everything downstream keys off.

    ``nodes`` is ``(Ni+1, Nj+1, 2)``: one more node line than cells in *both*
    directions, because neither wraps. The O-grid's ``(Ni, Nj+1, 2)`` has one per
    cell in ``i`` precisely because the last is shared with the first.
    """

    nodes: np.ndarray
    name: str = "rectangle"
    reference_length: float = 1.0
    #: Where a moment is taken about. The leading edge for a plate, which is what
    #: a boundary-layer case would quote a moment about if it quoted one.
    moment_reference: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )
    notes: list[str] = field(default_factory=list)
    #: Which ``j = 0`` faces are solid wall, or ``None`` if all of them are.
    #: Carried by the grid because it is a fact about the geometry -- where the
    #: plate starts -- and both the metrics and the boundary conditions need it.
    wall_mask: np.ndarray | None = None

    #: The ``i`` direction has two ends rather than wrapping. See
    #: :attr:`fluidsolver.mesh.metrics.Metrics.periodic_i`.
    periodic_i: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        """``(Ni, Nj)`` -- cells along the wall, cells in the wall-normal direction."""
        return self.nodes.shape[0] - 1, self.nodes.shape[1] - 1

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def wall(self) -> np.ndarray:
        """``(Ni+1, 2)`` nodes on the ``j = 0`` boundary."""
        return self.nodes[:, 0]


def rectilinear_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``(len(x), len(y), 2)`` nodes of the tensor product of two distributions.

    Both must increase, which is the only way the result can have positive cell
    volumes -- and checking it here means the mesh quality report never has to
    explain an inverted rectangle.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for name, values in (("x", x), ("y", y)):
        if values.ndim != 1 or len(values) < 2:
            raise MeshError(f"{name} needs at least two nodes, got {len(values)}")
        if not np.all(np.diff(values) > 0.0):
            raise MeshError(f"{name} node positions must increase")
    return np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)


def wall_normal_nodes(first_layer: float, height: float, growth: float = 1.15):
    """Node positions from a wall at ``y = 0`` out to ``height``.

    The same geometric distribution the O-grid marches with, so a boundary layer
    resolved to a given ``y+`` here is resolved the same way there and the two
    are comparable.
    """
    return np.concatenate(([0.0], np.cumsum(geometric_layers(first_layer, height, growth))))


def clustered_nodes(
    start: float, end: float, count: int, first_spacing: float
) -> np.ndarray:
    """``count + 1`` nodes from ``start`` to ``end``, finest at ``start``.

    A geometric distribution again, solved for the growth ratio that fits
    ``count`` cells into the span with the first one of the requested size. Used
    along a plate, where the boundary layer is thinnest at the leading edge and
    the streamwise resolution should follow it.

    ``first_spacing`` of zero or a span that a uniform distribution already
    matches gives uniform spacing, rather than a growth ratio of one being solved
    for numerically and coming back as one plus rounding.
    """
    span = end - start
    if span <= 0.0:
        raise MeshError(f"the span from {start} to {end} is not positive")
    if count < 1:
        raise MeshError(f"need at least one cell, got {count}")

    uniform = span / count
    if first_spacing <= 0.0 or np.isclose(first_spacing, uniform):
        return start + uniform * np.arange(count + 1)
    if first_spacing >= span:
        raise MeshError(
            f"a first spacing of {first_spacing:g} does not fit in a span of {span:g}"
        )

    # sum_{k<count} first * r^k = span, solved for r by bisection. The sum is
    # monotone in r, so bisection cannot miss and needs no derivative.
    def total(ratio: float) -> float:
        if np.isclose(ratio, 1.0):
            return first_spacing * count
        return first_spacing * (ratio**count - 1.0) / (ratio - 1.0)

    low, high = (1.0, 2.0) if first_spacing < uniform else (0.5, 1.0)
    while total(high) < span if first_spacing < uniform else total(low) > span:
        if first_spacing < uniform:
            high *= 2.0
            if high > 1e6:
                raise MeshError("no growth ratio spans this distance")
        else:
            low *= 0.5
            if low < 1e-6:
                raise MeshError("no growth ratio spans this distance")

    for _ in range(200):
        middle = 0.5 * (low + high)
        if total(middle) < span:
            low = middle
        else:
            high = middle
    ratio = 0.5 * (low + high)

    steps = first_spacing * ratio ** np.arange(count)
    # Rescale so the last node lands exactly on ``end`` rather than within the
    # bisection's tolerance of it: a boundary that is nearly where it was asked
    # to be is a boundary in the wrong place.
    steps *= span / steps.sum()
    return start + np.concatenate(([0.0], np.cumsum(steps)))


def flat_plate_grid(
    *,
    plate_length: float = 2.0,
    upstream: float = 0.33,
    height: float = 1.0,
    first_layer: float = 1.0e-5,
    growth: float = 1.15,
    upstream_cells: int = 24,
    plate_cells: int = 128,
) -> RectilinearGrid:
    """The NASA TMR ``2DZP`` geometry: a symmetry plane, then a flat plate.

    The domain runs from ``-upstream`` to ``plate_length`` in ``x`` and from the
    wall to ``height`` in ``y``. The plate starts at ``x = 0``; everything ahead
    of it is a symmetry plane, which is what keeps the leading edge from being a
    singularity and is why the TMR grids include it.

    The leading edge is where the two ``j = 0`` conditions meet, so a node is
    placed exactly at ``x = 0`` and the split falls on a face rather than through
    the middle of a cell. That is not cosmetic: a cell that is half symmetry and
    half wall has no consistent boundary condition at all.

    Streamwise spacing is clustered towards the leading edge from both sides,
    because that is where the boundary layer is thinnest and where ``Cf`` varies
    fastest. Wall-normal spacing is the same geometric distribution the O-grid
    marches with, so a given ``y+`` means the same thing on both.
    """
    if plate_length <= 0.0 or upstream <= 0.0 or height <= 0.0:
        raise MeshError("plate_length, upstream and height must all be positive")

    # Ahead of the leading edge, finest at x = 0: build it reversed and flip.
    ahead = clustered_nodes(0.0, upstream, upstream_cells, upstream / (4.0 * upstream_cells))
    symmetry_x = -ahead[::-1]
    plate_x = clustered_nodes(
        0.0, plate_length, plate_cells, plate_length / (8.0 * plate_cells)
    )
    x = np.concatenate((symmetry_x[:-1], plate_x))
    y = wall_normal_nodes(first_layer, height, growth)

    grid = RectilinearGrid(
        nodes=rectilinear_grid(x, y),
        name="flat plate",
        reference_length=plate_length,
        moment_reference=np.zeros(2),
    )
    grid.wall_mask = plate_wall_mask(grid)
    grid.notes.append(
        f"flat plate: symmetry from x = {-upstream:g} to 0, wall from 0 to "
        f"{plate_length:g}; {len(y) - 1} layers to y = {height:g} from a first "
        f"layer of {first_layer:g}"
    )
    return grid


def plate_wall_mask(grid: RectilinearGrid) -> np.ndarray:
    """True on the ``j = 0`` faces that are solid wall, False where symmetry.

    The face centres decide, so the mask follows the mesh rather than being
    computed twice from the geometry that built it.
    """
    centres = 0.5 * (grid.nodes[:-1, 0] + grid.nodes[1:, 0])
    return centres[:, 0] > 0.0
