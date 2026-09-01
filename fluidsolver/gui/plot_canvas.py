"""Matplotlib canvases embedded in Qt, and the field-plotting they do.

Plotting a solution on this mesh needs a little care in two places.

The mesh is *periodic* in ``i``, and neither ``pcolormesh`` nor a contour routine
knows that. Left alone they leave a wedge-shaped gap along the seam where the
grid wraps, which lands in the wake of an aerofoil -- exactly where someone is
looking. The node array is closed explicitly before drawing.

Streamlines need a regular grid, which a body-fitted mesh is the opposite of. The
velocity is resampled onto a Cartesian grid for that one purpose, with points
inside the body masked out so streamlines do not run through the aerofoil.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy

# Fields offered on the run page, with the label and the accessor.
FIELDS = {
    "Velocity magnitude": ("speed", "|U|  [m/s]"),
    "x velocity": ("u", "u  [m/s]"),
    "y velocity": ("v", "v  [m/s]"),
    "Pressure": ("pressure", "p  [Pa]"),
    "Pressure coefficient": ("cp", "Cp"),
    "Turbulent kinetic energy": ("k", "k  [m2/s2]"),
    "Specific dissipation": ("omega", "omega  [1/s]"),
    "Eddy viscosity ratio": ("viscosity_ratio", "mu_t / mu"),
    "Vorticity": ("vorticity", "dv/dx - du/dy  [1/s]"),
}


class Canvas(FigureCanvasQTAgg):
    """A matplotlib figure sized to fill its Qt parent."""

    def __init__(self, width=5.0, height=4.0, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self):
        self.figure.clear()


def close_seam(array: np.ndarray) -> np.ndarray:
    """Repeat the first row so a periodic array draws without a gap."""
    return np.concatenate((array, array[:1]), axis=0)


def draw_mesh(
    axes,
    nodes: np.ndarray,
    *,
    stride_i: int = 1,
    stride_j: int = 1,
    colour: str = "#3b6ea5",
    width: float = 0.3,
) -> None:
    """Draw grid lines. Strides thin them out for a zoomed-out view."""
    from matplotlib.collections import LineCollection

    closed = close_seam(nodes)
    segments = []
    for j in range(0, closed.shape[1], stride_j):
        line = closed[::stride_i, j]
        segments.extend(zip(line[:-1], line[1:]))
    for i in range(0, closed.shape[0] - 1, stride_i):
        line = closed[i]
        segments.extend(zip(line[:-1], line[1:]))

    axes.add_collection(
        LineCollection(segments, linewidths=width, colors=colour, alpha=0.85)
    )


def draw_body(axes, wall: np.ndarray, colour: str = "#c0392b", width: float = 1.4) -> None:
    closed = close_seam(wall)
    axes.plot(closed[:, 0], closed[:, 1], lw=width, color=colour, zorder=5)
    axes.fill(closed[:, 0], closed[:, 1], color="white", zorder=4)


def draw_field(
    axes,
    nodes: np.ndarray,
    values: np.ndarray,
    *,
    label: str = "",
    colourmap: str = "RdYlBu_r",
    levels: int = 40,
    symmetric: bool = False,
):
    """Filled contours of a cell field on the curvilinear mesh.

    ``pcolormesh`` takes node coordinates and cell values directly, which is
    exactly the finite-volume data layout, and unlike a contour routine it does
    not interpolate the field onto a triangulation first -- so what is shown is
    what the solver actually holds.
    """
    # Only the *nodes* get the seam closed. Adding a row of nodes turns the Ni
    # node rows into Ni+1, which is exactly the one-more-than-the-cells that
    # pcolormesh wants; closing the value array too would add a cell that does
    # not exist.
    x = close_seam(nodes[..., 0])
    y = close_seam(nodes[..., 1])
    field = values

    finite = field[np.isfinite(field)]
    if len(finite) == 0:
        return None
    low, high = np.percentile(finite, [1.0, 99.0])
    if symmetric:
        extreme = max(abs(low), abs(high))
        low, high = -extreme, extreme
    if high <= low:
        low, high = float(finite.min()), float(finite.min()) + 1.0

    mesh = axes.pcolormesh(
        x, y, field, cmap=colourmap, vmin=low, vmax=high, shading="flat", rasterized=True
    )
    if label:
        axes.figure.colorbar(mesh, ax=axes, label=label, shrink=0.85)
    return mesh


def draw_streamlines(
    axes,
    centroid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    wall: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    density: float = 1.2,
    resolution: int = 220,
) -> None:
    """Streamlines, via a temporary Cartesian resampling.

    ``streamplot`` requires a regular grid. Sampling is nearest-neighbour rather
    than a smooth interpolation: it is only for drawing, and nearest sampling
    cannot invent velocity inside the body the way a smooth fit would.
    """
    from matplotlib.path import Path
    from scipy.spatial import cKDTree

    xmin, xmax, ymin, ymax = bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, resolution),
        np.linspace(ymin, ymax, resolution),
    )
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    _, index = cKDTree(centroid.reshape(-1, 2)).query(points)
    sampled_u = u.ravel()[index].reshape(grid_x.shape)
    sampled_v = v.ravel()[index].reshape(grid_x.shape)

    inside = Path(close_seam(wall)).contains_points(points).reshape(grid_x.shape)
    sampled_u = np.where(inside, np.nan, sampled_u)
    sampled_v = np.where(inside, np.nan, sampled_v)

    axes.streamplot(
        grid_x, grid_y, sampled_u, sampled_v,
        density=density, linewidth=0.6, color="#222222", arrowsize=0.7,
    )


def field_values(case, name: str) -> np.ndarray:
    """Extract a named field from a case, computing the derived ones."""
    state = case.state
    if name == "speed":
        return state.speed
    if name == "cp":
        return state.pressure / case.freestream.dynamic_pressure(case.fluid)
    if name == "viscosity_ratio":
        return state.eddy_viscosity / case.fluid.viscosity
    if name == "vorticity":
        gradient = case.coupling.gradient
        far = case.boundaries.far_velocity(state.u, state.v, state.flux_j[:, -1])
        zero = np.zeros(case.faces.shape[0])
        grad_u = gradient(state.u, zero, far[0])
        grad_v = gradient(state.v, zero, far[1])
        return grad_v[..., 0] - grad_u[..., 1]
    return getattr(state, name)
