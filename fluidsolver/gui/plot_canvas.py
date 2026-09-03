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
from PySide6.QtCore import Qt, QTimer, Signal
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

# Colour maps offered on the run page. ``None`` means "whatever suits the field",
# which is the rule in :func:`field_style`.
COLOURMAPS = {
    "Automatic": None,
    "Red-blue": "RdBu_r",
    "Blue-red (cool-warm)": "coolwarm",
    "Red-yellow-blue": "RdYlBu_r",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Inferno": "inferno",
    "Turbo": "turbo",
    "Greyscale": "gray",
}

# How the field is painted. See :func:`draw_field` for why this is a control the
# user gets rather than a decision made for them: "Per cell" is the only honest
# view of what the solver holds, and it is the one that answers "is that
# oscillation real?", but on a graded polar mesh it renders a perfectly smooth
# field as concentric rings and radial spokes that are not in the solution.
SHADING = {
    "Smooth": "gouraud",
    "Per cell": "flat",
}

# Fields whose zero is meaningful rather than incidental. These are drawn on a
# range symmetric about zero so that the sign reads off the colour directly; on
# an autoscaled range a field that never changes sign looks identical to one that
# does, which is exactly the thing worth seeing in a wake.
SIGNED_FIELDS = frozenset({"v", "vorticity"})


def field_style(name: str, colourmap: str | None = None) -> tuple[str, bool]:
    """``(colourmap, symmetric)`` for a field, honouring an explicit choice.

    An explicit colour map only changes the colours; whether the range is
    centred on zero follows from the field itself either way.
    """
    symmetric = name in SIGNED_FIELDS
    if colourmap is None:
        colourmap = "RdBu_r" if symmetric else "viridis"
    return colourmap, symmetric


class Canvas(FigureCanvasQTAgg):
    """A matplotlib figure sized to fill its Qt parent."""

    def __init__(self, width=5.0, height=4.0, dpi=100, layout="constrained"):
        self.figure = Figure(figsize=(width, height), dpi=dpi, layout=layout)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self):
        self.figure.clear()

    def pixels(self) -> tuple[float, float]:
        """The figure size in pixels. Follows the widget as it is resized."""
        figure = self.figure
        return (
            figure.get_figwidth() * figure.dpi,
            figure.get_figheight() * figure.dpi,
        )


class InteractiveCanvas(Canvas):
    """A canvas the user can pan and zoom with the mouse.

    The wheel zooms about the cursor, a left-drag pans, and a double-click asks
    for the view to be put back where the page had it.

    The canvas only moves the axes limits; it has no idea what is drawn on them.
    It therefore tells the page whenever the view moved, and the page re-applies
    the same limits on its next redraw -- without that, every solver snapshot
    would snap a zoomed-in view back to the preset, which makes watching a wake
    develop impossible.
    """

    view_changed = Signal()
    home_requested = Signal()
    resized = Signal()

    #: Room left around the plot, in pixels: ``(left, right, bottom, top)``. The
    #: left and bottom hold the tick labels, the top the title, and the right the
    #: colour bar with its own labels.
    margins = (66, 122, 48, 40)

    #: The colour bar: how wide it is, and how far it sits from the plot.
    colourbar = (22, 16)

    # One wheel click. Large enough to get somewhere, small enough to stop on.
    zoom_step = 1.3

    # How long the widget has to sit still after a resize before the page is
    # asked to redraw. Dragging a window edge produces a resize event per pixel,
    # and a full field redraw cannot keep up with that.
    resize_settle_ms = 150

    def __init__(self, width=5.0, height=4.0, dpi=100):
        # Deliberately not a constrained layout. Constrained layout sizes the
        # cell from the axes and the axes from the cell, so with a fixed aspect
        # a page that fits its view to the space it measures chases its own
        # tail: the view grows slightly on every redraw and the plot zooms
        # itself out. Fixed margins make the space a plain function of the
        # canvas size, which the page can work out before it draws anything.
        super().__init__(width, height, dpi, layout=None)
        self._pan = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(self.resize_settle_ms)
        self._resize_timer.timeout.connect(self.resized)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.OpenHandCursor)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        # Qt resizes the widget while it is being constructed, before this
        # subclass has its timer.
        timer = getattr(self, "_resize_timer", None)
        if timer is not None:
            timer.start()

    # -- what the page asks of it --------------------------------------

    def plot_aspect(self) -> float | None:
        """Width over height of the space the plot gets, or ``None`` if none.

        Known without drawing anything, which is the point of the fixed margins:
        the page can fit its view to the canvas before it plots into it.
        """
        width, height = self.pixels()
        left, right, bottom, top = self.margins
        span_x, span_y = width - left - right, height - bottom - top
        if span_x < 1.0 or span_y < 1.0:
            return None
        return span_x / span_y

    def field_axes(self):
        """Fresh plot and colour-bar axes, laid out at the fixed margins."""
        width, height = self.pixels()
        left, right, bottom, top = self.margins
        bar_width, gap = self.colourbar
        span_x = max(1.0, width - left - right)
        span_y = max(1.0, height - bottom - top)

        axes = self.figure.add_axes(
            (left / width, bottom / height, span_x / width, span_y / height)
        )
        bar = self.figure.add_axes(
            (
                (left + span_x + gap) / width, bottom / height,
                bar_width / width, span_y / height,
            )
        )
        return axes, bar

    def plot_axes(self):
        """The axes the field is drawn on, or ``None`` before the first draw.

        The colour bar is an axes too, and it is added second, so the first one
        is always the plot.
        """
        return self.figure.axes[0] if self.figure.axes else None

    def current_bounds(self) -> tuple[float, float, float, float] | None:
        axes = self.plot_axes()
        if axes is None:
            return None
        (left, right), (bottom, top) = axes.get_xlim(), axes.get_ylim()
        return (left, right, bottom, top)

    def zoom_by(self, factor: float) -> None:
        """Zoom about the middle of the view. ``factor < 1`` zooms in."""
        axes = self.plot_axes()
        if axes is None:
            return
        left, right = axes.get_xlim()
        bottom, top = axes.get_ylim()
        self._zoom(axes, factor, 0.5 * (left + right), 0.5 * (bottom + top))

    def clear(self):
        # A redraw replaces the axes, so a pan in progress is now about an axes
        # that is no longer in the figure.
        self._end_pan()
        super().clear()

    # -- mouse ----------------------------------------------------------

    def _zoom(self, axes, factor: float, x: float, y: float) -> None:
        left, right = axes.get_xlim()
        bottom, top = axes.get_ylim()
        axes.set_xlim(x + (left - x) * factor, x + (right - x) * factor)
        axes.set_ylim(y + (bottom - y) * factor, y + (top - y) * factor)
        self.draw_idle()
        self.view_changed.emit()

    def _on_scroll(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        factor = 1.0 / self.zoom_step if event.button == "up" else self.zoom_step
        self._zoom(event.inaxes, factor, event.xdata, event.ydata)

    def _on_press(self, event) -> None:
        if event.dblclick:
            self.home_requested.emit()
            return
        if event.button != 1 or event.inaxes is None:
            return

        axes = event.inaxes
        box = axes.bbox
        if box.width <= 0 or box.height <= 0:
            return

        # The pixels-per-unit scale is captured now and held for the whole drag.
        # Reading it back from the transform after each move would be wrong: the
        # transform changes as the limits do, and the view would accelerate away
        # from the cursor.
        left, right = axes.get_xlim()
        bottom, top = axes.get_ylim()
        self._pan = (
            axes, event.x, event.y, (left, right), (bottom, top),
            (right - left) / box.width, (top - bottom) / box.height,
        )
        self.setCursor(Qt.ClosedHandCursor)

    def _on_motion(self, event) -> None:
        if self._pan is None or event.x is None:
            return
        axes, x, y, (left, right), (bottom, top), scale_x, scale_y = self._pan
        if axes not in self.figure.axes:
            self._end_pan()
            return

        dx = (event.x - x) * scale_x
        dy = (event.y - y) * scale_y
        axes.set_xlim(left - dx, right - dx)
        axes.set_ylim(bottom - dy, top - dy)
        self.draw_idle()
        self.view_changed.emit()

    def _on_release(self, event) -> None:
        if self._pan is not None:
            self._end_pan()
            self.view_changed.emit()

    def _end_pan(self) -> None:
        self._pan = None
        self.setCursor(Qt.OpenHandCursor)


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


def field_limits(
    values: np.ndarray, *, symmetric: bool = False
) -> tuple[float, float] | None:
    """The colour range for a field, or ``None`` if it holds nothing finite.

    The extremes are trimmed at the 1st and 99th percentiles. A single cell at a
    stagnation point or a stray spike on a skewed cell would otherwise take the
    whole range and leave the rest of the field one flat colour.
    """
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None

    low, high = np.percentile(finite, [1.0, 99.0])
    if symmetric:
        extreme = max(abs(low), abs(high))
        low, high = -extreme, extreme
    if high <= low:
        low, high = float(finite.min()), float(finite.min()) + 1.0
    return float(low), float(high)


def _cell_centred_grid(
    nodes: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates and values co-located at the data points, for gouraud shading.

    Gouraud wants one coordinate per value rather than the corners of each cell,
    so the cell centres are used. Two things have to be put back or the picture
    develops holes where the mesh does not:

    * The wall and far-field rows are re-added from the boundary face centres,
      carrying the adjacent cell's value. Without them the plot stops half a cell
      short of each boundary -- invisible at the wall, where the first cell is
      microns thick, but a bare ring a dozen reference lengths wide at the far
      field.
    * The ``i`` seam is closed, as it is for flat shading. Leaving it open
      strands a wedge one cell wide, and since ``i = 0`` sits on the downstream
      axis for a circle, that wedge lands straight down the wake.
    """
    corners = close_seam(nodes)
    centres = 0.25 * (
        corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:]
    )
    wall = 0.5 * (corners[:-1, 0] + corners[1:, 0])
    far_field = 0.5 * (corners[:-1, -1] + corners[1:, -1])

    points = np.concatenate((wall[:, None], centres, far_field[:, None]), axis=1)
    field = np.concatenate((values[:, :1], values, values[:, -1:]), axis=1)

    points = close_seam(points)
    field = close_seam(field)
    return points[..., 0], points[..., 1], field


def draw_field(
    axes,
    nodes: np.ndarray,
    values: np.ndarray,
    *,
    label: str = "",
    colourmap: str = "RdYlBu_r",
    levels: int = 40,
    symmetric: bool = False,
    limits: tuple[float, float] | None = None,
    colourbar_axes=None,
    shading: str = "gouraud",
):
    """Filled contours of a cell field on the curvilinear mesh.

    ``pcolormesh`` takes the finite-volume data directly, without interpolating
    it onto a triangulation first, so what is shown is what the solver holds.

    ``shading`` chooses how it is painted, and the choice matters more than it
    looks. ``"flat"`` gives every cell one solid colour -- honest, in that no
    value on screen was invented, but on a polar O-grid whose radial cells grow
    15% per layer it draws a smooth field as a set of concentric rings, and a
    smooth wake as a fan of radial spokes, one per surface point. Those are
    entirely an artefact of the painting: measured on a cylinder at Re = 2e6, the
    cell-to-cell alternating content of ``|U|`` is 0.001 to 0.01% of freestream,
    while the rings read as several m/s. ``"gouraud"`` interpolates between cell
    centres and shows the same data smooth, so it is the default; ``"flat"``
    stays available because seeing the actual cells is exactly what is wanted
    when the question is whether an oscillation is real.

    ``limits`` fixes the colour range instead of taking it from these values.
    Replaying a run needs that: a range recomputed per frame rescales as the
    solution develops, so the colours shift under a field that is not changing
    and nothing can be compared between one frame and the next.
    """
    if limits is None:
        limits = field_limits(values, symmetric=symmetric)
    if limits is None:
        return None
    low, high = limits

    if shading == "gouraud":
        x, y, field = _cell_centred_grid(nodes, values)
    else:
        # Only the *nodes* get the seam closed. Adding a row of nodes turns the
        # Ni node rows into Ni+1, which is exactly the one-more-than-the-cells
        # that pcolormesh wants; closing the value array too would add a cell
        # that does not exist.
        x = close_seam(nodes[..., 0])
        y = close_seam(nodes[..., 1])
        field = values

    mesh = axes.pcolormesh(
        x, y, field, cmap=colourmap, vmin=low, vmax=high, shading=shading, rasterized=True
    )
    if label:
        if colourbar_axes is not None:
            axes.figure.colorbar(mesh, cax=colourbar_axes, label=label)
        else:
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


def field_values(case, name: str, state=None) -> np.ndarray:
    """Extract a named field from a case, computing the derived ones.

    ``state`` defaults to the case's own, and is passed explicitly to draw a
    snapshot -- a replay frame, or the copy handed over by the solver thread.
    The case is used only for the geometry and the operators, which do not
    change during a run, so a snapshot can be evaluated against it safely while
    the solver keeps working on the live state.
    """
    state = case.state if state is None else state
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
