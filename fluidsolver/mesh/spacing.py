"""Wall-normal spacing: how thick the first cell must be, and how the rest grow.

The first cell height is not a free meshing parameter. Integrating k-omega SST to
the wall requires the first cell centre to sit at ``y+ ~ 1``, inside the viscous
sublayer, because that is where the model's wall boundary condition
``omega_w = 6 nu / (beta1 d1^2)`` is asymptotically valid. Put the first cell at
``y+ = 30`` instead and the model is being asked to resolve a sublayer it never
sees, which shows up as wrong skin friction and, on an aerofoil, a wrong
separation point.

So the mesher works backwards: from the flow conditions and a target ``y+``, it
computes the physical spacing that lands the first cell centre there.
"""

from __future__ import annotations

import numpy as np


class SpacingError(ValueError):
    """Raised when a wall-normal distribution cannot be built as requested."""


def friction_velocity(
    velocity: float, length: float, density: float, viscosity: float
) -> float:
    """Estimate the friction velocity ``u_tau`` from a flat-plate correlation.

    Uses the 1/7-power turbulent skin-friction law

        C_f = 0.026 Re^(-1/7),   tau_w = C_f rho U^2 / 2,   u_tau = sqrt(tau_w / rho)

    This is only an estimate -- the real ``u_tau`` varies along the surface and
    depends on the pressure gradient, which is not known until the case is
    solved. It is the standard way to *size* a mesh, and it is accurate enough
    for that: a factor-of-two error in ``u_tau`` moves ``y+`` by the same factor,
    which a ``y+ = 1`` target absorbs comfortably.

    The solver reports the ``y+`` distribution actually achieved, and that is the
    number to trust.
    """
    _require_positive(velocity=velocity, length=length, density=density, viscosity=viscosity)

    reynolds = density * velocity * length / viscosity
    skin_friction = 0.026 * reynolds ** (-1.0 / 7.0)
    wall_shear = 0.5 * skin_friction * density * velocity**2
    return float(np.sqrt(wall_shear / density))


def first_layer_thickness(
    y_plus: float,
    velocity: float,
    length: float,
    density: float,
    viscosity: float,
) -> float:
    """Thickness of the wall-adjacent cell that puts its *centre* at ``y_plus``.

    ``y+ = rho u_tau y / mu`` is defined at a distance ``y`` from the wall. On a
    cell-centred finite-volume mesh the first solution point is at half the cell
    height, so the cell must be twice as thick as the distance implied by ``y+``.
    Forgetting that factor of two is a common way to end up with ``y+ ~ 2`` while
    believing the mesh delivers 1.
    """
    if y_plus <= 0.0:
        raise SpacingError(f"y_plus must be positive, got {y_plus}")

    u_tau = friction_velocity(velocity, length, density, viscosity)
    centre_distance = y_plus * viscosity / (density * u_tau)
    return float(2.0 * centre_distance)


def y_plus_of(
    layer_thickness: float,
    velocity: float,
    length: float,
    density: float,
    viscosity: float,
) -> float:
    """Inverse of :func:`first_layer_thickness`: the ``y+`` a given first cell implies."""
    u_tau = friction_velocity(velocity, length, density, viscosity)
    return float(0.5 * layer_thickness * density * u_tau / viscosity)


def laminar_first_layer(
    velocity: float,
    length: float,
    density: float,
    viscosity: float,
    *,
    cells_in_layer: int = 25,
    growth: float = 1.15,
) -> float:
    """First cell thickness that puts ``cells_in_layer`` cells inside a laminar
    boundary layer.

    The ``y+`` route is a *turbulent* correlation and is meaningless here. Applied
    to a cylinder at Re = 40 it asks for a first cell more than half a diameter
    thick, because at that Reynolds number the notion of a viscous sublayer
    distinct from the boundary layer does not exist.

    What a laminar case needs instead is simply enough cells across the layer.
    Blasius gives its thickness as ``delta = 5 L / sqrt(Re)``, and distributing
    ``n`` geometrically stretched cells across that gives the first one as
    ``delta (r - 1) / (r^n - 1)``.
    """
    _require_positive(velocity=velocity, length=length, density=density, viscosity=viscosity)
    if cells_in_layer < 3:
        raise SpacingError(f"need at least 3 cells in the layer, got {cells_in_layer}")

    reynolds = density * velocity * length / viscosity
    thickness = 5.0 * length / np.sqrt(reynolds)

    if np.isclose(growth, 1.0):
        return float(thickness / cells_in_layer)
    return float(thickness * (growth - 1.0) / (growth**cells_in_layer - 1.0))


# ----------------------------------------------------------------------
# Wall-normal distributions
# ----------------------------------------------------------------------


def geometric_layers(
    first: float, total: float, growth: float, *, max_layers: int = 100_000
) -> np.ndarray:
    """Layer thicknesses growing geometrically from ``first`` out to ``total``.

    A geometric series is the right shape for a boundary-layer mesh: cells stay
    small where the gradients are steep and coarsen smoothly outward, and a
    constant ratio is exactly the bounded-gradation condition the discretisation
    wants.

    The layer count follows from summing the series,
    ``total = first (r^n - 1) / (r - 1)``, so

        n = log(1 + total (r - 1) / first) / log(r)

    rounded up, after which the ratio is nudged back down so the layers land
    exactly on ``total`` instead of overshooting it.
    """
    _require_positive(first=first, total=total)
    if growth < 1.0:
        raise SpacingError(f"growth ratio must be at least 1, got {growth}")
    if first >= total:
        raise SpacingError(
            f"first layer ({first:g}) must be thinner than the total distance ({total:g})"
        )

    if np.isclose(growth, 1.0):
        count = int(np.ceil(total / first))
    else:
        count = int(np.ceil(np.log1p(total * (growth - 1.0) / first) / np.log(growth)))

    count = max(count, 2)
    if count > max_layers:
        raise SpacingError(
            f"{count} layers needed to span {total:g} from a {first:g} first cell at "
            f"growth {growth:g}. Increase the growth ratio, relax the y+ target, or "
            f"bring the far-field boundary closer."
        )

    ratio = _ratio_for_exact_span(first, total, count, growth)
    thicknesses = first * ratio ** np.arange(count)
    # Absorb the last sliver of rounding into a uniform rescale, so the outer
    # boundary sits exactly where it was asked to.
    return thicknesses * (total / thicknesses.sum())


def _ratio_for_exact_span(first: float, total: float, count: int, guess: float) -> float:
    """Growth ratio for which ``count`` layers span exactly ``total``.

    Solves ``first (r^count - 1) / (r - 1) = total`` by bisection. The sum rises
    monotonically with r, so bisection is guaranteed to converge and cannot
    produce the negative or wildly large ratio that a Newton step occasionally
    would here.
    """
    if count * first >= total:
        return 1.0

    def span(ratio: float) -> float:
        if np.isclose(ratio, 1.0):
            return first * count
        return first * (ratio**count - 1.0) / (ratio - 1.0)

    low, high = 1.0, max(guess, 1.0) * 2.0
    while span(high) < total:
        high *= 1.5
        if high > 100.0:
            raise SpacingError("no growth ratio spans the requested distance")

    for _ in range(200):
        mid = 0.5 * (low + high)
        if span(mid) < total:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def layers_for_count(
    first: float, total: float, count: int, *, max_growth: float = 1.5
) -> np.ndarray:
    """Layer thicknesses for a *fixed* number of layers, spanning ``total`` exactly.

    Used when the far-field distance and the mesh size are both pinned and the
    growth ratio is whatever falls out. Raises if that ratio would exceed
    ``max_growth``, since beyond roughly 1.3 the truncation error from the
    stretching starts to dominate the solution.
    """
    _require_positive(first=first, total=total)
    if count < 2:
        raise SpacingError(f"need at least 2 layers, got {count}")
    if first * count > total:
        raise SpacingError(
            f"{count} layers of at least {first:g} already exceed {total:g}; "
            "use fewer layers or a thinner first cell"
        )

    ratio = _ratio_for_exact_span(first, total, count, 1.1)
    if ratio > max_growth:
        raise SpacingError(
            f"spanning {total:g} in {count} layers from a {first:g} first cell needs a "
            f"growth ratio of {ratio:.3f}, above the {max_growth:g} limit. "
            "Add layers, or move the far-field boundary closer."
        )

    thicknesses = first * ratio ** np.arange(count)
    return thicknesses * (total / thicknesses.sum())


def node_distances(thicknesses: np.ndarray) -> np.ndarray:
    """Cumulative wall distance of each grid *node*, starting at 0 on the wall."""
    return np.concatenate(([0.0], np.cumsum(thicknesses)))


def describe(thicknesses: np.ndarray) -> dict:
    """Summary of a wall-normal distribution, for the mesh page's readout."""
    return {
        "layers": int(len(thicknesses)),
        "first_layer": float(thicknesses[0]),
        "last_layer": float(thicknesses[-1]),
        "total": float(thicknesses.sum()),
        "max_growth": float(np.max(thicknesses[1:] / thicknesses[:-1]))
        if len(thicknesses) > 1
        else 1.0,
    }


def _require_positive(**values) -> None:
    for name, value in values.items():
        if not value > 0.0:
            raise SpacingError(f"{name} must be positive, got {value}")
