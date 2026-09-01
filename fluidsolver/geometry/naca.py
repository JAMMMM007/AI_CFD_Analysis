"""NACA 4-digit aerofoil sections, generated from the analytic definition.

The 4-digit series is defined by a thickness distribution superimposed
perpendicular to a two-arc camber line. Digits ``MPXX`` mean:

    M   maximum camber, in per cent of chord
    P   chordwise position of maximum camber, in tenths of chord
    XX  maximum thickness, in per cent of chord

So ``2412`` is 2% camber at 0.4c with a 12% thick section, and ``0012`` is the
symmetric 12% section used for the turbulent validation case.
"""

from __future__ import annotations

import numpy as np

from fluidsolver.geometry.contour import Contour, ContourError

# Thickness-distribution coefficients. The last one selects the trailing edge:
#
#   -0.1015  leaves the trailing edge open, with a finite base thickness
#   -0.1036  closes it to a knife edge
#
# The open form is the default here. An O-grid wrapped round a knife edge
# collapses into a degenerate zero-area cell at the trailing edge, and the
# resulting metric singularity poisons the whole wake. The open form leaves a
# base 2 * 0.0105 * t chords thick -- 0.25% of chord for a 12% section -- which
# is enough for the mesh to wrap cleanly and is closer to a real manufactured
# trailing edge anyway.
_A0, _A1, _A2, _A3 = 0.2969, -0.1260, -0.3516, 0.2843
_A4_OPEN, _A4_CLOSED = -0.1015, -0.1036


def naca4(
    code: str,
    n_points: int = 257,
    *,
    chord: float = 1.0,
    closed_trailing_edge: bool = False,
    n_trailing_edge: int = 4,
) -> Contour:
    """Build a NACA 4-digit section as a closed contour.

    Points are cosine-clustered along the chord, ``x = (1 - cos(beta)) / 2`` for
    ``beta`` uniform on ``[0, pi]``. Uniform chordwise spacing would badly
    under-resolve the leading-edge radius, where the surface curvature and the
    pressure gradient are both largest.

    Note this is the textbook distribution, chosen so the output can be compared
    against published coordinate tables. It is not a graded mesh distribution:
    ``dx/dbeta`` vanishes at both ends, so the cells right at the trailing edge
    come out very small relative to their neighbours. The mesher does not consume
    this directly -- it calls :meth:`~fluidsolver.geometry.Contour.resample` to
    build the wall line with a bounded sizing field. Generate at a high
    ``n_points`` and let the resampler grade it.

    The loop starts at the upper trailing edge, runs forward over the upper
    surface to the leading edge, back along the lower surface, and closes across
    the trailing-edge base. Starting the seam at the trailing edge is the usual
    convention and puts the mesh's periodic boundary in the wake, away from the
    region where accuracy matters most.

    Parameters
    ----------
    code
        Four digits, e.g. ``"2412"``.
    n_points
        Approximate total number of points on the closed loop.
    chord
        Chord length. Also becomes the contour's reference length, so the
        Reynolds number and the force coefficients are chord-based.
    closed_trailing_edge
        Use the knife-edge coefficient instead. Provided for comparison against
        published coordinates; not recommended for meshing (see above).
    n_trailing_edge
        Number of segments spanning the blunt trailing-edge base. Ignored when
        ``closed_trailing_edge`` is set.

    Returns
    -------
    Contour
        Counter-clockwise, validated, with ``reference_length == chord``.
    """
    m, p, t = _parse_code(code)
    if chord <= 0.0:
        raise ContourError(f"chord must be positive, got {chord}")

    n_base = 1 if closed_trailing_edge else max(2, n_trailing_edge)
    # The loop holds n_side points per surface sharing one leading-edge point,
    # plus the interior points of the trailing-edge base.
    n_side = max(20, (n_points - (n_base - 1) + 1) // 2)

    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_side)))
    y_t = _thickness(x, t, closed_trailing_edge)
    y_c, dy_c = _camber(x, m, p)

    # The thickness is applied normal to the camber line, not vertically. For a
    # cambered section this shifts the surface points in x as well as y.
    theta = np.arctan(dy_c)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    upper = np.column_stack((x - y_t * sin_t, y_c + y_t * cos_t))
    lower = np.column_stack((x + y_t * sin_t, y_c - y_t * cos_t))

    # Upper surface reversed runs trailing edge -> leading edge, which is
    # counter-clockwise; the lower surface then returns leading edge -> trailing
    # edge. Slicing off lower[0] drops the shared leading-edge point.
    loop = np.vstack((upper[::-1], lower[1:]))

    if not closed_trailing_edge:
        loop = np.vstack((loop, _base_points(loop[-1], loop[0], n_base)))

    return Contour(
        loop * chord,
        name=f"NACA {code}",
        reference_length=chord,
    )


def _parse_code(code: str) -> tuple[float, float, float]:
    """Split ``"2412"`` into camber, camber position and thickness as fractions of chord."""
    digits = str(code).strip()
    if len(digits) != 4 or not digits.isdigit():
        raise ContourError(f"NACA 4-digit code must be exactly four digits, got {code!r}")

    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0

    if t <= 0.0:
        raise ContourError(f"NACA {code}: thickness must be greater than zero")
    # A non-zero camber with its maximum at the leading edge divides by zero in
    # the camber-line formula, and describes no real section.
    if m > 0.0 and p <= 0.0:
        raise ContourError(
            f"NACA {code}: camber is non-zero but its position is 0; use a code like 2412"
        )
    return m, p, t


def _thickness(x: np.ndarray, t: float, closed: bool) -> np.ndarray:
    """Half-thickness distribution, measured normal to the camber line.

    ``y_t = 5t (a0 sqrt(x) + a1 x + a2 x^2 + a3 x^3 + a4 x^4)``

    The leading square-root term gives the section its finite leading-edge radius
    of ``1.1019 t^2``; note its slope is infinite at ``x = 0``, which is exactly
    why the chordwise points are cosine-clustered there.
    """
    a4 = _A4_CLOSED if closed else _A4_OPEN
    return 5.0 * t * (
        _A0 * np.sqrt(x) + _A1 * x + _A2 * x**2 + _A3 * x**3 + a4 * x**4
    )


def _camber(x: np.ndarray, m: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Camber line and its slope: two parabolic arcs meeting with matched slope at ``x = p``.

    Fore of the crest and aft of it,

        y_c = (m / p^2)(2 p x - x^2)                     0 <= x <= p
        y_c = (m / (1-p)^2)((1 - 2p) + 2 p x - x^2)      p <= x <= 1

    and both branches share the slope ``dy_c/dx = 2m (p - x) / <denominator>``,
    so the surface has no kink at the crest.
    """
    if m == 0.0:
        return np.zeros_like(x), np.zeros_like(x)

    fore = x <= p
    denom = np.where(fore, p**2, (1.0 - p) ** 2)

    y_c = np.where(
        fore,
        m / denom * (2.0 * p * x - x**2),
        m / denom * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
    )
    dy_c = 2.0 * m / denom * (p - x)
    return y_c, dy_c


def _base_points(lower_te: np.ndarray, upper_te: np.ndarray, n_base: int) -> np.ndarray:
    """Interior points spanning the blunt trailing-edge base.

    The endpoints already exist as the last and first points of the loop, so only
    the ``n_base - 1`` points strictly between them are returned. Spreading a few
    points across the base keeps the wall cells there comparable in size to their
    neighbours instead of leaving one long face against many short ones.
    """
    fractions = np.linspace(0.0, 1.0, n_base + 1)[1:-1]
    return lower_te + fractions[:, None] * (upper_te - lower_te)
