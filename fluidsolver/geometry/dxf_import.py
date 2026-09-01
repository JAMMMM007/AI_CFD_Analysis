"""Import a body contour from a DXF drawing.

A DXF is a bag of disconnected entities, not a boundary. Turning one into
something a mesher can use means three things: flattening every curve type to
line segments, chaining those segments into closed loops across the small gaps
CAD packages leave behind, and deciding which loop is the body.

Everything here is deliberately tolerant on input and strict on output: whatever
comes back is a validated :class:`~fluidsolver.geometry.Contour`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fluidsolver.geometry.contour import Contour, ContourError

# $INSUNITS header codes, mapped to metres. Anything not listed (unitless,
# astronomical) leaves the drawing unscaled and is reported as unknown.
_INSUNITS_TO_METRES = {
    1: 0.0254,        # inches
    2: 0.3048,        # feet
    4: 1e-3,          # millimetres
    5: 1e-2,          # centimetres
    6: 1.0,           # metres
    7: 1e3,           # kilometres
    10: 0.9144,       # yards
    13: 1e-6,         # microns
    14: 1e-1,         # decimetres
}
_INSUNITS_NAMES = {
    0: "unitless", 1: "inches", 2: "feet", 3: "miles", 4: "millimetres",
    5: "centimetres", 6: "metres", 7: "kilometres", 10: "yards",
    13: "microns", 14: "decimetres",
}

# Entity types that can bound a region. Anything else -- text, dimensions,
# hatches, points -- is ignored rather than treated as an error, because real
# drawings are full of annotation that has nothing to do with the profile.
_BOUNDARY_TYPES = "LINE ARC CIRCLE ELLIPSE LWPOLYLINE POLYLINE SPLINE"


class DxfImportError(ContourError):
    """Raised when a DXF cannot yield a usable body contour."""


def describe_dxf(path: str | Path) -> dict:
    """Summarise a DXF without committing to an import.

    The setup page calls this to populate the layer list and to show what was
    detected, so the user can pick a layer and confirm the units before anything
    is meshed.
    """
    doc = _read_document(path)
    msp = doc.modelspace()

    counts: dict[str, int] = {}
    layers: dict[str, int] = {}
    for entity in _boundary_entities(msp):
        counts[entity.dxftype()] = counts.get(entity.dxftype(), 0) + 1
        layer = str(entity.dxf.layer)
        layers[layer] = layers.get(layer, 0) + 1

    code = int(doc.header.get("$INSUNITS", 0))
    return {
        "path": str(path),
        "entity_counts": counts,
        "layers": layers,
        "insunits_code": code,
        "units": _INSUNITS_NAMES.get(code, f"code {code}"),
        "metres_per_unit": _INSUNITS_TO_METRES.get(code),
    }


def read_contours(
    path: str | Path,
    *,
    layers: list[str] | None = None,
    gap_tolerance: float | None = None,
    flatten_tolerance: float = 1e-4,
    scale: float | None = None,
    min_area_fraction: float = 1e-4,
) -> list[Contour]:
    """Extract every closed loop from a DXF, largest enclosed area first.

    Parameters
    ----------
    layers
        Restrict to these layer names. Drawings routinely carry the profile on
        one layer and construction lines, hatching and a title block on others.
    gap_tolerance
        Endpoints closer than this are treated as joined. Defaults to 1e-4 of the
        drawing's diagonal, which bridges the rounding gaps left by CAD exports
        without welding genuinely separate features together.
    flatten_tolerance
        Maximum sagitta when converting arcs, ellipses and splines to line
        segments, as a fraction of the drawing diagonal. The default resolves a
        full circle to a few hundred segments.
    scale
        Multiplier applied to every coordinate. Defaults to the factor implied by
        the drawing's ``$INSUNITS`` header so the result is in metres; pass 1.0 to
        keep drawing units, or an explicit number to override.
    min_area_fraction
        Loops enclosing less than this fraction of the largest loop's area are
        discarded as drawing artefacts. Area is the right filter here, not point
        count: a rectangle is stored as a four-vertex closed polyline, so a
        minimum point count would throw away the most common profile there is.

    Returns
    -------
    list[Contour]
        Validated contours, sorted by descending area. Empty if nothing closed.
    """
    doc = _read_document(path)
    entities = list(_boundary_entities(doc.modelspace(), layers))
    if not entities:
        raise DxfImportError(
            f"{path}: no boundary geometry found"
            + (f" on layer(s) {layers}" if layers else "")
            + f". Looked for {_BOUNDARY_TYPES.replace(' ', ', ')}."
        )

    diagonal = _drawing_diagonal(entities)
    polylines = _flatten_entities(entities, diagonal * flatten_tolerance)
    if not polylines:
        raise DxfImportError(f"{path}: entities were found but none produced any geometry")

    tolerance = gap_tolerance if gap_tolerance is not None else diagonal * 1e-4
    factor = _resolve_scale(doc, scale)

    contours = []
    for loop in _chain_loops(polylines, tolerance):
        try:
            contour = Contour(loop * factor, name=Path(path).stem)
            contour.validate()
        except ContourError:
            # A loop that closes geometrically but crosses itself is not a body.
            # Skip it rather than failing the whole import: the drawing may well
            # also contain the profile we actually want.
            continue
        contours.append(contour)

    contours.sort(key=lambda c: c.area, reverse=True)
    if not contours:
        return []
    cutoff = contours[0].area * min_area_fraction
    return [c for c in contours if c.area >= cutoff]


def read_contour(path: str | Path, **kwargs) -> Contour:
    """Return the single largest closed loop in a DXF.

    The convenience path for the common case of a drawing holding one profile.
    Raises if nothing closes, with a message aimed at the usual cause.
    """
    contours = read_contours(path, **kwargs)
    if not contours:
        raise DxfImportError(
            f"{path}: geometry was found but no closed loop could be assembled from it. "
            "The profile is probably left open by a gap between entity endpoints -- "
            "raise gap_tolerance, or close the outline in the CAD package."
        )
    return contours[0]


# ----------------------------------------------------------------------
# Reading and flattening
# ----------------------------------------------------------------------


def _read_document(path: str | Path):
    """Open a DXF, translating ezdxf's failures into our own error type."""
    import ezdxf
    from ezdxf import DXFError

    if not Path(path).is_file():
        raise DxfImportError(f"{path}: file not found")
    try:
        return ezdxf.readfile(str(path))
    except (DXFError, IOError, UnicodeDecodeError) as exc:
        raise DxfImportError(f"{path}: could not be read as a DXF ({exc})") from exc


def _boundary_entities(msp, layers: list[str] | None = None):
    """Yield boundary-forming entities, expanding block references in place.

    Profiles are often drawn inside a block and placed with an INSERT, so a plain
    modelspace query would come back empty on a drawing that visibly contains the
    shape. ``virtual_entities`` resolves the block's contents with the insert's
    scale, rotation and offset already applied.
    """
    wanted = set(layers) if layers else None

    for entity in msp.query(_BOUNDARY_TYPES):
        if wanted is None or str(entity.dxf.layer) in wanted:
            yield entity

    for insert in msp.query("INSERT"):
        if wanted is not None and str(insert.dxf.layer) not in wanted:
            continue
        try:
            for entity in insert.virtual_entities():
                if entity.dxftype() in _BOUNDARY_TYPES:
                    yield entity
        except Exception:
            # A malformed or missing block definition should not sink an import
            # that has perfectly good modelspace geometry alongside it.
            continue


def _drawing_diagonal(entities) -> float:
    """Bounding-box diagonal of the geometry, used to set relative tolerances."""
    from ezdxf import bbox

    try:
        extents = bbox.extents(entities, fast=True)
        if extents.has_data:
            size = extents.size
            diagonal = float(np.hypot(size.x, size.y))
            if diagonal > 0.0:
                return diagonal
    except Exception:
        pass
    return 1.0


def _flatten_entities(entities, sagitta: float) -> list[np.ndarray]:
    """Convert every entity to an (N, 2) polyline.

    ``ezdxf.path.make_path`` gives one uniform representation for lines, arcs,
    ellipses, splines and polylines, so the curve-specific tessellation maths does
    not have to be written out here. ``flattening`` then subdivides adaptively
    until no chord deviates from the true curve by more than ``sagitta``.
    """
    from ezdxf.path import make_path

    polylines = []
    for entity in entities:
        try:
            vertices = list(make_path(entity).flattening(distance=sagitta, segments=4))
        except Exception:
            continue
        if len(vertices) < 2:
            continue
        # Drop the z coordinate: this is a 2-D solver, and a profile drawn on a
        # non-zero plane projects onto xy perfectly well.
        polylines.append(np.array([(v.x, v.y) for v in vertices], dtype=float))
    return polylines


def _resolve_scale(doc, scale: float | None) -> float:
    """Coordinate multiplier: explicit if given, else implied by $INSUNITS."""
    if scale is not None:
        if scale <= 0.0:
            raise DxfImportError(f"scale must be positive, got {scale}")
        return float(scale)
    code = int(doc.header.get("$INSUNITS", 0))
    # An unset or exotic $INSUNITS means the drawing does not say what its units
    # are. Leaving it alone is the honest default; the setup page shows the
    # detected units so the user can correct it.
    return _INSUNITS_TO_METRES.get(code, 1.0)


# ----------------------------------------------------------------------
# Chaining segments into closed loops
# ----------------------------------------------------------------------


def _chain_loops(polylines: list[np.ndarray], tolerance: float) -> list[np.ndarray]:
    """Join polylines end to end and return those that close on themselves.

    Each polyline is a chain with two free ends. Starting from an unused chain,
    the walk repeatedly looks for another chain beginning (or, reversed, ending)
    within ``tolerance`` of the current head, until either the head returns to the
    start -- a closed loop -- or nothing matches, which leaves an open run that is
    discarded.

    Endpoints go into a KD-tree so the search does not degrade to comparing every
    end against every other, which matters on drawings with thousands of little
    segments.
    """
    from scipy.spatial import cKDTree

    chains = [p for p in polylines if len(p) >= 2]
    if not chains:
        return []

    loops = []
    # A polyline that is already closed needs no chaining at all. Circles and
    # closed LWPOLYLINEs arrive this way, which is the common case.
    remaining = []
    for chain in chains:
        if np.hypot(*(chain[-1] - chain[0])) <= tolerance and len(chain) >= 4:
            loops.append(chain[:-1])
        else:
            remaining.append(chain)

    if not remaining:
        return loops

    # Two rows per chain: row 2k is its start, row 2k+1 its end.
    endpoints = np.vstack([[c[0], c[-1]] for c in remaining])
    tree = cKDTree(endpoints)
    used = [False] * len(remaining)

    for seed in range(len(remaining)):
        if used[seed]:
            continue
        used[seed] = True
        loop = [remaining[seed]]
        head = remaining[seed][-1]
        start = remaining[seed][0]

        while True:
            if np.hypot(*(head - start)) <= tolerance and len(loop) > 1:
                loops.append(_concatenate(loop, tolerance))
                break

            nxt = _next_chain(tree, endpoints, used, head, tolerance, remaining)
            if nxt is None:
                break  # open run: not a body outline, so drop it
            index, piece = nxt
            used[index] = True
            loop.append(piece)
            head = piece[-1]

    return loops


def _next_chain(tree, endpoints, used, head, tolerance, chains):
    """Find an unused chain with a free end at ``head``, oriented to continue from it."""
    for row in tree.query_ball_point(head, tolerance):
        index, is_end = divmod(row, 2)
        if used[index]:
            continue
        piece = chains[index]
        # If the match landed on the chain's *end*, walk it backwards so the
        # joined-up loop keeps a single consistent direction.
        return index, (piece[::-1] if is_end else piece)
    return None


def _concatenate(pieces: list[np.ndarray], tolerance: float) -> np.ndarray:
    """Splice chained polylines, dropping the duplicated point at every join."""
    out = [pieces[0]]
    for piece in pieces[1:]:
        previous_end = out[-1][-1]
        out.append(piece[1:] if np.hypot(*(piece[0] - previous_end)) <= tolerance else piece)
    joined = np.vstack(out)
    # The final point coincides with the first: the closure is implicit in a
    # Contour, so it must not appear twice.
    if np.hypot(*(joined[-1] - joined[0])) <= tolerance:
        joined = joined[:-1]
    return joined
