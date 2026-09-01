"""Precomputed face geometry for the finite-volume discretisation.

Every transported quantity crosses the same faces with the same geometry, so the
geometric work is done once here and reused by momentum, pressure and both
turbulence equations.

The mesh is structured, so faces come in two families. ``i``-faces separate a cell
from its neighbour at ``i-1`` and wrap around the body; ``j``-faces separate it
from ``j-1`` and terminate on the wall at one end and the far field at the other.
Interior faces have a cell on both sides; boundary faces have one, and their far
side is supplied by a boundary condition.

Two geometric quantities do the real work, and both exist because a body-fitted
mesh is not orthogonal:

``diffusion_factor``
    The implicit part of a diffusive flux. Writing the area vector as
    ``S = g d + T`` with ``d`` the centroid-to-centroid vector, the component
    along ``d`` gives a flux proportional to ``phi_N - phi_P`` -- a two-point
    coupling the matrix can hold. This is ``g``, in the over-relaxed form
    ``g = |S|^2 / (d . S)``. The over-relaxed choice (rather than minimum-
    correction or orthogonal-correction) keeps the implicit part largest as
    non-orthogonality grows, which is what keeps the outer iteration stable on a
    skewed mesh.

``cross``
    The leftover ``T = S - g d``. It multiplies the full face gradient, which is
    not a two-point quantity, so it goes to the right-hand side and is lagged.
    On an orthogonal mesh it is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluidsolver.mesh.metrics import Metrics


@dataclass(frozen=True)
class InteriorFaces:
    """Faces with a cell on both sides.

    ``owner`` is the cell on the high-index side, ``neighbour`` the low-index one,
    and :attr:`area` points from neighbour to owner.
    """

    area: np.ndarray
    centre: np.ndarray
    delta: np.ndarray
    diffusion_factor: np.ndarray
    cross: np.ndarray
    weight: np.ndarray
    skew_offset: np.ndarray
    to_face_from_owner: np.ndarray
    to_face_from_neighbour: np.ndarray

    def interpolate(self, owner: np.ndarray, neighbour: np.ndarray) -> np.ndarray:
        """Linear interpolation to the face, weighted by centroid distance."""
        w = self.weight if owner.ndim == self.weight.ndim else self.weight[..., None]
        return w * owner + (1.0 - w) * neighbour


@dataclass(frozen=True)
class BoundaryFaces:
    """Faces with a cell on one side only.

    :attr:`area` points *out* of the fluid cell, so a positive ``u . area`` is
    outflow, and :attr:`delta` runs from the cell centroid to the face centre.
    """

    area: np.ndarray
    centre: np.ndarray
    delta: np.ndarray
    diffusion_factor: np.ndarray
    cross: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        """Unit outward normals."""
        length = np.linalg.norm(self.area, axis=-1, keepdims=True)
        return self.area / np.where(length > 0.0, length, 1.0)

    @property
    def length(self) -> np.ndarray:
        """Face lengths -- areas per unit depth."""
        return np.linalg.norm(self.area, axis=-1)

    @property
    def wall_normal_distance(self) -> np.ndarray:
        """Perpendicular distance from the cell centroid to the face.

        Not ``|delta|``: on a non-orthogonal mesh the centroid is not directly
        under the face centre, and it is the perpendicular distance that belongs
        in a wall-gradient or a ``y+``.
        """
        return np.abs(np.sum(self.delta * self.normal, axis=-1))


@dataclass(frozen=True)
class FaceGeometry:
    """All the faces of a mesh, grouped by family."""

    i_faces: InteriorFaces
    j_faces: InteriorFaces
    wall: BoundaryFaces
    far_field: BoundaryFaces
    metrics: Metrics

    @property
    def shape(self) -> tuple[int, int]:
        return self.metrics.shape


def build_faces(metrics: Metrics) -> FaceGeometry:
    """Precompute the geometry of every face in the mesh."""
    centroid = metrics.centroid

    # i-faces: owner (i, j), neighbour (i-1, j). Periodic, so every one is
    # interior and there are exactly Ni of them per row.
    i_faces = _interior(
        area=metrics.face_i_area,
        centre=metrics.face_i_centre,
        owner=centroid,
        neighbour=np.roll(centroid, 1, axis=0),
    )

    # j-faces: owner (i, j), neighbour (i, j-1), for j = 1 .. Nj-1. The first and
    # last j-faces are boundaries and are handled separately.
    j_faces = _interior(
        area=metrics.face_j_area[:, 1:-1],
        centre=metrics.face_j_centre[:, 1:-1],
        owner=centroid[:, 1:],
        neighbour=centroid[:, :-1],
    )

    # face_j points towards increasing j, so at the wall that is *into* the fluid
    # and the outward-from-the-cell direction is its negative. At the far field
    # the two already agree.
    wall = _boundary(
        area=-metrics.face_j_area[:, 0],
        centre=metrics.face_j_centre[:, 0],
        cell=centroid[:, 0],
    )
    far_field = _boundary(
        area=metrics.face_j_area[:, -1],
        centre=metrics.face_j_centre[:, -1],
        cell=centroid[:, -1],
    )

    return FaceGeometry(i_faces, j_faces, wall, far_field, metrics)


def _interior(
    area: np.ndarray, centre: np.ndarray, owner: np.ndarray, neighbour: np.ndarray
) -> InteriorFaces:
    delta = owner - neighbour
    factor, cross = _decompose(area, delta)

    # Interpolation weight, as the projection of the face centre onto the
    # centroid-to-centroid line. Zero puts the face on the neighbour, one on the
    # owner. Projecting rather than using plain distances keeps the weight
    # consistent with how the diffusion term splits the same vector.
    along = np.sum(delta * delta, axis=-1)
    weight = np.clip(
        np.sum((centre - neighbour) * delta, axis=-1)
        / np.where(along > 0.0, along, 1.0),
        0.0,
        1.0,
    )

    return InteriorFaces(
        area=area,
        centre=centre,
        delta=delta,
        diffusion_factor=factor,
        cross=cross,
        weight=weight,
        # A linearly interpolated face value is really the value at the point on
        # the centroid-to-centroid line that the weight picks out, not at the face
        # centre. On a skewed face those differ, and the gap is what a
        # skewness correction has to extrapolate across.
        skew_offset=centre - (neighbour + weight[..., None] * delta),
        # Upwind extrapolation needs the offset from whichever cell is upwind, so
        # both are kept.
        to_face_from_owner=centre - owner,
        to_face_from_neighbour=centre - neighbour,
    )


def _boundary(area: np.ndarray, centre: np.ndarray, cell: np.ndarray) -> BoundaryFaces:
    delta = centre - cell
    factor, cross = _decompose(area, delta)
    return BoundaryFaces(
        area=area, centre=centre, delta=delta, diffusion_factor=factor, cross=cross
    )


def _decompose(area: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an area vector into an implicit part along ``delta`` and a remainder.

    ``S = g d + T`` with ``g = |S|^2 / (d . S)``.

    A vanishing ``d . S`` would mean the centroid-to-centroid line lies in the
    face plane -- a degenerate cell. The mesh quality check rejects those before
    the solver ever sees them, so guarding here is only to keep a bad mesh from
    producing infinities instead of a diagnosable number.
    """
    projection = np.sum(delta * area, axis=-1)
    safe = np.where(np.abs(projection) > 1e-300, projection, 1e-300)
    factor = np.sum(area * area, axis=-1) / safe
    return factor, area - factor[..., None] * delta
