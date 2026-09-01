"""Finite-volume metrics for a structured O-grid.

The solver never touches node coordinates. It works entirely in terms of cell
volumes, face area-vectors and centroid positions, which is what this module
computes once up front.

Indexing, fixed here and assumed everywhere downstream:

* Cells are ``(i, j)`` for ``i`` in ``0 .. Ni-1`` and ``j`` in ``0 .. Nj-1``.
  ``i`` wraps around the body and is periodic; ``j`` runs outward from the wall.
* ``face_i[i, j]`` is the face between cell ``(i-1, j)`` and cell ``(i, j)``, with
  its area-vector pointing in the ``+i`` direction. There are ``Ni`` of them per
  row, and ``face_i[0]`` is the wrap-around face.
* ``face_j[i, j]`` is the face between cell ``(i, j-1)`` and cell ``(i, j)``, with
  its area-vector pointing in ``+j``. There are ``Nj + 1`` per column;
  ``face_j[:, 0]`` is the wall and ``face_j[:, Nj]`` the far field.

Area-vectors carry the face length in their magnitude, so a flux is simply
``u . S`` with no separate length factor to forget. In 2-D the "area" is a length,
the depth being unity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Metrics:
    """Geometric quantities the finite-volume discretisation needs.

    Attributes
    ----------
    volume
        ``(Ni, Nj)`` cell volumes -- areas, for unit depth.
    centroid
        ``(Ni, Nj, 2)`` cell centroids. These are true area centroids, not means
        of the four corners; on the stretched cells of a boundary-layer mesh the
        two differ enough to matter to the gradient reconstruction.
    face_i_area, face_j_area
        ``(Ni, Nj, 2)`` and ``(Ni, Nj+1, 2)`` face area-vectors, pointing towards
        increasing ``i`` and ``j``.
    face_i_centre, face_j_centre
        Face midpoints, matching shapes.
    wall_distance
        ``(Ni, Nj)`` shortest distance from each cell centroid to the body
        surface. The k-omega SST model needs this in its blending functions and
        its wall boundary condition.
    """

    volume: np.ndarray
    centroid: np.ndarray
    face_i_area: np.ndarray
    face_j_area: np.ndarray
    face_i_centre: np.ndarray
    face_j_centre: np.ndarray
    wall_distance: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.volume.shape

    @property
    def total_volume(self) -> float:
        return float(self.volume.sum())

    def wall_face_area(self) -> np.ndarray:
        """``(Ni, 2)`` area-vectors of the wall faces, pointing *into* the solid.

        ``face_j`` points towards increasing ``j``, which is away from the wall,
        so the wall's outward-from-the-fluid normal is the negative of it. Forces
        on the body are integrated with this, so the sign convention matters:
        pressure acting on the body pushes along ``-face_j``.
        """
        return -self.face_j_area[:, 0]


def compute_metrics(nodes: np.ndarray, wall_samples: int = 8) -> Metrics:
    """Build the finite-volume metrics for a node array.

    Parameters
    ----------
    nodes
        ``(Ni, Nj+1, 2)`` grid nodes, periodic in ``i``, as produced by
        :func:`fluidsolver.mesh.ogrid.build_ogrid`.
    wall_samples
        Sub-samples per wall segment used when measuring wall distance. The
        surface is a polyline; measuring only to its vertices overestimates the
        distance for cells sitting opposite the middle of a long segment.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 3 or nodes.shape[2] != 2 or nodes.shape[1] < 2:
        raise ValueError(f"nodes must have shape (Ni, Nj+1, 2), got {nodes.shape}")

    # Corners of every cell, all shaped (Ni, Nj, 2).
    p00 = nodes[:, :-1]
    p10 = np.roll(nodes, -1, axis=0)[:, :-1]
    p11 = np.roll(nodes, -1, axis=0)[:, 1:]
    p01 = nodes[:, 1:]

    volume, centroid = _polygon_volume_and_centroid([p00, p10, p11, p01])

    # A face in the i-direction runs along j: from node (i, j) to node (i, j+1).
    # Rotating that edge by -90 degrees gives a normal pointing towards +i.
    edge_i = nodes[:, 1:] - nodes[:, :-1]
    face_i_area = np.stack((edge_i[..., 1], -edge_i[..., 0]), axis=-1)
    face_i_centre = 0.5 * (nodes[:, 1:] + nodes[:, :-1])

    # A face in the j-direction runs along i: from node (i, j) to node (i+1, j).
    # Rotating by +90 degrees gives a normal pointing towards +j, i.e. outward.
    edge_j = np.roll(nodes, -1, axis=0) - nodes
    face_j_area = np.stack((-edge_j[..., 1], edge_j[..., 0]), axis=-1)
    face_j_centre = 0.5 * (np.roll(nodes, -1, axis=0) + nodes)

    return Metrics(
        volume=volume,
        centroid=centroid,
        face_i_area=face_i_area,
        face_j_area=face_j_area,
        face_i_centre=face_i_centre,
        face_j_centre=face_j_centre,
        wall_distance=_wall_distance(centroid, nodes[:, 0], wall_samples),
    )


def _polygon_volume_and_centroid(corners: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Signed area and area centroid of quadrilaterals given corner-by-corner.

    Both come from the same shoelace sum, so they are computed together:

        A   = 1/2 sum (x_k y_{k+1} - x_{k+1} y_k)
        C   = 1/(6A) sum (p_k + p_{k+1}) (x_k y_{k+1} - x_{k+1} y_k)

    Positive area means the corners were given counter-clockwise, which for this
    grid's ``(i, j)`` ordering is the correct, non-inverted orientation.
    """
    area = np.zeros(corners[0].shape[:-1])
    moment = np.zeros(corners[0].shape)

    for current, following in zip(corners, corners[1:] + corners[:1]):
        cross = (
            current[..., 0] * following[..., 1] - following[..., 0] * current[..., 1]
        )
        area += cross
        moment += (current + following) * cross[..., None]

    area *= 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = moment / (6.0 * area[..., None])
    return area, centroid


def _wall_distance(
    centroid: np.ndarray, wall: np.ndarray, samples: int
) -> np.ndarray:
    """Shortest distance from each cell centroid to the body surface.

    Uses the true minimum distance to the surface polyline rather than the
    marching distance carried by the grid's ``j`` index. Those agree only where
    the grid is orthogonal and the surface is smooth; near a convex corner the
    marching distance is larger than the real one, and feeding that to the SST
    blending functions would mis-place the switch between its two model regimes.

    The polyline is sub-sampled and queried with a KD-tree. Sub-sampling matters:
    against vertices alone, a cell opposite the middle of a long wall segment
    reads as further away than it is.
    """
    from scipy.spatial import cKDTree

    closed = np.vstack((wall, wall[:1]))
    fractions = np.linspace(0.0, 1.0, samples, endpoint=False)
    dense = (
        closed[:-1, None, :] + fractions[None, :, None] * np.diff(closed, axis=0)[:, None, :]
    ).reshape(-1, 2)

    distance, _ = cKDTree(dense).query(centroid.reshape(-1, 2))
    return distance.reshape(centroid.shape[:-1])


def cell_face_areas(metrics: Metrics) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four *outward* area-vectors of every cell, as (west, east, south, north).

    ``face_i`` and ``face_j`` are stored once per face and shared between the two
    cells either side of it, pointing consistently towards increasing index. A
    cell's own outward normals therefore flip sign on its low-index faces.
    Summing these four gives zero for a closed polygon, which is the identity the
    tests check.
    """
    west = -metrics.face_i_area
    east = np.roll(metrics.face_i_area, -1, axis=0)
    south = -metrics.face_j_area[:, :-1]
    north = metrics.face_j_area[:, 1:]
    return west, east, south, north
