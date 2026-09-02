"""Mesh quality assessment.

Every number here maps onto a specific way a finite-volume discretisation can go
wrong, so the report is worth reading rather than glancing at:

* **Negative volumes** -- an inverted cell. Nothing downstream can recover; the
  run is blocked.
* **Non-orthogonality** -- the angle between a face normal and the line joining
  the two cell centroids across it. The diffusion term splits into an implicit
  orthogonal part and an explicit cross-diffusion correction, and the correction
  grows with this angle, as ``tan``: at 45 degrees the two are equal, and past
  roughly 70 the lagged half dominates and the outer iteration stops converging.
  Reported three ways -- peak, mean, and the share of faces past the warning
  angle -- because they say different things. A peak is one cell; a mean is a
  region. It was a *mean* of 8 degrees with a 99th percentile of 65, under a peak
  that sat just below the old 70-degree threshold, that stopped this solver
  converging at any Reynolds number.
* **Skewness** -- how far the face midpoint sits from where the centroid-to-centroid
  line crosses the face, as a fraction of the face length. Face values are
  interpolated at the crossing point, so skewness is a direct error in every
  convective flux.
* **Aspect ratio** -- the cell's long side over its short side. Boundary-layer
  cells are deliberately extreme here (ratios of thousands at ``y+ = 1`` are
  normal and fine) but the number belongs in the report so a genuinely bad cell
  elsewhere is visible.
* **Expansion ratio** -- volume jump between neighbours. Abrupt jumps break the
  cancellation that gives the scheme its second-order truncation error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluidsolver.mesh.metrics import Metrics

# Above this the non-orthogonal correction stops being a correction. Meshes are
# not rejected for it -- plenty of useful grids have a few bad cells -- but the
# report says so.
#
# Set at 60 rather than the 70 where the correction formally overtakes the
# implicit part, because a threshold at the failure point does not warn about
# anything until it is too late. The mesh that stopped this solver converging
# peaked at 69.69 degrees and passed silently, while carrying a *mean* of 8.3 and
# a 99th percentile of 64.6 -- the peak was never the interesting number.
_ORTHOGONALITY_WARNING_DEG = 60.0

# Non-orthogonality is not a defect of one cell but of a region, and a peak can
# be one cell in fifty thousand. This is the share of faces allowed past
# ``_ORTHOGONALITY_WARNING_DEG`` before the report says the mesh has a bad
# *region* rather than a bad cell.
_ORTHOGONALITY_WARNING_FRACTION = 0.02
_SKEWNESS_WARNING = 0.5


@dataclass(frozen=True)
class QualityReport:
    """Summary of a mesh's fitness for the finite-volume solver."""

    cells: int
    min_volume: float
    max_volume: float
    negative_volumes: int
    max_non_orthogonality_deg: float
    mean_non_orthogonality_deg: float
    non_orthogonal_fraction: float
    max_skewness: float
    max_aspect_ratio: float
    max_expansion_ratio: float
    min_wall_distance: float

    @property
    def is_usable(self) -> bool:
        """Whether the solver may run on this mesh at all."""
        return self.negative_volumes == 0 and self.min_volume > 0.0

    @property
    def warnings(self) -> list[str]:
        """Human-readable concerns, worst first. Empty means the mesh is clean."""
        issues = []
        if self.negative_volumes:
            issues.append(
                f"{self.negative_volumes} inverted cells (negative volume). "
                "The mesh cannot be solved on; regenerate it."
            )
        if self.non_orthogonal_fraction > _ORTHOGONALITY_WARNING_FRACTION:
            issues.append(
                f"{self.non_orthogonal_fraction:.1%} of faces are more than "
                f"{_ORTHOGONALITY_WARNING_DEG:.0f} deg non-orthogonal "
                f"(peak {self.max_non_orthogonality_deg:.1f}, mean "
                f"{self.mean_non_orthogonality_deg:.1f}). Cross-diffusion is "
                "explicit, so the outer iteration will be slow and may not "
                "converge at all."
            )
        elif self.max_non_orthogonality_deg > _ORTHOGONALITY_WARNING_DEG:
            issues.append(
                f"peak non-orthogonality {self.max_non_orthogonality_deg:.1f} deg "
                f"(above {_ORTHOGONALITY_WARNING_DEG:.0f}), on "
                f"{self.non_orthogonal_fraction:.2%} of faces. Localised, so "
                "probably survivable, but it is where trouble will start."
            )
        if self.max_skewness > _SKEWNESS_WARNING:
            issues.append(
                f"peak skewness {self.max_skewness:.2f} (above {_SKEWNESS_WARNING}). "
                "Face interpolation is first-order accurate where this is large."
            )
        return issues

    def summary(self) -> str:
        """Multi-line report for the mesh page and the console."""
        lines = [
            f"cells                 {self.cells}",
            f"volume                {self.min_volume:.4g} .. {self.max_volume:.4g}",
            f"inverted cells        {self.negative_volumes}",
            f"non-orthogonality     {self.mean_non_orthogonality_deg:.1f} deg mean, "
            f"{self.max_non_orthogonality_deg:.1f} deg peak, "
            f"{self.non_orthogonal_fraction:.2%} of faces above "
            f"{_ORTHOGONALITY_WARNING_DEG:.0f}",
            f"skewness              {self.max_skewness:.3f} peak",
            f"aspect ratio          {self.max_aspect_ratio:.4g} peak",
            f"expansion ratio       {self.max_expansion_ratio:.3f} peak",
            f"first cell off wall   {self.min_wall_distance:.4g}",
        ]
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def assess(metrics: Metrics, nodes: np.ndarray) -> QualityReport:
    """Measure the quality of a mesh from its metrics and node coordinates."""
    volume = metrics.volume
    orthogonality, skewness = _face_quality(metrics)

    return QualityReport(
        cells=int(volume.size),
        min_volume=float(volume.min()),
        max_volume=float(volume.max()),
        negative_volumes=int((volume <= 0.0).sum()),
        max_non_orthogonality_deg=float(orthogonality.max()),
        mean_non_orthogonality_deg=float(orthogonality.mean()),
        non_orthogonal_fraction=float(
            (orthogonality > _ORTHOGONALITY_WARNING_DEG).mean()
        ),
        max_skewness=float(skewness.max()),
        max_aspect_ratio=float(_aspect_ratio(nodes).max()),
        max_expansion_ratio=float(_expansion_ratio(volume).max()),
        min_wall_distance=float(metrics.wall_distance.min()),
    )


def _face_quality(metrics: Metrics) -> tuple[np.ndarray, np.ndarray]:
    """Non-orthogonality angle (degrees) and skewness for every interior face.

    Boundary faces are excluded: they have only one cell, so there is no
    centroid-to-centroid line to measure against, and their treatment lives in
    the boundary conditions rather than in the interior scheme.
    """
    centroid = metrics.centroid

    angles, skews = [], []
    for area, centre, own, neighbour in (
        (
            metrics.face_i_area,
            metrics.face_i_centre,
            centroid,
            np.roll(centroid, 1, axis=0),
        ),
        (
            metrics.face_j_area[:, 1:-1],
            metrics.face_j_centre[:, 1:-1],
            centroid[:, 1:],
            centroid[:, :-1],
        ),
    ):
        joining = own - neighbour
        length = np.linalg.norm(joining, axis=-1)
        normal_length = np.linalg.norm(area, axis=-1)
        valid = (length > 0.0) & (normal_length > 0.0)

        cosine = np.abs(np.sum(joining * area, axis=-1)) / np.where(
            valid, length * normal_length, 1.0
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))[valid])

        # Where the centroid-to-centroid line actually crosses the face plane,
        # compared with the face midpoint. Interpolation happens at the crossing;
        # the offset between the two is the interpolation error.
        to_face = centre - neighbour
        travel = np.sum(to_face * area, axis=-1) / np.where(
            valid, np.sum(joining * area, axis=-1), 1.0
        )
        crossing = neighbour + travel[..., None] * joining
        face_size = np.sqrt(normal_length)
        skews.append(
            (np.linalg.norm(centre - crossing, axis=-1) / np.where(valid, face_size, 1.0))[
                valid
            ]
        )

    return np.concatenate(angles), np.concatenate(skews)


def _aspect_ratio(nodes: np.ndarray) -> np.ndarray:
    """Longest cell edge over shortest, per cell."""
    along_i = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=-1)
    along_j = np.linalg.norm(nodes[:, 1:] - nodes[:, :-1], axis=-1)

    edges = np.stack(
        (
            along_i[:, :-1],
            along_i[:, 1:],
            along_j,
            np.roll(along_j, -1, axis=0),
        ),
        axis=-1,
    )
    smallest = edges.min(axis=-1)
    return edges.max(axis=-1) / np.where(smallest > 0.0, smallest, np.inf)


def _expansion_ratio(volume: np.ndarray) -> np.ndarray:
    """Volume ratio between each cell and its neighbours, always at least 1."""
    ratios = []
    for neighbour in (
        np.roll(volume, 1, axis=0),
        np.roll(volume, -1, axis=0),
        np.concatenate((volume[:, :1], volume[:, :-1]), axis=1),
        np.concatenate((volume[:, 1:], volume[:, -1:]), axis=1),
    ):
        safe = np.where(np.abs(neighbour) > 0.0, neighbour, np.nan)
        ratios.append(np.abs(volume / safe))

    stacked = np.stack(ratios, axis=-1)
    return np.nanmax(np.maximum(stacked, 1.0 / stacked), axis=-1)
