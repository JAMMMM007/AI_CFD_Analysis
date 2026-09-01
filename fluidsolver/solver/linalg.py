"""Assembly and solution of the five-point systems the discretisation produces.

Every transport equation on this mesh has the same shape,

    a_P phi_P + a_W phi_(i-1,j) + a_E phi_(i+1,j)
              + a_S phi_(i,j-1) + a_N phi_(i,j+1) = b

so one coefficient container and one assembler serve momentum, pressure
correction, ``k`` and ``omega`` alike. Boundary faces never appear as a
neighbour: their contribution is folded into ``a_P`` and ``b`` by the boundary
conditions, which is why ``a_S`` is unused on the wall row and ``a_N`` on the
far-field row.

Unknowns are numbered ``k = i * Nj + j``, so a column of cells at fixed ``i`` --
a line running out from the wall, along which the boundary layer's steepest
gradients lie -- is contiguous in memory and contiguous in the matrix. That is
what makes the incomplete-LU preconditioner effective here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Coefficients:
    """The five bands of a structured five-point operator, plus its source."""

    centre: np.ndarray
    west: np.ndarray
    east: np.ndarray
    south: np.ndarray
    north: np.ndarray
    source: np.ndarray

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> "Coefficients":
        return cls(*(np.zeros(shape) for _ in range(6)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.centre.shape

    def apply(self, field: np.ndarray) -> np.ndarray:
        """Matrix-vector product, without going through the sparse matrix.

        Used for residuals and for the Rhie-Chow velocity reconstruction, where
        building a matrix just to multiply by it would be wasteful. Neighbour
        terms are rolled rather than indexed: ``i`` wraps, and the ``j`` rolls are
        masked off at the boundaries where those coefficients are zero anyway.
        """
        result = self.centre * field
        result += self.west * np.roll(field, 1, axis=0)
        result += self.east * np.roll(field, -1, axis=0)
        result[:, 1:] += self.south[:, 1:] * field[:, :-1]
        result[:, :-1] += self.north[:, :-1] * field[:, 1:]
        return result

    def under_relax(self, field: np.ndarray, factor: float) -> None:
        """Apply implicit (Patankar) under-relaxation in place.

        ``a_P/alpha`` on the diagonal with ``(1-alpha)/alpha a_P phi_old`` added to
        the source. Written this way the converged solution is untouched -- at
        convergence ``phi = phi_old`` and the two added terms cancel exactly -- so
        the relaxation factor changes only the path, never the answer.
        """
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"relaxation factor must be in (0, 1], got {factor}")
        if factor == 1.0:
            return
        self.centre /= factor
        self.source += (1.0 - factor) * self.centre * field

    def residual(self, field: np.ndarray) -> float:
        """Scaled residual of the current field, in the usual finite-volume sense.

        The raw imbalance ``|b - A phi|`` has the units of the equation and says
        nothing on its own -- it is small for a momentum equation on a fine mesh
        whether or not the solution is converged. Normalising by the variation the
        operator produces across the field gives a number that starts near one and
        falls as the solution settles, comparably between equations.
        """
        imbalance = np.abs(self.source - self.apply(field))
        uniform = self.apply(np.full_like(field, field.mean()))
        scale = np.abs(self.apply(field) - uniform) + np.abs(self.source - uniform)

        total = scale.sum()
        if total <= 0.0:
            # A uniform field satisfying a uniform equation: nothing to scale by,
            # and nothing left to converge either.
            return float(imbalance.sum())
        return float(imbalance.sum() / total)


class StructuredMatrix:
    """Turns :class:`Coefficients` into a sparse matrix, reusing the sparsity pattern.

    The pattern is fixed by the mesh, so it is worked out once. Only the values
    change from one outer iteration to the next, and there are five of those per
    iteration, so re-deriving the pattern each time would dominate the cost of
    assembly.
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        n_i, n_j = shape
        self.size = n_i * n_j

        index = np.arange(self.size).reshape(shape)
        rows, cols, order = [], [], []

        def band(row_mask, column, slot):
            rows.append(index[row_mask])
            cols.append(column[row_mask])
            order.append(np.full(int(row_mask.sum()), slot))

        everywhere = np.ones(shape, dtype=bool)
        interior_south = np.ones(shape, dtype=bool)
        interior_south[:, 0] = False
        interior_north = np.ones(shape, dtype=bool)
        interior_north[:, -1] = False

        band(everywhere, index, 0)
        band(everywhere, np.roll(index, 1, axis=0), 1)
        band(everywhere, np.roll(index, -1, axis=0), 2)
        band(interior_south, np.roll(index, 1, axis=1), 3)
        band(interior_north, np.roll(index, -1, axis=1), 4)

        self._rows = np.concatenate(rows)
        self._cols = np.concatenate(cols)
        self._slot = np.concatenate(order)
        self._mask = [
            everywhere.ravel(),
            everywhere.ravel(),
            everywhere.ravel(),
            interior_south.ravel(),
            interior_north.ravel(),
        ]

        # Assembling once with the entry numbers as data lets scipy tell us where
        # each entry ends up after it sorts and merges into CSR. That permutation
        # is then reused to fill the values directly.
        marker = sp.coo_matrix(
            (np.arange(1, len(self._rows) + 1, dtype=float), (self._rows, self._cols)),
            shape=(self.size, self.size),
        ).tocsr()
        self._indices = marker.indices
        self._indptr = marker.indptr
        self._permutation = marker.data.astype(np.int64) - 1

    def build(self, coefficients: Coefficients) -> sp.csr_matrix:
        """Assemble the sparse matrix for these coefficients."""
        bands = (
            coefficients.centre,
            coefficients.west,
            coefficients.east,
            coefficients.south,
            coefficients.north,
        )
        values = np.concatenate(
            [band.ravel()[mask] for band, mask in zip(bands, self._mask)]
        )
        # The permutation says which entry ends up in each CSR slot, so filling
        # the matrix is a gather from the entries. Scattering *into* it with the
        # same array applies the mapping backwards and silently produces a matrix
        # with the right sparsity pattern and the wrong values everywhere.
        return sp.csr_matrix(
            (values[self._permutation], self._indices, self._indptr),
            shape=(self.size, self.size),
        )


def solve(
    matrix: sp.csr_matrix,
    source: np.ndarray,
    guess: np.ndarray,
    *,
    tolerance: float = 0.1,
    max_iterations: int = 200,
    preconditioner=None,
) -> tuple[np.ndarray, int]:
    """Solve one linear system, reducing its residual by ``tolerance``.

    The tolerance is deliberately loose. Inside a SIMPLE loop the matrix itself is
    only an approximation -- the coefficients are rebuilt from a velocity field
    that is about to change -- so driving the inner solve to machine precision
    buys nothing. Reducing the residual by one order per outer iteration is the
    usual, and cheapest, compromise.

    The reduction is measured against *this solve's own starting residual*, which
    is the whole point and is not what a bare ``rtol`` does. SciPy compares
    against ``|b|``, and in a converging SIMPLE run the starting residual is
    already far below that: the solver then reports success having performed no
    iterations at all, the outer loop stops advancing, and the case sits at a
    residual plateau looking as though the physics were at fault.

    Returns the solution and the solver's status code (0 for converged).
    """
    right_hand_side = source.ravel()
    start = guess.ravel()

    initial_residual = float(np.linalg.norm(matrix @ start - right_hand_side))
    if initial_residual == 0.0:
        return guess.copy(), 0

    solution, info = spla.bicgstab(
        matrix,
        right_hand_side,
        x0=start,
        rtol=0.0,
        atol=tolerance * initial_residual,
        maxiter=max_iterations,
        M=preconditioner,
    )
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError(
            "the linear solver returned a non-finite result. The equations have "
            "diverged -- usually too large a relaxation factor, or a mesh with "
            "inverted cells."
        )
    return solution.reshape(guess.shape), int(info)


def jacobi_preconditioner(matrix: sp.csr_matrix) -> spla.LinearOperator:
    """Diagonal scaling. Cheap, and enough for the diagonally dominant equations.

    Momentum and the turbulence transport equations are strongly dominated by
    their diagonal -- convection and the sink terms both feed it -- so there is
    little for a stronger preconditioner to do.
    """
    diagonal = matrix.diagonal()
    inverse = np.where(np.abs(diagonal) > 0.0, 1.0 / diagonal, 1.0)
    return spla.LinearOperator(matrix.shape, matvec=lambda x: inverse * x)


def incomplete_lu_preconditioner(
    matrix: sp.csr_matrix, drop_tolerance: float = 1e-4, fill_factor: float = 10.0
):
    """Incomplete LU, for the pressure correction.

    The pressure equation is a Poisson problem: no diagonal dominance to speak
    of, and an error field that is global rather than local, so a diagonal
    preconditioner leaves the iteration count scaling with mesh size. ILU
    propagates information along the whole matrix band, and with the wall-normal
    lines contiguous in the ordering that band follows the direction the
    stiffness actually lies in.

    Falls back to diagonal scaling if the factorisation fails, which it can do on
    a badly conditioned matrix; a slower solve beats no solve.
    """
    try:
        factorisation = spla.spilu(
            matrix.tocsc(), drop_tol=drop_tolerance, fill_factor=fill_factor
        )
    except (RuntimeError, ValueError):
        return jacobi_preconditioner(matrix)
    return spla.LinearOperator(matrix.shape, matvec=factorisation.solve)
