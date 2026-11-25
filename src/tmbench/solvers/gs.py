"""Gerchberg–Saxton amplitude projections for a sample-major probe matrix."""

from dataclasses import dataclass

import numpy as np

from .prvamp.linear import matvec


@dataclass
class GSResult:
    x: np.ndarray
    z: np.ndarray
    history: list


class GerchbergSaxton:
    """Cache a numerical Moore–Penrose pseudoinverse for all detector rows."""

    def __init__(self, X, *, rcond):
        dtype = np.complex128 if np.iscomplexobj(X) else np.float64
        self.X = np.asarray(X, dtype=dtype)
        if self.X.ndim != 2 or min(self.X.shape) == 0 or not np.isfinite(self.X).all():
            raise ValueError('X must be a nonempty finite matrix')
        if not np.isfinite(rcond) or not 0 <= rcond < 1:
            raise ValueError('pinv_rcond must be in [0, 1)')

        U, singular_values, Vh = np.linalg.svd(self.X, full_matrices=False)
        self.cutoff = float(rcond * singular_values[0])
        self.rank = int(np.count_nonzero(singular_values > self.cutoff))
        if self.rank == 0:
            raise ValueError('X has zero numerical rank')
        scaled_vectors = Vh[:self.rank].conj().T / singular_values[:self.rank]
        self.pseudoinverse = scaled_vectors @ U[:, :self.rank].conj().T
        if not np.isfinite(self.pseudoinverse).all():
            raise FloatingPointError('Nonfinite pseudoinverse')

    def fit(self, y, *, max_iter, seed):
        """Run a fixed number of full projection cycles from seeded complex coefficients."""
        sample_count, mode_count = self.X.shape
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (sample_count,) or not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError('y must contain one finite nonnegative amplitude per sample')
        amplitude_norm = np.linalg.norm(y)
        if not np.isfinite(amplitude_norm) or amplitude_norm == 0:
            raise ValueError('Finite positive amplitude energy required')
        if type(max_iter) is not int or max_iter < 1:
            raise ValueError('max_iter must be a positive integer')

        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(mode_count) + 1j * rng.standard_normal(mode_count)) / np.sqrt(2)
        z = matvec(self.X, x)
        history = []
        for iteration in range(1, max_iter + 1):
            magnitude = np.abs(z)
            # Any phase projects a zero field to the amplitude circle. Use phase zero.
            phase = np.divide(z, magnitude, out=np.ones_like(z), where=magnitude > 0)
            x = matvec(self.pseudoinverse, y * phase)
            z = matvec(self.X, x)
            residual = float(np.linalg.norm(np.abs(z) - y) / amplitude_norm)
            if not np.isfinite(x).all() or not np.isfinite(residual):
                raise FloatingPointError(f'Nonfinite GS iterate at iteration {iteration}')
            history.append(dict(iteration=iteration, relative_amplitude_residual=residual))
        return GSResult(x, z, history)
