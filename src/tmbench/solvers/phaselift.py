"""Dense bounded PhaseLift for the sample-major field X @ a."""

from dataclasses import dataclass

import numpy as np
from ._convex import BACKEND, UnsupportedSize, load_scs


MAX_MODES = 32
MAX_PATTERNS = 512


@dataclass(frozen=True)
class PhaseLiftConfig:
    relative_noise_bound: float = 0.05
    max_iter: int = 5000
    solver_tolerance: float = 1e-6
    feasibility_tolerance: float = 2e-5
    rank_rtol: float = 1e-4

    def __post_init__(self):
        if not np.isfinite(self.relative_noise_bound) or not 0 <= self.relative_noise_bound < 1:
            raise ValueError('relative_noise_bound must be in [0, 1)')
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer')
        for name in ('solver_tolerance', 'feasibility_tolerance', 'rank_rtol'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name} must be between zero and one')

    @property
    def variant(self):
        return 'exact' if self.relative_noise_bound == 0 else 'bounded_intensity_noise'


@dataclass
class PhaseLiftResult:
    status: str
    diagnostics: dict
    x: np.ndarray | None = None
    lifted: np.ndarray | None = None


class PhaseLift:
    def __init__(self, X, config):
        X = np.asarray(X)
        if X.ndim != 2 or min(X.shape) == 0:
            raise ValueError('X must be a nonempty matrix')
        self.P, self.N = X.shape
        if self.N > MAX_MODES or self.P > MAX_PATTERNS:
            raise UnsupportedSize(
                f'{self.P} patterns and {self.N} modes exceed the dense PhaseLift '
                f'limits of {MAX_PATTERNS} patterns and {MAX_MODES} modes',
            )
        self.X = np.asarray(X, dtype=complex)
        if not np.isfinite(self.X).all() or not np.any(self.X):
            raise ValueError('X must have finite entries and nonzero probe energy')
        cp, self.backend_versions = load_scs('PhaseLift')
        self._solver_error = cp.SolverError
        self.config = config
        self._intensities = cp.Parameter(self.P, nonneg=True)
        # A scalar Hermitian matrix is real; avoid CVXPY's complex scalar reduction.
        self._lifted = cp.Variable((self.N, self.N), hermitian=self.N > 1, symmetric=self.N == 1)

        # A(M)_p = sum_ij X[p,i] M[i,j] conj(X[p,j]) in C order.
        operator = self.X[:, :, None] * self.X[:, None, :].conj()
        if self.N == 1:
            operator = operator.real
        predicted = operator.reshape(self.P, -1) @ cp.vec(self._lifted, order='C')
        trace = cp.trace(self._lifted)
        if self.N > 1:
            predicted = cp.real(predicted)
            trace = cp.real(trace)
        constraints = [self._lifted >> 0]
        self._radius = None
        if config.relative_noise_bound == 0:
            constraints.append(predicted == self._intensities)
        else:
            self._radius = cp.Parameter(nonneg=True)
            constraints.append(cp.norm(predicted - self._intensities, 2) <= self._radius)
        self._problem = cp.Problem(cp.Minimize(trace), constraints)

    def fit(self, intensities):
        if np.iscomplexobj(intensities):
            raise ValueError('Intensities must be real')
        I = np.asarray(intensities, dtype=float)
        if I.shape != (self.P,) or not np.isfinite(I).all() or np.any(I < 0):
            raise ValueError('Expected one finite nonnegative intensity per sample')
        scale = float(I.mean())
        norm = float(np.linalg.norm(I))
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(norm) or norm <= 0:
            raise ValueError('Finite positive intensity energy required')
        bound = self.config.relative_noise_bound * norm
        self._intensities.value = I / scale
        if self._radius is not None:
            self._radius.value = bound / scale

        diagnostics = dict(backend=BACKEND, feasible=False, intensity_bound=bound, intensity_scale=scale)
        try:
            self._problem.solve(solver=BACKEND, eps=self.config.solver_tolerance,
                                max_iters=self.config.max_iter, warm_start=False)
        except self._solver_error as error:
            diagnostics.update(backend_status='solver_error', error=str(error))
            return PhaseLiftResult('solver_error', diagnostics)
        status = self._problem.status
        stats = self._problem.solver_stats
        diagnostics.update(backend_status=status, backend_iterations=stats.num_iters,
                           backend_solve_seconds=stats.solve_time, backend_setup_seconds=stats.setup_time)
        if status not in ('optimal', 'optimal_inaccurate') or self._lifted.value is None:
            return PhaseLiftResult(status, diagnostics)

        lifted = self._lifted.value * scale
        if not np.isfinite(lifted).all():
            return PhaseLiftResult('invalid_solution', diagnostics)
        lifted = (lifted + lifted.conj().T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(lifted)
        predicted = np.sum((self.X @ lifted) * self.X.conj(), axis=1).real
        residual = float(np.linalg.norm(predicted - I))
        violation = max(0., residual - bound) / norm
        minimum = float(eigenvalues[0])
        largest = float(eigenvalues[-1])
        psd_violation = max(0., -minimum) / max(scale, float(np.max(np.abs(eigenvalues))))
        feasible = max(violation, psd_violation) <= self.config.feasibility_tolerance
        rank = int(np.count_nonzero(eigenvalues > max(0., largest) * self.config.rank_rtol))
        diagnostics.update(feasible=feasible, minimum_eigenvalue=minimum, largest_eigenvalue=largest,
                           numerical_rank=rank, lifted_trace=float(eigenvalues.sum()),
                           lifted_intensity_residual_norm=residual,
                           relative_constraint_violation=violation, relative_psd_violation=psd_violation)
        if not feasible:
            return PhaseLiftResult('constraint_violation', diagnostics)
        if largest <= 0:
            return PhaseLiftResult('rank_zero', diagnostics)
        x = np.sqrt(largest) * eigenvectors[:, -1]
        diagnostics.update(
            leading_positive_trace_fraction=largest / float(np.maximum(eigenvalues, 0).sum()),
            rank1_intensity_residual_norm=float(np.linalg.norm(np.abs(self.X @ x)**2 - I)),
        )
        return PhaseLiftResult(status, diagnostics, x, lifted)
