"""Dense bounded PhaseLift for the sample-major field X @ a."""

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import eigh

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


@dataclass(frozen=True)
class OperatorPhaseLiftConfig:
    regularization_fraction: float = 0.001
    max_iter: int = 50
    solver_time_limit_seconds: float = 120.
    solver_tolerance: float = 1e-6
    rank_rtol: float = 1e-4

    def __post_init__(self):
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer')
        if not np.isfinite(self.solver_time_limit_seconds) or self.solver_time_limit_seconds <= 0:
            raise ValueError('solver_time_limit_seconds must be finite and positive')
        for name in ('regularization_fraction', 'solver_tolerance', 'rank_rtol'):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f'{name} must be between zero and one')

    @property
    def variant(self):
        return 'trace_regularized_intensity_loss'


class OperatorPhaseLift:
    """Solve the trace-regularized PSD problem with implicit measurement products."""

    backend = 'scipy_proximal_gradient'

    def __init__(self, X, config):
        self.X = np.asarray(X, dtype=complex if np.iscomplexobj(X) else float)
        if self.X.ndim != 2 or min(self.X.shape) == 0 or not np.isfinite(self.X).all():
            raise ValueError('X must be a finite, nonempty matrix')
        self.P, self.N = self.X.shape
        self.adjoint_design = self.X.conj().T if np.iscomplexobj(self.X) else self.X.T
        self.config = config
        # A A* has nonnegative entries |X[p] @ X[q].H|**2. Its maximum
        # row sum bounds its spectral norm and thus the gradient Lipschitz constant.
        self.lipschitz_bound = float(self.forward(self.adjoint(np.ones(self.P))).max())
        if not np.isfinite(self.lipschitz_bound) or self.lipschitz_bound <= 0:
            raise ValueError('X must have finite, positive probe energy')

    def forward(self, lifted):
        return np.einsum('pi,pi->p', self.X @ lifted, self.adjoint_design.T).real

    def adjoint(self, values):
        gram = self.adjoint_design @ (values[:, None] * self.X)
        return (gram + gram.conj().T) / 2

    def fit(self, intensities):
        if np.iscomplexobj(intensities):
            raise ValueError('Intensities must be real')
        intensity = np.asarray(intensities, dtype=float)
        if intensity.shape != (self.P,) or not np.isfinite(intensity).all() or np.any(intensity < 0):
            raise ValueError('Expected one finite nonnegative intensity per sample')
        scale = float(intensity.mean())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Finite positive intensity energy required')

        cfg = self.config
        start = time.perf_counter()
        target = intensity / scale
        adjoint_target = self.adjoint(target)
        largest = eigh(adjoint_target, subset_by_index=[self.N - 1, self.N - 1], eigvals_only=True)[0]
        penalty = float(cfg.regularization_fraction * largest)
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError('Positive intensities must overlap illuminated probes')
        step = 1 / self.lipschitz_bound
        estimate = np.zeros_like(adjoint_target)
        extrapolated = estimate
        momentum = 1.
        status = 'max_iter_reached'

        for iteration in range(1, cfg.max_iter + 1):
            gradient = self.adjoint(self.forward(extrapolated)) - adjoint_target
            candidate = extrapolated - step * gradient
            eigenvalues, eigenvectors = eigh(candidate)
            eigenvalues = np.maximum(eigenvalues - step * penalty, 0.)
            positive = eigenvalues > 0
            basis = eigenvectors[:, positive]
            updated = (basis * eigenvalues[positive]) @ basis.conj().T
            change = float(np.linalg.norm(updated - extrapolated) / max(np.linalg.norm(updated), 1e-15))
            next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
            extrapolated = updated + ((momentum - 1) / next_momentum) * (updated - estimate)
            estimate, momentum = updated, next_momentum
            if change <= cfg.solver_tolerance:
                status = 'converged'
                break
            if time.perf_counter() - start >= cfg.solver_time_limit_seconds:
                status = 'timed_out'
                break

        eigenvalues *= scale
        lifted = estimate * scale
        largest = float(eigenvalues[-1])
        diagnostics = dict(backend=self.backend, backend_status=status, backend_iterations=iteration,
                           backend_solve_seconds=time.perf_counter() - start, intensity_scale=scale,
                           trace_penalty=penalty * scale, relative_proximal_step=change,
                           lifted_trace=float(eigenvalues.sum()),
                           numerical_rank=int(np.count_nonzero(eigenvalues > largest * cfg.rank_rtol)))
        if largest <= 0:
            return PhaseLiftResult('rank_zero', diagnostics)
        # With real probes, Re(a a.H) can have rank two. Its first two
        # eigenvectors recover the real and imaginary components up to O(2).
        x = np.sqrt(largest) * eigenvectors[:, -1]
        if not np.iscomplexobj(self.X) and self.N > 1:
            x = x + 1j * np.sqrt(eigenvalues[-2]) * eigenvectors[:, -2]
        diagnostics.update(
            lifted_intensity_residual_norm=float(np.linalg.norm(self.forward(lifted) - intensity)),
            rank1_intensity_residual_norm=float(np.linalg.norm(np.abs(self.X @ x)**2 - intensity)),
        )
        return PhaseLiftResult(status, diagnostics, x, lifted)
