"""Bounded PhaseMax with an explicit complex objective anchor."""

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._convex import BACKEND, UnsupportedSize, load_scs
from .prvamp.linear import matvec


MAX_MODES = 128
MAX_PATTERNS = 2048


@dataclass(frozen=True)
class PhaseMaxConfig:
    amplitude_padding_fraction: float = 0.
    max_iter: int = 5000
    solver_time_limit_seconds: float = 10.
    solver_tolerance: float = 1e-6
    feasibility_tolerance: float = 2e-5

    def __post_init__(self):
        if not np.isfinite(self.amplitude_padding_fraction) or self.amplitude_padding_fraction < 0:
            raise ValueError('amplitude_padding_fraction must be finite and nonnegative')
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer')
        if not np.isfinite(self.solver_time_limit_seconds) or self.solver_time_limit_seconds <= 0:
            raise ValueError('solver_time_limit_seconds must be finite and positive')
        for name in ('solver_tolerance', 'feasibility_tolerance'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name} must be between zero and one')

    @property
    def variant(self):
        if self.amplitude_padding_fraction > 0:
            return 'padded_amplitude_bounds'
        return 'phasemax'


@dataclass
class PhaseMaxResult:
    status: str
    diagnostics: dict
    x: np.ndarray | None = None


def _fit_parameters(amplitudes, anchor, patterns, modes, padding_fraction):
    if np.iscomplexobj(amplitudes):
        raise ValueError('Amplitudes must be real')
    y = np.asarray(amplitudes, dtype=float)
    if y.shape != (patterns,) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('Expected one finite nonnegative amplitude per probe')
    scale = float(np.linalg.norm(y) / np.sqrt(patterns))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Finite positive amplitude energy required')
    anchor = np.asarray(anchor, dtype=complex)
    if anchor.shape != (modes,) or not np.isfinite(anchor).all():
        raise ValueError('Anchor must have one finite coefficient per mode')
    anchor_norm = float(np.linalg.norm(anchor))
    if not np.isfinite(anchor_norm) or anchor_norm <= 0:
        raise ValueError('Anchor must have finite positive norm')
    padding = padding_fraction * scale
    bounds = y + padding
    scaled_bounds = bounds / scale
    if not np.isfinite(scaled_bounds).all():
        raise ValueError('Amplitude bounds must be finite')
    return scale, anchor / anchor_norm, padding, bounds, scaled_bounds


class PhaseMax:
    def __init__(self, X, config):
        X = np.asarray(X)
        if X.ndim != 2 or min(X.shape) == 0:
            raise ValueError('X must be a nonempty matrix')
        self.P, self.N = X.shape
        if self.N > MAX_MODES or self.P > MAX_PATTERNS:
            raise UnsupportedSize(
                f'{self.P} patterns and {self.N} modes exceed the dense PhaseMax '
                f'limits of {MAX_PATTERNS} patterns and {MAX_MODES} modes',
            )
        dtype = np.complex128 if np.iscomplexobj(X) else np.float64
        self.X = np.asarray(X, dtype=dtype)
        if not np.isfinite(self.X).all() or not np.any(self.X):
            raise ValueError('X must have finite entries and nonzero probe energy')
        self.config = config
        cp, self.backend_versions = load_scs('PhaseMax')
        self._solver_error = cp.SolverError
        self._anchor = cp.Parameter(2 * self.N)
        self._bounds = cp.Parameter(self.P, nonneg=True)
        self._coefficients = cp.Variable(2 * self.N)

        # Split the complex field into two real coordinates per amplitude disk.
        real = self._coefficients[:self.N]
        imag = self._coefficients[self.N:]
        field_real = self.X.real @ real
        field_imag = self.X.real @ imag
        if np.iscomplexobj(self.X):
            field_real -= self.X.imag @ imag
            field_imag += self.X.imag @ real
        magnitudes = cp.norm(cp.vstack([field_real, field_imag]), 2, axis=0)
        self._problem = cp.Problem(cp.Maximize(self._anchor @ self._coefficients),
                                   [magnitudes <= self._bounds])

    def fit(self, amplitudes, anchor):
        scale, direction, padding, bounds, scaled_bounds = _fit_parameters(
            amplitudes, anchor, self.P, self.N, self.config.amplitude_padding_fraction
        )
        self._bounds.value = scaled_bounds
        self._anchor.value = np.concatenate([direction.real, direction.imag])
        diagnostics = dict(backend=BACKEND, feasible=False, amplitude_scale=scale,
                           amplitude_padding=padding, backend_warm_start=False)
        try:
            self._problem.solve(solver=BACKEND, eps=self.config.solver_tolerance,
                                max_iters=self.config.max_iter, warm_start=False,
                                time_limit_secs=self.config.solver_time_limit_seconds)
        except self._solver_error as error:
            diagnostics.update(backend_status='solver_error', error=str(error))
            return PhaseMaxResult('solver_error', diagnostics)

        status = self._problem.status
        stats = self._problem.solver_stats
        detail = (stats.extra_stats or {}).get('info', {}).get('status')
        diagnostics.update(backend_status=status, backend_detail=detail,
                           backend_iterations=stats.num_iters, backend_solve_seconds=stats.solve_time,
                           backend_setup_seconds=stats.setup_time)
        if status not in ('optimal', 'optimal_inaccurate') or self._coefficients.value is None:
            return PhaseMaxResult(status, diagnostics)

        values = self._coefficients.value
        x = scale * (values[:self.N] + 1j * values[self.N:])
        predicted = np.abs(self.X @ x)
        objective = float(np.vdot(direction, x).real)
        if not np.isfinite(x).all() or not np.isfinite(predicted).all() or not np.isfinite(objective):
            return PhaseMaxResult('invalid_solution', diagnostics)
        violation = max(0., float(np.max(predicted - bounds)))
        relative_violation = violation / scale
        feasible = relative_violation <= self.config.feasibility_tolerance
        diagnostics.update(feasible=feasible, max_amplitude_violation=violation,
                           relative_amplitude_violation=relative_violation, anchor_objective=objective)
        if not feasible:
            return PhaseMaxResult('constraint_violation', diagnostics)
        return PhaseMaxResult(status, diagnostics, x)


class OperatorPhaseMax:
    """Solve the amplitude-disk problem with ADMM and a shared Gram factor."""

    backend = 'scipy_admm'

    def __init__(self, X, config):
        self.X = np.asarray(X, dtype=complex if np.iscomplexobj(X) else float)
        if self.X.ndim != 2 or min(self.X.shape) == 0 or not np.isfinite(self.X).all():
            raise ValueError('X must be a finite, nonempty matrix')
        self.P, self.N = self.X.shape
        self.adjoint_design = self.X.conj().T if np.iscomplexobj(self.X) else self.X.T
        self.config = config
        if self.P < self.N:
            raise ValueError('Operator PhaseMax requires a full-column-rank probe matrix')
        gram = (self.adjoint_design @ self.X) / self.P
        try:
            self.factor = cho_factor(gram, lower=True)
        except np.linalg.LinAlgError as error:
            raise ValueError('Operator PhaseMax requires a full-column-rank probe matrix') from error
        pivots = np.abs(self.factor[0].diagonal())
        if pivots.min() <= pivots.max() * np.sqrt(np.finfo(float).eps * max(self.P, self.N)):
            raise ValueError('Operator PhaseMax requires a numerically full-column-rank probe matrix')

    def fit(self, amplitudes, anchor, *, stop=None):
        cfg = self.config
        scale, direction, padding, _, bounds = _fit_parameters(
            amplitudes, anchor, self.P, self.N, cfg.amplitude_padding_fraction
        )
        start = time.perf_counter()
        # Use rho = 1/P, so the linear system has Gram X.H @ X / P.
        auxiliary = np.zeros(self.P, dtype=complex)
        dual = np.zeros_like(auxiliary)
        bound_norm = max(float(np.linalg.norm(bounds)), 1.)
        status = 'max_iter_reached'

        for iteration in range(1, cfg.max_iter + 1):
            rhs = direction + matvec(self.adjoint_design, auxiliary - dual) / self.P
            if np.iscomplexobj(self.X):
                estimate = cho_solve(self.factor, rhs, check_finite=False)
            else:
                parts = np.column_stack((rhs.real, rhs.imag))
                solved = cho_solve(self.factor, parts, check_finite=False)
                estimate = solved[:, 0] + 1j * solved[:, 1]
            field = matvec(self.X, estimate)
            if not np.isfinite(field).all():
                raise FloatingPointError('PhaseMax produced a nonfinite field')
            point = field + dual
            magnitudes = np.abs(point)
            projection_scale = np.ones(self.P)
            outside = magnitudes > bounds
            projection_scale[outside] = bounds[outside] / magnitudes[outside]
            previous_auxiliary = auxiliary
            auxiliary = point * projection_scale
            primal = field - auxiliary
            dual = dual + primal
            relative_primal = float(np.linalg.norm(primal) / bound_norm)
            dual_residual = float(np.linalg.norm(
                matvec(self.adjoint_design, auxiliary - previous_auxiliary) / self.P
            ))
            relative_violation = max(0., float(np.max(np.abs(field) - bounds)))
            feasible = relative_violation <= cfg.feasibility_tolerance
            if feasible and max(relative_primal, dual_residual) <= cfg.solver_tolerance:
                status = 'converged'
                break
            if ((stop is not None and stop())
                    or time.perf_counter() - start >= cfg.solver_time_limit_seconds):
                status = 'timed_out'
                break

        x = scale * estimate
        diagnostics = dict(
            backend=self.backend, backend_status=status, backend_iterations=iteration,
            backend_solve_seconds=time.perf_counter() - start, backend_warm_start=False,
            amplitude_scale=scale, amplitude_padding=padding, feasible=feasible,
            relative_primal_residual=relative_primal, dual_stationarity_residual=dual_residual,
            max_amplitude_violation=scale * relative_violation,
            relative_amplitude_violation=relative_violation,
            anchor_objective=float(np.vdot(direction, x).real),
        )
        return PhaseMaxResult(status, diagnostics, x)
