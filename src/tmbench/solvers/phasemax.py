"""Bounded PhaseMax with an explicit complex objective anchor."""

from dataclasses import dataclass

import numpy as np

from ._convex import BACKEND, UnsupportedSize, load_scs


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
        if np.iscomplexobj(amplitudes):
            raise ValueError('Amplitudes must be real')
        y = np.asarray(amplitudes, dtype=float)
        if y.shape != (self.P,) or not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError('Expected one finite nonnegative amplitude per probe')
        scale = float(np.linalg.norm(y) / np.sqrt(self.P))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Finite positive amplitude energy required')
        anchor = np.asarray(anchor, dtype=complex)
        if anchor.shape != (self.N,) or not np.isfinite(anchor).all():
            raise ValueError('Anchor must have one finite coefficient per mode')
        anchor_norm = float(np.linalg.norm(anchor))
        if not np.isfinite(anchor_norm) or anchor_norm <= 0:
            raise ValueError('Anchor must have finite positive norm')
        direction = anchor / anchor_norm

        padding = self.config.amplitude_padding_fraction * scale
        bounds = y + padding
        scaled_bounds = bounds / scale
        if not np.isfinite(scaled_bounds).all():
            raise ValueError('Amplitude bounds must be finite')
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
