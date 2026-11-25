"""Wirtinger Flow for intensity observations and sample-major probes."""

from dataclasses import dataclass

import numpy as np

from .prvamp.linear import matvec


@dataclass(frozen=True)
class WFConfig:
    policy: str
    max_iter: int
    spectral_iterations: int = 50
    mu_max: float = 0.2
    ramp_tau: float = 330.0
    gradient_tolerance: float = 1e-8

    def __post_init__(self):
        if self.policy not in ('original', 'adaptive'):
            raise ValueError('policy must be original or adaptive')
        for name in ('max_iter', 'spectral_iterations'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('mu_max', 'ramp_tau', 'gradient_tolerance'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


@dataclass
class WFResult:
    x: np.ndarray
    z: np.ndarray
    status: str
    history: list
    initial_loss: float
    final_loss: float


class WirtingerFlow:
    def __init__(self, X):
        dtype = np.complex128 if np.iscomplexobj(X) else np.float64
        self.X = np.asarray(X, dtype=dtype)
        if self.X.ndim != 2 or min(self.X.shape) == 0 or not np.isfinite(self.X).all():
            raise ValueError('X must be a nonempty finite matrix')
        self.P, self.N = self.X.shape
        self.adjoint = self.X.conj().T if np.iscomplexobj(self.X) else self.X.T
        self.probe_energy = float(np.vdot(self.X, self.X).real)
        if not np.isfinite(self.probe_energy) or self.probe_energy <= 0:
            raise ValueError('X must have finite positive probe energy')

    def _intensities(self, intensities):
        if np.iscomplexobj(intensities):
            raise ValueError('Intensities must be real')
        values = np.asarray(intensities, dtype=float)
        if values.shape != (self.P,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Expected one finite nonnegative intensity per sample')
        energy = float(values.mean())
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError('Finite positive mean intensity required')
        return values, energy

    def spectral_initialize(self, intensities, *, iterations, seed):
        """Power iteration on X.H diag(I) X / P, with the paper's energy scale."""
        if type(iterations) is not int or iterations < 1:
            raise ValueError('iterations must be a positive integer')
        values, _ = self._intensities(intensities)
        rng = np.random.default_rng(seed)
        direction = rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N)
        direction /= np.linalg.norm(direction)
        for _ in range(iterations):
            field = matvec(self.X, direction)
            direction = matvec(self.adjoint, values * field) / self.P
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm <= 0:
                raise FloatingPointError('Invalid spectral power iterate')
            direction /= norm
        scale_squared = self.N * values.sum() / self.probe_energy
        if not np.isfinite(scale_squared) or scale_squared <= 0:
            raise FloatingPointError('Invalid spectral initialization scale')
        return direction * np.sqrt(scale_squared)

    def loss_gradient(self, x, intensities):
        """Return f and d f / d conjugate(x), half the real-coordinate gradient."""
        field = matvec(self.X, x)
        loss, residual = self._loss_residual(field, intensities)
        return loss, self._gradient(field, residual)

    def _loss_residual(self, field, intensities):
        residual = np.abs(field)**2 - intensities
        loss = float(residual @ residual / (2 * self.P))
        return loss, residual

    def _gradient(self, field, residual):
        return matvec(self.adjoint, residual * field) / self.P

    def fit(self, intensities, initial, config):
        values, mean_intensity = self._intensities(intensities)
        x = np.asarray(initial, dtype=np.complex128)
        if x.shape != (self.N,) or not np.isfinite(x).all():
            raise ValueError('Invalid initial coefficient vector')
        norm_squared = float(np.vdot(x, x).real)
        if not np.isfinite(norm_squared) or norm_squared <= 0:
            raise ValueError('Initial coefficients must have finite positive energy')

        field = matvec(self.X, x)
        loss, residual = self._loss_residual(field, values)
        initial_loss = loss
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite initial intensity loss')
        history = []
        status = 'max_iter_reached'
        adaptive = config.policy == 'adaptive'
        step_cap = config.mu_max / norm_squared
        step = step_cap
        for iteration in range(1, config.max_iter + 1):
            gradient = self._gradient(field, residual)
            grad_norm_squared = float(np.vdot(gradient, gradient).real)
            if not np.isfinite(grad_norm_squared):
                raise FloatingPointError(f'Nonfinite gradient at iteration {iteration}')
            relative_gradient = np.sqrt(grad_norm_squared * norm_squared) / mean_intensity**2
            if relative_gradient <= config.gradient_tolerance:
                status = 'converged'
                break

            if adaptive:
                step = min(2 * step, step_cap)
            else:
                mu = min(-np.expm1(-iteration / config.ramp_tau), config.mu_max)
                step = mu / norm_squared

            for backtracks in range(21 if adaptive else 1):
                with np.errstate(over='ignore', invalid='ignore'):
                    candidate = x - step * gradient
                    next_field = matvec(self.X, candidate)
                    next_loss, next_residual = self._loss_residual(next_field, values)
                if np.isfinite(next_loss) and (
                    not adaptive or next_loss <= loss - 2e-4 * step * grad_norm_squared
                ):
                    break
                step *= 0.5
            else:
                if adaptive:
                    status = 'line_search_failed'
                    break
                raise FloatingPointError(f'Nonfinite original WF update at iteration {iteration}')

            x, field, residual, loss = candidate, next_field, next_residual, next_loss
            history.append(dict(iteration=iteration, intensity_loss=loss,
                                step_size=float(step), backtracks=backtracks,
                                gradient_norm_before_step=float(np.sqrt(grad_norm_squared))))
        return WFResult(x, field, status, history, initial_loss, loss)
