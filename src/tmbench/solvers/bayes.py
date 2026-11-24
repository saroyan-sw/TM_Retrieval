"""Complex Bayesian row solvers. Source equations and deviations: docs/bayes_source_audit.md."""
from dataclasses import dataclass, field
import numpy as np

from .prvamp.linear import matvec
from .prvamp.moments import bessel_ratio, positive_scalar


class Design:
    """Reusable X-only preparation; sample-major forward map X @ a."""
    def __init__(self, X, gram=False):
        self.X = np.asarray(X, dtype=np.complex128 if np.iscomplexobj(X) else np.float64, order='F')
        if self.X.ndim != 2 or min(self.X.shape) == 0 or not np.isfinite(self.X).all():
            raise ValueError('X must be a nonempty finite matrix')
        self.P, self.N = self.X.shape
        self.H = self.X.conj().T if np.iscomplexobj(self.X) else self.X.T
        self.power = np.abs(self.X)**2
        self.colnorm = self.power.sum(axis=0)
        if np.any(self.colnorm == 0):
            raise ValueError('Unilluminated input columns are unsupported')
        self.probe_energy = float(self.colnorm.sum() / self.P)
        self.gram = self.H @ self.X if gram else None

    def forward(self, a):
        return matvec(self.X, a)

    def adjoint(self, z):
        return matvec(self.H, z)


@dataclass
class Result:
    x: np.ndarray
    variance: np.ndarray
    z: np.ndarray
    status: str
    history: list
    noise_variance: float
    prior_variance: float
    metadata: dict = field(default_factory=dict)

    @property
    def converged(self):
        return self.status == 'converged'


def channel(p, V, y, noise):
    """Exact Rician posterior, total complex variance, vector incoming variances."""
    S = V + noise
    radius = np.abs(p)
    k = 2 * y * radius / S
    ratio = bessel_ratio(k)
    direction = np.divide(p, radius, out=np.zeros_like(p), where=radius > 0)
    mean = noise / S * p + V / S * y * ratio * direction
    inv = np.divide(1., k, out=np.zeros_like(k), where=k > 0)
    complement = np.where(k > 1e6, inv / 2 + inv**2 / 8 + inv**3 / 8, 1 - ratio)
    variance = noise * V / S + (V / S)**2 * y**2 * complement * (1 + ratio)
    return mean, variance


def initialize(design, y, noise, prior, config, seed, initial=None):
    y = np.asarray(y, dtype=float)
    if y.shape != (design.P,) or not np.isfinite(y).all() or np.any(y < 0) or not np.any(y > 0):
        raise ValueError('Positive-energy finite amplitude vector required')
    noise, prior = positive_scalar(noise, 'noise'), positive_scalar(prior, 'prior')
    if not 0 < config.get('damping', 1.) <= 1:
        raise ValueError('damping is the new-value fraction in (0, 1]')
    if not isinstance(config['max_iter'], int) or config['max_iter'] < 1:
        raise ValueError('max_iter must be a positive integer')
    if not 1 <= config.get('min_iter', 1) <= config['max_iter'] or config.get('tol', 1e-6) <= 0:
        raise ValueError('Invalid stopping configuration')
    rng = np.random.default_rng(seed)
    if initial is None:
        a = np.sqrt(prior / 2) * (rng.standard_normal(design.N) + 1j*rng.standard_normal(design.N))
        c = np.full(design.N, prior)
    else:
        a, c = np.array(initial.x, copy=True), np.array(initial.variance, copy=True)
        if a.shape != (design.N,) or c.shape != a.shape or np.any(c <= 0) or not np.isfinite(a).all() or not np.isfinite(c).all():
            raise ValueError('Invalid warm-start moments')
    return y, noise, prior, rng, a, c


def history_entry(design, a, old, y, iteration, stabilized, **extra):
    if not np.isfinite(a).all():
        raise FloatingPointError('Nonfinite estimate')
    change = float(np.linalg.norm(a-old) / max(np.linalg.norm(old), 1e-30))
    residual = float(np.linalg.norm(np.abs(design.forward(a))-y) / np.linalg.norm(y))
    if not np.isfinite(change + residual):
        raise FloatingPointError('Nonfinite diagnostic')
    return dict(iteration=iteration, relative_change=change,
                relative_amplitude_residual=residual, stabilized=int(stabilized), **extra)


def prgamp(design, y, noise, prior, config, seed):
    """Damped sum-product GAMP, Schniter/Rangan Table I, Gaussian dense prior."""
    y, noise, prior, rng, a, c = initialize(design, y, noise, prior, config, seed)
    s = np.zeros(design.P, complex)
    ts = np.zeros(design.P)
    history, status = [], 'max_iter_reached'
    damping = config.get('damping', 1.)
    floor = 1e-12 / prior
    for iteration in range(1, config['max_iter'] + 1):
        old, old_c, old_s, old_ts = a.copy(), c.copy(), s.copy(), ts.copy()
        V = np.maximum(design.power @ c, 1e-12 * noise)
        p = design.forward(a) - V * s
        z, vz = channel(p, V, y, noise)
        s = (1-damping)*old_s + damping*(z-p)/V
        ts = (1-damping)*old_ts + damping*(1-vz/V)/V
        precision = design.power.T @ ts
        stabilized = np.count_nonzero(precision < floor)
        vr = 1 / np.maximum(precision, floor)
        r = a + vr * design.adjoint(s)
        a = (1-damping)*old + damping*prior/(prior+vr)*r
        c = (1-damping)*old_c + damping*prior*vr/(prior+vr)
        entry = history_entry(design, a, old, y, iteration, stabilized)
        history.append(entry)
        # Monitor all iterative messages, not just a potentially damped mean.
        state_change = max(entry['relative_change'], float(np.linalg.norm(c-old_c)/np.linalg.norm(old_c)),
                           float(np.linalg.norm(s-old_s)/max(np.linalg.norm(old_s), 1e-30)),
                           float(np.linalg.norm(ts-old_ts)/max(np.linalg.norm(old_ts), 1e-30)))
        entry['state_relative_change'] = state_change
        if iteration >= config.get('min_iter', 1) and state_change < config.get('tol', 1e-6):
            status = 'converged'
            break
    return Result(a, c, design.forward(a), status, history, noise, prior)
