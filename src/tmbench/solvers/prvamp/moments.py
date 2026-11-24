"""Exact scalar moments used by the two nonlinear prVAMP blocks.

All variances are TOTAL complex variances E|u - E u|**2.
Physical noise acts BEFORE magnitude: y = |z + e|, e ~ CN(0, noise_variance).
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import i0e, i1e


def positive_scalar(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


@dataclass
class Moments:
    mean: np.ndarray
    variance: np.ndarray
    sensitivity: float


def gaussian_prior(r, gamma, prior_mean=0.0, prior_variance=1.0):
    """Block 1: p(x) CN(x; r, 1/gamma), with an iid Gaussian p(x)."""
    gamma = positive_scalar(gamma, "gamma")
    prior_variance = positive_scalar(prior_variance, "prior_variance")
    r = np.asarray(r, dtype=np.complex128)
    mu = np.broadcast_to(np.asarray(prior_mean, dtype=np.complex128), r.shape)
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(mu)):
        raise ValueError("r and prior_mean must be finite.")
    variance = 1.0 / (1.0 / prior_variance + gamma)
    mean = variance * (mu / prior_variance + gamma * r)
    return Moments(mean, np.full(r.shape, variance), gamma * variance)


def bessel_ratio(kappa):
    """I1(kappa)/I0(kappa), evaluated without exponential overflow.

    For very large kappa use its asymptotic expansion. That branch also
    avoids inf/inf in scipy's scaled functions for extremely large inputs.
    """
    k = np.asarray(kappa, dtype=float)
    if np.any(np.isnan(k)) or np.any(k < 0):
        raise ValueError("kappa must be nonnegative and not NaN.")
    large = k > 1e6
    safe = np.where(large, 1.0, k)
    regular = i1e(safe) / i0e(safe)
    inv = np.divide(1.0, k, out=np.zeros_like(k), where=k > 0)
    asymptotic = 1.0 - inv / 2.0 - inv**2 / 8.0 - inv**3 / 8.0
    return np.where(large, asymptotic, regular)


def rician_output(p, tau, y, noise_variance):
    """Block 2: exact scalar MMSE moments, then average sensitivity.

    q_j(z) proportional to Rice(y_j; |z|, noise_variance) CN(z; p_j, 1/tau).
    The scalar posterior is generally neither Gaussian nor circular.
    Returned sensitivity can exceed one; do not silently interpret it as
    a valid positive-precision Gaussian quotient in that case.
    """
    tau = positive_scalar(tau, "tau")
    v = positive_scalar(noise_variance, "noise_variance")
    p = np.asarray(p, dtype=np.complex128)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("p and y must have the same nonempty shape.")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("p must be finite; amplitudes y must be finite and nonnegative.")
    V = 1.0 / tau
    S = V + v
    amplitude = np.abs(p)
    kappa = 2.0 * y * amplitude / S
    R = bessel_ratio(kappa)
    direction = np.divide(p, amplitude, out=np.zeros_like(p), where=amplitude > 0)
    mean = (v / S) * p + (V / S) * y * R * direction
    # Evaluate 1-R^2 as (1-R)(1+R). For very large kappa evaluate the
    # small difference directly from the same asymptotic expansion.
    inv = np.divide(1.0, kappa, out=np.zeros_like(kappa), where=kappa > 0)
    one_minus_R = np.where(kappa > 1e6, inv / 2 + inv**2 / 8 + inv**3 / 8, 1 - R)
    angular_variance = one_minus_R * (1 + R)
    variance = v * V / S + (V / S)**2 * y**2 * angular_variance
    return Moments(mean, variance, float(tau * np.mean(variance)))
