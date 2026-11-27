"""Readable scalar-variance, sum-product complex prVAMP implementation.

Core equations follow Erik's VAMP_Completed_Handwritten_Derivations.pdf.
Stabilization, natural-parameter damping, optional spectral initialization,
and convergence checks are explicit implementation choices, not a claim of
bit-for-bit reproduction of the original MATLAB implementation.
"""

from dataclasses import dataclass, field
import numpy as np
from .moments import gaussian_prior, rician_output, positive_scalar
from .messages import extrinsic, damp_message
from .linear import SVDLinear
from .utils import complex_normal, amplitude_residual


@dataclass
class VAMPConfig:
    max_iter: int = 300
    min_iter: int = 20
    tolerance: float = 1e-7
    damping: float = 0.5  # new-message fraction in NATURAL parameters
    initializer: str = "random"  # also "spectral"
    stabilize: bool = True
    sensitivity_epsilon: float = 1e-8
    precision_min: float = 1e-12
    precision_max: float = 1e12
    keep_iterates: bool = False

    def __post_init__(self):
        if not isinstance(self.max_iter, int) or not isinstance(self.min_iter, int):
            raise ValueError("Iteration counts must be integers.")
        if not 1 <= self.min_iter <= self.max_iter:
            raise ValueError("Require 1 <= min_iter <= max_iter.")
        positive_scalar(self.tolerance, "tolerance")
        if not 0 < self.damping <= 1:
            raise ValueError("damping must be in (0, 1].")
        if self.initializer not in ("random", "spectral"):
            raise ValueError("initializer must be 'random' or 'spectral'.")
        if not 0 < self.sensitivity_epsilon < 0.5:
            raise ValueError("sensitivity_epsilon must be in (0, 0.5).")
        if not 0 < self.precision_min < self.precision_max < np.inf:
            raise ValueError("Invalid precision bounds.")


@dataclass
class VAMPResult:
    x: np.ndarray
    z: np.ndarray
    history: list
    converged: bool
    status: str
    iterates: list = field(default_factory=list)
    restart_residuals: list = field(default_factory=list)


class PrVAMP:
    """Recover x from amplitudes y = |phi @ x + e|.

    noise_variance is E|e|**2, not variance of amplitude or intensity noise.
    prior_variance is E|x_n - prior_mean_n|**2 (one shared iid variance).
    Construct once for a fixed phi to share its SVD between detector pixels.
    """

    def __init__(self, phi, *, noise_variance, prior_variance=1.0,
                 prior_mean=0.0, config=None):
        self.linear = phi if isinstance(phi, SVDLinear) else SVDLinear(phi)
        self.noise_variance = positive_scalar(noise_variance, "noise_variance")
        self.prior_variance = positive_scalar(prior_variance, "prior_variance")
        self.prior_mean = np.broadcast_to(np.asarray(prior_mean, dtype=np.complex128),
                                          (self.linear.N,)).copy()
        if not np.all(np.isfinite(self.prior_mean)):
            raise ValueError("prior_mean must be finite.")
        self.config = config if config is not None else VAMPConfig()

    def _initial_x(self, y, rng):
        N = self.linear.N
        if self.config.initializer == "random":
            return self.prior_mean + complex_normal(rng, (N,), self.prior_variance)
        # Optional teaching initializer: leading eigenvector of phi^H diag(y^2) phi.
        # No ground truth used. Capping weights limits unusually bright samples.
        weights = np.minimum(y**2, 3.0 * np.mean(y**2))
        matrix = self.linear.phi.conj().T @ (weights[:, None] * self.linear.phi)
        _, eigenvectors = np.linalg.eigh(matrix)
        direction = eigenvectors[:, -1]
        predicted = np.abs(self.linear.forward(direction))
        scale = np.dot(predicted, y) / max(np.dot(predicted, predicted), 1e-30)
        return scale * direction

    def fit(self, y, *, seed=0, x0=None, n_restarts=1, stop=None):
        """Run prVAMP; optional restarts selected by measured amplitude residual.

        No correctness is inferred from convergence alone. This nonconvex
        algorithm can reach a poor fixed point. Inspect residuals, status,
        stabilization counts and held-out measurements on real experiments.
        """
        y = np.asarray(y, dtype=float)
        if y.shape != (self.linear.P,) or not np.all(np.isfinite(y)) or np.any(y < 0):
            raise ValueError("y must be finite nonnegative amplitudes with shape (P,).")
        if not np.any(y):
            raise ValueError("All-zero amplitudes are uninformative for this implementation.")
        if not isinstance(n_restarts, int) or n_restarts < 1:
            raise ValueError("n_restarts must be a positive integer.")
        if x0 is not None and n_restarts != 1:
            raise ValueError("Use one run when providing x0; otherwise all restarts are identical.")
        rng = np.random.default_rng(seed)
        candidates = []
        for _ in range(n_restarts):
            candidates.append(self._fit_once(y, rng, x0, stop))
            if stop is not None and stop():
                break
        residuals = [float(np.linalg.norm(np.abs(result.z) - y) / np.linalg.norm(y))
                     for result in candidates]
        best = candidates[int(np.argmin(residuals))]
        best.restart_residuals = residuals
        return best

    def _fit_once(self, y, rng, x0, stop=None):
        cfg = self.config
        x_initial = self._initial_x(y, rng) if x0 is None else np.asarray(x0, dtype=np.complex128).copy()
        if x_initial.shape != (self.linear.N,) or not np.all(np.isfinite(x_initial)):
            raise ValueError("x0 must be finite with shape (N,).")
        p1 = self.linear.forward(x_initial)
        if np.linalg.norm(p1) == 0:
            raise ValueError("Initial field is zero: magnitude data cannot choose a phase from symmetry.")
        r1 = x_initial.copy()
        gamma1 = 1.0 / self.prior_variance
        mean_field_variance = self.prior_variance * np.sum(self.linear.s**2) / self.linear.P
        tau1 = float(np.clip(1.0 / mean_field_variance, cfg.precision_min, cfg.precision_max))
        history, iterates = [], []
        previous_x = x_initial
        status = "max_iter_reached"
        options = dict(stabilize=cfg.stabilize, epsilon=cfg.sensitivity_epsilon,
                       precision_min=cfg.precision_min, precision_max=cfg.precision_max)

        for iteration in range(cfg.max_iter):
            # BLOCK 1: Gaussian prior moments. Its exact extrinsic factor is
            # just the prior itself, for EVERY incoming (r1, gamma1).
            signal = gaussian_prior(r1, gamma1, self.prior_mean, self.prior_variance)
            r2 = self.prior_mean.copy()
            gamma2 = 1.0 / self.prior_variance

            # BLOCK 2: exact Rician scalar moments, then Gaussian projection
            # and information subtraction with the averaged sensitivity.
            output = rician_output(p1, tau1, y, self.noise_variance)
            to_linear = extrinsic(output.mean, p1, tau1, output.sensitivity, **options)
            p2, tau2 = to_linear.center, to_linear.precision

            # BLOCKS 3 AND 4: same inputs, same belief, two outputs.
            linear = self.linear.solve(r2, gamma2, p2, tau2)
            to_prior = extrinsic(linear.x, r2, gamma2, linear.alpha, **options)
            to_output = extrinsic(linear.z, p2, tau2, linear.beta, **options)

            # Damping mixes centers weighted by precision (natural parameters).
            next_r1, next_gamma1 = damp_message(r1, gamma1, to_prior.center,
                                                to_prior.precision, cfg.damping)
            next_p1, next_tau1 = damp_message(p1, tau1, to_output.center,
                                                to_output.precision, cfg.damping)
            relative_change = float(np.linalg.norm(linear.x - previous_x)
                                    / max(np.linalg.norm(previous_x), 1e-30))
            gap = float(np.linalg.norm(linear.x - signal.mean)
                        / max(np.linalg.norm(linear.x), 1e-30))
            message_change = max(
                float(np.linalg.norm(next_p1 - p1) / max(np.linalg.norm(p1), 1e-30)),
                abs(next_tau1 - tau1) / tau1,
                float(np.linalg.norm(next_r1 - r1) / max(np.linalg.norm(r1), 1e-30)),
                abs(next_gamma1 - gamma1) / gamma1)
            history.append(dict(
                iteration=iteration + 1,
                amplitude_residual=float(np.linalg.norm(np.abs(linear.z) - y) / np.linalg.norm(y)),
                relative_change=relative_change, belief_gap=gap, message_change=message_change,
                alpha1=signal.sensitivity, beta1_raw=output.sensitivity,
                beta1_used=to_linear.sensitivity_used, alpha2=linear.alpha, beta2=linear.beta,
                gamma1=gamma1, gamma2=gamma2, tau1=tau1, tau2=tau2,
                variance_x=linear.variance_x, variance_z=linear.variance_z,
                stabilized=int(to_linear.stabilized) + int(to_prior.stabilized) + int(to_output.stabilized)))
            if cfg.keep_iterates:
                iterates.append(linear.x.copy())
            r1, gamma1, p1, tau1 = next_r1, next_gamma1, next_p1, next_tau1
            previous_x = linear.x
            if iteration + 1 >= cfg.min_iter and max(relative_change, gap, message_change) < cfg.tolerance:
                status = "fixed_point_tolerance"
                break
            if stop is not None and stop():
                status = "timed_out"
                break

        return VAMPResult(linear.x.copy(), linear.z.copy(), history,
                          status == "fixed_point_tolerance", status, iterates)

    def fit_tm(self, amplitudes, *, seed=0, n_restarts=1):
        """Rows of amplitudes: detector pixels; columns: P displayed patterns.

        Use phi = patterns.T for patterns of shape (N, P), with NO conjugation.
        Return A_hat of shape (M, N) and per-row results. Each row has its own
        undetermined global phase, and real probes also allow conjugation.
        """
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.ndim != 2 or amplitudes.shape[0] == 0 or amplitudes.shape[1] != self.linear.P:
            raise ValueError("amplitudes must have shape (M, P), with M >= 1.")
        if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0):
            raise ValueError("amplitudes must be finite and nonnegative.")
        seeds = np.random.SeedSequence(seed).spawn(amplitudes.shape[0])
        results = [self.fit(row, seed=row_seed, n_restarts=n_restarts)
                   for row, row_seed in zip(amplitudes, seeds)]
        return np.stack([result.x for result in results]), results
