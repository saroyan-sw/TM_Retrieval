"""Solver dispatch and measurements shared by the I11 and I12 campaigns."""

from dataclasses import dataclass
import time

import numpy as np

from .solvers.bayes import Design, prgamp, prsamp, prvbem
from .solvers.gs import GerchbergSaxton
from .solvers.phaselift import OperatorPhaseLift, OperatorPhaseLiftConfig
from .solvers.phasemax import OperatorPhaseMax, PhaseMaxConfig
from .solvers.prvamp import PrVAMP, SVDLinear, VAMPConfig
from .solvers.prvamp.linear import matvec
from .solvers.prvamp.utils import complex_normal
from .solvers.wf import WFConfig, WirtingerFlow


PAPER_ITERATIONS = {
    'prgamp': 50, 'prsamp': 50, 'prvbem': 50, 'prvamp': 50,
    'gs50': 50, 'gs100': 100, 'wf50': 50, 'wf500': 500,
    'phaselift': 50, 'phasemax': 50,
}
AMP_METHODS = frozenset(('prgamp', 'prvbem', 'prsamp', 'prvamp'))


def method_family(method):
    if method.startswith('gs'):
        return 'gs'
    if method.startswith('wf'):
        return 'wf'
    return method


def prepare(family, X):
    if family == 'prvamp':
        return SVDLinear(X)
    if family in ('prvbem', 'prsamp', 'prgamp'):
        return Design(X, gram=family != 'prgamp')
    if family == 'gs':
        return GerchbergSaxton(X, rcond=1e-12)
    if family == 'wf':
        return WirtingerFlow(X)
    if family == 'phaselift':
        return OperatorPhaseLift(X, OperatorPhaseLiftConfig(solver_time_limit_seconds=3600))
    if family == 'phasemax':
        model = OperatorPhaseMax(X, PhaseMaxConfig(solver_time_limit_seconds=3600))
        return model, WirtingerFlow(model.X)
    raise ValueError(f'Unknown benchmark method: {family}')


@dataclass
class Solve:
    x: np.ndarray | None
    metrics: dict
    history: list


def training_variances(X, y, noise_fraction=0.01):
    energy = float(np.mean(y**2))
    probe_energy = float(np.einsum('ij,ij->', X.conj(), X).real / len(X))
    return noise_fraction * energy, (1 - noise_fraction) * energy / probe_energy


def fit(method, prepared, y, seed, variances, *, simulated, max_iter, cpu_seconds=None):
    """Fit one detector or simulated target with an independent CPU budget."""
    family = method_family(method)
    y = np.asarray(y, dtype=float)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    stop = None if cpu_seconds is None else lambda: time.process_time() - cpu_start >= cpu_seconds
    noise, prior = variances
    initial_cpu = initial_wall = 0.
    details = {}

    if family == 'prvamp':
        cfg = VAMPConfig(max_iter=max_iter, min_iter=min(10, max_iter), tolerance=1e-7, damping=.2)
        result = PrVAMP(prepared, noise_variance=noise, prior_variance=prior, config=cfg).fit(
            y, seed=seed, stop=stop
        )
    elif family in ('prgamp', 'prvbem', 'prsamp'):
        cfg = dict(max_iter=max_iter, min_iter=min(10, max_iter), tol=1e-7,
                   damping={'prgamp': .2, 'prsamp': .1, 'prvbem': 1.}[family],
                   learn_noise=not simulated, learn_prior=not simulated)
        if family == 'prsamp':
            warm_cfg = dict(cfg, max_iter=50, min_iter=10, damping=1.,
                            learn_noise=False, learn_prior=False)
            warm = prvbem(prepared, y, noise, prior, warm_cfg, [*seed, 606], stop=stop)
            initial_cpu = time.process_time() - cpu_start
            initial_wall = time.perf_counter() - wall_start
            result = prsamp(prepared, y, noise, prior, cfg, seed, initial=warm, stop=stop)
            details['warm_iterations'] = len(warm.history)
            details['warm_status'] = warm.status
            details['completed_sweeps'] = sum(
                len(entry['coordinate_order']) == prepared.N for entry in result.history
            )
            details['coordinate_updates'] = sum(len(entry['coordinate_order']) for entry in result.history)
            if not details['coordinate_updates']:
                result.status = 'timed_out_warm_start'
        else:
            solver = prgamp if family == 'prgamp' else prvbem
            result = solver(prepared, y, noise, prior, cfg, seed, stop=stop)
        details.update(final_noise_variance=result.noise_variance, final_prior_variance=result.prior_variance)
    elif family == 'gs':
        result = prepared.fit(y, max_iter=max_iter, seed=seed, stop=stop)
    elif family == 'wf':
        cfg = WFConfig(policy='original', max_iter=max_iter)
        intensity = y**2
        initial = prepared.spectral_initialize(intensity, iterations=cfg.spectral_iterations, seed=seed)
        initial_cpu = time.process_time() - cpu_start
        initial_wall = time.perf_counter() - wall_start
        result = prepared.fit(intensity, initial, cfg, stop=stop)
    elif family == 'phaselift':
        prepared.config = OperatorPhaseLiftConfig(max_iter=max_iter, solver_time_limit_seconds=3600)
        result = prepared.fit(y**2, seed=seed, stop=stop)
    else:
        model, initializer = prepared
        model.config = PhaseMaxConfig(max_iter=max_iter, solver_time_limit_seconds=3600)
        if simulated:
            anchor = complex_normal(np.random.default_rng(seed), model.N)
        else:
            anchor = initializer.spectral_initialize(y**2, iterations=50, seed=seed)
        initial_cpu = time.process_time() - cpu_start
        initial_wall = time.perf_counter() - wall_start
        result = model.fit(y, anchor, stop=stop)

    diagnostics = getattr(result, 'diagnostics', {})
    history = getattr(result, 'history', [])
    x = result.x if result.status != 'timed_out_warm_start' else None
    if x is not None and not np.isfinite(x).all():
        raise FloatingPointError('Nonfinite estimate')
    status = result.status
    return Solve(x, dict(
        status=status, converged=status in ('converged', 'fixed_point_tolerance', 'optimal'),
        iterations=diagnostics.get('backend_iterations', len(history)),
        solve_cpu_seconds=time.process_time()-cpu_start, solve_wall_seconds=time.perf_counter()-wall_start,
        initialization_cpu_seconds=initial_cpu, initialization_wall_seconds=initial_wall,
        stabilized_updates=sum(entry.get('stabilized', 0) for entry in history),
        constraint_violation=diagnostics.get('relative_amplitude_violation'),
        feasible=diagnostics.get('feasible'), lifted_rank=diagnostics.get('numerical_rank'),
        **details,
    ), history)


def prediction_metrics(X, estimate, observed, valid):
    y = observed[valid]
    prediction = np.abs(matvec(X, estimate))[valid]
    if not len(y) or not np.isfinite(prediction).all():
        return None
    error = float(np.sum((prediction-y)**2))
    energy = float(y @ y)
    return dict(count=len(y), squared_error=error, energy=energy, amplitude_mse=error / len(y),
                relative_residual=float(np.sqrt(error / energy)) if energy > 0 else None)
