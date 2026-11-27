"""Run the cross-solver I11 verification campaign and save the evidence."""

import argparse
from pathlib import Path
import json
import platform

import cvxpy as cp
import h5py
import numpy as np
import scipy

from tmbench.adapters.rice_phase import load
from tmbench.benchmark import fit, method_family, prepare, training_variances
from tmbench.fit import write_json
from tmbench.solvers.bayes import Design, prvbem
from tmbench.solvers.phaselift import OperatorPhaseLift, OperatorPhaseLiftConfig
from tmbench.solvers.phasemax import OperatorPhaseMax, PhaseMax, PhaseMaxConfig
from tmbench.solvers.prvamp.utils import complex_normal, phase_nmse


METHODS = ('prgamp', 'prsamp', 'prvbem', 'prvamp', 'gs50', 'wf50', 'phaselift', 'phasemax')


def numerical_campaign():
    rng = np.random.default_rng(8611)
    patterns, modes = 72, 6
    gaussian = complex_normal(rng, (patterns, modes))
    signed = rng.choice([-1., 1.], (patterns, modes))
    amplitude = (signed + 1) / 2
    correlated = gaussian.copy()
    correlated[:, 1] = correlated[:, 0] + 1e-4 * correlated[:, 1]
    singular = gaussian.copy()
    singular[:, 1] = singular[:, 0]
    cases = {
        'complex_gaussian': (gaussian, 0.01), 'signed': (signed, 0.01),
        'binary_amplitude': (amplitude, 0.01), 'noisy': (gaussian, 2.),
        'correlated': (correlated, 0.01), 'rank_deficient': (singular, 0.01),
        'zero_observations': (signed, 0.),
    }
    truth = complex_normal(rng, modes, variance=1/modes)
    report = []
    for case, (X, noise) in cases.items():
        measured = np.abs(X @ truth + complex_normal(rng, patterns, noise))
        # Include an exactly zero sample, supported by the physical amplitude model.
        measured[0] = 0.
        if case == 'zero_observations':
            measured[:] = 0.
        for method in METHODS:
            context = dict(case=case, method=method, seed=[811, 4])
            try:
                model = prepare(method_family(method), X)
                kwargs = dict(simulated=True, max_iter=15)
                variances = (max(noise, 1e-8), 1/modes)
                a = fit(method, model, measured, context['seed'], variances, **kwargs)
                b = fit(method, model, measured, context['seed'], variances, **kwargs)
            except ValueError as error:
                expected = (case == 'zero_observations'
                            or method == 'phasemax' and case == 'rank_deficient')
                if not expected:
                    raise
                report.append(dict(context, status='expected_rejection', reason=str(error)))
                continue
            assert case != 'zero_observations' and a.x is not None and b.x is not None, context
            np.testing.assert_array_equal(a.x, b.x, err_msg=str(context))
            assert a.metrics['status'] == b.metrics['status']
            train_residual = np.linalg.norm(np.abs(X @ a.x)-measured)/np.linalg.norm(measured)
            metrics = {key: value for key, value in a.metrics.items() if not key.endswith('_seconds')}
            report.append(dict(context, **metrics, repeatable=True, train_residual=float(train_residual),
                               phase_nmse=phase_nmse(a.x, truth),
                               ambiguity_nmse=phase_nmse(a.x, truth, allow_conjugation=np.isrealobj(X))))
    return report


def convex_references():
    rng = np.random.default_rng(1112)
    X = complex_normal(rng, (27, 3))
    truth = complex_normal(rng, 3)
    intensities = np.abs(X @ truth)**2
    lift = OperatorPhaseLift(X, OperatorPhaseLiftConfig(
        max_iter=3000, regularization_fraction=.01, solver_tolerance=1e-9
    )).fit(intensities, seed=23)
    variable = cp.Variable((3, 3), hermitian=True)
    dense = np.array([np.outer(row, row.conj()).reshape(-1) for row in X])
    prediction = cp.real(dense @ cp.vec(variable, order='C'))
    penalty = lift.diagnostics['trace_penalty']
    objective = cp.sum_squares(prediction-intensities)/2 + penalty * cp.real(cp.trace(variable))
    reference = cp.Problem(cp.Minimize(objective), [variable >> 0])
    reference.solve(solver='SCS', eps=1e-8, max_iters=15000)
    lift_objective = lift.diagnostics['lifted_intensity_residual_norm']**2/2 + penalty*np.trace(lift.lifted).real
    np.testing.assert_allclose(lift_objective, reference.value, rtol=1e-6)

    anchor = truth + complex_normal(rng, 3, .05)
    cfg = PhaseMaxConfig(max_iter=10000, solver_tolerance=1e-7, solver_time_limit_seconds=15)
    bounded = PhaseMax(X, cfg).fit(np.sqrt(intensities), anchor)
    split = OperatorPhaseMax(X, cfg).fit(np.sqrt(intensities), anchor)
    assert bounded.x is not None and split.x is not None and split.diagnostics['feasible']
    np.testing.assert_allclose(split.diagnostics['anchor_objective'],
                               bounded.diagnostics['anchor_objective'], rtol=1e-5)
    return dict(phaselift=dict(operator_objective=float(lift_objective), scs_objective=reference.value,
                              operator_status=lift.status, reference_status=reference.status),
                phasemax=dict(operator_objective=split.diagnostics['anchor_objective'],
                              scs_objective=bounded.diagnostics['anchor_objective'],
                              operator_status=split.status, reference_status=bounded.status,
                              constraint_violation=split.diagnostics['relative_amplitude_violation']))


def rice_variance_diagnostic(data_dir):
    manifest = json.loads(Path('data/manifests/rice_phase_slm_40x40.json').read_text())
    policy = json.loads(Path('configs/public/rice_phase_slm_40x40.json').read_text())
    train = load(data_dir, manifest, policy).train
    model = Design(train.X, gram=True)
    detector = int(train.detector_ids[0])
    y = train.amplitudes[:, 0]
    noise, prior = training_variances(model.X, y)
    outcomes = []
    for learn_noise, learn_prior in [(True, True), (True, False), (False, False)]:
        cfg = dict(max_iter=50, min_iter=50, learn_noise=learn_noise, learn_prior=learn_prior)
        result = prvbem(model, y, noise, prior, cfg, [20261124, detector, 501])
        outcomes.append(dict(config=cfg, status=result.status, coefficient_norm=float(np.linalg.norm(result.x)),
                             initial_noise=noise, final_noise=result.noise_variance,
                             initial_prior=prior, final_prior=result.prior_variance,
                             training_relative_residual=result.history[-1]['relative_amplitude_residual'],
                             final_variational_bound=result.history[-1]['variational_bound']))
    return dict(detector=detector, training_shape=list(train.X.shape), outcomes=outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw/phase_slm_40x40'))
    args = parser.parse_args()
    record = dict(platform=platform.platform(), versions=dict(
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        cvxpy=cp.__version__, h5py=h5py.__version__,
    ))
    record['controls'] = numerical_campaign()
    record['convex_references'] = convex_references()
    record['rice_training_only'] = rice_variance_diagnostic(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, record)
    print(f"Verified {len(record['controls'])} controlled cases, both convex references and the Rice diagnostic")


if __name__ == '__main__':
    main()
