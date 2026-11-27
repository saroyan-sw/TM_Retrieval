import json

import numpy as np
import pytest

from tmbench.data import CalibrationSplit
from tmbench.phaselift_fit import fit_rows
from tmbench.solvers.phaselift import OperatorPhaseLift, OperatorPhaseLiftConfig, PhaseLift, PhaseLiftConfig


@pytest.fixture
def backend():
    cvxpy = pytest.importorskip('cvxpy')
    if 'SCS' not in cvxpy.installed_solvers():
        pytest.skip('PhaseLift tests require the SCS backend')
    return cvxpy


def test_exact_complex_lift_and_principal_factor(backend):
    X = np.array([[1, 0], [0, 1], [1, 1], [1, 1j], [1, -1j]])
    truth = np.array([1+.2j, .6-1.3j])
    I = np.abs(X @ truth)**2
    result = PhaseLift(X, PhaseLiftConfig(relative_noise_bound=0)).fit(I)
    assert result.status == 'optimal'
    assert result.diagnostics['backend'] == 'SCS'
    assert result.diagnostics['feasible']
    assert result.diagnostics['numerical_rank'] == 1
    np.testing.assert_allclose(result.lifted, np.outer(truth, truth.conj()), atol=1e-5)
    np.testing.assert_allclose(np.abs(X @ result.x)**2, I, atol=1e-5)


def test_noisy_bound_matches_analytical_scalar_solution(backend):
    X = np.ones((2, 1))
    I = np.array([1., 1.2])
    bound_fraction = .2
    bound = bound_fraction * np.linalg.norm(I)
    expected = I.mean() - np.sqrt((bound**2 - np.sum((I - I.mean())**2)) / len(I))
    constrained = PhaseLift(X, PhaseLiftConfig(relative_noise_bound=0))
    assert constrained.fit(np.ones(2)).x is not None
    inconsistent = constrained.fit(I)
    assert inconsistent.status == 'infeasible'
    assert not inconsistent.diagnostics['feasible']
    assert inconsistent.x is None

    relaxed = PhaseLift(X, PhaseLiftConfig(relative_noise_bound=bound_fraction)).fit(I)
    assert relaxed.diagnostics['feasible']
    assert relaxed.diagnostics['intensity_bound'] == pytest.approx(bound)
    np.testing.assert_allclose(relaxed.lifted, [[expected]], atol=1e-5)


def test_lifted_and_rank_one_residuals_are_saved_separately(backend, tmp_path):
    X = np.array([[1., 0], [0, 1], [1, 1], [1, -1]])
    I = np.abs(X @ np.array([1, 1j]))**2
    measurements = np.column_stack([I, I])
    valid = np.ones_like(measurements, dtype=bool)
    valid[0, 1] = False
    train = CalibrationSplit(X, measurements, valid, np.arange(4), np.array([7, 11]))
    report = fit_rows(train, dict(relative_noise_bound=0), tmp_path)
    assert report['backend_started'] and report['finite_estimate_rows'] == 1
    row = report['rows'][0]
    assert row['backend_status'] == 'optimal' and row['backend_iterations'] > 0
    assert row['feasible'] and row['numerical_rank'] == 2
    assert row['lifted_intensity_residual_norm'] < 1e-5
    assert row['rank1_intensity_residual_norm'] > 0.5
    assert report['rows'][1]['status'] == 'unsupported_mask'
    assert json.loads((tmp_path / 'status.json').read_text()) == report
    with np.load(tmp_path / row['estimate_file'], allow_pickle=False) as arrays:
        assert arrays['lifted'].shape == (2, 2) and arrays['A_row'].shape == (2,)


@pytest.mark.parametrize('shape', [(5, 33), (513, 2)])
def test_size_cap_precedes_backend_import(shape, tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, 'cvxpy', None)
    X = np.ones(shape)
    I = np.ones((len(X), 2))
    train = CalibrationSplit(X, I, np.ones_like(I, bool), np.arange(len(X)), np.array([1, 2]))
    status = fit_rows(train, {}, tmp_path)
    assert status['preparation_status'] == 'unsupported_size'
    assert not status['backend_started']
    assert status['attempted_rows'] == 2 and status['finite_estimate_rows'] == 0
    assert [row['status'] for row in status['rows']] == ['unsupported_size'] * 2
    assert list(tmp_path.glob('*.npz')) == []


def test_missing_optional_backend_has_explicit_status(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, 'cvxpy', None)
    X = np.ones((3, 1))
    I = np.ones((3, 1))
    train = CalibrationSplit(X, I, np.ones_like(I, bool), np.arange(3), np.array([1]))
    status = fit_rows(train, {}, tmp_path)
    assert status['rows'][0]['status'] == 'backend_unavailable'
    assert 'tmbench[convex]' in status['reason']


def test_operator_phase_lift_complex_adjoint_and_noiseless_recovery():
    X = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 1j], [1, -1j]])
    truth = np.array([1 + .2j, .6 - 1.3j])
    expected_lift = np.outer(truth, truth.conj())
    intensity = np.abs(X @ truth)**2
    cfg = OperatorPhaseLiftConfig(max_iter=300, regularization_fraction=1e-5)
    solver = OperatorPhaseLift(X, cfg)

    weights = np.array([1., .5, -.2, .3, -.7, .1])
    explicit_adjoint = sum(weight * np.outer(probe.conj(), probe) for weight, probe in zip(weights, X))
    np.testing.assert_allclose(solver.adjoint(weights), explicit_adjoint, atol=1e-14)
    np.testing.assert_allclose(solver.forward(expected_lift), intensity)
    assert float(weights @ solver.forward(expected_lift)) == pytest.approx(
        np.vdot(solver.adjoint(weights), expected_lift).real
    )

    result = solver.fit(intensity)
    assert result.status == 'converged'
    assert result.diagnostics['numerical_rank'] == 1
    np.testing.assert_allclose(result.lifted, expected_lift, atol=4e-5)


def test_operator_phase_lift_accepts_larger_design_and_retains_rank_two(tmp_path):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(520, 2))
    truth = np.array([1., 1j])
    intensity = np.abs(X @ truth)**2
    split = CalibrationSplit(X, intensity[:, None], np.ones((len(X), 1), bool),
                             np.arange(len(X)), np.array([7]))
    summary = fit_rows(split, dict(implementation='operator', max_iter=300,
                                  regularization_fraction=1e-5), tmp_path)
    row = summary['rows'][0]
    assert summary['backend'] == 'scipy_proximal_gradient'
    assert row['converged'] and row['numerical_rank'] == 2
    with np.load(tmp_path / row['estimate_file'], allow_pickle=False) as arrays:
        np.testing.assert_allclose(np.abs(X @ arrays['A_row'])**2, intensity, rtol=1e-4, atol=1e-6)
