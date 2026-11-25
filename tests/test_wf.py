import json

import numpy as np

from tmbench.data import CalibrationSplit
from tmbench.solvers.wf import WFConfig, WirtingerFlow
from tmbench.wf_fit import fit_rows


def test_wirtinger_gradient_matches_complex_finite_difference():
    X = np.array([[1, 2j], [-1+1j, 3], [2, -1j]])
    x = np.array([.3+.2j, -.1+.5j])
    intensities = np.array([1.4, .7, .2])
    solver = WirtingerFlow(X)
    loss, gradient = solver.loss_gradient(x, intensities)

    def direct_loss(point):
        return np.mean((np.abs(X @ point)**2 - intensities)**2) / 2

    np.testing.assert_allclose(loss, direct_loss(x))
    epsilon = 1e-6
    for direction in [np.array([.2+.5j, -.4j]), np.array([-.1j, .3+.2j])]:
        numerical = (direct_loss(x + epsilon*direction) - direct_loss(x - epsilon*direction)) / (2*epsilon)
        np.testing.assert_allclose(2*np.vdot(gradient, direction).real, numerical, rtol=1e-7, atol=1e-9)


def test_spectral_initializer_matches_dense_eigenvector_and_scale():
    X = np.array([[1, 2j], [-1j, 3], [2, -.5j], [1j, -1]])
    intensities = np.array([2., 6., .4, 0.])
    solver = WirtingerFlow(X)
    a = solver.spectral_initialize(intensities, iterations=50, seed=[4, 15])
    covariance = X.conj().T @ np.diag(intensities) @ X / len(X)
    _, vectors = np.linalg.eigh(covariance)
    np.testing.assert_allclose(abs(np.vdot(vectors[:, -1], a))/np.linalg.norm(a), 1., atol=1e-12)
    np.testing.assert_allclose(np.vdot(a, a).real, X.shape[1]*intensities.sum()/np.sum(abs(X)**2))
    np.testing.assert_array_equal(a, solver.spectral_initialize(intensities, iterations=50, seed=[4, 15]))


def test_original_step_uses_ramp_and_adaptive_step_backtracks():
    solver = WirtingerFlow(np.eye(2))
    initial = np.array([3+1j, .5j])
    intensities = np.array([3., .8])
    initial_loss, gradient = solver.loss_gradient(initial, intensities)
    norm_squared = np.vdot(initial, initial).real
    original = solver.fit(intensities, initial, WFConfig('original', 1))
    step = (1 - np.exp(-1/330)) / norm_squared
    np.testing.assert_allclose(original.x, initial - step*gradient)
    assert original.status == 'max_iter_reached'
    adaptive = solver.fit(intensities, initial, WFConfig('adaptive', 2, mu_max=20.))
    assert adaptive.history[0]['backtracks'] > 0
    for entry in adaptive.history:
        expected_decrease = 2e-4 * entry['step_size'] * entry['gradient_norm_before_step']**2
        assert entry['intensity_loss'] <= initial_loss - expected_decrease
        initial_loss = entry['intensity_loss']
    stationary = solver.fit(intensities, np.sqrt(intensities), WFConfig('adaptive', 2))
    assert stationary.status == 'converged'
    assert stationary.history == []


def test_fit_records_intensity_policy_and_invalid_rows(tmp_path):
    X = np.array([[1., 1], [1, -1], [-1, 1], [-1, -1]])
    values = np.array([[220, 220, 0], [170, 170, 0], [170, 170, 0], [220, 220, 0]], dtype=np.uint8)
    valid = np.ones_like(values, dtype=bool)
    valid[0, 1] = False
    split = CalibrationSplit(X, values, valid, np.arange(4), np.array([3, 7, 9]))
    status = fit_rows(split, dict(seed=4, policy='adaptive', max_iter=2), tmp_path)
    assert status['policy'] == 'adaptive'
    assert status['finite_estimate_coverage'] == 1/3
    assert [row['status'] for row in status['rows']] == ['max_iter_reached', 'unsupported_mask', 'failed']
    row = status['rows'][0]
    with np.load(tmp_path / row['estimate_file'], allow_pickle=False) as saved:
        predicted = np.abs(X @ saved['A_row'])**2
    np.testing.assert_allclose(row['final_intensity_loss'], np.mean((predicted-values[:, 0])**2)/2)
    history = json.loads((tmp_path / 'row_3_history.json').read_text())
    assert history[-1]['intensity_loss'] == row['final_intensity_loss']
