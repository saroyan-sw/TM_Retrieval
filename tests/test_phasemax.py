import sys

import numpy as np
import pytest

from tmbench.data import CalibrationSplit
from tmbench.phasemax_fit import fit_rows
from tmbench.solvers.phasemax import MAX_MODES, MAX_PATTERNS, OperatorPhaseMax, PhaseMax, PhaseMaxConfig


@pytest.fixture
def backend():
    pytest.importorskip('cvxpy')
    pytest.importorskip('scs')


def test_complex_solution_matches_independent_disk_maximizers(backend):
    X = np.array([[1, 2j], [.3-1j, 1.2]])
    amplitudes = np.array([1.3, .8])
    anchor = np.array([.4+.7j, 1-.2j])
    # Express the anchor objective in field coordinates and optimize each disk.
    field_anchor = np.linalg.solve(X.conj().T, anchor)
    field = amplitudes * field_anchor / abs(field_anchor)
    expected = np.linalg.solve(X, field)

    result = PhaseMax(X, PhaseMaxConfig()).fit(amplitudes, anchor)
    assert result.status == 'optimal' and result.diagnostics['feasible']
    assert result.diagnostics['backend_warm_start'] is False
    np.testing.assert_allclose(result.x, expected, atol=1e-5)
    np.testing.assert_allclose(result.diagnostics['anchor_objective'],
                               np.vdot(anchor / np.linalg.norm(anchor), expected).real, atol=1e-5)


def test_padded_bounds_match_scalar_optimum(backend):
    X = np.array([[1.], [2.], [.5]])
    y = np.array([1., 1.8, .75])
    padding = .2 * np.sqrt(np.mean(y**2))
    expected = np.min((y + padding) / X[:, 0])
    result = PhaseMax(X, PhaseMaxConfig(amplitude_padding_fraction=.2)).fit(y, [1j])
    assert result.diagnostics['feasible']
    assert result.diagnostics['amplitude_padding'] == pytest.approx(padding)
    np.testing.assert_allclose(result.x, [1j * expected], atol=1e-5)
    assert np.max(abs(X @ result.x) - y) > 0


def test_nullspace_anchor_returns_unbounded_after_success(backend):
    solver = PhaseMax(np.array([[1., 0], [2, 0]]), PhaseMaxConfig())
    assert solver.fit([1., 3.], [1., 0]).x is not None
    result = solver.fit([1., 3.], [0., 1j])
    assert result.status == result.diagnostics['backend_status'] == 'unbounded'
    assert result.x is None and not result.diagnostics['feasible']
    with pytest.raises(ValueError, match='positive norm'):
        solver.fit([1., 3.], [0., 0])


def test_training_adapter_saves_anchor_separately_from_estimate(backend, tmp_path):
    X = np.array([[1., 0], [1, 1], [1, -1], [0, 1]])
    I = np.column_stack([abs(X @ np.array([1., .2]))**2] * 2)
    valid = np.ones_like(I, dtype=bool)
    valid[0, 1] = False
    train = CalibrationSplit(X, I, valid, np.arange(4), np.array([7, 11]))
    config = dict(seed=23, spectral_iterations=50)
    summary = fit_rows(train, config, tmp_path)
    first, second = summary['rows']
    assert first['status'] == 'optimal' and second['status'] == 'unsupported_mask'
    assert summary['anchor_policy'] == 'training_intensity_spectral'
    assert first['anchor_seed'] == [23, 7] and first['anchor_status'] == 'complete'
    assert first['anchor_file'] != first['estimate_file']
    assert first['backend_warm_start'] is False and first['backend_iterations'] > 0
    with np.load(tmp_path / first['anchor_file'], allow_pickle=False) as stored:
        anchor = stored['anchor']
        assert int(stored['detector_id']) == 7
    with np.load(tmp_path / first['estimate_file'], allow_pickle=False) as stored:
        estimate = stored['A_row']
    _, vectors = np.linalg.eigh(X.T @ np.diag(I[:, 0]) @ X)
    np.testing.assert_allclose(abs(np.vdot(vectors[:, -1], anchor)), 1., atol=1e-10)
    np.testing.assert_allclose(first['anchor_objective'], np.vdot(anchor, estimate).real)
    assert np.max(abs(X @ estimate) - np.sqrt(I[:, 0])) < 1e-5


@pytest.mark.parametrize('shape, expected', [
    ((4, MAX_MODES + 1), 'unsupported_size'),
    ((MAX_PATTERNS + 1, 2), 'unsupported_size'),
    ((4, 2), 'backend_unavailable'),
])
def test_size_caps_precede_backend_and_anchor_setup(shape, expected, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'cvxpy', None)
    X = np.ones(shape)
    I = np.ones((len(X), 2))
    train = CalibrationSplit(X, I, np.ones_like(I, bool), np.arange(len(X)), np.array([1, 2]))
    summary = fit_rows(train, dict(seed=23, spectral_iterations=50), tmp_path)
    assert summary['preparation_status'] == expected
    assert summary['attempted_rows'] == 2 and summary['finite_estimate_rows'] == 0
    assert [row['status'] for row in summary['rows']] == [expected, expected]
    assert not summary['backend_started']
    assert list(tmp_path.glob('*.npz')) == []


def test_operator_phase_max_matches_complex_disk_oracle_with_zero_bound():
    X = np.array([[1, 2j], [.3 - 1j, 1.2]])
    bounds = np.array([1.3, 0.])
    anchor = np.array([.4 + .7j, 1 - .2j])
    field_anchor = np.linalg.solve(X.conj().T, anchor)
    expected = np.linalg.solve(X, bounds * field_anchor / abs(field_anchor))
    solver = OperatorPhaseMax(X, PhaseMaxConfig(solver_tolerance=1e-8))

    result = solver.fit(bounds, anchor)
    assert result.status == 'converged' and result.diagnostics['feasible']
    np.testing.assert_allclose(result.x, expected, atol=1e-7)
    with pytest.raises(ValueError, match='full-column-rank'):
        OperatorPhaseMax(np.ones((3, 2)), PhaseMaxConfig())


def test_operator_phase_max_retains_budget_limited_estimate_above_dense_cap(tmp_path):
    X = np.tile(np.eye(2), (MAX_PATTERNS // 2 + 1, 1))
    intensity = np.abs(X @ np.array([1., .3]))**2
    train = CalibrationSplit(X, intensity[:, None], np.ones((len(X), 1), bool),
                             np.arange(len(X)), np.array([7]))
    summary = fit_rows(train, dict(implementation='operator', seed=23, spectral_iterations=5,
                                  max_iter=1), tmp_path)
    row = summary['rows'][0]
    assert summary['backend'] == 'scipy_admm'
    assert summary['finite_estimate_rows'] == 1
    assert row['status'] == 'max_iter_reached' and row['backend_iterations'] == 1
    assert not row['converged'] and not row['feasible']
    assert row['max_amplitude_violation'] > 0
