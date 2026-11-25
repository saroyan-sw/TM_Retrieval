import numpy as np

from tmbench.data import CalibrationSplit
from tmbench.gs_fit import fit_rows
from tmbench.solvers.gs import GerchbergSaxton


def test_complex_rank_deficient_projections_match_direct_least_squares():
    X = np.array([[1, 1j, 1], [-1j, 2, -1j], [0, 1, 0], [0, 0, 0]])
    y = np.array([2., 0., 1., 0.5])
    rng = np.random.default_rng(17)
    expected = (rng.standard_normal(3) + 1j * rng.standard_normal(3)) / np.sqrt(2)
    residuals = []
    for _ in range(5):
        phase = np.exp(1j * np.angle(X @ expected))
        expected, *_ = np.linalg.lstsq(X, y * phase, rcond=1e-12)
        residuals.append(np.linalg.norm(np.abs(X @ expected) - y) / np.linalg.norm(y))

    solver = GerchbergSaxton(X, rcond=1e-12)
    fitted = solver.fit(y, max_iter=5, seed=17)
    assert solver.rank == 2
    np.testing.assert_allclose(fitted.x, expected, atol=1e-12)
    np.testing.assert_allclose(fitted.z, X @ expected, atol=1e-12)
    np.testing.assert_allclose([h['relative_amplitude_residual'] for h in fitted.history], residuals)
    assert np.all(np.diff(residuals) <= 1e-12)


def test_zero_field_uses_phase_zero_for_nonzero_target(monkeypatch):
    class ZeroInitializer:
        def standard_normal(self, size):
            return np.zeros(size)

    monkeypatch.setattr(np.random, 'default_rng', lambda seed: ZeroInitializer())
    solver = GerchbergSaxton(np.eye(2), rcond=1e-12)
    fitted = solver.fit(np.array([2., 0.]), max_iter=1, seed=0)
    np.testing.assert_array_equal(fitted.x, [2., 0.])


def test_training_rows_share_pseudoinverse_and_retain_unsupported_masks(tmp_path):
    X = np.array([[1., 1], [1, -1], [-1, 1], [-1, -1]])
    intensities = np.array([[1., 2., 2., 0.], [2., 1., 1., 0.],
                            [2., 1., 1., 0.], [1., 2., 2., 0.]])
    valid = np.ones_like(intensities, dtype=bool)
    valid[0, 2] = False
    train = CalibrationSplit(X, intensities, valid, np.arange(4), np.array([8, 9, 10, 11]))

    status = fit_rows(train, dict(seed=4, max_iter=2, pinv_rcond=1e-12), tmp_path)

    assert status['pseudoinverse_count'] == 1
    assert status['attempted_rows'] == 4
    assert status['finite_estimate_rows'] == 2
    assert [row['status'] for row in status['rows']] == [
        'max_iter_reached', 'max_iter_reached', 'unsupported_mask', 'failed',
    ]
    assert [row['seed'] for row in status['rows'][:2]] == [[4, 8], [4, 9]]
    for row in status['rows'][:2]:
        with np.load(tmp_path / row['estimate_file'], allow_pickle=False) as estimate:
            assert estimate['A_row'].shape == (2,)
            assert int(estimate['detector_id']) == row['detector_id']
