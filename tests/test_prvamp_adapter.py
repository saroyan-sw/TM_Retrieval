import json
import numpy as np
import pytest
from tmbench.data import CalibrationSplit
from tmbench.fit import fit_rows, hyperparameters
from tmbench.solvers.prvamp import PrVAMP, VAMPConfig, SVDLinear


def test_real_optimization_matches_complex_reference():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(48, 8))
    y = np.abs(X @ (rng.normal(size=8) + 1j * rng.normal(size=8)))
    cfg = VAMPConfig(max_iter=25, min_iter=20)
    real = PrVAMP(X, noise_variance=.1, config=cfg).fit(y, seed=6)
    reference = PrVAMP(X.astype(complex), noise_variance=.1, config=cfg).fit(y, seed=6)
    np.testing.assert_allclose(real.x, reference.x, rtol=1e-8, atol=1e-9)


def test_shared_svd_and_every_row_status(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    X = rng.choice([-1, 1], size=(48, 4))
    I = np.abs(X @ (rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))))**2
    I[:, 2] = 0
    valid = np.ones_like(I, dtype=bool)
    valid[0, 1] = False
    split = CalibrationSplit(X, I, valid, np.arange(48), np.array([8, 16, 24]))
    original = np.linalg.svd
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(np.linalg, 'svd', counted)
    report = fit_rows(split, dict(seed=4, noise_fraction=.01, max_iter=2, min_iter=1), tmp_path)
    assert len(calls) == 1
    assert report['attempted_rows'] == 3
    assert [r['status'] for r in report['rows']] == ['max_iter_reached', 'unsupported_mask', 'failed']
    assert report['finite_estimate_coverage'] == 1 / 3
    assert json.loads((tmp_path / 'status.json').read_text()) == report
    assert np.load(tmp_path / 'row_8.npz')['A_row'].shape == (4,)
    with pytest.raises(FileExistsError):
        fit_rows(split, dict(seed=4, noise_fraction=.01), tmp_path)


def test_energy_scaling_uses_float_and_total_complex_variance():
    params = hyperparameters(np.full((5, 4), 255, dtype=np.uint8), np.full(5, 10.), .01)
    assert params['noise_variance'] == 1
    assert params['prior_variance'] == pytest.approx(99 / (4 * 255**2))
