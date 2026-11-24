import numpy as np
import pytest
from tmbench.solvers.bayes import Design, channel, prgamp
from tmbench.solvers.prvamp.moments import rician_output


def test_vector_channel_matches_independently_tested_scalar_moments():
    p = np.array([0, 1+2j, -0.2j, 1e7+1j], complex)
    y, V = np.array([1, 3, 2, 1e7]), np.array([1., 2., .3, .1])
    mean, variance = channel(p, V, y, .2)
    for i in range(len(p)):
        reference = rician_output(p[i:i+1], 1/V[i], y[i:i+1], .2)
        np.testing.assert_allclose(mean[i], reference.mean[0])
        np.testing.assert_allclose(variance[i], reference.variance[0])


def test_gamp_undamped_step_and_complex_adjoint():
    X = np.array([[1, 2j], [-1j, 3], [2, -1]], complex)
    d, y = Design(X), np.array([1., 2., 3.])
    prior, noise, seed = .7, .2, 42
    rng = np.random.default_rng(seed)
    a = np.sqrt(prior/2)*(rng.standard_normal(2)+1j*rng.standard_normal(2))
    V = abs(X)**2 @ np.full(2, prior)
    z, vz = channel(X@a, V, y, noise)
    s, ts = (z-X@a)/V, (1-vz/V)/V
    vr = 1/np.maximum(abs(X).T**2 @ ts, 1e-12/prior)
    expected = prior/(prior+vr)*(a+vr*(X.conj().T @ s))
    result = prgamp(d, y, noise, prior, {'max_iter': 1}, seed)
    np.testing.assert_allclose(result.x, expected)
    np.testing.assert_allclose(result.variance, prior*vr/(prior+vr))
    assert result.status == 'max_iter_reached'


def test_gamp_repeatability_and_input_rejection():
    d = Design(np.array([[1., 1], [-1, 1], [1, -1], [2, 1]]))
    cfg = dict(max_iter=3, damping=.2)
    a = prgamp(d, np.ones(4), .1, .5, cfg, [1, 2])
    b = prgamp(d, np.ones(4), .1, .5, cfg, [1, 2])
    np.testing.assert_array_equal(a.x, b.x)
    with pytest.raises(ValueError):
        prgamp(d, np.zeros(4), .1, .5, cfg, 0)
    with pytest.raises(ValueError):
        prgamp(d, np.ones(4), .1, .5, dict(max_iter=3, damping=0), 0)


def test_gamp_second_step_includes_onsager():
    X = np.array([[1, 2j], [-1j, 3], [2, -1]], complex)
    d, y = Design(X), np.array([1., 2., 3.])
    prior, noise, seed = .7, .2, 42
    rng = np.random.default_rng(seed)
    a0 = np.sqrt(prior/2)*(rng.standard_normal(2)+1j*rng.standard_normal(2))
    V0 = abs(X)**2 @ np.full(2, prior)
    s0 = (channel(X@a0, V0, y, noise)[0]-X@a0)/V0
    first = prgamp(d, y, noise, prior, {'max_iter': 1}, seed)
    V = abs(X)**2 @ first.variance
    p = X@first.x - V*s0
    z, vz = channel(p, V, y, noise)
    vr = 1/np.maximum(abs(X).T**2 @ ((1-vz/V)/V), 1e-12/prior)
    expected = prior/(prior+vr)*(first.x+vr*(X.conj().T @ ((z-p)/V)))
    second = prgamp(d, y, noise, prior, {'max_iter': 2}, seed)
    np.testing.assert_allclose(second.x, expected)


def test_bayes_adapter_retains_failure_rows(tmp_path):
    from tmbench.bayes_fit import fit_rows
    from tmbench.data import CalibrationSplit
    X = np.array([[1., 1], [-1, 1], [1, -1], [2, 1]])
    I = np.ones((4, 3))
    I[:, 2] = 0
    valid = np.ones_like(I, dtype=bool)
    valid[0, 1] = False
    split = CalibrationSplit(X, I, valid, np.arange(4), np.array([8, 16, 24]))
    cfg = dict(seed=4, noise_fraction=.01, max_iter=2)
    result = fit_rows(split, cfg, tmp_path, 'prgamp')
    assert [r['status'] for r in result['rows']] == ['max_iter_reached', 'unsupported_mask', 'failed']
    assert result['finite_estimate_coverage'] == 1/3


def test_gamp_small_complex_recovery_fixture():
    # Numerical solver check only; never included in public benchmark reports.
    rng = np.random.default_rng(83)
    X = (rng.normal(size=(100, 6))+1j*rng.normal(size=(100, 6)))/10
    a = rng.normal(size=6)+1j*rng.normal(size=6)
    y = abs(X@a)
    result = prgamp(Design(X), y, 1e-4, 2., dict(max_iter=200, min_iter=20, damping=.2, tol=1e-7), 2)
    assert np.linalg.norm(abs(X@result.x)-y)/np.linalg.norm(y) < .001
