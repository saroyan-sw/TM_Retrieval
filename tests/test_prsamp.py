import json
import numpy as np
import pytest
from tmbench.solvers.bayes import Design, Result, prsamp
from tmbench.solvers.prvamp.moments import rician_output


def direct_sweeps(X, y, a0, c0, noise, prior, seed, damping, sweeps):
    """Slow oracle: scalar moments and recomputed fields, no incremental updates."""
    a, c = a0.copy(), c0.copy()
    g = np.zeros(len(y), complex)
    sigma, r = np.zeros(len(a)), np.zeros(len(a), complex)
    rng = np.random.default_rng(seed)
    orders = []
    def output(omega, V):
        moments = [rician_output(np.array([w]), 1/v, np.array([amp]), noise)
                   for w, v, amp in zip(omega, V, y)]
        means = np.array([m.mean[0] for m in moments])
        variances = np.array([m.variance[0] for m in moments])
        return (means-omega)/V, (variances/V-1)/V
    for sweep in range(sweeps):
        old, V0 = a.copy(), abs(X)**2@c
        omega0 = X@a-V0*g
        g, dg = output(omega0, V0)
        frozen_g = g.copy()
        order = rng.permutation(len(a))
        orders.append(order.tolist())
        for i in order:
            precision = -np.sum(abs(X[:, i])**2*dg)
            v = 1/max(precision, 1e-12/prior)
            d = 1. if sweep == 0 else damping
            sigma[i] = (1-d)*sigma[i]+d*v
            r[i] = (1-d)*r[i]+d*(a[i]+sigma[i]*np.sum(X[:, i].conj()*g))
            a[i] = prior*r[i]/(prior+sigma[i])
            c[i] = prior*sigma[i]/(prior+sigma[i])
            V = abs(X)**2@c
            omega = omega0+X@(a-old)-(V-V0)*frozen_g
            g, dg = output(omega, V)
    return a, c, orders


@pytest.mark.parametrize('kind', ['binary', 'signed', 'complex'])
def test_swept_updates_against_recomputed_reference(kind):
    X = np.array([[1., 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0]])
    if kind == 'signed':
        X = 2*X-1
    if kind == 'complex':
        X = X + 1j*np.array([[1, -1, 2], [2, 1, -1], [0, -2, 1], [1, 1, -1], [1, -1, 1]])
    a0, c0 = np.array([.1+.2j, -.1j, .3+.1j]), np.array([.3, .4, .5])
    y = np.array([.3, .4, .5, .6, .8])
    noise, prior, seed, damping = .3, .4, 77, .2
    initial = Result(a0, c0, X@a0, 'max_iter_reached', [], noise, prior)
    result = prsamp(Design(X), y, noise, prior, dict(max_iter=2, damping=damping), seed, initial)
    expected, variance, orders = direct_sweeps(X, y, a0, c0, noise, prior, seed, damping, 2)
    np.testing.assert_allclose(result.x, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(result.variance, variance, rtol=1e-11, atol=1e-12)
    assert [h['coordinate_order'] for h in result.history] == orders
    np.testing.assert_array_equal(initial.x, a0)
    np.testing.assert_array_equal(initial.variance, c0)


def test_samp_warm_start_adapter_cost_artifacts_and_repeatability(tmp_path):
    from tmbench.data import CalibrationSplit
    from tmbench.bayes_fit import fit_rows
    rng = np.random.default_rng(9)
    X = rng.choice([-1., 1], size=(30, 5))
    I = abs(X@(rng.normal(size=(5, 2))+1j*rng.normal(size=(5, 2))))**2
    split = CalibrationSplit(X, I, np.ones_like(I, bool), np.arange(30), np.array([6, 10]))
    cfg = dict(seed=2, noise_fraction=.01, max_iter=3, min_iter=3, damping=.1,
               warm_start=dict(max_iter=3, learn_noise=True, learn_prior=True))
    first = fit_rows(split, cfg, tmp_path/'first', 'prsamp')
    second = fit_rows(split, cfg, tmp_path/'second', 'prsamp')
    assert first['attempted_rows'] == first['finite_estimate_rows'] == 2
    assert first['warm_start_seconds'] > 0 and first['swept_seconds'] > 0
    for row in first['rows']:
        assert row['solve_seconds'] >= row['warm_start_seconds']+row['swept_seconds']
        assert row['warm_start_status'] == 'max_iter_reached'
        assert row['warm_start_iterations'] == 3
        assert (tmp_path/'first'/row['warm_start_file']).exists()
        a = np.load(tmp_path/'first'/row['estimate_file'])['A_row']
        b = np.load(tmp_path/'second'/row['estimate_file'])['A_row']
        np.testing.assert_array_equal(a, b)
        history = json.loads((tmp_path/'first'/f"row_{row['detector_id']}_history.json").read_text())
        assert all(sorted(h['coordinate_order']) == list(range(5)) for h in history)


def test_fixed_variance_warm_start_and_sweeps_recover_complex_fixture():
    # Numerical verification only, not a generated benchmark dataset.
    from tmbench.solvers.bayes import prvbem
    rng = np.random.default_rng(83)
    X = (rng.normal(size=(100, 6))+1j*rng.normal(size=(100, 6)))/10
    truth = rng.normal(size=6)+1j*rng.normal(size=6)
    y = abs(X@truth)
    design = Design(X, gram=True)
    warm = prvbem(design, y, 1e-4, 2.,
                  dict(max_iter=20, learn_noise=False, learn_prior=False), 2)
    result = prsamp(design, y, 1e-4, 2., dict(max_iter=5, min_iter=5, damping=.1), 3, warm)
    assert warm.noise_variance == result.noise_variance == 1e-4
    assert warm.prior_variance == result.prior_variance == 2.
    assert result.history[-1]['relative_amplitude_residual'] < .001
    phase = np.vdot(truth, result.x)
    aligned = result.x * phase.conjugate()/abs(phase)
    assert np.linalg.norm(aligned-truth)/np.linalg.norm(truth) < .001
