import numpy as np
from scipy.special import i0, i1
from tmbench.solvers.bayes import Design, phase_mean, prvbem


def test_phase_factor_by_angular_quadrature():
    theta = np.linspace(0, 2*np.pi, 100000, endpoint=False)
    field, y, noise = np.array([.7+.3j]), np.array([1.2]), .8
    weights = np.exp(2*y[0]*np.real(np.exp(-1j*theta)*field[0])/noise)
    expected = np.sum(y[0]*np.exp(1j*theta)*weights)/np.sum(weights)
    np.testing.assert_allclose(phase_mean(field, y, noise)[0][0], expected, atol=1e-12)


def test_vbem_coordinate_sweep_and_em_against_direct_residual():
    X = np.array([[1., 2j], [1j, -2], [2., 1j], [-1j, 1.]])
    y, noise, prior = np.array([1., 2., .8, 1.5]), .8, .5
    rng = np.random.default_rng(14)
    a = np.sqrt(prior/2)*(rng.standard_normal(2)+1j*rng.standard_normal(2))
    z = X@a
    k = 2*y*abs(z)/noise
    pseudo = y*z/abs(z)*i1(k)/i0(k)
    c = 1/(1/prior+np.sum(abs(X)**2, axis=0)/noise)
    for i in range(2):
        residual = pseudo-X@a+X[:, i]*a[i]
        a[i] = c[i]/noise*np.vdot(X[:, i], residual)
    z = X@a
    expected_noise = (y@y+np.vdot(z,z).real-2*np.vdot(pseudo,z).real+np.sum(abs(X)**2@c))/len(y)
    expected_prior = np.mean(abs(a)**2+c)
    actual = prvbem(Design(X, gram=True), y, noise, prior,
                    dict(max_iter=1, learn_noise=True, learn_prior=True), 14)
    np.testing.assert_allclose(actual.x, a)
    np.testing.assert_allclose(actual.variance, c)
    np.testing.assert_allclose(actual.noise_variance, expected_noise)
    np.testing.assert_allclose(actual.prior_variance, expected_prior)


def test_vbem_bound_is_monotone_and_repeatable():
    rng = np.random.default_rng(4)
    X = (rng.normal(size=(80, 5))+1j*rng.normal(size=(80, 5)))/8
    y = abs(X@(rng.normal(size=5)+1j*rng.normal(size=5)))
    cfg = dict(max_iter=100, learn_noise=True, learn_prior=True, tol=1e-8)
    design = Design(X, gram=True)
    result = prvbem(design, y, .01, 1., cfg, 10)
    repeat = prvbem(design, y, .01, 1., cfg, 10)
    np.testing.assert_array_equal(result.x, repeat.x)
    assert np.min(np.diff([h['variational_bound'] for h in result.history])) > -1e-8
    assert result.history[-1]['relative_amplitude_residual'] < .01


def test_pilot_training_only_selection_cost_and_failure(tmp_path):
    from tmbench.data import CalibrationSplit
    from tmbench.pilots import estimate_variances
    from tmbench.bayes_fit import fit_rows
    X = np.array([[1., -1], [1, 1], [-1, 1], [2, 1]])
    I = np.array([[1., 2, 3], [2, 3, 4], [1, 2, 3], [4, 5, 6]])
    split = CalibrationSplit(X, I, np.ones_like(I, dtype=bool), np.array([2, 7, 11, 15]), np.array([9, 5, 20]))
    cfg = dict(rows=2, max_iter=3, seed=1)
    design = Design(X, gram=True)
    result = estimate_variances(split, design, cfg)
    assert result['status'] == 'complete'
    assert [r['detector_id'] for r in result['rows']] == [9, 5]
    assert result['seconds'] >= sum(r['seconds'] for r in result['rows']) > 0
    assert result['training_sample_count'] == 4
    assert result['noise_fraction'] > 0 and result['prior_factor'] > 0
    # Changing a nonpilot row cannot change the hyperparameters.
    I[:, 2] = 10000
    repeat = estimate_variances(split, design, cfg)
    assert repeat['noise_fraction'] == result['noise_fraction']
    assert repeat['prior_factor'] == result['prior_factor']
    split.valid[0, 0] = False
    failed = fit_rows(split, dict(seed=1, max_iter=3, noise_fraction=.01, pilot=cfg), tmp_path, 'prvbem')
    assert failed['finite_estimate_rows'] == 0
    assert [r['status'] for r in failed['rows']] == ['pilot_failed']*3
