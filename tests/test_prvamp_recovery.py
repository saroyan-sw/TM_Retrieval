import numpy as np
import pytest
from tmbench.solvers.prvamp import PrVAMP, VAMPConfig, complex_normal, phase_nmse


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["gaussian", "phase_only", "binary"])
def test_recovery_multiple_seeds_and_probe_types(seed, kind):
    rng = np.random.default_rng(seed)
    N, P, v = 20, 240, 1e-3
    if kind == "gaussian":
        phi = complex_normal(rng, (P,N), 1/N)
    elif kind == "phase_only":
        phi = np.exp(2j*np.pi*rng.random((P,N))) / np.sqrt(N)
    else:
        phi = rng.integers(0,2,(P,N)) / np.sqrt(N/2)
    x = complex_normal(rng,(N,))
    y = np.abs(phi@x + complex_normal(rng,(P,),v))
    result = PrVAMP(phi,noise_variance=v).fit(y,seed=10)
    assert phase_nmse(result.x,x,allow_conjugation=kind=="binary") < 2e-3
    assert result.history[-1]["amplitude_residual"] < 0.06
    assert np.all(np.isfinite(result.x))
    np.testing.assert_allclose(result.z, phi@result.x, atol=1e-12)


def test_tm_orientation_and_reusing_complex_probe_svd():
    rng = np.random.default_rng(40)
    N, P, M, v = 12, 144, 3, 1e-4
    patterns = np.exp(2j*np.pi*rng.random((N,P))) / np.sqrt(N)
    A = complex_normal(rng,(M,N))
    amplitudes = np.abs(A@patterns + complex_normal(rng,(M,P),v))
    model = PrVAMP(patterns.T,noise_variance=v)
    U_before = model.linear.U
    A_hat, results = model.fit_tm(amplitudes,seed=17)
    assert model.linear.U is U_before
    assert A_hat.shape == (M,N)
    assert max(phase_nmse(a,b) for a,b in zip(A_hat,A)) < 5e-4
    assert len(results) == M


def test_optional_spectral_initializer_and_restart_metadata():
    rng = np.random.default_rng(10)
    phi = complex_normal(rng,(192,16),1/16)
    x = complex_normal(rng,(16,))
    y = np.abs(phi@x + complex_normal(rng,(192,),1e-3))
    model = PrVAMP(phi,noise_variance=1e-3,config=VAMPConfig(initializer="spectral"))
    result = model.fit(y,n_restarts=1)
    assert phase_nmse(result.x,x) < 1e-3
    random_model = PrVAMP(phi,noise_variance=1e-3)
    restarted = random_model.fit(y,n_restarts=2)
    assert len(restarted.restart_residuals) == 2
    assert restarted.history[-1]["amplitude_residual"] == pytest.approx(min(restarted.restart_residuals))


def test_reproducibility_and_status():
    rng = np.random.default_rng(20)
    phi = complex_normal(rng,(64,8),1/8)
    y = np.abs(phi @ complex_normal(rng,(8,)))
    cfg = VAMPConfig(max_iter=1,min_iter=1,keep_iterates=True)
    model = PrVAMP(phi,noise_variance=1e-4,config=cfg)
    one, two = model.fit(y,seed=3), model.fit(y,seed=3)
    np.testing.assert_array_equal(one.x,two.x)
    assert not one.converged and one.status == "max_iter_reached"
    assert len(one.iterates) == 1


def test_invalid_inputs_and_zero_initialization_are_explicit():
    phi = np.ones((10,3))
    with pytest.raises(ValueError):
        PrVAMP(phi,noise_variance=0)
    model = PrVAMP(phi,noise_variance=0.1)
    with pytest.raises(ValueError,match="Initial field is zero"):
        model.fit(np.ones(10),x0=np.zeros(3))
    with pytest.raises(ValueError):
        model.fit(-np.ones(10))
    with pytest.raises(ValueError):
        model.fit(np.ones(3))
    with pytest.raises(ValueError):
        model.fit(np.zeros(10))
    with pytest.raises(ValueError):
        VAMPConfig(damping=0)
