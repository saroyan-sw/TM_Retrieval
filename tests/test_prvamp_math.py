import numpy as np
import pytest
from scipy.special import i0e

from tmbench.solvers.prvamp import (gaussian_prior, rician_output, extrinsic, damp_message,
                    SVDLinear, complex_normal, phase_nmse, bessel_ratio)


def test_prior_example_and_exact_information_subtraction():
    posterior = gaussian_prior(np.array([3.0]), 1.0, prior_variance=4.0)
    np.testing.assert_allclose(posterior.mean, [2.4])
    np.testing.assert_allclose(posterior.variance, [0.8])
    message = extrinsic(posterior.mean, np.array([3.0]), 1.0, posterior.sensitivity)
    np.testing.assert_allclose(message.center, [0.0], atol=1e-14)
    assert message.precision == pytest.approx(0.25)
    assert not message.stabilized


def test_y_equals_three_example_from_notes():
    post = rician_output(np.array([2j]), 1.0, np.array([3.0]), 1.0)
    np.testing.assert_allclose(post.mean, [2.3685389565j], atol=1e-10)
    np.testing.assert_allclose(post.variance, [0.8771011245], atol=1e-10)
    outgoing = extrinsic(post.mean, np.array([2j]), 1.0, post.sensitivity)
    np.testing.assert_allclose(outgoing.center, [4.9987170746j], atol=1e-9)
    assert outgoing.precision == pytest.approx(0.1401193912, abs=1e-9)


def test_rician_moments_against_independent_cartesian_integration():
    # Independently integrate p(y|z) exp(-|z-p|^2/V) over a 2D complex grid.
    p, V, v, y = 0.8 + 0.6j, 1.3, 0.7, 1.8
    coordinates = np.linspace(-7.0, 7.0, 601)
    real, imag = np.meshgrid(coordinates, coordinates)
    z = real + 1j * imag
    a = np.abs(z)
    log_density = -(a - y)**2 / v + np.log(i0e(2 * a * y / v)) - np.abs(z - p)**2 / V
    density = np.exp(log_density - np.max(log_density))
    weights = density / density.sum()
    mean = np.sum(weights * z)
    variance = np.sum(weights * np.abs(z - mean)**2)
    post = rician_output(np.array([p]), 1/V, np.array([y]), v)
    np.testing.assert_allclose(post.mean, [mean], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(post.variance, [variance], rtol=1e-7, atol=1e-8)


def test_complex_divergence_matches_variance_identity():
    p = np.array([0.8 + 0.6j, -0.4 + 0.2j, 0j])
    y, tau, v, h = np.array([1.8, 0.9, 0.7]), 0.8, 0.7, 1e-5
    derivative_real = (rician_output(p + h, tau, y, v).mean.real
                       - rician_output(p - h, tau, y, v).mean.real) / (2*h)
    derivative_imag = (rician_output(p + 1j*h, tau, y, v).mean.imag
                       - rician_output(p - 1j*h, tau, y, v).mean.imag) / (2*h)
    divergence = np.mean(derivative_real + derivative_imag) / 2
    assert rician_output(p, tau, y, v).sensitivity == pytest.approx(divergence, rel=1e-8)


def test_zero_field_and_large_bessel_argument():
    zero = rician_output(np.array([0j]), 1.0, np.array([3.0]), 1.0)
    np.testing.assert_allclose(zero.mean, [0.0])
    np.testing.assert_allclose(zero.variance, [2.75])
    assert zero.sensitivity > 1
    with pytest.raises(ValueError, match="Positive Gaussian quotient"):
        extrinsic(zero.mean, np.array([0j]), 1.0, zero.sensitivity)
    clipped = extrinsic(zero.mean, np.array([0j]), 1.0, zero.sensitivity, stabilize=True)
    assert clipped.stabilized and clipped.precision > 0
    np.testing.assert_allclose(bessel_ratio(np.array([0, 1e3, 1e8, np.inf]))[0], 0)
    large = rician_output(np.array([1+0j]), 1e8, np.array([1.0]), 1e-10)
    assert np.all(np.isfinite(large.variance)) and np.all(large.variance > 0)


@pytest.mark.parametrize("P,N,rank_deficient", [(11,4,False), (4,9,False), (7,5,True)])
def test_svd_against_direct_linear_system_and_covariance(P, N, rank_deficient):
    rng = np.random.default_rng(32)
    phi = complex_normal(rng, (P,N))
    if rank_deficient:
        phi[:,-1] = phi[:,0]
    r2, p2 = complex_normal(rng,(N,)), complex_normal(rng,(P,))
    gamma2, tau2 = 0.6, 1.7
    stage = SVDLinear(phi)
    result = stage.solve(r2, gamma2, p2, tau2)
    Cx, Cz = stage.dense_covariances(gamma2, tau2)
    direct = Cx @ (gamma2*r2 + tau2*phi.conj().T@p2)
    np.testing.assert_allclose(result.x, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.z, phi @ direct, rtol=1e-12, atol=1e-12)
    assert result.variance_x == pytest.approx(np.trace(Cx).real/N)
    assert result.variance_z == pytest.approx(np.trace(Cz).real/P)
    assert N*result.alpha + P*result.beta == pytest.approx(N)


def test_scalar_linear_example_from_notes():
    result = SVDLinear(np.array([[2.0]])).solve(np.array([1+1j]), 4, np.array([3+1j]), 1)
    np.testing.assert_allclose(result.x, [1.25+0.75j])
    np.testing.assert_allclose(result.z, [2.5+1.5j])
    assert result.variance_z == pytest.approx(0.5)
    message = extrinsic(result.z, np.array([3+1j]), 1, result.beta)
    np.testing.assert_allclose(message.center, [2+2j])
    assert message.precision == pytest.approx(1)


def test_natural_parameter_damping():
    center, precision = damp_message(np.array([1j]), 2, np.array([3j]), 6, 0.5)
    assert precision == pytest.approx(4)
    np.testing.assert_allclose(center, [2.5j])


def test_phase_metric_and_real_probe_conjugation():
    x = np.array([1+2j, -3+0.7j])
    assert phase_nmse(x*np.exp(0.9j),x) < 1e-28
    assert phase_nmse(x.conj(),x,allow_conjugation=True) < 1e-28
    assert phase_nmse(2*x,x) == pytest.approx(1.0)  # no scale cheating
