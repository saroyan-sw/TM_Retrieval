# Bayesian baseline source audit

All implementations use sample-major X, the forward field X @ a, and its Hermitian
adjoint for inference. Noise and coefficient variances mean E|z-Ez|², the **total**
complex variance. Amplitudes are sqrt(camera counts). The circular Gaussian
field-noise likelihood is an approximation to this quantized camera dataset.
Neither released fitted TMs nor held-out responses enter fitting or pilots.

## I04 — prGAMP

Primary source: Schniter and Rangan, [Compressive Phase Retrieval via Generalized
Approximate Message Passing](https://arxiv.org/abs/1405.5618), Table I and equations
8–13, 18–19. Local verified PDF SHA256:
`b5562b9d4fea50b127f1bd28605ef7b22f89605f04d6ab678b8f17a5ac4e33d2`.

The implementation is an independent dense-Gaussian-prior specialization, not a
claim of MATLAB/reference-package parity. With B=|X|²:

- V=Bc; p=Xa−V*s_previous (Onsager correction).
- Exact Rician posterior mean z and total variance v give s=(z−p)/V and
  t_s=(1−v/V)/V. This derivative can have either sign.
- v_r=1/(B.T @ t_s); r=a+v_r*(X.H @ s).
- Gaussian input posterior a=prior*r/(prior+v_r), c=prior*v_r/(prior+v_r).

Fixed damping is a **new-value fraction**, applied to s, t_s, a and c. This is a
declared stabilization variant; the paper's adaptive damping, restart selection,
sparse priors and EM noise learning are not implemented in this I04 baseline.
The precision sum is floored at 1e-12/prior, and each affected coordinate is
counted in history. V has a 1e-12*noise floor. Output variances retain the exact
Rician moments before precision projection. Initial coefficients are seeded
CN(0,prior), c=prior and s=t_s=0. Noise is 0.01*mean training intensity; the prior
uses the energy moment (paper equation 19 at sparsity=1). The configuration is
fixed before running I04 and is not selected by held-out scores.

Stopping requires relative changes in mean, variance, s and t_s below tolerance
after min_iter. Finite max_iter estimates remain labelled max_iter_reached.
Nonfinite arithmetic/invalid inputs fail explicitly, and missing measurement
masks are unsupported in this initial shared-design adapter. No failed row is
removed from evaluation denominators.

Focused tests check vector channel moments against the I02 scalar implementation
(already checked by independent Cartesian integration), an explicit complex
one-step update, Hermitian adjoints, invalid inputs and seeded repeatability.
Full source-package parity and difficult-case verification remain I11 work.

## I05 — prVBEM and training pilots

Primary method source: Dremeau and Krzakala, [Phase Recovery from a Bayesian Point
of View: the Variational Approach](https://arxiv.org/abs/1410.1368), equations 8–18.
PDF SHA256 `776f5434416f32bd24c1df4b7832523caa3d497e92e191ea67c95ffb481fa28d`.
Optical application: [Reference-less measurement of the transmission matrix of a
highly scattering material using a DMD and phase retrieval
techniques](https://arxiv.org/abs/1502.03324), SHA256
`e0550699e849b2fcbc6560888691997607a2521e72576ab52d8e1a6fdae77c8f`.

The independent implementation uses the fully factorized complex Gaussian
coefficient posterior and circular phase factors. With field f=Xa, the phase
pseudo-observation is b=y*direction(f)*I1(k)/I0(k), k=2*y*abs(f)/noise.
Unlike GAMP, there is **no incoming field variance** in that denominator.
Coefficient variance c_i=(1/prior+||X_i||²/noise)^-1. A sequential coordinate
sweep updates a_i=c_i/noise * X_i.H @ (b−sum_{j!=i}X_j*a_j).
The implementation uses a cached Gram matrix to evaluate the same residual.
Equation 11's printed residual.H @ column is conjugated relative to the
conditional mean under f=Xa; we use column.H @ residual, derived directly from
the complex Gaussian conditional and tested on complex probes. No claim of
original author-code parity is made.

Noise EM is equation 18: (||y||²+||Xa||²−2 Re(b.H Xa)+sum_i||X_i||²*c_i)/P.
It is evaluated as ||Xa−b||²+sum(y²−|b|²)+sum_i||X_i||²*c_i to avoid cancellation.
The angular variance uses a stable asymptotic Bessel complement at large k.
Optional prior EM, mean(|a|²+c), is our explicit dense Gaussian EM extension.
Both fitted variances have an energy-scaled 1e-12 lower bound; activations are
counted. Random CN initialization, relative-state stopping (mean, variance,
noise and prior) and optional damping are implementation choices, distinct from
the paper's KL-difference stopping and original initializer. The public config
uses no VBEM damping and learns both variances. The variational lower bound is
saved for diagnostics and is independently checked for monotonicity on a fixture.

The Rice ICCP paper's section 4.3.2 uses prVBEM on one real-data row to estimate
signal/noise variance for other AMP rows. Our declared variant uses the **first
four already-frozen training detector IDs**, 20 iterations each, separate seeded
initializations, and median normalized noise/mean_intensity and
prior*mean_probe_energy/mean_intensity. These ratios are rescaled by each row's
training intensity. This is a robust aggregation policy, not the paper's exact
single-row transfer. Pilot selection and iteration budget are fixed before
held-out evaluation. No failed pilot is silently omitted; any failed pilot makes
all dependent fits pilot_failed. Finite iteration-limit pilots are allowed and
retain their nonconvergence status. The pilot routine accepts only a training
split; its source-index hash, per-row parameters/history/status and elapsed time
are saved in pilot.json. X-only Gram preparation is timed separately. Pilots are
included in end-to-end time and can also be used by prGAMP via a pilot config.

Focused tests independently integrate the phase posterior over angle, verify a
complex Gauss-Seidel sweep and both EM updates using direct sample residuals,
check the lower bound and recovery on a small numerical fixture, and check fixed
pilot selection, repeatability, cost accounting and failure propagation. The
standalone public run uses four pilots and then 50 VBEM iterations per detector.
