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

The [Rice ICCP paper](https://rice.box.com/s/4jipt0ebr9lm03ccg53fqgf4tsodcpco),
section 4.3.2, uses prVBEM on one real-data row to estimate
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

## I06 — prSAMP with prVBEM warm start

Primary source: Rajaei et al., [Robust Phase Retrieval with the Swept Approximate
Message Passing (prSAMP) Algorithm](https://www.ipol.im/pub/art/2017/178/),
Algorithms 1–3, [article PDF](https://www.ipol.im/pub/art/2017/178/article.pdf)
SHA256 `a33421f53e001f5033c16bf91a15be937a4748adff96075142fa151bf1414a6f`.
The [reference archive](https://www.ipol.im/pub/art/2017/178/prsamp_2.0.zip), SHA256
`6024033d37f538ffdc7d8d3beed1a6b64db7f97019648b56f227d92c279f0735`, was inspected
locally, including calibration/cgamp.c, cpr.c and cgb.c. Its C code is GPL-3.0-or-later;
it is not vendored here. This Python implementation follows the mathematical
algorithm with the deviations below, rather than reproducing the binary sparse
C implementation or claiming reference executable parity.

At sweep start, V=|X|²c, omega=Xa−V*g_previous. The same verified Rician moments
give g=(z−omega)/V and dg=(posterior_variance/V−1)/V. Draw a seeded permutation
of every coefficient. For each i, form sigma_i=1/(−sum |X_i|²*dg), then
r_i=a_i+sigma_i*(X_i.H @ g), and apply the dense Gaussian posterior. Damping
applies to sigma and r, with an **undamped first sweep**, as in the reference C.
The configured damping=0.1 is the new-value fraction (reference old fraction 0.9).

After each coordinate update, V accumulates |X_i|²*(new_c_i−old_c_i) and omega
accumulates X_i*(new_a_i−old_a_i)−(new_V−old_V)*g_sweep_start. Recompute the
output channel immediately. The fixed sweep-start g in this correction is not
replaced by the evolving g. The paper's displayed omega line omits the previous
omega term; calibration/cgamp.c confirms accumulation with +=. Our independent
tests recompute the entire field and variance after each update to check this.

Supported designs are finite dense real signed, binary or complex matrices with
nonzero column energy. Complex designs use |X|² and X.H, extending the released
binary sparse calibration implementation; a sparse {0,1}-only kernel would be
incorrect for this Rice ±1 dataset. All 1600 illuminated modes are retained.
Gaussian complex unknowns are supported; the paper's binary-object prior,
2D median regularization, block-designed probes, MPI/OMP implementation and
automatic restarts are not included. Relative mean/variance stopping replaces
the reference's absolute average-change criterion. Precision is floored at
1e-12/prior and V at 1e-12*noise, with counts recorded, instead of the reference's
fixed negative-variance substitution. Nonfinite results fail explicitly.

The warm-start policy comes from Rice ICCP section 4.3.2, where prSAMP is
initialized by prVBEM. That paper's PDF SHA256 is
`320ae232f5582110f90a58c9e9bc56b5ae859043590d39c2954c712b21ce22e0`.
Our bounded integration config uses the same four training-only variance pilots
as I05, then 20 prVBEM iterations for each row, followed by five full prSAMP
sweeps. The iteration budgets are declared smoke-run budgets, not the paper's
50-iteration configuration or an I12 matched comparison. prVBEM's mean **and
diagonal variance** initialize prSAMP (variance transfer is our explicit
extension). prSAMP's noise/prior remain the training-pilot values; the warm
VBEM's fitted parameters are saved separately and do not silently replace them.
Warm seeds use [seed, detector_id, 606]; sweep seeds use [seed, detector_id].
Every actual coordinate permutation is saved in row history. Nonconverged finite
warm starts are retained and labelled. Warm arrays/history, status, iterations
and seconds are saved before the swept solver starts. Row solve_seconds includes
both stages; preparation, pilots, warm starts, swept time and end-to-end time are
reported separately. Tests verify this accounting and deterministic replay.

### Observed pilot collapse and explicit I06 follow-up

The I05 public run's learned mean fields collapsed almost to zero. Pilot median
noise/mean_intensity was 1.011745 and prior*probe_energy/mean_intensity 0.011746;
the training-only I06 pilot trial also produced near-zero estimates. The original
trial is preserved, including five completed rows, one interruption and 26 unrun
rows. A short training-only fixed-variance burn-in diagnostic still subsequently
collapsed under EM; it is recorded as an unsuccessful diagnostic.

The follow-up **prsamp_i06_fixed_variance.json** is an explicit configuration
change, now the CLI default: no pilots; noise and prior use the same declared
training-energy assumptions as I04, and VBEM's optional noise/prior M-steps are
disabled during the 20-iteration warm start. Five swept iterations follow. The
mean-field coordinate and phase updates are unchanged. This tests I06 with a
training initialization without transferring unusable pilot estimates. It does
not claim to solve variance estimation on this camera, nor does it reproduce
the paper's pilot transfer. Both attempts and costs are kept; no configuration
is selected using validation/test responses.
