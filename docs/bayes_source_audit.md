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
