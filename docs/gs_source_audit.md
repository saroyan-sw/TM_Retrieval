# I07 Gerchberg–Saxton

The [Metzler et al. ICCP paper](https://compphotolab.northwestern.edu/wordpress/wp-content/uploads/2017/08/ICCP_paper_uploaded_5_12_17.pdf)
describes using a measurement pseudoinverse for alternating projections in
section 2.2.1. Section 4.3.2 and Table 3 specify random vector initialization
and 50- and 100-iteration budgets. The local paper copy used for the earlier
checkpoints has SHA-256
`320ae232f5582110f90a58c9e9bc56b5ae859043590d39c2954c712b21ce22e0`.

The solver uses the shared forward model `z = X @ a`, where X is sample-major
and a is a complex detector row. The target amplitudes y are the square roots
of valid training intensity counts. Each full iteration replaces |z| with y,
then estimates the minimum-norm a whose field best matches that amplitude
projection:

1. `z_amp = y * phase(z)` with `phase(z) = z / abs(z)` for nonzero z and
   `phase(0) = 1`. A zero target amplitude produces a zero projected field.
2. `a = X_pinv @ z_amp` and `z = X @ a`. The product `X @ X_pinv` projects
   onto the retained column space of X. The saved estimate is a, and the
   residual is measured on its forward field.

For each run the solver materializes X_pinv once from an economy SVD and shares
it across detector rows and iterations. It retains singular values strictly
above `pinv_rcond * largest_singular_value` and records the cutoff and numerical
rank. Both presets use `pinv_rcond = 1e-12`. The SVD uses a Hermitian transpose
for complex X. The public signed real probe matrix remains real in storage and
all 1600 input modes enter the fit. Rows with incomplete training masks receive
`unsupported_mask`. Zero amplitude energy receives `failed` because the
relative amplitude residual has a zero denominator. Every attempted detector
remains in the coverage denominator.

The paper specifies random initialization without a sampling distribution.
This implementation uses independent standard normal real and imaginary
coefficient samples divided by sqrt(2), from NumPy's default generator seeded
with `[20261124, detector_id]`. Both budgets start from the same initial
coefficients and execute the exact requested number of cycles. Completed rows
are labelled `max_iter_reached`. The convergence count reflects that these
presets use an iteration budget only. The cutoff, seeded Gaussian initialization
and zero-field phase convention are local choices. This implements the paper's
projection approach. Reference-code parity remains a separate verification task.

Focused checks compare the complex rank-deficient iterations with direct
least-squares solves, exercise the zero-field phase convention, and check how
the training adapter records shared preparation and unsupported or invalid
rows. Public runs use the existing frozen training split for preparation and
fitting, and reserve validation and test data for prediction reports.
