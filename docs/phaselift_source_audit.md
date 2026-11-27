# I09 Bounded PhaseLift

For one detector, the field is `z = X @ a` with sample-major X. The
lifted variable is the Hermitian matrix `M = a a.H`. Its p-th intensity
prediction is `A(M)[p] = X[p] @ M @ X[p].conj()`, which equals
`abs(X[p] @ a)**2` for a rank-one lift. The PSD relaxation minimizes
`trace(M)` over `M >= 0` in the semidefinite order. The intensity
constraint is either:

- Exact when `relative_noise_bound = 0`: `A(M) = I`.
- Noisy when `relative_noise_bound > 0`:
  `norm(A(M) - I, 2) <= relative_noise_bound * norm(I, 2)`.

The trace objective, intensity constraints and eigenvector extraction follow
equations (1.5) and (1.9) in the [PhaseLift paper](https://arxiv.org/abs/1109.4499).
The 5% relative bound in the public preset is a declared intensity-model
tolerance. The camera noise level has not been measured. Both variants divide
intensities and the bound by mean training intensity for the solve and restore
the original units before extracting the coefficient vector.

The optional `convex` extra supplies CVXPY and SCS. The solver uses a Hermitian
variable with an explicit PSD constraint and a vectorized intensity operator.
For a single mode the Hermitian variable is scalar and real. A reusable
parameterized problem shares the design across rows. SCS uses the preset
iteration cap and numerical tolerance with warm starts disabled. See the
[CVXPY solver documentation](https://www.cvxpy.org/tutorial/solvers/).

After the backend returns an optimal or optimal-inaccurate solution, the
adapter preserves that status and checks the lifted constraints numerically.
It reports the minimum eigenvalue, spectral trace, numerical rank, and
bound and PSD violations. The dimensionless intensity violation is
`max(0, norm(A(M)-I)-bound) / norm(I)`. The PSD violation is the negative
part of the minimum eigenvalue divided by the larger of mean intensity and
the largest absolute eigenvalue. Both must be at most the feasibility tolerance.
Rank counts positive eigenvalues above `rank_rtol * largest_eigenvalue`.
Extraction uses `sqrt(largest_eigenvalue) * leading_eigenvector`. The row
file saves the lift and extracted coefficients. Their intensity residuals
are recorded separately because extraction can violate the intensity bound
even when the lifted solution is feasible. This can occur with real probes,
which constrain only the real part of a Hermitian lift. The common runner
evaluates the extracted coefficients on held-out amplitude predictions.
Only accepted feasible solutions with backend status `optimal` count as
converged. Zero-energy rows and incomplete training masks keep explicit
statuses in the full attempted-row denominator.

The dense implementation admits at most 32 input modes and 512 training
patterns. The bounds limit the complex measurement operator to 8 MiB before
CVXPY's real cone conversion and SCS workspace. The size check runs before
the optional backend import and lifted operator allocation. The frozen Rice
PhaseSLM_40x40 design has 17,280 training patterns and 1,600 illuminated
modes. Its dense operator alone would require about 659 GiB. The audited
Rice AmpSLM_16x16 alternative has 256 modes and 3,072 released training
patterns and also exceeds both caps. The existing [dataset inventory](dataset_inventory.md)
documents these releases. A detector subset reduces the number of independent
SDPs. An input-column subset changes the predicted fields for the original
measurements. The runner retains all frozen modes and records out-of-cap rows
as `unsupported_size`.

The numerical fixtures verify complex lifting orientation, rank-one
extraction, distinction between lifted and extracted residuals, an analytical
noisy bound, infeasibility, missing backend, and size rejection. These fixtures
verify solver mathematics. Public-data evidence comes from the Rice runner.

## Implicit operator preset

`phaselift_operator.json` solves
`min_(M >= 0) 0.5 * ||A(M) - I||^2 + lambda * trace(M)`.
This is the penalized form used by [PhasePack's PhaseLift solver](https://github.com/tomgoldstein/phasepack-matlab/blob/master/solvers/solvePhaseLift.m).
The Python implementation uses accelerated proximal gradient with a zero
initial lift. It evaluates `A(M)` by multiplying the probes by `M` and
taking rowwise inner products. Its adjoint is `X.H @ (weights[:, None] * X)`.
The maximum row sum of `A A*` bounds the gradient Lipschitz constant and
sets the step size. Each proximal update shifts the eigenvalues by
`lambda * step` and clips them to zero. The preset sets `lambda` to
0.001 times the largest eigenvalue of `A*(I)`. Intensity normalization
cancels when the matrix and penalty are restored to the original units.

Memory scales as O(PN + N²). The eigendecomposition remains computationally
expensive at large N. For real probes, the real lifted matrix can represent
a complex coefficient vector with two real eigencomponents. Extract the two
leading components as its real and imaginary parts. For complex probes,
extract the leading component. Save the lifted matrix and the extracted
vector and report both intensity residuals. The relative proximal step
determines convergence. A time or iteration limit retains a finite estimate
with that status.
