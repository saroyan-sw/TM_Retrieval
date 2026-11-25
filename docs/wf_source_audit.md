# I08 Wirtinger Flow

The original policy follows equations (2.1)–(2.5), the spectral scale in
Algorithm 1 and the power method in Appendix B. The presets use the 0.2 cap and
330-iteration ramp from the original paper's section 4.2. Both use a local
50-update budget for the public integration run. See [Candès, Li and
Soltanolkotabi, Phase Retrieval via Wirtinger Flow](https://arxiv.org/abs/1407.1065).

For sample-major X and complex detector coefficients a, the forward field is
`z = X @ a`. Each row fits the released training intensity I directly:

- Loss `f(a) = sum((abs(z)**2 - I)**2) / (2 * P)`.
- Wirtinger derivative `g = X.H @ ((abs(z)**2 - I) * z) / P`.
- For a complex perturbation h, the directional derivative is
  `2 * real(vdot(g, h))`. The adjoint X.H uses the Hermitian transpose.

Both presets start with 50 power iterations on `X.H @ diag(I) @ X / P` using
matrix-vector products. The start direction has independent Gaussian real and
imaginary parts from NumPy's default generator with seed `[20261124, detector_id]`.
The returned vector has squared norm `N * sum(I) / sum(abs(X)**2)`. The power
iterations approximate one leading eigenvector, so their accuracy depends on
the eigengap. For real probes a simple leading eigenvector is real up to a
global phase, which can make the start nearly real even with a complex draw.
Phase-and-conjugation ambiguities remain for real probes. The saved scores
measure amplitude prediction on the existing validation and test splits.

The two policies share the loss, gradient and initializer:

| Policy | Step used in `a_new = a - step * g` |
|---|---|
| `original` | `min(1 - exp(-k / 330), 0.2) / norm(a_init)**2`, with k starting at 1 |
| `adaptive` | Start at the same cap. Propose up to twice the last accepted step, bounded by that cap. Halve up to 20 times until `f(a_new) <= f(a) - 2e-4 * step * norm(g)**2`. |

The adaptive policy is a local Armijo backtracking variant. The factor in its
sufficient-decrease test accounts for the real directional derivative along
`-g`. Both policies declare stationarity when
`norm(g) * norm(a_init) / mean(I)**2 <= 1e-8` before an update. An exhausted
adaptive line search preserves the last finite estimate with status
`line_search_failed`. Nonfinite original updates fail explicitly. Rows with
incomplete training masks or invalid intensities keep explicit status entries
in the full detector denominator. All settings are declared in the presets
or resolved solver config. The two presets use the same 50-update limit and
fitting data. The archived provenance, run status and evaluation identify the
policy, and row histories save accepted steps, losses and backtrack counts.

The checks compare the complex derivative with centered directional differences
and the initializer with a dense Hermitian eigensolve. Additional checks exercise
the original step, the adaptive sufficient-decrease test and invalid-row
handling. The archive verifier recalculates the final intensity loss and training
amplitude residual before replaying the held-out report from saved coefficients.
