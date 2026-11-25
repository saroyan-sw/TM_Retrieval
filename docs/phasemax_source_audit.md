# I10 Bounded PhaseMax

For sample-major X, the field at one detector is `X @ a`. PhaseMax
maximizes `real(vdot(anchor, a))` subject to `abs(X @ a) <= b`
componentwise. The unpadded bounds are `b = sqrt(I_train)`. These are
amplitude upper bounds. The benchmark residual still compares predicted
and observed amplitudes in both directions. The formulation follows
[Goldstein and Studer, PhaseMax](https://arxiv.org/abs/1610.07531).

The solver accepts a finite nonzero complex anchor and normalizes it to
unit norm. The training adapter uses 50 iterations of the existing
intensity-weighted spectral power method on `X.H @ diag(I_train) @ X`.
The starting vector is seeded with `[20261124, detector_id]` in the public
preset. Only valid training observations enter that computation. Each
computed anchor is saved in a separate row file, with its seed, timing
and status. The anchor defines the objective and SCS warm starts are
disabled. Recovery quality depends on anchor alignment. A real design
can yield an almost real spectral anchor, and an objective component
in the probe nullspace can make the conic program unbounded.
See the [spectral initializer audit](wf_source_audit.md).

An optional nonnegative `amplitude_padding_fraction` adds
`padding_fraction * rms(sqrt(I_train))` to all amplitude bounds. It is a
local one-sided relaxation and its setting is user supplied. The public
preset uses zero padding. The code scales bounds by the measured RMS
amplitude and rescales estimated coefficients to the original units.
The unknown has 2N real coordinates. Each complex amplitude cap is a
second-order cone on the two real coordinates of its field. A shared
CVXPY model uses parameterized bounds and anchor, and the optional
`convex` extra supplies SCS. The preset uses a 5,000-iteration ceiling
and 10 seconds of SCS solve time per detector. SCS timing excludes
canonicalization and setup and is checked periodically.
See the [SCS settings](https://www.cvxgrp.org/scs/api/settings.html).

The saved row includes backend status and detail, iterations, the
real anchor objective, and the largest positive amplitude-cap violation.
An optimal or optimal-inaccurate estimate is saved only if it is finite
and the maximum violation divided by measured RMS amplitude is within
the feasibility tolerance. Only accepted optimal statuses count as
converged. Missing masks, nonpositive amplitude energy and solver failures
receive explicit row statuses. The archive verifier checks the saved anchor
norm and recomputes the objective and upper-bound violation.

The dense CPU implementation caps a design at 128 input modes and 2,048
training patterns. This limits the complex probe matrix to 4 MiB before
the real conic representation and workspace. The full supported shape
was exercised with a numerical fixture made from a seeded complex
unitary block repeated 16 times, with an analytical disk solution.
SCS returned an optimal feasible solution. Smaller fixtures check
complex orientation, padding, nullspace unboundedness, anchor artifacts
and size or backend rejection. These are numerical checks.

The frozen Rice phase design has 1,600 illuminated modes and 17,280
training patterns. The audited Rice amplitude candidate has 256 modes
and 3,072 released training patterns. Both exceed these caps. The size
gate runs before anchor preparation and optional backend import, with
all original modes and attempted detectors recorded.
See the [dataset inventory](dataset_inventory.md).
