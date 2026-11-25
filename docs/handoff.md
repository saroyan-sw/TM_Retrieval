# Handoff — 2026-11-25

## I01 complete
Recovered the existing OneDrive loader in commit 6a0cbac, then inspected all four public phase files, corrected stored field scaling (-255/+255 to -1/+1), pinned SHA256 hashes and froze explicit split/detector IDs. Dataset provenance and remaining acquisition uncertainties are in dataset_inventory.md.

Run from this repository:
```powershell
.\.venv\Scripts\python.exe -m tmbench inspect-data
.\.venv\Scripts\python.exe -m pytest -q
```
Verified: X training (17280,1600), validation (1920,1600), test (5,1600); 32 fixed detectors; 552960/61440/160 valid intensity observations; no train/test probe overlap. Seven focused tests pass. Raw public files remain local and untracked; download URLs and hashes are in the dataset manifest. Python packages are in the isolated .venv.

## I02 complete — execution checkpoint, not a convergence claim
The supplied ZIP's solver and all original mathematical/recovery tests are integrated. A sample-major training-only adapter shares one SVD across the fixed 32 rows, uses declared training-energy prior scaling and assumed noise fraction, and saves each row immediately. See prvamp_source_audit.md for source hash, formula conventions, changes and likelihood mismatch. All 34 tests pass.

Two real-data runs finished. Both retained finite estimates for all 32 attempted detectors. Initial damping=0.5/100 iterations had 1 converged row and 31 iteration limits; median training relative amplitude residual was 133.26989, exposing severe instability. The training-only follow-up used damping=0.1/300 iterations: 0 converged rows and 32 iteration limits, residual range 0.138539–0.570115 and median 0.313051. This is improved numerical behavior, not evidence of accurate TM recovery. No detector was dropped. Noise fraction remained an assumed 0.01; it was not fitted to the camera.

Commands for the stronger-damping configuration and saved-artifact verification:

```powershell
.\.venv\Scripts\python.exe -m tmbench fit-prvamp --config configs/public/prvamp_i02_damped.json --output runs/prvamp_i02_repeat
$env:OPENBLAS_NUM_THREADS='4'
.\.venv\Scripts\python.exe scripts/verify_i02.py runs/prvamp_i02 runs/prvamp_i02_damped --archive-estimates
```

Use a fresh --output directory; existing results are protected. The initial run is runs/prvamp_i02; the follow-up is runs/prvamp_i02_damped. Both contain inspection, provenance, per-row estimates/history and status. All 64 estimates were reloaded and training residuals recomputed directly with X, independently of the SVD field. Maximum discrepancies were 5.69e-14 (initial) and 1.12e-16 (follow-up).

Tracked docs/results/i02_summary.json includes every row, runtime versions, source commits, artifact hashes and timings. Compact (32,1600) matrices with detector IDs and statuses are committed as docs/results/prvamp_i02_estimates.npz and prvamp_i02_damped_estimates.npz. Full histories and raw calibration files remain local. These arrays contain estimated coefficients, not independently measured reference TMs.

Preparation used one SVD per run (10.87 s initial, 6.27 s follow-up). End-to-end times were 211.65 s and 373.30 s. Peak traced allocation was about 1.264 GB, excluding untraced native BLAS workspace; it is not a process-RSS measurement. Runs used CPU, Python 3.12.14, NumPy 2.5.5, SciPy 1.18.1 and four configured BLAS threads.

## I03 complete
The runner command `python -m tmbench.runner run --output runs/i03_prvamp` loads,
fits, evaluates held-out splits and saves. `python -m tmbench.runner report
--output runs/i03_prvamp` regenerates evaluation without fitting. Metrics retain
per-row SSE, amplitude energy, valid counts and the full attempted-row denominator.
The one-command public run runs/i03_prvamp completed for all 32 detectors with
max_iter_reached at 300 iterations. Validation relative residual 0.586235 and
test 0.575469, both 32/32 coverage, reproduce the prior I02 estimate evaluation.
The artifact verifier regenerates the report byte-for-byte and independently
recomputes training residuals. Compact estimates and provenance/status/evaluation
are archived in docs/results/i03_prvamp_*; full histories remain local.
These scores do not certify convergence.

## I04 complete
prGAMP passed its equation, Onsager, complex-adjoint, recovery, repeatability and
failure-denominator tests (41 tests at the I04 code checkpoint). The public run
runs/i04_prgamp used all 32 fixed detectors and all 1600 input modes. All 32 rows
produced finite estimates and reached the 100-iteration limit; none converged.
Validation/test relative amplitude residuals are 0.632898/0.655574. These are
bounded integration runs, not a matched algorithm ranking. See bayes_source_audit.md
for fixed damping, precision floors and reference deviations. Compact estimates
and replay verification are archived under docs/results/i04_prgamp_*.

## I05 complete — degenerate real-data outcome retained
prVBEM and training-only variance pilots are integrated and checked against
independent phase integration, coordinate updates, EM formulas and a variational
bound. The run runs/i05_prvbem completed four 20-iteration training pilots and
32 standalone 50-iteration fits. All were finite iteration-limit results, but
their coefficient means collapsed near zero: validation/test relative amplitude
residuals are both 1.0. This is an unsuccessful calibration outcome, not convergence
or accuracy. Pilot normalized noise/prior were 1.011745/0.011746; the noise model
absorbed essentially all signal. The full report replayed byte-for-byte and
training residuals were independently recomputed. Evidence is archived in
docs/results/i05_prvbem_*. Pilot runtime and all pilot histories are included.

A training-only diagnostic on detector 8208 held the initial variances fixed for
10 coordinate sweeps before 20 EM sweeps. Its training residual worsened from
0.42 at iteration 10 to 0.72 at iteration 30; it did not resolve collapse. This
experimental schedule was not retained as a solver feature. The diagnostic is
saved locally and in the failure evidence. No held-out score selected a setting.

## I06 integration checkpoint
The seeded swept solver and prVBEM warm-start adapter are integrated; 49 tests
pass. Tests compare two full sweeps against a separate scalar-channel/direct-field
oracle for binary, signed and complex probes. Warm starts and every coordinate
permutation are persisted; timing includes preparation, pilots, warm starts and
sweeps. The initial pilot-based run runs/i06_prsamp was stopped after five finite
rows also gave training residual approximately 1.0. All 32 attempted rows remain
recorded: five completed, one interrupted, 26 not run after pilot failure. This
is an incomplete trial, not I06 completion. Its configuration remains available
as configs/public/prsamp_i06.json.

The explicit follow-up config prsamp_i06_fixed_variance.json uses noise=0.01 times
training intensity and the training-energy prior, with 20 fixed-variance VBEM
warm-start iterations and five prSAMP sweeps. It does not silently consume the
degenerate pilot estimates. This is now the runner default for prsamp. Execute
`python -m tmbench.runner run --method prsamp --output runs/i06_prsamp_fixed`
and archive it before declaring I06 complete.

The fixed-variance run is already active (execution session 63389). Do not launch
another copy into that directory. Inspect runs/i06_prsamp_fixed/status.json or
poll the existing session; eight rows were complete at this handoff checkpoint.
The implementation and 51 focused tests are in place; only completion of the
public run, final artifact verification and result/handoff commit remain.

## Scope
No synthetic benchmark data, reference-TM tuning or published algorithm rankings.
Tests use small numerical fixtures only. No remote push has been performed.
