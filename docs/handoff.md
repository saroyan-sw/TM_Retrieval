# Handoff — 2026-11-24

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

## Next entry point: I04
prGAMP implementation and focused checks are in progress. Preserve all
nonconvergence/instability outcomes. Neither held-out observations nor released
comparator TMs are used for solver tuning.

## Scope
No synthetic benchmark data, reference-TM tuning or published algorithm rankings.
Tests use small numerical fixtures only. No remote push has been performed.
