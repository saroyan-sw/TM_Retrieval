# TM Retrieval

Public-data transmission-matrix calibration benchmark using verified Rice
PhaseSLM_40x40 pairs, with prVAMP, prGAMP, prVBEM and warm-started prSAMP adapters.

## Setup and inspect
Python 3.10+:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,hdf5]"
.\.venv\Scripts\python.exe -m tmbench inspect-data
.\.venv\Scripts\python.exe -m pytest -q
```

Place the four calibration MAT files in data/raw/phase_slm_40x40. Exact public download URLs and SHA256 hashes are in data/manifests/rice_phase_slm_40x40.json. The current checkout already has the downloaded files locally.

Inspection verifies hashes, decodes signed fields, loads the predetermined 32 detectors, checks pairing and frozen splits, and writes runs/rice_phase_slm_40x40/inspection.json. All 1600 input modes are preserved; measurements remain in released intensity counts.

See [dataset inventory](docs/dataset_inventory.md) for acquisition limitations and [handoff](docs/handoff.md) for checkpoint status.

## Fit prVAMP on the fixed public subset

```powershell
.\.venv\Scripts\python.exe -m tmbench fit-prvamp --config configs/public/prvamp_i02_damped.json --output runs/prvamp_i02_repeat
```

This uses only training measurements, one common SVD, and the settings in
configs/public/prvamp_i02_damped.json. Each detector writes an estimate, iteration
history and status into the selected output folder. Existing outputs are protected; use
--output with a fresh directory for a new run. See
[source audit](docs/prvamp_source_audit.md) for assumptions and variance conventions.

I01 and I02 checkpoints are complete. Both real-data configurations and all
32 attempted rows are recorded in [results](docs/results/i02_summary.json).
The stronger-damping run produced finite estimates for every row, but all
reached the 300-iteration limit.

## Reproducible fitting and held-out evaluation

```powershell
$env:OPENBLAS_NUM_THREADS='4'
.\.venv\Scripts\python.exe -m tmbench.runner run --output runs/my_prvamp
.\.venv\Scripts\python.exe -m tmbench.runner report --output runs/my_prvamp
.\.venv\Scripts\python.exe scripts/archive_run.py runs/my_prvamp
```

The run command loads, checks and preprocesses the frozen data, fits training
rows, and evaluates validation/test amplitudes. The report command reloads saved
estimates without fitting. The archive command checks byte-identical report
regeneration and recomputes training residuals, then preserves a compact matrix
and provenance/report summary under docs/results. Full row histories remain in
the run directory. Failed rows remain in the attempted-row denominator. Finite
estimates at the iteration limit are not labelled converged.

## Bayesian baselines

```powershell
.\.venv\Scripts\python.exe -m tmbench.runner run --method prgamp --output runs/my_prgamp
.\.venv\Scripts\python.exe -m tmbench.runner run --method prvbem --output runs/my_prvbem
.\.venv\Scripts\python.exe -m tmbench.runner run --method prsamp --output runs/my_prsamp
```

Defaults are versioned in configs/public/*_i04.json, *_i05.json and *_i06.json;
override with --config. All commands use the same frozen train/validation/test
split and detector selection. prVBEM estimates variances from training data.
prVBEM defaults run four predetermined training pilots. The initial learned
variances collapsed on this camera subset; that failed calibration is retained.
prSAMP now defaults to an explicit fixed-variance configuration using training
energy, and saves each prVBEM warm start before sweeping. The original pilot
configuration remains available with --config configs/public/prsamp_i06.json.
Timings include all initialization costs. prSAMP's
five-sweep default is a bounded integration check, not a convergence guarantee.

See the [Bayesian source audit](docs/bayes_source_audit.md) for equations, complex
variance conventions, supported probes, safeguards and deviations from reference
implementations. These local implementations and smoke runs do not establish
published algorithm rankings; comprehensive parity and matched benchmarking
remain I11/I12. The [handoff](docs/handoff.md) records verified checkpoint results.
