# TM Retrieval

Public-data transmission-matrix calibration benchmark. I01 imports verified Rice PhaseSLM_40x40 pairs; I02 adapts the existing prVAMP implementation.

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
