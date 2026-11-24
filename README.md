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
.\.venv\Scripts\python.exe -m tmbench fit-prvamp
```

This uses only training measurements, one common SVD, and the settings in
configs/public/prvamp_i02.json. Each detector writes an estimate, iteration
history and status into runs/prvamp_i02. Existing outputs are protected; use
--output with a fresh directory for a new run. See
[source audit](docs/prvamp_source_audit.md) for assumptions and variance conventions.
