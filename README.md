# TM Retrieval

Benchmark transmission-matrix retrieval from known illumination patterns and measured optical outputs. The first dataset candidate is the Rice `PhaseSLM_40x40` release. I01 verifies and imports the public measurements, I02 adapts prVAMP, and I03 evaluates predictions on held-out patterns.

## Set up on Windows

Run from the repository root with Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

If inspection shows the release uses HDF5-based MATLAB files, install the HDF5 extra with `-e ".[dev,hdf5]"`.

## Check the public data files

Place the four calibration files from the [Rice phase subset](https://rice.app.box.com/v/TransmissionMatrices/folder/42189283459) in `data/raw/phase_slm_40x40/`. The command checks filenames and sizes. Array contents and their optical meaning are verified in I01.

```powershell
.\.venv\Scripts\python.exe -m tmbench check-files
```

See the [dataset inventory](docs/dataset_inventory.md) and [handoff](docs/handoff.md) for the first checkpoint. Source code, configs, and manifests are tracked. Downloaded measurements and run outputs stay local.
