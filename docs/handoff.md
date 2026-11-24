# Handoff — 2026-11-24

## I01 complete
Recovered the existing OneDrive loader in commit 6a0cbac, then inspected all four public phase files, corrected stored field scaling (-255/+255 to -1/+1), pinned SHA256 hashes and froze explicit split/detector IDs. Dataset provenance and remaining acquisition uncertainties are in dataset_inventory.md.

Run from this repository:
```powershell
.\.venv\Scripts\python.exe -m tmbench inspect-data
.\.venv\Scripts\python.exe -m pytest -q
```
Verified: X training (17280,1600), validation (1920,1600), test (5,1600); 32 fixed detectors; 552960/61440/160 valid intensity observations; no train/test probe overlap. Seven focused tests pass. Raw public files remain local and untracked; download URLs and hashes are in the dataset manifest. Python packages are in the isolated .venv.

## I02 integration checkpoint
The user supplied prVAMP_TM_Retrieval_Python.zip. Its solver and all original mathematical/recovery tests are integrated. A sample-major training-only adapter shares one SVD across the fixed 32 rows, uses declared training-energy prior scaling and assumed noise fraction, and saves each row immediately. See prvamp_source_audit.md for source hash, formula conventions, changes and likelihood mismatch. All 34 tests pass. The public-data fit is the next verification step:

```powershell
.\.venv\Scripts\python.exe -m tmbench fit-prvamp
```

The default output is runs/prvamp_i02; choose --output for another fresh directory. Existing results are protected from overwrites. No I02 public fit is yet claimed at this integration checkpoint.

## Scope
No synthetic benchmark data, reference-TM tuning, published algorithm rankings or I03 completion claims. Tests use small numerical fixtures only. No remote push has been performed.
