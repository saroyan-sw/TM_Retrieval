# Handoff — 2026-11-24

## I01 complete
Recovered the existing OneDrive loader in commit 6a0cbac, then inspected all four public phase files, corrected stored field scaling (-255/+255 to -1/+1), pinned SHA256 hashes and froze explicit split/detector IDs. Dataset provenance and remaining acquisition uncertainties are in dataset_inventory.md.

Run from this repository:
```powershell
.\.venv\Scripts\python.exe -m tmbench inspect-data
.\.venv\Scripts\python.exe -m pytest -q
```
Verified: X training (17280,1600), validation (1920,1600), test (5,1600); 32 fixed detectors; 552960/61440/160 valid intensity observations; no train/test probe overlap. Seven focused tests pass. Raw public files remain local and untracked; download URLs and hashes are in the dataset manifest. Python packages are in the isolated .venv.

## I02 next
Recover supplied prVAMP ZIP from previous project chats (user confirmed its location there). Adapt its likelihood and solver with a single shared training-X SVD, preserve held-out isolation, record per-row statuses and mismatch between quantized camera intensities and Rician field noise. No fit has yet been claimed. The user did not authorize an amplitude-regime substitution; the selected data is phase-only.

## Scope
No synthetic benchmark data, reference-TM tuning, published algorithm rankings or I03 completion claims. Tests use small numerical fixtures only. No remote push has been performed.
