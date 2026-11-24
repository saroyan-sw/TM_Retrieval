# Handoff

The Python project layout and initial Rice file manifest are ready. The file check reports missing and empty inputs.

## I01 entry point

1. Put the four Rice `PhaseSLM_40x40` calibration files in `data/raw/phase_slm_40x40/` and run `python -m tmbench check-files` from the repository root.
2. Inspect the MATLAB arrays and release documentation. Update the manifest and inventory with checksums, file variables, orientation, encoding, observation domain, and known preprocessing.
3. Implement the first adapter and freeze the training, validation, test, and detector selections. Use sample-major `X` of shape `(P, N)` and observations of shape `(P, M)`, where `P` counts patterns, `N` input modes, and `M` detector outputs. Compute predicted field as `X @ A.T` for `A` of shape `(M, N)`.
4. Verify the loader on the public arrays and record the command and result. I02 starts after that checkpoint.

Research can proceed independently in `research/`.
