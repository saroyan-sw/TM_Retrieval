# Public calibration inventory — inspected 2026-11-24

## Selected: Rice PhaseSLM_40x40

All four paired files were downloaded from the [public Box folder](https://rice.app.box.com/v/TransmissionMatrices/folder/42189283459). Files are dated 2017-11-21; no semantic release version or explicit dataset license was found. Public access requires no authentication. Raw files remain local in data/raw/phase_slm_40x40. The dataset manifest pins SHA256, byte counts, variable names, download URLs and orientation.

| File / variable | HDF5 storage shape | Type | Meaning |
|---|---|---|---|
| XH_train | (1600,19200) | float32 | Known fields stored as -255/+255 |
| YH_squared_train | (65536,19200) | uint8 | Paired intensity counts |
| XH_test | (1600,5) | float64 | Held-out fields stored as -255/+255 |
| YH_squared_test | (65536,5) | uint8 | Held-out intensity counts |

These are MATLAB v7.3 files. Shapes above are h5py storage shapes, not MATLAB logical dimensions. Transpose to sample-major arrays. Keep all input modes. Divide stored fields by 255 to define unit-magnitude inputs and the corresponding TM scale. This does not change camera values. The [paper linked by Rice](https://rice.box.com/s/4jipt0ebr9lm03ccg53fqgf4tsodcpco), section 5.2, specifies binary 0/pi phases and -1/+1 fields, consistent with the signed arrays up to the documented scale.

Camera values are released intensity counts; amplitudes are sqrt(counts). Absolute radiometric calibration, exposure, gain, dark frames, flat-field corrections, gamma/linearity calibration and timestamps are not provided in the inspected files. Release preprocessing remains unknown. No further intensity scaling, subtraction, denoising or clipping is applied. Negative/nonfinite values and counts >=255 are invalid (the latter conservatively treated as potential saturation); original values are retained. Rician field-noise likelihoods are a modeling approximation to these quantized measurements.

Paper sections 4.3.1 and 5.4 describe a 632.8 nm setup, diffuser and camera. Section 5.4 identifies a reflective phase SLM and 40x40 reconstruction. The earlier amplitude setup lists LC2012, a Thorlabs 5-degree diffuser, Newport X10 objective and Point Grey Grasshopper2 camera. Do not assign the amplitude SLM model to the phase device or infer missing per-file capture settings.

The pre-existing 32-detector grid is fixed in configs/public/rice_phase_slm_40x40.json before fitting. IDs are zero-based release indices; their physical image coordinate mapping is not newly verified. Identical probes and global-sign opposites stay grouped. The seeded 90/10 split gives 17,280 training and 1,920 validation patterns; all five test patterns remain held out. No training/test overlap was found. Explicit indices and file hashes are frozen in data/manifests/rice_phase_slm_40x40_split.json and checked on load.

Inspection finds 552,960 valid training, 61,440 validation and 160 test observations. Pairing checks establish matching sample counts in release order, not independent verification of frame identity (frame IDs are unavailable). No invalid observations occurred in this selection. Unselected detectors are not covered by this statement.

A_GS and A_prVAMP are fitted comparators, not independent reference TMs; they are not loaded for calibration or tuning. No independently measured reference TM was identified. Real probes retain per-row phase and conjugation ambiguity.

## Alternative: Rice AmpSLM_16x16

The [amplitude folder](https://rice.app.box.com/v/TransmissionMatrices/folder/42189311436) is accessible. Actual XH_train has h5py shape (256,3072), uint8 values {0,255}. XH_test has shape (256,5), with grayscale entries, not only binary. YH_squared_test has shape (65536,5), uint8. The 153,755,929-byte training observation file is listed but not downloaded in this checkpoint; its pairing is not newly verified. Grayscale test encoding needs a separate field-calibration audit. The amplitude regime is not selected because usable phase-only pairs are available.

## Eligibility

Known probes paired with measured outputs support unknown-TM calibration. A fitted matrix alone, or a known-TM unknown-image task, does not. No synthetic measurements supply these inspection results.

## ICCP 2017 benchmark protocol

### Published simulation experiments

| Evidence | Probe coefficients | Unknown modes N | Measurements P | Reporting |
|---|---|---:|---:|---|
| Table 1 | i.i.d. circular complex Gaussian, CN(0,1) | 64, 256, 1,024 | 12N | Solve time in seconds. Table 2 lists SVD and pseudoinverse preparation separately. |
| Fig. 2 | Complex Gaussian | 1,024 | 12,288 | 32 x 32 smiley reconstruction at median error over 10 trials. |
| Fig. 3 | Complex Gaussian | 256 | N through 12N in increments of N | Median coefficient NMSE over 50 trials per sampling ratio. The plot labels the ratio M/N, where M is the measurement count in that figure. |
| Fig. 7 | Uniform binary amplitude, {0, 1} | 1,024 | 12,288 | Smiley reconstruction at median error over 10 trials. |
| Fig. 10 | Uniform binary phase, {-1, +1} | 1,024 | 12,288 | Smiley reconstruction at median error over 10 trials. |
| Figs. 6 and 11 | Uniform binary amplitude and uniform binary phase, respectively | 256 | 3,072 | Median coefficient NMSE over 50 trials at each plotted noise level. The panels label sigma = 0.1, 1, 5, and 15. |

Section 2.2.4 aligns a global phase to minimize reconstruction error before
reporting NMSE. The calibration model applies additive complex field noise
before taking magnitudes. The exact simulation target normalization,
random seeds, and code mapping the plotted noise labels to variance are
unspecified. The table and figures support a protocol-level replication.
[ICCP paper, Table 1 and §§2.2.4, 4.2, 5.3](https://compphotolab.northwestern.edu/wordpress/wp-content/uploads/2017/08/ICCP_paper_uploaded_5_12_17.pdf).

| Method | Reported initialization and budget |
|---|---|
| prGAMP and prVAMP | Random vector, 50 iterations, damping parameter 0.8. |
| prSAMP | prVBEM estimate, 50 sweeps, damping parameter 0.9. |
| prVBEM | Built-in initializer, 50 iterations, specified noise variance and otherwise defaults. |
| Gerchberg–Saxton | Random vector, 50 and 100 iterations. |
| Wirtinger Flow | Spectral initializer, 50 and 500 iterations. |
| PhaseLift and PhaseMax | Random vector, 50 iterations, defaults and no signal prior. Exact solver backend, tolerances and PhaseMax anchor construction are unspecified. |

The authors set the AMP signal means to zero and used oracle signal/noise
variances in simulations. On experimental data they estimated variances
with prVBEM on one TM row and reused them to tune prGAMP, prVAMP and prSAMP
on other rows. Resolve the damping convention against the reference code
before matching the printed values. The reported PhaseLift timings were
5.00, 201.00 and 12,321 seconds across the three Gaussian sizes. PhaseMax
took 0.48, 2.85 and 104.46 seconds. The project links prVAMP source and a
demo in GAMPmatlab. PhasePack supplies independent PhaseLift and PhaseMax
references. [ICCP paper, Tables 1 and 3 and §4.3.2](https://compphotolab.northwestern.edu/wordpress/wp-content/uploads/2017/08/ICCP_paper_uploaded_5_12_17.pdf#page=10)
· [Rice project](https://dsp.rice.edu/research/transmissionmatrices/)
· [PhasePack](https://www.cs.umd.edu/~tomg/projects/phasepack/).

PhasePack's empirical demo reconstructs an unknown image using a supplied
TM. Here the known design is the probe matrix and the unknown is a TM
row. A shared data release supports both experiments.
[PhasePack empirical benchmark](https://www.cs.umd.edu/~tomg/projects/phasepack/#:~:text=PhasePack%20can%20also%20reconstruct%20images).

### Published camera experiments

- The 64 x 64 amplitude-SLM demonstration used 49,152 calibration patterns
  and 256 x 256 camera outputs. Figure 9 uses prVAMP calibration. The authors
  excluded rows with residual above 0.4 from subsequent imaging and reported
  around 98% below that threshold. They used BM3D after image reconstruction.
  The stated parallel cluster had 80 dual-processor nodes with 16 cores and
  32–128 GB RAM per node. [ICCP paper, §§4.3.1–4.3.4](https://compphotolab.northwestern.edu/wordpress/wp-content/uploads/2017/08/ICCP_paper_uploaded_5_12_17.pdf#page=9).
- The 40 x 40 phase-SLM demonstration used a reflective SLM with 0/pi phase
  commands. Figure 13 shows prGAMP calibration and 256 x 256 camera speckle
  for the shown inputs. The paper applies BM3D after reconstruction. It gives
  no measured PhaseLift or PhaseMax result. The corresponding release has
  19,200 calibration and five test patterns. Section 6 also mentions captured
  128 x 128 data that the authors had not processed.
  [ICCP paper, §§5.2–6](https://compphotolab.northwestern.edu/wordpress/wp-content/uploads/2017/08/ICCP_paper_uploaded_5_12_17.pdf#page=12).

### Frozen R05 protocol for I12

The [benchmark manifest](../configs/benchmarks/iccp2017_i12.json) fixes
the seed, dimensions, trial counts and CPU limits. The
[I11 verification](results/i11_verification.json) defines the numerical
coverage of the supported implementations.

- **Simulation inputs.** Pair the same target, probes, and noise across
  all methods. Generate zero-mean complex Gaussian coefficients with
  unit total variance for vector runs. Use a procedural unit-RMS smiley
  for image runs. Each Gaussian ratio is a prefix of the same generated
  matrix, and each noise level scales the same circular complex draw
  by the plotted sigma, so total field-noise variance is sigma squared.
  Noiseless observations have a small positive likelihood floor for AMP.
  These choices fill gaps in the paper's published generator details.
- **Simulation runs.** Use all eight families, including separate GS-50,
  GS-100, WF-50 and WF-500 runs. Reproduce 50 trials per oversampling or
  noise setting and 10 per image regime. Run a separate sequential timing
  pass at each table size. Report median coefficient NMSE with quartiles,
  phase-only alignment as in the paper, and phase-plus-conjugation
  alignment separately when probes are real. Save the median-error image
  trial and all finite estimates. At Gaussian 12N, repeat the AMP methods
  with observation-derived variance estimates to expose the effect of
  the paper's oracle inputs.
- **Solver settings.** Treat the paper's reported 0.8 and 0.9 damping as
  old-state weights, giving 0.2 and 0.1 in the local new-state convention.
  This is an explicit reconstruction assumption. Use the local VBEM
  initialization, spectral WF, prVBEM warm-started prSAMP, and seeded
  random starts for VAMP, GAMP, GS and PhaseLift. The simulation PhaseMax
  anchor is a seeded random vector to reflect the published random-start
  description, whose precise anchor semantics are unspecified. PhaseLift
  uses the implicit trace-penalized objective. PhaseMax uses the
  existing ADMM implementation. Report these variants explicitly.
- **Measured calibration.** Freeze the 1,600 modes, 32 detector IDs,
  17,280 training patterns, 1,920 validation patterns, and supplied five
  test patterns. The first training detector supplies a single 50-step
  prVBEM pilot, whose absolute noise and prior variances are reused in
  prGAMP, prSAMP and prVAMP. Preserve a degenerate pilot outcome.
  Standalone VBEM fits its variances, and PhaseMax uses the training
  spectral anchor. Both arms use the same frozen settings and split,
  without selection using validation or test scores. Report amplitude MSE
  and RMSE relative to measured RMS amplitude, pooled over valid evaluated
  observations. Compute coverage and failure denominators over all
  attempted detectors. The five previously evaluated test patterns serve
  as a repeatability check. Leave BM3D and residual-based detector
  exclusion to a separate image reconstruction experiment.
- **Compute.** Use one BLAS thread per process. The simulation and
  isolated timing passes have a 20-second process CPU limit per solve.
  Measured runs use published nominal iterations with a 35-second CPU
  ceiling and a separate eight-second matched arm for every family.
  Check budgets after updates and within prSAMP sweeps. Report the
  actual completed updates, CPU and wall time, preparation,
  initialization and pilot costs. Save process peak resident memory as
  a worker high-water mark. Record iteration and time limits, violated
  constraints, unsupported designs, and numerical failures explicitly.
  Compare these capped results as a protocol-level replication.

Use Python with the HDF5 extra from the project root and an output folder
on the same disk as its checkpoint files.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python scripts/run_i12.py all --workers 20 --output runs/i12
python scripts/run_i12.py report --output runs/i12 --summary docs/results/i12_summary.json --archive docs/results/i12_trials.jsonl.gz
```

Rerun the same command to resume. Each simulation trial and measured
row is saved after completion. Output provenance includes the solver
and data-contract fingerprint.
