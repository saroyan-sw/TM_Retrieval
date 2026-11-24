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
