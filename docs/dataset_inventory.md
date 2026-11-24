# Dataset inventory

## Rice PhaseSLM 40×40

The release listing shows paired training and test files. The [paper](https://compphotolab.northwestern.edu/wordpress/wp-content/uploads/2017/08/ICCP_paper_uploaded_5_12_17.pdf) describes a 40×40 phase-only SLM with binary phase probes and 256×256 detector images. The downloadable arrays still need inspection. [Release listing](https://rice.app.box.com/v/TransmissionMatrices/folder/42189283459).

| Split | Known patterns | Observations |
| --- | --- | --- |
| Training | `XH_train.mat` | `YH_squared_train.mat` |
| Test | `XH_test.mat` | `YH_squared_test.mat` |

`A_GS.mat` and `A_prVAMP.mat` are fitted comparators. Independent reference TM availability is unverified.

I01 records checksums, format, variable names, actual shapes, observation domain, probe encoding, orientation, units, preprocessing, and any release or license details. Preserve the supplied test split. Define validation within training, grouping repeated patterns, and record the detector selection before solver tuning.
