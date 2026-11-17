# PulseLab BIDMC trained artifact bundle

Run ID: pulselab_bidmc_patient_disjoint_20261117T144120Z_2489
Project version at training: 0.7.0
Dataset: BIDMC PPG and Respiration Dataset 1.0.0

## Distribution and installation

The BIDMC trained models and evaluation outputs are distributed separately from the source repository because the repository intentionally does not track this binary artifact bundle.

Use the verified artifact ZIP for this run. Extract it **at the PulseLab repository root**, preserving the models/ and experiments/ paths. The expected paths are:

- models/pulselab_bidmc_patient_disjoint_20261117T144120Z_2489/bidmc_rough_fuzzy_ecg_ppg_rr_mean.joblib
- models/pulselab_bidmc_patient_disjoint_20261117T144120Z_2489/bidmc_rough_fuzzy_ppg_only_rr_mean.joblib
- experiments/pulselab_bidmc_patient_disjoint_20261117T144120Z_2489/ — run manifest, model comparison, test predictions, per-patient metrics and report
- docs/reproducibility/bidmc_training_environment.json — source and artifact checksum manifest (provided by the matching PulseLab source release)

Do not overwrite the repository's docs/reproducibility/bidmc_training_environment.json with a stale training-environment copy from an older artifact ZIP. The source release's file contains the trusted SHA-256 hashes for the matching binary bundle.

## Contents and model roles

- **ECG + PPG** is the primary Rough-Set + Fuzzy estimator.
- **PPG-only** is the comparison estimator.
- RESP-derived predictors and ECG/PPG-to-RESP correlations are excluded from respiratory-rate estimation because the reference breath annotations are made on RESP.
- The final estimators were refitted on all labelled windows after held-out evaluation. Replaying a record through these final models is retrospective/in-sample, not independent test evaluation.

## Training and evaluation split

The saved evaluation used a patient-disjoint GroupShuffleSplit over 46 MIMIC-II subject IDs:

- 36 training patients
- 10 held-out patients
- 53 recordings
- 1,537 total windows, 1,533 labelled windows for the mean-annotator target
- 32-second windows with a 16-second stride

The held-out mean-annotator comparison is modest: ECG+PPG Rough-Fuzzy MAE 3.260 BPM, PPG-only Rough-Fuzzy MAE 3.130 BPM, and median baseline MAE 3.167 BPM. These results do not establish clinically reliable respiratory-rate prediction.

## Integrity and compatibility checks

Before loading either Joblib model, the application verifies:

1. SHA-256 hashes of VERSION, modules/bidmc_pipeline.py, modules/bidmc_model.py, modules/rough_set.py and modules/fuzzy_logic.py against training provenance.
2. SHA-256 checksums of both Joblib models and every allow-listed model/evaluation artifact.
3. The run ID and project version shared by the training environment and run manifest.
4. The expected model type and predictor feature schema; RESP-derived feature leakage is rejected.

Do not edit, recompress, replace or reserialize these files after extraction. A failed checksum blocks model loading until the trusted bundle is restored.

## Patient identity safeguard

The CSVs do not provide a reliable patient identifier by themselves. PulseLab parses the MIMIC-II subject ID from each corresponding WFDB header. The /api/bidmc/status endpoint distinguishes verified IDs from missing, unparseable, unreadable or ambiguous headers. Record-level fallback labels are not accepted as patient identities. Patient-disjoint feature exports are blocked until every indexed recording has a verified subject ID. When the dataset is installed, the saved training/test patient and record lists are also checked against the local 53-record/46-patient mapping.

## Runtime

Install the exact versions in requirements/bidmc-model-runtime.txt when loading the supplied Joblib estimators. The dataset itself is not bundled; extract BIDMC v1.0.0 under data/bidmc/ so PulseLab can index the signal CSVs and WFDB headers.

Academic research prototype only; not clinically validated and not for medical decision-making.
