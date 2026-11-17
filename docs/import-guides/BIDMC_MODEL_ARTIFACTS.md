PulseLab BIDMC Kaggle model export — integrated into PulseLab 0.7.0
Run ID: pulselab_bidmc_patient_disjoint_20261117T144120Z_2489
Project version: 0.7.0
Dataset: BIDMC PPG and Respiration Dataset 1.0.0

CONTENTS IN THE PROJECT
- models/pulselab_bidmc_patient_disjoint_20261117T144120Z_2489/: final Rough-Set + Fuzzy Joblib models (ECG+PPG primary and PPG-only comparison)
- experiments/pulselab_bidmc_patient_disjoint_20261117T144120Z_2489/: patient-disjoint comparison, held-out predictions, per-patient metrics, feature tables, manifest and research report
- docs/reproducibility/bidmc_training_environment.json: Python/package/source SHA-256 provenance
- requirements/bidmc-model-runtime.txt: key versions used in Kaggle
- modules/bidmc_inference.py: source-hash verification, model loading and real-record inference
- server.py and web/app.js: BIDMC model API and frontend integration

INTEGRATION STATUS
The trained models are already integrated into PulseLab 0.7.0. No manual artifact copying is required when using the complete PulseLab project ZIP. The application loads the models from the versioned models/ directory and verifies the Kaggle source hashes before making them available.

RESEARCH DESIGN
- The primary estimator uses ECG + PPG. The PPG-only estimator is retained as a comparison.
- RESP is used for reference/annotation and is excluded from respiratory-rate predictors to avoid target leakage.
- Evaluation used 36 training patients and 10 held-out patients (46 MIMIC-II patient groups; 53 recordings).
- The 32-second windows use a 16-second stride.
- The final estimators were refitted on all labelled windows after held-out evaluation. Predictions made by replaying a record through these final estimators are retrospective and are not independent test results.
- Held-out MAE on the mean-annotator target: ECG+PPG Rough-Fuzzy 3.260 BPM; PPG-only Rough-Fuzzy 3.130 BPM; median baseline 3.167 BPM. Results are modest and close to baseline.

COMPATIBILITY
The loader verifies SHA-256 hashes for VERSION, modules/bidmc_pipeline.py, modules/bidmc_model.py, modules/rough_set.py and modules/fuzzy_logic.py against the Kaggle export metadata. Do not alter those model-source files without retraining or deliberately migrating and revalidating the models. The runtime versions used during training are documented in requirements/bidmc-model-runtime.txt. It also verifies trusted SHA-256 checksums for both Joblib estimators and all allow-listed model/evaluation artifacts before deserializing the models. Do not edit, recompress or replace these files without regenerating the trusted checksum manifest from a verified training export. Patient-disjoint feature exports require verified MIMIC-II patient IDs from every WFDB header; record-level fallback IDs are not accepted as subject IDs.

The dataset itself is not bundled in the project ZIP. Extract BIDMC v1.0.0 under data/bidmc/ so the application can index the CSV files.

Academic research prototype only; not clinically validated and not for medical decision-making.
