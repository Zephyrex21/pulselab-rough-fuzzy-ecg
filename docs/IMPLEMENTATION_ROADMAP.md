# Implementation Roadmap

## Phase 1 — Foundation and interface system

**Status: Implemented.**

- PulseLab branding and responsive navigation.
- Persistent light/dark themes.
- Design tokens, architecture notes and research-prototype safeguards.

## Phase 2 — MIT-BIH research engine

**Status: Implemented.**

- Python fuzzy classifier, Rough Set + fuzzy hybrid, canonical 47-feature MIT-BIH schema.
- Experiment pipeline, model comparisons, metrics, model saving/loading and Streamlit console.
- Training-only imputation / feature selection and patient-split checks.

## Phase 3 — Real MIT-BIH validation and reproducible experiments

**Status: Implemented; full-data execution is a local step because the dataset is not bundled.**

- Dataset inventory, AAMI annotation counts, class support, WFDB file checks and SHA-256 provenance.
- Strict subject-disjoint default split and explicit warning for historical record-wise compatibility.
- Automatic detector-based feature extraction for training and test records.
- Separate R-peak detection metrics, conditional matched-beat classifier metrics, per-record results and record-cluster bootstrap intervals.

## Phase 4 — Python API and saved-model inference

**Status: Implemented.**

- FastAPI serves the browser interface and exposes MIT-BIH dataset, experiment, saved-model and inference endpoints.
- CSV/TXT ECG upload validation, automatic preprocessing, R-peak detection, beat segmentation and saved-model predictions.
- Beat-level AAMI predictions, model-specific score summaries and fuzzy rule explanations where supported.
- Explainability, calibration diagnostics, feature drift reports and research artifacts.

## Phase 5 — BIDMC multimodal research engine

**Status: Implemented; Kaggle training artifacts and saved-model inference are integrated.**

- BIDMC CSV discovery, dataset inventory, record/channel metadata and patient grouping from source WFDB headers when available.
- Synchronized 125 Hz ECG Lead II, PPG/PLETH and impedance RESP loading, with dual manual breath annotations.
- Modality-specific window feature extraction and multimodal feature fusion.
- Separate full multimodal feature export and leakage-controlled respiratory-rate estimation feature export.
- Continuous Rough-Set + fuzzy respiratory-rate regressor with two source-hash-verified trained models (ECG+PPG primary and PPG-only comparison).
- Patient-disjoint Kaggle metrics, reports, final model artifacts and per-record Python prediction endpoint.
- Frontend model status, held-out metrics, model selector and real-record inference view. Record replay is clearly separated from held-out evaluation.

## Phase 6 — Unified two-engine browser interface

**Status: Implemented.**

- Browser is now an API-backed presentation/control interface only.
- Removed the former synthetic ECG generator, in-browser training/classifiers, simulated charts and browser demo smoke test.
- Added separate MIT-BIH ECG and BIDMC multimodal workspaces with engine-specific status, controls and results.
- Included the supplied patient-disjoint MIT-BIH model artifacts and their experiment reports.

## Next planned work

- Improve BIDMC respiratory-rate features and modeling, because the first held-out results are close to the median baseline.
- Investigate whether PPG-only and ECG+PPG add reproducible value on more than one fixed patient split; retain the two annotator-specific evaluations and RESP leakage safeguards.
- Add repeat patient-grouped evaluation / robustness analysis and, where possible, independent external validation.
- Re-train MIT-BIH with the current adaptive R-peak detector before treating the supplied historical artifacts as matched to current preprocessing.

## Research boundaries

PulseLab is an academic prototype, not a medical device. BIDMC outputs are academic estimates from the installed trained model. Record-replay predictions are not independent test results because final models were refitted on all labelled windows. Dataset-derived windows must be split by patient group to prevent leakage. Internal performance does not establish clinical validity or external generalisation.
