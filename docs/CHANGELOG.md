# Changelog

## 0.7.0 — Dual real-data research engines

**Repository organization**
- Consolidated architecture, methodology, implementation notes, changelog, model-import guides and training provenance under `docs/`.
- Moved pinned model-runtime dependency sets into `requirements/`, leaving only the main `README.md` and primary `requirements.txt` at the repository root.
- Updated the BIDMC inference loader and setup documentation to use the reorganized provenance and runtime-requirements paths.

**Browser application**
- Removed the browser-only synthetic ECG generator, simulated waveform renderer, browser-side model training and simulated prediction path.
- Rebuilt the web interface as an API client for the Python research engines.
- Added separate MIT-BIH ECG and BIDMC multimodal workspaces; BIDMC displays real dataset availability and extracted signal previews, not untrained predictions.

**BIDMC research pipeline**
- Added BIDMC dataset discovery, metadata inventory, patient-group identification and loading of ECG II, PPG/PLETH, RESP, monitor numerics and dual manual breath annotations.
- Added aligned window-level features for each modality and multimodal fusion.
- Added a leakage-controlled respiratory-rate feature export that excludes RESP-derived predictors when the target is derived from RESP breath annotations.
- Added the Rough-Set + fuzzy respiratory-rate regression implementation; the trained Kaggle estimators and inference integration are recorded in the post-training entry below.
- Added BIDMC status, signal-preview and asynchronous feature-export API endpoints.

**MIT-BIH artifacts and reproducibility**
- Bundled the supplied patient-disjoint MIT-BIH model artifacts and experiment reports.
- Preserved the original training manifest and prominently documented the source-version mismatch: the supplied models were trained using 0.6.0, while the shipped preprocessing uses the adaptive detector from 0.6.1. Re-training is recommended.

**Validation**
- Removed the obsolete browser demo smoke test and replaced it with API/front-end and BIDMC pipeline tests.
- Verified the BIDMC loader and feature extractor against a real uploaded BIDMC record.


**Post-training BIDMC model integration**
- Added the Kaggle-trained ECG+PPG primary and PPG-only comparison Rough-Fuzzy regression artifacts under the versioned BIDMC run ID.
- Preserved the patient-disjoint comparison, per-patient metrics, test predictions, run manifest, source/runtime metadata, annotation agreement and full feature exports.
- Added source SHA-256 verification before loading the Joblib estimators, plus model-status and safe report-download API endpoints.
- Added real-record BIDMC prediction through the Python engine and a frontend estimator selector with per-window results.
- Clearly labels final-model record replay as retrospective/in-sample; held-out results remain the source of generalization metrics.
- Documented the modest held-out performance and its limited difference from the median baseline.

## 0.6.1 — Bug-fix and UI pass

**Signal processing**
- Replaced the single global R-peak threshold with a locally adaptive threshold (candidate energy compared with the median of neighbouring beats, plus a noise floor). The old detector missed ~10% of MIT-BIH beats overall and almost all beats in records with strongly varying QRS amplitude (record 228: 14% sensitivity, 114: 22%). Across the 43 study records, sensitivity rose from ~89.7% to ~99.1% with PPV ~99.4%. Because the missed beats were not random, previously trained models should be retrained.

**Inference / API**
- ECG upload parser no longer treats a time or sample-index column as an ECG lead when the column is called e.g. "sample interval" or "ms", or has no header: monotonic, evenly stepped columns are detected from the data and skipped.
- The experiment summary now exposes `n_train_patients`, `n_test_patients` and record counts (the "held-out patients" tile previously always showed a dash).
- Only one experiment can run at a time (HTTP 409 otherwise).
- Audit, R-peak validation, experiment and dataset-status requests now default to the full dataset (`quick=false`); quick runs must be requested explicitly.

**Web interface**
- Fixed a crash (`inferenceExplanation is not defined`) that stopped the Model Inference results from rendering after any successful run.
- Backend and Inference forms keep their values (quick subset, tolerance, selected file, sampling rate) when the view re-renders after a refresh, job completion or beat selection.
- Dataset status now follows the Quick subset / split controls instead of always reporting the quick subset; Quick subset is unchecked by default.
- Removed the "Simulation mode" card from the sidebar; the sidebar is now scrollable so the bottom entries are never clipped on short screens.
- New collapsible sidebar (icon rail with tooltips, Ctrl/Cmd+B shortcut, remembered between visits) with navigation grouped into "Synthetic demo" and "MIT-BIH · Python".
- Page subtitle chip now reflects the current workspace; stale "0.5" title and "Phases 4–5" wording removed; a failure while training the browser demo no longer leaves the loading screen stuck.

## 0.6.0 — Phase 6: Explainability and Score Validation

- Added grouped out-of-fold scalar temperature scaling using patient-grouped folds where patient IDs are available and record-grouped folds otherwise; held-out labels are never used to fit the transform.
- Added fold-specific training-only median imputation during calibration OOF fits to avoid validation-fold feature statistics influencing the fold model.
- Added held-out multiclass Brier score, log loss, top-label ECE, class-wise reliability bins and per-class score diagnostics.
- Added per-patient held-out metrics and patient-cluster bootstrap intervals to make patient-level variation explicit.
- Added local single-feature training-median reference ablation and global Random Forest / fuzzy-rule feature summaries, with explicit interpretation limitations.
- Persisted training-only feature quantile references and added per-beat / per-feature inference drift reports.
- Added selected-beat R-amplitude-normalised waveform previews with an in-memory-only bounded cache; preview samples are not written to disk.
- Extended the inference UI and research report with calibration status, reliability metrics, waveform previews and explainability summaries.
- Added deterministic tests for grouped calibration, held-out diagnostics, feature drift, local explanations and ephemeral waveform access.
- Documented that internal score calibration and explainability do not establish external validation, clinical probability calibration or causal explanations.

## 0.5.0 — Phase 5: Saved-Model ECG Inference

- Added a trusted saved-model catalog backed by MIT-BIH experiment manifests and existing model artifacts.
- Persisted training-only feature medians in new run manifests for reproducible inference-time feature cleaning.
- Added validated CSV/TXT ECG upload with size, duration, sampling-rate, channel, finite-value and constant-signal checks.
- Added Python inference using the same preprocessing, automatic R-peak detection, beat segmentation and feature extraction pipeline as training.
- Added beat-level predictions, class score summaries, review flags, derived-feature / prediction artifacts and on-demand fuzzy rule explanations.
- Added the Model Inference workspace to the browser application, retaining a clear boundary between synthetic demonstration results and saved Python model outputs.
- Raw uploaded signal samples are processed in memory and not persisted; derived features and prediction artifacts are saved locally.
- Added `python-multipart` and automated tests for CSV parsing, saved-model discovery, path safety, multipart API upload, predictions and explanations.
- Kept the local API unauthenticated and documented that it must remain bound to loopback; no clinical or real-time diagnostic claims are made.



## 0.4.0 — Phase 4: Unified browser and Python API

- Added a local FastAPI service that serves the web UI and exposes health, dependency and MIT-BIH dataset status endpoints.
- Added asynchronous dataset-audit and experiment jobs with progress tracking and persisted outputs.
- Added saved-run discovery, experiment summaries and safe JSON / CSV / Markdown artifact downloads.
- Added a Research Backend page to the Apple-inspired browser interface.
- Added frontend integration while keeping synthetic browser demonstrations explicitly separate from real MIT-BIH results.
- Added API tests for health checks, static serving, dataset status, background job outcomes and artifact path validation.
- Added FastAPI, Uvicorn and HTTPX dependencies.
- API is intended for local loopback use only; it does not provide authentication.


## 0.3.0 — Phase 3: Real MIT-BIH Validation

- Added dataset integrity and provenance auditing for WFDB files, annotation counts, AAMI class support and SHA-256 checksums.
- Corrected the default evaluation protocol to strict patient-disjoint mode by excluding record 202 from test because records 201 and 202 come from the same subject.
- Retained the classic de Chazal 22/22 record-wise list as an explicit compatibility option and marked its subject overlap in reports.
- Added patient identifiers to extracted feature rows and enforced patient-level split checks before training.
- Changed the default experiment feature extraction to automatic R-peak detections; classifier scores are reported conditionally on one-to-one matched detected beats.
- Added independent one-to-one R-peak validation with TP, FP, FN, sensitivity, PPV, F1 and timing error.
- Added per-record classifier metrics, held-out predictions and record-cluster bootstrap confidence intervals.
- Added a human-readable Markdown experiment report and reproducibility artifacts.
- Reworked the Streamlit research console into dataset QA, detector validation, classification and results tabs.
- Added dataset-audit and R-peak-validation command-line tools.
- Added dependency-injected full-record pipeline tests and detected-peak-to-reference label matching tests.
- Documented that current feature extraction is offline / retrospective because it uses zero-phase filters, record-level amplitude scaling and next-RR context.
- Added tests for strict split logic, same-subject overlap, detection matching, record-level metrics and report generation.

## 0.2.0 — Phase 2: Research Engine Completion

- Added the Python fuzzy rule classifier with quantile memberships, beam-searched antecedents, coverage-based rule selection and sample-level explanations.
- Added reproducible record-wise feature collection and a four-model experiment pipeline.
- Added canonical Python feature schema and reusable AAMI metrics.
- Completed fuzzy-only and Rough-Set + fuzzy model wrappers, including model save/load support.
- Added record-overlap checks, training-only imputation and experiment manifests.
- Made `train.py` a functional CLI and replaced the incomplete Streamlit imports in `app.py` with a research console.
- Improved Rough Set handling of string labels, missing values and zero-dependency edge cases.
- Improved R-peak matching to enforce one-to-one reference/detection matches.
- Restricted the dominant-frequency feature to the documented 0.5–40 Hz band.
- Added deterministic unit and pipeline tests.
- Kept the browser's synthetic demonstration clearly separate from the MIT-BIH Python engine.

## 0.1.0 — Phase 1: Foundation and UI Refresh

- Introduced PulseLab branding and an Apple-inspired responsive browser interface.
- Added persistent light and dark appearance modes.
- Added design-system and architecture documentation.
- Added browser-core smoke test and Phase 1 notes.
