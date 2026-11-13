# Changelog


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
