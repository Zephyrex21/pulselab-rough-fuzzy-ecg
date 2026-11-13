# Implementation Roadmap

## Phase 1 — Foundation and interface system

**Status: Implemented.**

- PulseLab branding and Apple-inspired responsive navigation.
- Persistent light/dark themes.
- Design tokens, architecture notes and browser demo safeguards.

## Phase 2 — Research engine completion

**Status: Implemented.**

- Python fuzzy classifier, Rough Set + fuzzy hybrid, canonical 47-feature Python schema.
- Experiment pipeline, model comparisons, metrics, model saving/loading and Streamlit console.
- Training-only imputation / feature selection and record split checks.

## Phase 3 — Real MIT-BIH validation and reproducible experiments

**Status: Implementation delivered; full-data execution remains a local step because MIT-BIH files are not bundled.**

- Dataset inventory, AAMI annotation counts, class support, WFDB file checks and SHA-256 provenance.
- Strict subject-disjoint default split; historical de Chazal record-wise compatibility retained with an explicit overlap warning for records 201/202.
- Automatic detector-based feature extraction for training and test records.
- Separate R-peak detection metrics, conditional matched-beat classifier metrics, per-record results and record-cluster bootstrap intervals.
- Dataset audit, validation and experiment CLI entry points plus Streamlit research-console views.
- Tests run on deterministic fixtures and feature tables; real MIT-BIH metrics require local download and execution.

## Phase 4 — Unified browser and Python API integration

**Status: Implemented for local research orchestration.**

- Added a local FastAPI service that serves the browser workspace and exposes health / dependency / dataset status.
- Added background jobs and progress polling for dataset audits, independent R-peak validation and MIT-BIH classifier experiments.
- Added saved experiment discovery, model-comparison summaries and allow-listed report downloads.
- Added the Research Backend page to the browser navigation while retaining the synthetic demonstration as a separately labelled model path.
- Added API integration tests and localhost-only run guidance.

**Explicitly deferred:** raw ECG file upload into the Python inference path, Python model-backed per-beat prediction / explanation inside the browser, user authentication, remote deployment and job cancellation. Those require a separately versioned inference contract, strict input limits and additional validation; Phase 4 does not claim they exist.

## Phase 5 — Inference integration and research readiness

- Define a versioned raw-signal / feature inference contract using the Python 47-feature schema and saved model artifacts.
- Add validated CSV / WFDB upload with file-size, sampling-rate, channel and finite-value checks.
- Show Python model-backed prediction, uncertainty and rule explanations only for compatible trained model artifacts.
- Add end-to-end browser + API tests in a supported local browser, including download and job lifecycle flows.
- Accessibility, browser compatibility, performance profiling and long-signal tests.
- Repeated patient-wise splits, ablation studies and external validation where suitable datasets and licences permit.
- Final research documentation, limitations and release packaging.
