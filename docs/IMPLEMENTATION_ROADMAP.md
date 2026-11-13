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

## Phase 4 — Unified application integration

- Keep one user-facing UI and one authoritative research engine.
- Define API schemas for upload validation, prediction, feature inspection, model explanation, experiment listing and report export.
- Move long-running training into an explicit experiment workflow with progress and cancellation.
- Persist experiment metadata and model artifacts; make simulated and real-data environments visually distinct.

## Phase 5 — Quality and research readiness

- Full browser, API and data-pipeline integration tests.
- Input validation, file-size limits, sampling-rate/channel handling and safe error messages.
- Accessibility and browser compatibility checks.
- Performance profiling and large-signal testing.
- Repeated patient-wise splits and external validation where suitable data licences permit.
- Research documentation, limitations, ablation studies and final packaging.
