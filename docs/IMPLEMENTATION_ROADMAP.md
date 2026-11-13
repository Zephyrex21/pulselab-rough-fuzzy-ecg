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

## Phase 5 — Saved-model ECG inference integration

**Status: Implemented and covered by automated tests; full MIT-BIH inference still requires a trained local model and the optional dataset / dependency setup.**

- Added a saved-model catalog limited to compatible MIT-BIH experiment manifests and model artifacts within the project `models/` directory.
- Persisted training-only feature medians in new experiment manifests so inference applies the training preprocessing policy without fitting on uploaded data.
- Added validated CSV/TXT parsing with file-size, duration, sampling-rate, numeric-channel, missing-value and constant-signal checks.
- Added Python inference using the same preprocessing, automatic R-peak detection, beat segmentation and canonical feature schema used by the research pipeline.
- Added beat-level AAMI predictions, model-specific score summaries, per-class counts, low-score-separation review flags and downloadable CSV/JSON artifacts.
- Added on-demand fuzzy rule / membership explanations for fuzzy-only and Rough-Set + fuzzy models; baseline models clearly report that rule explanations are unavailable.
- Added a Model Inference page to the Apple-inspired web application while keeping the earlier synthetic demonstration separately labelled.
- Raw uploaded ECG samples are processed in memory and are not persisted; derived features and prediction outputs are stored locally under `experiments/`.
- Added API, parser, model-registry, saved-artifact, inference and explanation tests.

**Constraints:** CSV/TXT only (no WFDB pair upload yet); one or two numeric ECG channels; 20 MB maximum; 3 seconds to 10 minutes; 100–2000 Hz. The API remains unauthenticated and must be bound to `127.0.0.1`. The inference output is not clinically validated.

**Deferred beyond Phase 5:** full browser end-to-end automation in a real browser, profiling on long real MIT-BIH signals, repeated patient-wise cross-validation, ablation studies, external validation, authentication and remote deployment.


## Phase 6 — Explainability and score validation

**Status: Implemented and covered by deterministic unit / API tests; real MIT-BIH execution remains a local validation step.**

- Added patient/record-grouped out-of-fold temperature scaling using training records only; the untouched held-out split is never used to fit the transform.
- Added fold-specific training-only median imputation within the calibration folds.
- Added held-out multiclass Brier / log-loss, top-label and classwise ECE, reliability bins, and model-specific score-semantics reporting.
- Added per-patient held-out metrics and patient-cluster bootstrap intervals, with record fallback when patient identifiers are not provided.
- Added local single-feature training-median reference ablation, plus Random Forest impurity and fuzzy-rule usage summaries where supported.
- Added training-only feature quantile references and per-beat / per-feature input drift reports for saved-model inference.
- Added selected-beat preprocessed waveform previews using a bounded process-memory cache; preview waveforms are not persisted to disk.
- Extended the browser interface, research report, API and test suite with these diagnostics.

**Interpretation boundary:** internal temperature scaling is not external / clinical calibration; local ablation is not causal attribution or SHAP; drift flags are descriptive only. MIT-BIH dataset performance requires a local real-data run and remains unclaimed until completed.
