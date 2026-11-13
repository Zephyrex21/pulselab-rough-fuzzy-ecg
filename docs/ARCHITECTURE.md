# System Architecture

## 1. Three layers and two model paths

PulseLab contains a browser-side synthetic demonstration and a Python MIT-BIH research engine. Phase 4 adds a local FastAPI integration layer that connects the browser workspace to Python experiment orchestration and saved research outputs. It does **not** make the synthetic browser classifier share weights or feature schemas with the Python model.

### Browser workspace

```text
web/index.html
   ├── web/styles.css       Apple-inspired design system, responsive layout, themes
   ├── web/app.js           Navigation, views, interactions, API client and charts
   └── web/core.js          Synthetic ECG generator, processing, features,
                            rough-set selector, fuzzy rules, evaluation
```

The original ECG Analysis, Feature Extraction, Rough Set, Fuzzy and Model Evaluation screens remain a client-side synthetic demonstration. They do not call Python models or claim MIT-BIH performance. The Research Backend screen calls the local API to display service / dataset status, start Python research jobs and inspect saved Python outputs. The Model Inference page discovers locally saved models, submits ECG uploads, and displays beat predictions, explainability and drift diagnostics. A selected-beat waveform preview is served from a bounded process-memory cache and is not persisted.

### Phase 4 API integration

```text
Browser /web/  ── same-origin JSON requests ──>  server.py (FastAPI)
                                                   ├── /api/health and dataset status
                                                   ├── dataset audit / R-peak / experiment jobs
                                                   ├── saved experiment summaries and reports
                                                   ├── saved model catalog and ECG inference
                                                   ├── beat-level explanation / waveform preview
                                                   └── allow-listed inference artifacts
                                                               │
                                                               ▼
                                                       modules.pipeline
                                                       modules.dataset_audit
                                                       modules.validation
                                                               │
                                                               ▼
                                                   experiments/ and models/
```

`server.py` serves the frontend at `/web/` and the JSON API from `/api/`. Long-running audit / validation / experiment actions use background jobs. Job state is held in process memory; completed research artifacts are persisted under `experiments/`. The API is for localhost only and intentionally has no authentication layer.

### Python research engine

```text
MIT-BIH / WFDB records
        ↓
modules.data_loader
        ↓
modules.preprocessing
        ↓
modules.rpeak_detection
        ↓
modules.segmentation
        ↓
modules.feature_extraction + feature_schema
        ↓
        ┌─────────────────────────────────────────┐
        │ Model A: fuzzy rules on all features    │
        │ Model B: Rough Set reduct → fuzzy rules │
        │ Model C1: Random Forest                 │
        │ Model C2: RBF-SVM                       │
        └─────────────────────────────────────────┘
        ↓
modules.metrics + pipeline
        ├── grouped OOF temperature calibration (training groups only)
        ├── held-out score reliability diagnostics
        ├── model-specific global explainability summaries
        └── training feature reference quantiles
        ↓
experiments/<run_id>/ + models/<run_id>/
```

`train.py` is the CLI entrypoint. `app.py` is an alternative Streamlit research console. `server.py` exposes the same audit, validation and `modules.pipeline` functionality to the browser. All Python execution paths reuse the canonical Python research modules.

## 2. Research engine contracts

- `modules.feature_schema` is the canonical ordered schema for Python MIT-BIH features. The current two-channel schema contains 7 RR features plus 20 features per channel (47 total).
- The browser's 25-feature synthetic schema remains separate until explicit integration work; similarly named features must not be assumed to be interchangeable.
- Preprocessing, R-peak locations, segmentation and feature extraction use the same Python functions for train and inference preparation.
- Training-only transformations include missing-value medians, rough-set discretisation cuts and fuzzy membership quantiles.
- The pipeline validates that record IDs do not overlap between train and test feature tables.
- Evaluation uses the fixed AAMI order N, S, V, F, Q and reports per-class metrics as well as aggregate scores.
- Every saved experiment includes a run manifest with split, schema, model, metrics and research limitations.

## 3. Python fuzzy classifier

The Python fuzzy model fits a three-term membership partition per feature using training-data quantiles (15th, 50th and 85th percentiles). It induces class-specific conjunction rules with a bounded beam search. Rule antecedents use distinct features and are limited to three by default. Rule selection uses a confidence-weighted marginal coverage criterion. At inference, antecedent memberships are combined using product t-norm; class rules are aggregated using probabilistic OR.

The resulting row-normalised fuzzy scores are useful for ranking and explanation, but are not calibrated probabilities.

## 4. Rough Set selector

The selector learns equal-frequency bin boundaries from a balanced training-only sample, computes indiscernibility classes and positive-region dependency, then constructs a greedy QuickReduct-style subset with backward pruning. It reports the full/reduct dependency, core, selected/removed attributes and approximation sizes.

The reduct is consistency-oriented, not directly optimised for predictive accuracy. The experiment therefore compares the hybrid model against the fuzzy-only model and conventional baselines.

## 5. Data and split policy

The Phase 3 default uses a strict patient-disjoint split: 22 DS1 training records and 21 DS2 test records. Record 202 is excluded because MIT-BIH records 201 and 202 are recordings from the same subject. The historical 22/22 de Chazal record-wise list remains an explicit compatibility mode and is marked as not patient-disjoint. The default experiment detects R-peaks on both train and test signals; classifier results apply to detected beats matched to reference annotations, while R-peak false positives and misses are reported separately. The dataset is downloaded locally from PhysioNet using WFDB and is not committed or bundled in this repository. See `PHASE_3_REAL_DATA_VALIDATION.md` for the exact protocol.

## 6. API security and lifecycle

The integrated API is intended to bind only to `127.0.0.1`. It has no authentication, user accounts or remote-access controls. Do not expose it to an internet-facing host or shared network. Background job metadata is process-local and disappears on server restart; finished experiment, audit and validation artifacts remain on disk.

## 7. Non-clinical status

PulseLab is an academic research prototype. It is not a medical device, and its predictions must not be used for diagnosis, triage, treatment or clinical decisions.

## 8. Phase 6 inference diagnostics

- Score temperature is fitted from grouped out-of-fold training predictions (patient ID where available, otherwise record ID); fold-local median imputation avoids using OOF validation-fold feature statistics. The final held-out test set is used only for reliability evaluation.
- Test reliability outputs include multiclass Brier / log loss, top-label ECE, classwise ECE and reliability bins. Scores keep their model-specific semantics and are not clinical probabilities.
- Local inference explanation uses one-at-a-time training-median reference ablation. Global summaries use Random Forest impurity importance or fuzzy-rule antecedent usage. Neither is causal attribution.
- New model manifests hold training-only feature quantile references. Inference compares new beat features to q01–q99 ranges as a descriptive domain-shift heuristic.
- Selected-beat preprocessed, R-amplitude-normalised waveform windows are held in a bounded process-local cache for UI display. They are not persisted; restart or cache eviction clears previews.
