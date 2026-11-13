# PulseLab — Hybrid Rough-Fuzzy ECG Research Platform

PulseLab is an academic research prototype for ECG signal processing, Rough Set Theory-based feature reduction, and interpretable fuzzy-rule arrhythmia classification. It contains a polished synthetic browser demonstration and a separate Python research engine for MIT-BIH experiments.

> **Research-use disclaimer:** PulseLab is not a medical device. Its outputs are not clinically validated and must not be used for diagnosis, treatment, triage, or clinical decision-making.

## Current release

**Version 0.6.0 — Phase 6: Explainability and Score Validation**

Phase 6 extends saved-model inference with grouped out-of-fold temperature scaling (training records only), held-out reliability diagnostics, local single-feature reference ablation, global model summaries, training-range drift checks, and selected-beat waveform previews. Waveform windows are kept in a bounded process-memory cache and are not written to disk. The original browser ECG Analysis / Rough Set / Fuzzy demonstration remains explicitly synthetic; its predictions and metrics are not relabelled as MIT-BIH outputs.

## 1. Run the integrated PulseLab application (recommended)

The integrated FastAPI service serves the browser interface and Python API from the same local origin. From the repository root in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
py -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/web/>. The API health check is at <http://127.0.0.1:8000/api/health> and interactive API documentation is at <http://127.0.0.1:8000/api/docs>.

The **Research Backend** navigation page checks the API and dependencies, audits local MIT-BIH files, runs independent R-peak validation, launches classifier experiments, tracks job progress, lists saved experiment runs, and downloads audit / validation / experiment artifacts. The **Model Inference** page lists saved compatible MIT-BIH models and runs beat-level inference on uploaded CSV/TXT signals.

The existing browser ECG Analysis, Feature Extraction, Rough Set, Fuzzy Logic and Model Evaluation pages still run the separate synthetic demonstration. Those client-side scores are never presented as MIT-BIH experiment results. The browser UI retains responsive layouts and persistent light / dark themes.

> Security: the API has no authentication and is intended for local use only. Keep Uvicorn bound to `127.0.0.1`; do not expose it to a public network.

### Optional: run only the static synthetic browser demo

For UI-only development, the browser can still be served without the Python API:

```powershell
py -m http.server 8000
```

Open <http://localhost:8000/web/>. The Research Backend page will correctly show that the API is offline in this mode.

## 2. Set up the Python research engine

Use Python 3.10 or newer. From the repository root in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The Python engine uses NumPy, pandas, SciPy, scikit-learn, WFDB, PyWavelets, Streamlit, Plotly and joblib. The MIT-BIH data is not bundled with the ZIP.

## 3. Audit and download MIT-BIH records

Quick engineering audit and download (8 training + 6 testing records):

```powershell
py audit_dataset.py --download --quick
```

Full strict patient-disjoint audit and download:

```powershell
py audit_dataset.py --download
```

The audit checks `.hea`, `.dat` and `.atr` file presence, readable signal metadata, sampling frequency, channel count, record duration, annotation coverage, AAMI class support, train/test patient separation and SHA-256 file hashes. It writes JSON and CSV reports under `experiments/`.

### Why the default split has 21 test records

The historical de Chazal list contains 22 DS1 training and 22 DS2 testing records, but records **201 and 202 are from the same subject**. PulseLab's default strict patient-disjoint split removes record 202 from test, giving 22 training and 21 testing records. The original 22/22 list is retained only as an explicit compatibility mode:

```powershell
py audit_dataset.py --download --split-protocol de-chazal-record-wise
```

This compatibility mode is not strictly patient-disjoint and the audit / experiment manifest mark it accordingly.

## 4. Validate R-peak detection separately

Quick test-record detector validation:

```powershell
py validate_rpeaks.py --download --quick
```

Full strict test split:

```powershell
py validate_rpeaks.py --download
```

The detector is compared with AAMI-mappable reference beat annotations using one-to-one matching (default tolerance: 75 ms). It reports TP, FP, FN, sensitivity, PPV, F1 and matched-beat timing error. These are R-peak detection metrics, not arrhythmia classification metrics.

## 5. Run the MIT-BIH classifier experiment

Quick automatic-detection engineering run:

```powershell
py train.py --download --quick
```

Complete strict patient-disjoint experiment:

```powershell
py train.py --download
```

By default, automatic R-peaks are detected on both training and test records. Supervised classifier features are retained only for detected beats matched one-to-one to reference annotations. The resulting classification metrics are conditional on matched detected beats; detector false positives and missed reference beats are reported separately in the same run.

For a controlled annotation-guided classifier benchmark (not an automatic end-to-end estimate):

```powershell
py train.py --peak-source annotations
```

For historical 22/22 record-wise comparability only:

```powershell
py train.py --split-protocol de-chazal-record-wise
```

To skip Random Forest and SVM:

```powershell
py train.py --no-baselines
```

To save metrics without serializing trained models:

```powershell
py train.py --no-save-models
```

## 6. Open the Python research console

After dependencies are installed:

```powershell
streamlit run app.py
```

The console contains tabs for:

- Overview and evaluation safeguards
- Dataset quality audit and class counts
- Independent R-peak detector validation
- AAMI beat-classification experiment
- Saved result comparison and per-record metrics

The static browser application remains separately available at `http://localhost:8000/web/`.

## 7. Run automated tests

```powershell
py -m pytest -q
node tests/browser_core_smoke.test.js
```

Phase 6 adds grouped calibration, held-out reliability, explainability, feature-drift, ephemeral waveform-preview, and report-generation tests, alongside the Phase 4–5 API and inference tests.

The Python tests use deterministic synthetic records / feature tables and dependency-injected detector functions for unit testing. They verify the algorithms and report generation; they do not substitute for running the full MIT-BIH dataset experiment.

## 8. Run saved-model inference on an uploaded ECG (Phases 5–6)

1. Complete a MIT-BIH experiment with **Save model artifacts** enabled in Research Backend, or run `py train.py --download` after the MIT-BIH dataset is ready.
2. Start the integrated service with `py -m uvicorn server:app --host 127.0.0.1 --port 8000`.
3. Open **Model Inference** in the sidebar. Refresh the model list, select a saved MIT-BIH model, upload a `.csv` or `.txt` file, and enter its sampling rate (100–2000 Hz).
4. The upload must contain one or two numeric ECG signal columns, matching the selected model's channel count. A header is optional. Common named time/index columns are ignored; with headerless files, the first numeric columns are treated as ECG leads. Use the same lead order used during training.
5. Review the detected beats, AAMI class outputs, score calibration status, training-range drift diagnostics and available rule / feature-sensitivity explanations. Select a beat to view its normalised waveform preview while the local API process remains running. Download `beat_predictions.csv` and the feature-drift reports for analysis.

Upload constraints: 20 MB maximum; 3 seconds to 10 minutes; 100–2000 Hz; short internal missing-sample gaps may be interpolated. Raw uploaded samples are processed in memory and are not saved. Derived beat features, predictions and summary metadata are written to `experiments/inference_<id>/`. The upload route requires `python-multipart`, installed through `requirements.txt`.

For Phase 6 details—including grouped out-of-fold temperature scaling, held-out reliability diagnostics, feature sensitivity, drift checks and ephemeral waveform preview retention—see [`docs/PHASE_6_EXPLAINABILITY_VALIDATION.md`](docs/PHASE_6_EXPLAINABILITY_VALIDATION.md).

Inference is limited to saved models whose run manifest confirms the current MIT-BIH feature schema and whose `.joblib` artifact exists under `models/`. Models trained with annotation-guided peaks are flagged because inference uses automatic peak detection. Model scores are ranking/normalised outputs and are not calibrated probabilities. This is research software, not a medical device.

## Outputs

Dataset audit:

```text
experiments/dataset_audit_<timestamp>/
    dataset_audit.json
    record_inventory.csv
    class_distribution.csv
    file_sha256_manifest.csv
```

R-peak validation:

```text
experiments/rpeak_validation_<timestamp>/
    rpeak_validation.json
    rpeak_metrics_per_record.csv
```

Classification experiment:

```text
experiments/<run_id>/
    run_manifest.json
    metrics.json
    model_comparison.csv
    per_class_metrics.csv
    per_record_model_metrics.csv
    per_patient_model_metrics.csv
    test_predictions.csv
    rpeak_detection_metrics.json
    rpeak_detection_per_record.csv
    *_confusion_matrix.csv
    *_rule_base.json
    research_report.md
```

When model saving is enabled, model artifacts are stored under `models/<run_id>/`. Local datasets and generated artifacts are ignored by Git by default.

Saved-model inference:

```text
experiments/inference_<id>/
    inference_summary.json
    beat_predictions.csv
    beat_features.csv
    feature_drift_per_beat.csv       # when training feature reference exists
    feature_drift_by_feature.csv     # when training feature reference exists
```

Selected-beat waveform previews are process-memory only and are not persisted. The cache retains up to two recent inference sessions and is cleared by an API restart.

## Metrics and interpretation

Reported classifier metrics include accuracy, balanced accuracy, macro/weighted F1, per-class precision, sensitivity/recall, specificity, confusion matrices and one-vs-rest ROC-AUC / average precision when mathematically defined. Record-cluster bootstrap intervals are reported for accuracy and fixed-five-class macro-F1 when held-out record IDs are available.

Where enough grouped training records and class support exist, Phase 6 fits a scalar temperature from grouped out-of-fold training scores; fold-specific imputation statistics are also fitted within each fold. The held-out test split is used only to calculate Brier score, log loss, top-label / classwise ECE and reliability bins. Inspect each model's calibration status and skip reason. This is internal calibration, not external or clinical validation. Fuzzy scores and SVM ranking scores retain model-specific semantics and are not calibrated clinical probabilities. In automatic-detection mode, classifier metrics describe only matched detected beats. Consider those alongside the separate detector sensitivity / PPV / F1 and the number of unmatched detections / reference beats.

A single MIT-BIH split, even with record- and patient-cluster intervals and score calibration, is not external validation or clinical validation. Inspect `per_patient_model_metrics.csv` and patient-cluster intervals to understand performance variation across test subjects; these summaries do not replace repeated patient-disjoint cross-validation. Local single-feature reference ablation is a sensitivity diagnostic rather than SHAP or causal explanation. Training-range drift flags are descriptive q01–q99 comparisons; they do not prove a signal is invalid or suitable. The current implementation is offline / retrospective: zero-phase filtering, record-level amplitude normalisation and the next-RR feature use future or whole-record context. It is not a validated streaming or real-time system.

## Repository map

```text
config.py                    Shared configuration, AAMI mapping and split protocols
app.py                        Streamlit research console (alternative UI)
server.py                     Phase 4–5 FastAPI service + static web host
train.py                     MIT-BIH classification CLI
audit_dataset.py             Dataset integrity / provenance audit CLI
validate_rpeaks.py           Standalone R-peak validation CLI
modules/
  data_loader.py              WFDB record loading and AAMI labels
  dataset_audit.py            File integrity, class support, checksums and split audit
  preprocessing.py            Baseline removal, filtering, normalisation and resampling
  rpeak_detection.py          R-peak detection, refinement and one-to-one matching
  validation.py               Independent detector validation and aggregate metrics
  segmentation.py             Beat-window extraction
  feature_schema.py            Canonical ordered Python feature schema
  feature_extraction.py        RR, morphology, statistical, spectral and wavelet features
  rough_set.py                 Discretisation, dependency, approximations and QuickReduct
  fuzzy_logic.py               Fuzzy membership functions, rules, inference and explanations
  hybrid_model.py              Fuzzy-only and Rough-Set + fuzzy model wrappers
  baseline_models.py           Random Forest and RBF-SVM baselines
  metrics.py                   AAMI metrics, record-level metrics and cluster bootstrap
  pipeline.py                  Reproducible feature, model and report orchestration
  inference.py                 Saved-model registry, ECG CSV parsing and beat inference
web/
  index.html                   Browser application shell
  styles.css                   Phase 1 design system and responsive themes
  app.js                       Browser views and interactions
  core.js                      Synthetic-data browser demonstration algorithms
tests/                         Python and JavaScript tests
docs/                          Architecture, methodology, design and phase notes
```

## Dataset citation

Moody, G. B., & Mark, R. G. (2005). *MIT-BIH Arrhythmia Database* (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2F305

Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.

See `docs/PHASE_3_REAL_DATA_VALIDATION.md`, `docs/PHASE_2_RESEARCH_ENGINE.md`, `docs/ARCHITECTURE.md` and `docs/METHODOLOGY.md` for further details.
