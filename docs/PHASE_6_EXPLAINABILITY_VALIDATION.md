# Phase 6 — Explainability and score validation

**Version:** 0.6.0  
**Scope:** saved Python models, MIT-BIH experiment outputs, and uploaded-signal inference.

## Overview

Phase 6 adds five connected research diagnostics to the Phase 5 inference workflow:

1. grouped out-of-fold scalar temperature scaling, fitted only within the outer training partition;
2. held-out score-reliability diagnostics and patient-cluster robustness summaries;
3. local and global model explainability summaries;
4. training-range feature drift checks;
5. selected-beat waveform previews.

These are research diagnostics. They do not make PulseLab a medical device and do not support diagnosis, treatment, triage, or clinical decisions.

## 1. Grouped out-of-fold temperature scaling

For each model that exposes a class-score interface, PulseLab creates out-of-fold scores using `StratifiedGroupKFold` on the **outer training feature table**. The grouping key is `patient_id` when present and otherwise the record identifier. Each outer-training sample is scored by a fold model that was not fitted using that sample's group.

Missing-value medians for each OOF fold are calculated from that fold's training rows only and then applied to the fold's validation rows. The final experiment model is subsequently fitted on all outer-training rows using the outer-training imputation statistics.

A single positive scalar temperature is optimised against OOF negative log loss. The resulting temperature is applied to the final model's normalised score vectors for the untouched held-out test split and later saved-model inference. If fewer than three groups are available, a fold loses an observed class, scores cannot be aligned, or optimisation fails, the experiment records `status: not_fitted` and retains temperature 1.0. The skip reason is stored in the model's calibration JSON.

This is internal score calibration, not external validation. A temperature fitted to OOF scores is then applied to a final model fitted on all outer-training records; the held-out test split remains independent, but finite-sample and model-refit differences may remain.

## 2. Held-out reliability diagnostics

After the optional training-only transform, the untouched test score matrix is evaluated against held-out labels. PulseLab stores:

- multiclass Brier score;
- multiclass log loss;
- top-label expected calibration error (ECE);
- top-label accuracy and mean top score;
- class-wise Brier score, ECE, reliability bins, and class support.

`score_reliability_diagnostics()` never fits a transform. The test labels are used only for evaluation. ECE depends on bin count and sample size, so inspect reliability bins and support alongside aggregate metrics. Score semantics remain model-specific: fuzzy rule scores, SVM decision-derived ranking scores, and Random Forest probability estimates are not interchangeable by default.

## 3. Local explainability and global feature summaries

### Local reference ablation

For a selected inference beat, PulseLab replaces one model input feature at a time with that feature's median from the outer training set, re-scores the beat, and records the change in the predicted-class score. The browser presents the largest absolute changes first.

This is a one-feature-at-a-time sensitivity analysis. It is not SHAP, a causal effect, a counterfactual, or a guarantee that a feature caused the prediction. ECG features are correlated; independent replacement can create combinations that do not occur naturally.

### Global summaries

- Random Forest: native impurity-based feature importance.
- Fuzzy / Rough-Fuzzy: aggregated antecedent weight multiplied by rule support.
- Other models: global native importance is marked unavailable rather than fabricated.

These measures are model summaries, not causal importance. Impurity importance can favour features with many possible split points; fuzzy antecedent usage represents rule construction, not held-out permutation importance.

## 4. Training-range feature drift

Each new experiment manifest stores training-only q01, q05, median, q95, q99, mean and standard deviation for each feature. Uploaded-signal inference compares extracted features to the training q01–q99 interval and reports out-of-range counts by beat and by feature.

The configured flag is a descriptive range heuristic, not a statistical hypothesis test. An out-of-range value does not prove that a signal is invalid; an in-range value does not prove that the model is suitable for that signal or patient population.

## 5. Selected-beat waveform preview

When a user selects a beat in Model Inference, the API may return the preprocessed, R-amplitude-normalised beat window (pre-R and post-R interval as defined by the project configuration) for display in the browser. The chart marks the R-peak sample and labels time relative to it.

The full uploaded ECG and beat waveform windows are not written to disk. Beat windows are held in a bounded, process-local memory cache (latest two inference sessions). A server restart or cache eviction removes previews; the browser reports that condition and asks the user to rerun inference. Derived beat features, predictions, drift tables and metadata continue to be written to the local inference folder.

The displayed waveform is in R-amplitude-normalised units, not calibrated millivolts. It is an input-processing visualisation, not a diagnosis or reference annotation.

## 6. Reproducibility artifacts

For each model, a new run may include:

```text
*_score_calibration.json
*_score_reliability.json
*_global_explainability.json
```

New run manifests also include `feature_reference`. The human-readable `research_report.md` includes calibration status, temperature, and held-out reliability metrics.

New inference folders may include:

```text
feature_drift_per_beat.csv
feature_drift_by_feature.csv
```

Waveform previews are deliberately **not** saved as report files or included in the inference summary.

## 7. Running Phase 6 locally

Install the full project requirements, then start the integrated local app:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
py -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Run a quick engineering experiment:

```powershell
py train.py --download --quick
```

Run the full default patient-disjoint experiment for the main research evaluation:

```powershell
py train.py --download
```

The Phase 6 grouped OOF calibration fits additional fold models and therefore increases training time. The quick split may not contain all five AAMI classes and may cause calibration to be skipped for a model; inspect each model's `score_calibration` artifact and skip reason.

## 8. Validation and limitations

Unit tests use deterministic synthetic data and fixture models to verify the calibration, explainability, drift, waveform-cache, API and report logic. They do not demonstrate MIT-BIH classification performance. The actual MIT-BIH dataset is not bundled with the repository; users must download it through WFDB before running a real experiment.

Per-patient held-out metrics and patient-cluster bootstrap intervals summarise heterogeneity within the selected test split; they do not replace repeated patient-disjoint cross-validation or establish external generalisation. A single patient-disjoint MIT-BIH split does not establish external generalisation. The feature workflow remains offline / retrospective: filtering is zero-phase, amplitude normalisation uses record context, and next-RR features use future-beat information. PulseLab has not been validated for live streaming or real-time use.

The local FastAPI app has no authentication. Keep it bound to `127.0.0.1` and do not expose it to a public network. Treat uploaded ECG and derived artifacts as sensitive research data even though the raw upload is not persisted.
