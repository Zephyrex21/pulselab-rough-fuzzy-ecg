# PulseLab experiment report

- **Run ID:** `pulselab_patient_disjoint_detected_20261114T053618Z_5383`
- **Created (UTC):** 2026-11-14T05:38:30.104682+00:00
- **Dataset:** MIT-BIH Arrhythmia Database
- **Split protocol:** `patient-disjoint`
- **Patient-disjoint:** `True`
- **Peak source:** `detected`
- **Training beats:** 44,288
- **Testing beats:** 44,157
- **Held-out patients:** 21
- **Input features:** 47

> Research prototype only. Not clinically validated. Do not use for diagnosis or treatment decisions.

## Model comparison

| model | accuracy | balanced_accuracy | macro_f1 | macro_f1_all_aami | weighted_f1 | macro_roc_auc | n_features_input | n_features_used | train_seconds | predict_seconds | accuracy_95ci_low | accuracy_95ci_high | macro_f1_all_aami_95ci_low | macro_f1_all_aami_95ci_high | patient_accuracy_95ci_low | patient_accuracy_95ci_high | patient_macro_f1_all_aami_95ci_low | patient_macro_f1_all_aami_95ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1: Random Forest | 0.9388 | 0.3600 | 0.3683 | 0.3683 | 0.9157 | 0.8423 | 47 | 47 | 54.9161 | 0.1910 | 0.8807 | 0.9821 | 0.3413 | 0.3856 | 0.8807 | 0.9821 | 0.3413 | 0.3856 |
| B: Rough Set + Fuzzy | 0.9325 | 0.3723 | 0.3568 | 0.3568 | 0.9106 | 0.7370 | 47 | 14 | 25.3437 | 0.0345 | 0.8708 | 0.9746 | 0.3037 | 0.3808 | 0.8708 | 0.9746 | 0.3037 | 0.3808 |
| A: Fuzzy (all features) | 0.9204 | 0.3637 | 0.3400 | 0.3400 | 0.8999 | 0.6280 | 47 | 47 | 8.7972 | 0.0965 | 0.8587 | 0.9629 | 0.2837 | 0.3672 | 0.8587 | 0.9629 | 0.2837 | 0.3672 |
| C2: SVM (RBF) | 0.7835 | 0.3944 | 0.3234 | 0.3234 | 0.8181 | 0.8507 | 47 | 47 | 5.4375 | 4.6058 | 0.6782 | 0.8914 | 0.2541 | 0.4097 | 0.6782 | 0.8914 | 0.2541 | 0.4097 |

## Train / test class distribution

| AAMI class | Train | Test |
|---|---:|---:|
| N | 40,915 | 39,204 |
| S | 845 | 1,691 |
| V | 2,122 | 2,886 |
| F | 403 | 372 |
| Q | 3 | 4 |

## Score calibration and reliability

| Model | Calibration status | Temperature | Held-out Brier | Held-out log loss | Top-label ECE |
|---|---|---:|---:|---:|---:|
| A: Fuzzy (all features) | fitted | 0.1481 | 0.1936 | 0.5932 | 0.1151 |
| B: Rough Set + Fuzzy | fitted | 0.2646 | 0.2351 | 0.5153 | 0.2273 |
| C1: Random Forest | fitted | 0.6414 | 0.1138 | 0.3416 | 0.0311 |
| C2: SVM (RBF) | fitted | 0.8662 | 0.3444 | 0.6266 | 0.0912 |

## Patient-level robustness

| Model | Held-out patients | Accuracy 95% interval | Macro-F1 (all AAMI) 95% interval |
|---|---:|---:|---:|
| A: Fuzzy (all features) | 21 | 0.8587–0.9629 | 0.2837–0.3672 |
| B: Rough Set + Fuzzy | 21 | 0.8708–0.9746 | 0.3037–0.3808 |
| C1: Random Forest | 21 | 0.8807–0.9821 | 0.3413–0.3856 |
| C2: SVM (RBF) | 21 | 0.6782–0.8914 | 0.2541–0.4097 |

Patient-cluster bootstrap samples whole held-out patients rather than individual beats. The intervals describe variation among patients in this split; they do not establish external generalisation.

Temperature scaling, when fitted, uses grouped out-of-fold training-record predictions and fold-specific training-only imputation. Held-out labels are used only for the reliability diagnostics above.
These are internal score diagnostics; they do not establish external or clinical calibration. Fuzzy and SVM score vectors retain model-specific semantics.

## Protocol notes

- Strict patient-disjoint runs exclude record 202 because records 201 and 202 belong to the same subject; historical compatibility mode explicitly allows that overlap.
- In automatic-detection mode, classifier metrics are conditional on detected beats matched one-to-one to reference beats; detector FP/FN are reported separately.
- Fuzzy normalised scores and SVM ranking scores are not calibrated clinical probabilities.
- Record-cluster bootstrap intervals resample held-out records, not individual beats; they are descriptive uncertainty summaries.
- Patient-cluster bootstrap intervals and per-patient metrics are also reported when patient identifiers are available (record identifiers are used as a documented fallback); these summaries describe held-out patient heterogeneity and are not external validation.
- The feature pipeline is offline / retrospective: zero-phase filtering, record-level R-amplitude scaling and next-RR features use full-record or future-beat context; it is not a validated streaming detector.
- A single MIT-BIH split is not external validation or clinical validation.
- When grouped training records support it, scalar temperature scaling is fitted only from grouped out-of-fold training scores; fold-specific imputation is fitted within each fold. The untouched test split is used only for evaluation. This is internal calibration, not external or clinical validation.
- Held-out score reliability diagnostics (Brier, log loss and ECE) are computed after the training-only transform. Fuzzy and SVM outputs remain model score estimates and must not be interpreted as clinical risk probabilities.
- Global feature summaries are model-native importance or fuzzy-rule usage summaries; they are not causal effects.

## Warnings

- Low training support (below 20 beats): {'Q': 3}

## Test-set R-peak detection summary

- Micro sensitivity: 0.9285
- Micro PPV: 0.9990
- Micro F1: 0.9624
- These are detection metrics, not classification metrics.
