# BIDMC regression baseline screen — 17 November 2026

## Purpose

The supplied BIDMC Rough-Fuzzy estimators show modest held-out performance close to a median baseline. This diagnostic screen asks whether a small set of standard regressors materially improves on the shipped results before changing or replacing the final estimators.

This is a research comparison, not a new model release. No estimator from this screen was promoted to the saved-model registry.

## Data and split

- Source: `features_rr_estimation_no_resp_input.csv` from run `pulselab_bidmc_patient_disjoint_20261117T144120Z_2489`.
- Target: `ref_rr_mean_bpm`.
- Predictors: PPG-only features or the ECG+PPG feature set. RESP-derived predictors, RESP correlations and reference/annotation columns were excluded.
- Training: 36 MIMIC-II patient groups.
- Held-out evaluation: 10 different patient groups.
- The manifest's original patient and record assignments were retained.
- Hyperparameters were selected with 3-fold `GroupKFold` over training patients only. The fixed held-out patients were not used for parameter selection.

## Results

The supplied run reports the following original methods on the same held-out split:

| Feature set | Method | Held-out MAE (BPM) | RMSE (BPM) | R² |
|---|---|---:|---:|---:|
| PPG-only | Rough-Set + Fuzzy | 3.1303 | 4.5810 | 0.0073 |
| PPG-only | Median baseline | 3.1668 | 4.6020 | -0.0018 |
| PPG-only | Random Forest | 3.7839 | 5.2333 | -0.2955 |
| ECG+PPG | Rough-Set + Fuzzy | 3.2598 | 4.6607 | -0.0275 |
| ECG+PPG | Median baseline | 3.1668 | 4.6020 | -0.0018 |
| ECG+PPG | Random Forest | 3.5305 | 4.7843 | -0.0827 |

Additional candidate baselines were trained using the same 36/10 patient split:

| Feature set | Candidate | Training GroupKFold MAE (BPM) | Held-out MAE (BPM) | Held-out R² |
|---|---|---:|---:|---:|
| PPG-only | Ridge | 2.1075 | 3.1306 | 0.0601 |
| PPG-only | ElasticNet | 1.9916 | 3.1652 | 0.0373 |
| PPG-only | SVR-RBF | 2.0920 | 3.6192 | -0.0820 |
| PPG-only | Extra Trees | 2.1820 | 3.5353 | -0.1716 |
| PPG-only | HistGradientBoosting | 2.3311 | 3.6082 | -0.1293 |
| ECG+PPG | Ridge | 2.6443 | 3.2636 | -0.0413 |
| ECG+PPG | ElasticNet | 2.1852 | 3.2359 | -0.0022 |
| ECG+PPG | SVR-RBF | 2.3576 | 3.4311 | -0.0749 |
| ECG+PPG | Extra Trees | 2.1482 | 3.2282 | -0.0772 |
| ECG+PPG | HistGradientBoosting | 2.1513 | 3.3699 | -0.0617 |

## Interpretation

The PPG-only Ridge model is essentially tied with the shipped PPG-only Rough-Fuzzy estimator: 3.1306 versus 3.1303 BPM MAE. Its 0.0362 BPM improvement over the median baseline is too small to justify replacing the existing estimator, especially given the fixed held-out set and the near-zero R² values.

None of the tested ECG+PPG candidates beat the median baseline on this split. Adding ECG features did not establish a predictive advantage in this experiment.

**Decision:** retain the existing saved Rough-Fuzzy estimators as research baselines, do not promote a new candidate, and do not describe the current BIDMC engine as a high-performing respiratory-rate estimator.

## Reproducibility and limitations

The baseline screen used Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0 in the analysis environment. This differs from the pinned training runtime (scikit-learn 1.6.1), so this is a diagnostic comparison rather than a replacement production artifact. The included `tools/evaluate_bidmc_baselines.py` script allows the comparison to be re-run from the supplied feature export and run manifest.

A fixed 10-patient holdout is not external validation. The result needs confirmation with repeated patient-group splits, pre-specified model selection and ideally an independent dataset. Performance is not clinically validated.
