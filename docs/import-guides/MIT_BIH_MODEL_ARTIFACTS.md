# PulseLab MIT-BIH trained model artifacts

## Current compatible run

- Run ID: `pulselab_patient_disjoint_adaptive_44143473651`
- Training date: 17 November 2026
- Project version: 0.7.0
- Source commit recorded in bundle: `24bfc2bd75e4a94089ebb7cb634494192252bfc7`
- Training runtime: Python 3.13.18; pinned packages in `requirements/mitbih-model-runtime.txt`
- Detector: `adaptive-local-threshold-v1`
- Input contract: `mitdb-input-pipeline-v1`
- Split: 22 training records / 21 held-out records; patient-disjoint
- Evaluation: 50,218 training beats and 47,416 held-out matched beats

This is the current model run for uploaded-signal inference. The earlier run `pulselab_patient_disjoint_detected_20261114T053618Z_5383` is retained as historical evidence only; its detector/input-pipeline contract does not match the current project.

## Four saved models

- `a_fuzzy_all_features.joblib` — fuzzy rule classifier using all 47 input features.
- `b_rough_set_fuzzy.joblib` — Rough-Set feature selection followed by fuzzy-rule classification; the current run selects 13 of 47 features.
- `c1_random_forest.joblib` — conventional comparison baseline.
- `c2_svm_rbf.joblib` — conventional RBF-SVM comparison baseline.

The model files live in `models/pulselab_patient_disjoint_adaptive_44143473651/`. The matching run manifest, predictions, confusion matrices, detector diagnostics, per-class/per-record/per-patient metrics and research report live in `experiments/pulselab_patient_disjoint_adaptive_44143473651/`.

## Install the artifact

1. Download the recovered artifact from [GitHub Actions run 44147194597](https://github.com/Zephyrex21/pulselab-rough-fuzzy-ecg/actions/runs/44147194597). Its artifact name is `pulselab-mitbih-recovered-pulselab_patient_disjoint_adaptive_44143473651`.
2. Extract the artifact at the PulseLab repository root, preserving the `models/` and `experiments/` paths.
3. Keep the run ID folder names unchanged; manifests refer to relative paths.
4. Install the pinned model runtime in a compatible Python 3.13 environment:

   ```powershell
   python -m pip install -r requirements/mitbih-model-runtime.txt
   ```

5. Start the API from the repository root:

   ```powershell
   python -m uvicorn server:app --host 127.0.0.1 --port 8000
   ```

6. Open `http://127.0.0.1:8000/`, select **MIT-BIH ECG**, and confirm that the model catalog lists the November 17 run. The API docs are at `http://127.0.0.1:8000/api/docs`.

The artifact includes a separate `docs/reproducibility/mitbih_training_environment_adaptive_44143473651.json`. Do not replace the older `mitbih_training_environment.json`; it documents the historical November 14 run.

## Evaluation: explain the results honestly

### R-peak detector on held-out records

- Reference beats: 47,576
- Detected peaks: 47,685
- One-to-one matched true positives: 47,435
- False positives: 250
- False negatives: 141
- Sensitivity: 99.70%
- Positive predictive value: 99.48%
- F1: 99.59%

### Arrhythmia classification on matched detected beats

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Fuzzy (all features) | 90.82% | 38.09% | 0.3285 |
| Rough Set + Fuzzy | 94.36% | 38.45% | 0.3693 |
| Random Forest | 94.40% | 37.66% | 0.3816 |
| SVM (RBF) | 84.05% | 46.79% | 0.4047 |

Accuracy is dominated by the Normal class. SVM has the highest Macro-F1 of these four models, while Random Forest has the highest raw accuracy. The Rough-Set + Fuzzy model reduces the feature set from 47 to 13, but its Macro-F1 is not the best in this run.

Minority-class performance remains a limitation. Class Q has only 3 training examples and 6 held-out examples, so performance for Q is not reliable; all four models have zero Q recall. S recall is also zero for the Fuzzy, Rough Set + Fuzzy and Random Forest models. The SVM provides some S recall, but it remains limited. Do not present the overall accuracy as evidence of robust arrhythmia detection.

Classifier metrics are conditional on detected beats matched one-to-one to reference annotations. Detector false positives and missed beats are reported separately. The held-out split is patient-disjoint, but one internal split is not external validation.

## Integrity and compatibility safeguards

- Every model has a SHA-256 checksum recorded in its run manifest.
- Model discovery and uploaded-signal inference verify the checksum before deserializing a model.
- The inference pipeline contract records the feature schema, detector, preprocessing, segmentation, feature extraction, sampling frequency, filters and beat-window geometry.
- Legacy manifests without a contract, mismatched detector/preprocessing contracts, and annotation-guided benchmark runs are rejected for uploaded-signal inference.
- Do not rename, recompress, reserialize or modify the model files after extraction.

## Scientific and safety limitations

This is an academic research prototype, not a medical device. The data are from MIT-BIH; results do not demonstrate performance on arbitrary ECG devices, external populations or a live clinical stream. The pipeline is offline/retrospective and is not clinically validated. Do not use it for diagnosis or treatment.
