# Reproducible MIT-BIH retraining

## Why this workflow exists

The older supplied MIT-BIH run was trained before the current adaptive R-peak detector and therefore cannot be used for uploaded-signal inference under the current input-pipeline contract. This workflow creates a new compatible run instead of relabelling historical metrics.

## Launch from GitHub

1. Merge the manual training workflow into main.
2. Open the repository's Actions tab.
3. Select MIT-BIH patient-disjoint training.
4. Select Run workflow on main.
5. Wait for the job to finish. The workflow downloads the full selected PhysioNet records, trains the models and runs a strict packaging validator.
6. Download the pulselab-mitbih artifact from the completed workflow run. The ZIP file is named PulseLab_MITBIH_<run-id>.zip.

The workflow uses Python 3.13.18 and the exact versions pinned in requirements/mitbih-model-runtime.txt. It uses a 350-minute job timeout and does not commit model binaries to the source repository.

## Experimental protocol

- Database: MIT-BIH Arrhythmia Database, PhysioNet WFDB.
- Training records: the 22-record DS1 split.
- Held-out records: the 21-record patient-disjoint DS2 split.
- Records 201 and 202 are assigned to the same patient group; the strict patient-disjoint split excludes record 202 from DS2.
- R-peak source: detected (automatic), never annotation-guided.
- Models: the current Rough-Set + Fuzzy and fuzzy-only models plus the configured conventional baselines.
- The classifier evaluation is conditional on detected beats matched one-to-one to reference beat annotations. R-peak sensitivity, PPV, false positives and false negatives are reported separately.

## Recovering a run whose training succeeded but packaging failed

Use **Actions → Recover MIT-BIH model package → Run workflow** and provide the original training workflow run ID. The recovery workflow downloads that run's already-trained model/report artifact, downloads the exact 43 configured WFDB records for dataset hashes, validates and packages the existing outputs, and uploads a verified ZIP. It does **not** retrain the models. The default input is the successful training run `44143473651`.

The packager can be invoked either as `python -m tools.package_mitbih_run` from the repository root or directly as `python tools/package_mitbih_run.py`; it bootstraps the repository root for either form.

## Packaging checks

The package step fails rather than publishing a run unless:

- The manifest identifies MIT-BIH, automatic R-peak detection and the strict patient-disjoint protocol.
- Training and held-out record lists exactly match the configured 22/21 split.
- The train/test patient sets are disjoint.
- All 43 records have complete WFDB headers, signal files and annotation files.
- All 43 records are represented in the R-peak detector diagnostics.
- Every serialized model has a SHA-256 checksum matching its saved file.
- The saved inference-pipeline contract matches the current detector, preprocessing, segmentation and feature-extraction versions.

The resulting training_environment.json records the source commit, Python/package versions, source-code hashes, checksums of the selected WFDB record files and checksums of packaged artifacts. The database files themselves are not redistributed in the model ZIP.

## Install the resulting model bundle

Extract the ZIP at the PulseLab repository root. It contains:

- experiments/<run-id>/ — run manifest, comparisons, class-wise results, predictions, detector diagnostics and research report.
- models/<run-id>/ — saved Joblib estimators.
- training_environment.json — reproducibility and integrity metadata.
- README.md — bundle-specific installation and interpretation notes.

Keep the source version identified by the bundle's source commit. The inference catalog checks the model's SHA-256 and input-pipeline contract before offering it for uploaded-signal prediction.

## Interpretation checklist

Do not judge the retrained system by accuracy alone. Review:

- Macro-F1 and balanced accuracy.
- Per-class precision, recall and F1, especially S, V, F and Q.
- Confusion matrix and per-record/per-patient results.
- R-peak sensitivity and PPV for every held-out record.
- Number of reference beats missed and false-positive detections.
- Missing-class warnings and minority-class support.
- Whether performance is stable across records rather than dominated by the normal class.

A single patient-disjoint split is not external validation. The pipeline is offline and uses zero-phase filtering and future-beat context; this is not a validated real-time ECG monitor. It is an academic research prototype, not clinically validated and not for medical decision-making.
