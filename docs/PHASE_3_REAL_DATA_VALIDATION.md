# Phase 3 — Real MIT-BIH validation and reproducible experiments

**Release:** PulseLab 0.3.0  
**Scope:** Move the Python research engine from an experiment scaffold to an auditable real-data workflow. The browser application remains a clearly labelled synthetic demonstration in this phase.

## What Phase 3 adds

- Strict patient-disjoint split as the default.
- A named compatibility mode for the historical 22/22 de Chazal record-wise split.
- Patient identity grouping for MIT-BIH records 201 and 202, which are recordings from the same subject.
- Dataset audit: local WFDB file completeness, sampling rate, channel count, record duration, annotation counts, per-class support, missing records and SHA-256 file provenance.
- Automatic R-peak detection for both classifier training and test feature extraction by default.
- One-to-one matching between detections and AAMI-mappable reference annotations.
- Separate R-peak detection metrics (sensitivity, positive predictive value, F1 and timing error) so missed and false detections are not hidden by classifier metrics.
- Classification metrics conditional on detected beats that match reference beats; an explicit annotation-guided benchmark option remains available for controlled comparison.
- Per-record model metrics and test predictions.
- Record-cluster bootstrap percentile intervals for accuracy and fixed-five-class macro-F1. Whole records are resampled rather than individual beats because beats within a recording are correlated.
- A Markdown research report alongside JSON and CSV experiment artifacts.
- Streamlit tabs for dataset QA, R-peak validation, beat classification and result inspection.

## Important split correction

The commonly copied de Chazal split contains 22 records for training (DS1) and 22 records for testing (DS2). However, MIT-BIH records **201 and 202 originate from the same subject**. Leaving 201 in DS1 and 202 in DS2 creates subject overlap even though the record IDs differ.

PulseLab therefore makes the default `patient-disjoint` protocol stricter: record 202 is removed from the test list. This yields 22 training records and 21 test records (43 unique subjects represented by 43 records). The historical list is retained as `de-chazal-record-wise` for explicit reproduction and comparative research. Runs using that compatibility option are labelled `patient_disjoint: false` and include a warning in their manifest.

Do not describe the historical 22/22 mode as strictly patient-independent.

## Two evaluation levels — report them separately

### 1. Automatic R-peak detection

The signal is preprocessed, R-peaks are found automatically, and detections are matched one-to-one to reference beat annotations using a configurable tolerance (default 75 ms). Each detection/reference event can contribute to at most one true positive.

Report:

- TP, FP and FN
- Sensitivity = TP / (TP + FN)
- PPV = TP / (TP + FP)
- F1 derived from sensitivity and PPV
- Mean signed and absolute timing error for matched pairs

The standalone `validate_rpeaks.py` workflow evaluates the test records without fitting an arrhythmia classifier.

### 2. Beat classification conditional on matched detections

The default training and test feature extraction uses automatically detected peaks. Only detected events matched to reference beat annotations are labelled for the supervised beat classifier. False-positive detections and missed reference beats are represented in the detector metrics, not silently re-labelled as one of the five AAMI beat classes.

Consequently, the reported classifier accuracy / F1 describes **classification on detected, matched beats**, while sensitivity / PPV / F1 describe R-peak detection. They must be considered together and must not be collapsed into an unsupported end-to-end clinical accuracy claim.

`--peak-source annotations` enables an annotation-guided benchmark. It refines reference annotation locations and classifies the resulting beat windows. This is useful for isolating classifier behaviour, but it is not an automatic detection-to-classification evaluation.

## Dataset and protocol

The MIT-BIH Arrhythmia Database contains 48 approximately half-hour, two-channel ECG recordings from 47 subjects, sampled at 360 Hz, with expert-reviewed beat annotations. Four paced-rhythm records configured by the project are excluded from the AAMI experiment lists.

AAMI mapping used by PulseLab:

| Class | MIT-BIH symbols |
|---|---|
| N | N, L, R, e, j |
| S | A, a, J, S |
| V | V, E |
| F | F |
| Q | /, f, Q |

Non-beat rhythm, waveform, signal-quality and comment annotations are excluded from beat classification.

## Windows commands

From the repository root, after setting up `.venv` and installing `requirements.txt`:

```powershell
# Download and audit the quick engineering split
py audit_dataset.py --download --quick

# Run detector validation on quick test records
py validate_rpeaks.py --download --quick

# Run the automatic-detection quick classification experiment
py train.py --download --quick

# Download and audit the full strict patient-disjoint split
py audit_dataset.py --download

# Run the complete strict patient-disjoint experiment
py train.py --download
```

Historical de Chazal list for compatibility research only:

```powershell
py audit_dataset.py --download --split-protocol de-chazal-record-wise
py train.py --download --split-protocol de-chazal-record-wise
```

Annotation-guided classifier benchmark:

```powershell
py train.py --peak-source annotations
```

Run the Streamlit console:

```powershell
streamlit run app.py
```

Run automated tests:

```powershell
py -m pytest -q
node tests/browser_core_smoke.test.js
```

## Output files

Dataset audit folder:

```text
experiments/dataset_audit_<timestamp>/
    dataset_audit.json
    record_inventory.csv
    class_distribution.csv
    file_sha256_manifest.csv
```

R-peak validation folder:

```text
experiments/rpeak_validation_<timestamp>/
    rpeak_validation.json
    rpeak_metrics_per_record.csv
```

Classification run folder:

```text
experiments/<run_id>/
    run_manifest.json
    metrics.json
    model_comparison.csv
    per_class_metrics.csv
    per_record_model_metrics.csv
    test_predictions.csv
    rpeak_detection_metrics.json
    rpeak_detection_per_record.csv
    *_confusion_matrix.csv
    *_rule_base.json
    research_report.md
```

Exact output files depend on the selected mode and whether models are saved.

## Limitations

- The database is not bundled in the project. WFDB downloads the requested MIT-BIH files to the local data directory.
- This sandbox handoff did not contain the MIT-BIH records and could not access package indexes to install WFDB / PyWavelets / Streamlit. Therefore, no real MIT-BIH experiment metrics are claimed in this release. The added dataset-dependent workflow must be run locally after dependencies and data are available.
- This is an offline / retrospective batch pipeline, not a streaming system. Zero-phase filtering uses future samples; record-level normalisation uses full-record detected peaks; and the `rr_next` feature requires the following heartbeat. Streaming use would require causal preprocessing, online normalisation, a latency-aware feature schema and its own evaluation.
- The R-peak detector is a research implementation and is not claimed to meet a medical-device standard.
- Detection metrics use the project's AAMI-mappable beat annotations as the reference set.
- The cluster bootstrap intervals are descriptive, based on one test split and only the number of test record clusters; they do not replace external validation or repeated independent cohorts.
- Feature extraction, detector, class mapping, and split choices remain research design decisions that should be justified and reported in any publication.
- PulseLab remains an academic prototype and is not for diagnosis, treatment, triage, or clinical decisions.

## Dataset citation

Moody, G. B., & Mark, R. G. (2005). *MIT-BIH Arrhythmia Database* (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2F305

Also cite the original database paper when publishing work based on MIT-BIH:

Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
