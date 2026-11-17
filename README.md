# PulseLab — Dual Physiological Research Engines

PulseLab is an academic research prototype with **two independent real-data research pipelines** under a shared Rough-Fuzzy methodology. The browser is now a presentation and control interface only: it does not generate synthetic ECG signals, train browser classifiers, or display simulated predictions.

## Research pipelines

### Pipeline 1 — MIT-BIH ECG arrhythmia research

- MIT-BIH Arrhythmia Database (WFDB records; 360 Hz).
- ECG preprocessing, R-peak detection, beat segmentation and 47-feature extraction.
- Rough Set feature selection and fuzzy-rule AAMI classification.
- Random Forest and RBF-SVM baselines.
- Patient-disjoint evaluation, saved experiment manifests and Python saved-model inference.
- The supplied patient-disjoint model artifacts and evaluation reports are included under `models/` and `experiments/`.

**Artifact compatibility note:** the supplied November 14, 2026 training export records source version 0.6.0. PulseLab 0.7.0 retains the 0.6.1 adaptive R-peak detector/inference code. That detector differs from the one used to create the supplied model artifacts, so the saved metrics should be treated as historical run results and the MIT-BIH models should be retrained before relying on predictions under the current detector.

### Pipeline 2 — BIDMC multimodal cardiorespiratory research

- BIDMC PPG and Respiration Dataset 1.0.0 (53 recordings; ECG Lead II, PPG/PLETH and impedance RESP sampled at 125 Hz).
- Separate modality preprocessing and quality checks.
- Synchronized 32-second windows with configurable stride.
- Per-modality feature extraction and full ECG + PPG + RESP fusion.
- Two independent manual breath annotations (ann1 and ann2) used to derive respiratory-rate reference targets.
- A separate leakage-controlled respiratory-rate feature export excludes RESP-derived predictors when the target is derived from RESP annotations.
- `RoughFuzzyRespiratoryRegressor` reuses the project's Rough Set selector and fuzzy rule classifier to produce continuous respiratory-rate estimates after a later training run.

**BIDMC model training is intentionally pending.** The current project includes dataset loading, waveform inspection, feature extraction, reference-label creation, export APIs and the Rough-Fuzzy regression model implementation. It does not display BIDMC predictions or performance metrics until trained artifacts are created using patient-grouped training/testing.

## Dataset setup

Datasets are not bundled in this source ZIP.

### MIT-BIH

Place WFDB files (`.hea`, `.dat`, `.atr`) under:

```text
data/mitdb/
```

The MIT-BIH engine can also use its existing PhysioNet download workflow through the API/Research Console.

### BIDMC

Download the official [BIDMC PPG and Respiration Dataset 1.0.0](https://physionet.org/content/bidmc/1.0.0/) ZIP and extract it under:

```text
data/bidmc/
```

The loader automatically locates the `bidmc_csv/` directory in the standard PhysioNet extraction layout. The folder should contain `bidmc_01_Signals.csv`, `bidmc_01_Breaths.csv`, `bidmc_01_Numerics.csv`, and equivalent files for the remaining recordings.

## Install and run

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The web interface displays the two Python research engines. API documentation is available at <http://127.0.0.1:8000/api/docs>.

The Streamlit research console can be launched separately with:

```bash
streamlit run app.py
```

## BIDMC workflow

1. Extract the official BIDMC dataset beneath `data/bidmc/`.
2. Open **BIDMC Multimodal** and check that all 53 signal/annotation CSVs are indexed.
3. Inspect an actual ECG/PPG/RESP recording in the waveform preview.
4. Export window-level feature tables. The API creates both `features_full_multimodal.csv` and `features_rr_estimation_no_resp_input.csv`.
5. Train/evaluate later in Kaggle with **patient-grouped splits**. Overlapping windows from the same patient must never cross train/test boundaries.
6. Export a BIDMC model artifact and connect the trained model to the inference interface in a subsequent integration step.

The dataset's MIMIC-II patient identifiers are read from accompanying WFDB headers when present. If the user has only CSV files and a subject ID cannot be identified, the loader uses an explicit record-level fallback identifier rather than guessing. Confirm patient grouping from the included metadata before publishing a patient-disjoint evaluation claim.

## API overview

- `GET /api/health` — local service and dataset availability.
- `GET /api/dataset/status` — MIT-BIH split and local record status.
- `POST /api/experiments` — run MIT-BIH experiments.
- `GET /api/models` — discover compatible saved MIT-BIH models.
- `POST /api/inference` — infer on an uploaded ECG CSV/TXT using a saved MIT-BIH model.
- `GET /api/bidmc/status` — inspect BIDMC dataset availability and records.
- `GET /api/bidmc/records/{record_id}/preview` — load real synchronized signal channels.
- `POST /api/bidmc/feature-exports` — extract and persist multimodal window features and reference labels.
- `GET /api/jobs/{job_id}` — poll a background research/export job.

## Important limitations

PulseLab is for retrospective academic research only and is not a medical device. It is not intended for diagnosis, treatment or urgent decisions. Model metrics must be interpreted with class imbalance, patient-level generalization, label quality, and the source-version compatibility of the preprocessing pipeline in mind.
