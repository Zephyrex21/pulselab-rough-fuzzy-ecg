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
- Two final `RoughFuzzyRespiratoryRegressor` artifacts are installed: a primary ECG+PPG model and a PPG-only comparison model. Both use the existing Rough Set selector and fuzzy rule classifier.
- Trained estimators load only after source-hash verification against the Kaggle run manifest. The browser calls the Python API for record-level inference; it does not run client-side model code.
- The Kaggle experiment, model comparison, per-patient metrics, held-out predictions, manifest, annotation-agreement summary and synchronized feature exports are preserved under `experiments/pulselab_bidmc_patient_disjoint_20261117T144120Z_2489/`.

**Training result caveat:** the held-out patient split contains 10 patients / 286 windows. On the mean annotator target, ECG+PPG Rough-Fuzzy achieved 3.260 BPM MAE and PPG-only achieved 3.130 BPM MAE, compared with 3.167 BPM for the median baseline. These are modest results near baseline (R² is near zero); further feature engineering and external validation are needed. The final saved models were subsequently refitted on all labelled windows, so record replay predictions are not independent held-out results.

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
# For the closest match to the Kaggle BIDMC model runtime:
pip install -r pulselab_bidmc_model_runtime_requirements.txt
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
5. The supplied Kaggle run used **patient-grouped splits**: 36 training patients and 10 held-out patients. Overlapping windows from one patient never cross the split.
6. The trained ECG+PPG and PPG-only estimators are integrated. Choose a local record and estimator in **BIDMC Multimodal** to run saved Python inference. Record replay is descriptive; use the held-out report for generalization estimates.

The dataset's MIMIC-II patient identifiers are read from accompanying WFDB headers when present. If the user has only CSV files and a subject ID cannot be identified, the loader uses an explicit record-level fallback identifier rather than guessing. Confirm patient grouping from the included metadata before publishing a patient-disjoint evaluation claim.

## API overview

- `GET /api/health` — local service and dataset availability.
- `GET /api/dataset/status` — MIT-BIH split and local record status.
- `POST /api/experiments` — run MIT-BIH experiments.
- `GET /api/models` — discover compatible saved MIT-BIH models.
- `POST /api/inference` — infer on an uploaded ECG CSV/TXT using a saved MIT-BIH model.
- `GET /api/bidmc/status` — inspect BIDMC dataset availability and records.
- `GET /api/bidmc/model/status` — trained model provenance, source-hash verification and held-out comparison.
- `GET /api/bidmc/records/{record_id}/predictions?model=ecg_ppg|ppg_only` — run a saved estimator on real 32-second windows.
- `GET /api/bidmc/model/artifacts/{filename}` — download allow-listed research reports / metrics.
- `GET /api/bidmc/records/{record_id}/preview` — load real synchronized signal channels.
- `POST /api/bidmc/feature-exports` — extract and persist multimodal window features and reference labels.
- `GET /api/jobs/{job_id}` — poll a background research/export job.

## Important limitations

PulseLab is for retrospective academic research only and is not a medical device. It is not intended for diagnosis, treatment or urgent decisions. Model metrics must be interpreted with class imbalance, patient-level generalization, label quality, and the source-version compatibility of the preprocessing pipeline in mind.
