# PulseLab — Dual Physiological Research Engines

PulseLab is an academic research prototype with **two independent, real-data research pipelines** under a shared Rough-Set + Fuzzy methodology. The browser is a presentation and control interface for the Python engines; it does not synthesize ECG signals, train browser-side classifiers, or display simulated predictions.

## Research pipelines

### 1. MIT-BIH — ECG arrhythmia research

- MIT-BIH Arrhythmia Database (WFDB; 360 Hz).
- ECG preprocessing, R-peak detection, heartbeat segmentation and 47-feature extraction.
- Rough-Set feature selection and fuzzy-rule AAMI classification.
- Random Forest and RBF-SVM comparison baselines.
- Patient-disjoint evaluation, reproducible experiment manifests and Python model inference.
- The latest compatible four-model export and its reports are distributed as a separate GitHub Actions artifact; they are installed under `models/` and `experiments/` locally, not committed as binary files.

**Current compatible model run:** `pulselab_patient_disjoint_adaptive_44143473651` was trained on 17 November 2026 using the adaptive detector (`adaptive-local-threshold-v1`) and the current versioned inference-pipeline contract. It uses 22 training records and 21 held-out records under a strict patient-disjoint split. The four-model bundle is distributed separately as the GitHub Actions artifact **`pulselab-mitbih-recovered-pulselab_patient_disjoint_adaptive_44143473651`**; extract its contents at the repository root. Model binaries are not tracked by Git. The older November 14 run remains historical and is intentionally excluded from uploaded-signal inference because its input-pipeline contract does not match current preprocessing. Current model discovery checks the run contract and each model's SHA-256 before loading.

### 2. BIDMC — multimodal cardiorespiratory research

- BIDMC PPG and Respiration Dataset 1.0.0 (53 recordings; ECG Lead II, PPG/PLETH and impedance RESP at 125 Hz).
- Independent signal preprocessing, quality handling, synchronized 32-second windows and multimodal feature fusion.
- Two manual breath annotations (`ann1`, `ann2`) provide respiratory-rate reference targets.
- Trained Rough-Set + Fuzzy regressors: **ECG + PPG** primary model and **PPG-only** comparison model.
- RESP-derived predictors are excluded from respiratory-rate model inputs to avoid target leakage.
- Model loading verifies the source SHA-256 hashes recorded by the Kaggle training run.
- The Kaggle manifest, held-out predictions, per-patient metrics, annotation agreement and research reports are preserved under `experiments/pulselab_bidmc_patient_disjoint_20261117T144120Z_2489/`.

**Training result caveat:** the held-out split contains 10 patients / 286 windows. On the mean-annotator target, ECG + PPG Rough-Fuzzy achieved 3.260 BPM MAE, PPG-only achieved 3.130 BPM MAE, and the median baseline achieved 3.167 BPM MAE. Performance is modest and close to baseline (R² near zero); additional feature engineering and external validation are needed. Final estimators were refitted on all labelled windows, so record replay is retrospective/in-sample rather than independent validation.

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Install the pinned runtimes used for the supplied trained artifacts:
pip install -r requirements/mitbih-model-runtime.txt
pip install -r requirements/bidmc-model-runtime.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The web interface provides both research engines. API documentation is at <http://127.0.0.1:8000/api/docs>.

The Streamlit research console can be launched separately:

```bash
streamlit run app.py
```

## Dataset setup

Raw datasets are **not included** in this source ZIP.

### MIT-BIH

Place WFDB records (`.hea`, `.dat`, `.atr`) under `data/mitdb/`. The MIT-BIH engine also supports its existing PhysioNet download workflow through the API / Research Console.

### BIDMC

Download the official [BIDMC PPG and Respiration Dataset 1.0.0](https://physionet.org/content/bidmc/1.0.0/) and extract it under `data/bidmc/`. The loader locates the `bidmc_csv/` folder in the standard PhysioNet extraction layout. It expects the per-record signal, breath-annotation and numerics CSV files.

## Application workflows

### MIT-BIH

Open the MIT-BIH ECG workspace to inspect dataset status, launch experiments, view saved model availability, and submit ECG CSV/TXT files for Python inference.

### BIDMC

1. Extract the BIDMC dataset beneath `data/bidmc/`.
2. Open **BIDMC Multimodal** and check that the 53 recordings are indexed.
3. Inspect real ECG / PPG / RESP waveform previews.
4. Use the trained ECG + PPG or PPG-only estimator to generate window-level respiratory-rate estimates.
5. Use the held-out report—not retrospective record replay—to assess generalization.

The BIDMC loader reads MIMIC-II patient identifiers from accompanying WFDB headers. `/api/bidmc/status` distinguishes verified patient IDs from missing, unreadable or ambiguous headers; record-level fallback IDs are never presented as patient IDs. Patient-disjoint feature exports are blocked unless every indexed recording has a verified subject ID. The installed BIDMC model status also checks the saved 36/10 split against the locally verified record-to-patient mapping when the dataset is present.

## API overview

- `GET /api/health` — service and dataset availability.
- `GET /api/dataset/status` — MIT-BIH dataset / split status.
- `POST /api/experiments` — run MIT-BIH experiments.
- `GET /api/models` — discover compatible MIT-BIH models.
- `POST /api/inference` — infer on uploaded ECG CSV/TXT data.
- `GET /api/bidmc/status` — BIDMC dataset availability and records.
- `GET /api/bidmc/model/status` — trained model provenance, source-hash verification and held-out comparison.
- `GET /api/bidmc/records/{record_id}/predictions?model=ecg_ppg|ppg_only` — run a saved estimator on real 32-second windows.
- `GET /api/bidmc/model/artifacts/{filename}` — download allow-listed research reports and metrics.
- `GET /api/bidmc/records/{record_id}/preview` — inspect synchronized real signal channels.
- `POST /api/bidmc/feature-exports` — create multimodal feature exports and reference labels.
- `GET /api/jobs/{job_id}` — poll a background research/export job.

## Documentation and project layout

Project guides, methodology, implementation notes, import guides, changelog and training-environment provenance are organized under `docs/`. Runtime lock files for the supplied model artifacts are under `requirements/`.

- [Architecture](docs/ARCHITECTURE.md)
- [Methodology](docs/METHODOLOGY.md)
- [Design system](docs/DESIGN_SYSTEM.md)
- [Implementation roadmap](docs/IMPLEMENTATION_ROADMAP.md)
- [Changelog](docs/CHANGELOG.md)
- [MIT-BIH artifact and compatibility guide](docs/import-guides/MIT_BIH_MODEL_ARTIFACTS.md)
- [BIDMC artifact and training guide](docs/import-guides/BIDMC_MODEL_ARTIFACTS.md)
- [Training provenance files](docs/reproducibility/)
- [BIDMC baseline screen and candidate evaluation](docs/reproducibility/BIDMC_BASELINE_SCREEN_20261117.md)
- [Patient-grouped BIDMC baseline evaluator](tools/evaluate_bidmc_baselines.py)
- [Manual MIT-BIH retraining workflow guide](docs/reproducibility/MITBIH_RETRAINING.md)
- [MIT-BIH run validator and artifact packager](tools/package_mitbih_run.py)
- [MIT-BIH package recovery workflow](.github/workflows/recover-mitbih-package.yml)

```text
.
├── docs/           # Architecture, methodology, guides, changelog and provenance
├── experiments/    # Reproducible experiment outputs and reports
├── models/         # Trained model artifacts
├── modules/        # Research pipelines, models and inference code
├── requirements/   # Pinned model-runtime dependency sets
├── tests/          # Automated tests
├── tools/          # Reproducible training and artifact packaging utilities
├── web/            # Browser interface (Python API client only)
├── README.md       # Main project overview and setup guide
├── requirements.txt
├── server.py
└── VERSION
```

## Research and safety limitations

PulseLab is an academic research prototype, not a medical device. It is not intended for diagnosis, treatment or urgent decisions. Results must be interpreted with attention to class imbalance, patient-level generalization, annotation quality and model/source compatibility.
