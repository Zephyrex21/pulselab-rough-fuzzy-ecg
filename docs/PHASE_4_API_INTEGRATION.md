> **Historical implementation note:** This document records an earlier development phase. The synthetic browser ECG demonstration described here was removed in PulseLab 0.7.0; the current browser only calls the real Python research APIs.

# Phase 4 — Unified browser and Python research API

**Release:** PulseLab 0.4.0  
**Scope:** Integrate the Apple-inspired browser workspace with the existing Python research engine through a local API, without mixing the synthetic demonstration model with MIT-BIH results.

## What changed

- Added `server.py`, a FastAPI service that also serves the existing `web/` static application at `/web/`.
- Added `/api/health` and `/api/dataset/status` for runtime, dependency and local dataset visibility.
- Added asynchronous, progress-reporting background jobs for dataset audits, R-peak validation and MIT-BIH classification experiments.
- Added job polling through `/api/jobs/{job_id}`.
- Added saved experiment discovery and summary retrieval, including model comparison metrics and reproducibility metadata.
- Added safe downloads for experiment CSV / JSON / Markdown artifacts, R-peak validation reports and dataset audit reports.
- Added a Research Backend navigation page in the browser UI.
- Kept the original browser-side synthetic ECG generator and classifiers as a separate simulation. It is not silently retrained from, or presented as, the Python MIT-BIH engine.

## Run the integrated application

From the project root in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
py -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open:

- Browser: http://127.0.0.1:8000/web/
- Health: http://127.0.0.1:8000/api/health
- API docs: http://127.0.0.1:8000/api/docs

Use the Research Backend page to refresh the service status, inspect the MIT-BIH data directory, run a dataset audit, run independent test-set R-peak validation, launch an experiment, poll progress, inspect saved run metrics and download reports.

The older UI-only command remains available, but it does not start the API:

```powershell
py -m http.server 8000
```

When using this static-only server, the Research Backend page correctly reports that the API is offline. Do not run both servers on port 8000 simultaneously.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Runtime, dependency and record-count status |
| GET | `/api/dataset/status` | Selected train/test record availability |
| POST | `/api/dataset/audits` | Queue a dataset file/provenance audit |
| POST | `/api/rpeak-validations` | Queue independent R-peak detector validation |
| POST | `/api/experiments` | Queue the Rough-Fuzzy and baseline experiment suite |
| GET | `/api/jobs/{job_id}` | Inspect status, progress, result or error |
| GET | `/api/experiments` | List saved experiment runs |
| GET | `/api/experiments/{run_id}/summary` | Read metrics, split metadata and model comparison |
| GET | `/api/experiments/{run_id}/artifacts/{name}` | Download a result artifact |
| GET | `/api/audits/{id}/summary` | Read a dataset audit report |
| GET | `/api/audits/{id}/artifacts/{name}` | Download allow-listed dataset audit artifacts |
| GET | `/api/rpeak-validations/{id}/summary` | Read R-peak detector validation results |
| GET | `/api/rpeak-validations/{id}/artifacts/{name}` | Download R-peak validation artifacts |
| GET | `/api/docs` | Interactive OpenAPI documentation |

Background job state is held in process memory and is lost if the API server is restarted. Completed run artifacts are written to `experiments/` and can be rediscovered after restart.

## Research and safety boundaries

- The browser's existing synthetic ECG charts, rule explanations and Model Evaluation page remain synthetic benchmark outputs.
- Real MIT-BIH model metrics appear only after a Python experiment has actually completed and its report has been saved.
- R-peak detector sensitivity / PPV / F1 are distinct from arrhythmia classification metrics.
- Automatic-mode classification metrics remain conditional on detected beats matched to reference annotations; unmatched detections and missed reference beats remain separate detector errors.
- Quick runs are engineering checks, not the final benchmark. Use the full patient-disjoint protocol for the principal experiment and inspect the per-class support and warnings.
- Dataset files are not included in the repository. Download them from PhysioNet using the project workflow.
- The API has no authentication. It is intended for localhost use only. Bind Uvicorn to `127.0.0.1` and do not expose the service to the internet or a shared network.
- The platform remains an offline academic research prototype, not a medical device. It must not be used for diagnosis, triage, treatment, or clinical decision-making.

## Verification

Automated tests cover service health, selected split metadata, static frontend serving, audit job persistence, missing-data experiment and validation errors, request validation, and artifact path traversal. They use isolated temporary directories and do not require a PhysioNet download.

```powershell
py -m pytest -q
node --check web/app.js
node --check web/app.js
```

These checks verify code paths and the integration contract. They do not establish real MIT-BIH performance; run the dataset audit and experiments locally to generate those results.
