"""PulseLab local API and web server (Phases 4–5).

Run from the repository root:
    py -m uvicorn server:app --host 127.0.0.1 --port 8000

The service is designed for local academic research use. Do not expose it to a
public network: there is intentionally no user authentication layer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json
import logging
import re
import threading
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import config
from modules import data_loader, dataset_audit, inference, pipeline, validation

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
EXPERIMENTS_DIR = config.EXPERIMENTS_DIR
logger = logging.getLogger("pulselab.api")

app = FastAPI(
    title="PulseLab Research API",
    description=(
        "Local API for dataset QA, R-peak validation, reproducible MIT-BIH "
        "Rough-Fuzzy experiments, and saved-model ECG inference. Research use only; not a medical device."
    ),
    version="0.5.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

# Jobs are process-local by design. Completed experiment outputs are persisted
# under experiments/, so the results survive an API restart.
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_TERMINAL_STATES = {"completed", "failed"}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class AuditRequest(BaseModel):
    quick: bool = True
    download_missing: bool = False
    split_protocol: Literal["patient-disjoint", "de-chazal-record-wise"] = "patient-disjoint"


class RPeakValidationRequest(BaseModel):
    quick: bool = True
    download_missing: bool = False
    split_protocol: Literal["patient-disjoint", "de-chazal-record-wise"] = "patient-disjoint"
    tolerance_ms: int = Field(default=75, ge=25, le=150)


class ExperimentRequest(BaseModel):
    quick: bool = True
    download_missing: bool = False
    split_protocol: Literal["patient-disjoint", "de-chazal-record-wise"] = "patient-disjoint"
    peak_source: Literal["detected", "annotations"] = "detected"
    include_baselines: bool = True
    save_models: bool = True
    run_name: str | None = Field(default=None, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")


def _split_records(protocol: str, quick: bool) -> tuple[list[str], list[str]]:
    train_ids, test_ids = config.get_record_split(protocol)
    if quick:
        return train_ids[:8], test_ids[:6]
    return train_ids, test_ids


def _available_deps() -> dict[str, bool]:
    return {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "scikit_learn": importlib.util.find_spec("sklearn") is not None,
        "wfdb": importlib.util.find_spec("wfdb") is not None,
        "pywavelets": importlib.util.find_spec("pywt") is not None,
        "fastapi": importlib.util.find_spec("fastapi") is not None,
        "uvicorn": importlib.util.find_spec("uvicorn") is not None,
        "python_multipart": importlib.util.find_spec("multipart") is not None,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    """Convert NumPy scalars and non-finite floats into strict JSON values."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _job_snapshot(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found. Jobs are cleared when the local API process restarts.")
        return dict(job)


def _create_job(kind: str, parameters: dict) -> str:
    job_id = uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "created_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "parameters": parameters,
            "result": None,
            "error": None,
        }
    return job_id


def _update_job(job_id: str, **updates) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at_utc"] = _utc_now()


def _progress_callback(job_id: str):
    def update(fraction: float, message: str) -> None:
        _update_job(job_id, status="running", progress=max(0.0, min(1.0, float(fraction))), message=str(message))
    return update


def _run_job(job_id: str, worker) -> None:
    _update_job(job_id, status="running", message="Starting", progress=0.01)
    try:
        result = worker(_progress_callback(job_id))
        _update_job(job_id, status="completed", progress=1.0, message="Completed", result=_json_safe(result), error=None)
    except Exception as exc:  # surfaced to local user; detailed traceback stays in server log
        logger.exception("PulseLab background job %s failed", job_id)
        _update_job(job_id, status="failed", message="Job failed", error=f"{type(exc).__name__}: {exc}")


def _run_dataset_audit(job_id: str, request: AuditRequest, progress) -> dict:
    train_ids, test_ids = _split_records(request.split_protocol, request.quick)
    wanted = list(dict.fromkeys(train_ids + test_ids))
    if request.download_missing:
        if not _available_deps()["wfdb"]:
            raise RuntimeError("WFDB is not installed. Run `pip install -r requirements.txt`, then restart the API.")
        data_loader.download_records(wanted, data_dir=config.DATA_DIR,
                                     progress=lambda p, m: progress(0.05 + 0.35 * p, m))
    progress(0.45, "Auditing record files, annotations, class support and provenance")
    report = dataset_audit.audit_dataset(
        record_ids=wanted,
        data_dir=config.DATA_DIR,
        train_record_ids=train_ids,
        test_record_ids=test_ids,
        split_protocol=request.split_protocol,
    )
    output = EXPERIMENTS_DIR / ("dataset_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
    dataset_audit.write_audit_report(report, output)
    progress(1.0, "Dataset audit saved")
    return {
        "audit_id": output.name,
        "output_dir": str(output),
        "n_records_requested": report["n_records_requested"],
        "n_records_ok": report["n_records_ok"],
        "n_records_failed": report["n_records_failed"],
        "missing_records": report["missing_records"],
        "patient_disjoint": report["patient_disjoint"],
        "class_counts": report["class_counts"],
        "warnings": report["warnings"],
        "artifacts": ["dataset_audit.json", "record_inventory.csv", "class_distribution.csv", "file_sha256_manifest.csv"],
    }


def _run_rpeak_validation_job(job_id: str, request: RPeakValidationRequest, progress) -> dict:
    _, test_ids = _split_records(request.split_protocol, request.quick)
    if request.download_missing:
        if not _available_deps()["wfdb"]:
            raise RuntimeError("WFDB is not installed. Run `pip install -r requirements.txt`, then restart the API.")
        data_loader.download_records(
            test_ids, data_dir=config.DATA_DIR,
            progress=lambda p, m: progress(0.02 + 0.30 * p, m),
        )
    available = set(data_loader.available_records(config.DATA_DIR))
    missing = [rid for rid in test_ids if rid not in available]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} MIT-BIH test records: {', '.join(missing[:12])}. "
            "Enable 'Download missing records' and ensure the internet connection and WFDB dependency are available."
        )
    output = EXPERIMENTS_DIR / ("rpeak_validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
    progress(0.34, "Validating detected R-peaks against reference annotations")
    saved = validation.run_rpeak_validation(
        test_ids,
        data_dir=config.DATA_DIR,
        output_dir=output,
        tolerance_seconds=request.tolerance_ms / 1000.0,
        progress=progress,
    )
    payload = json.loads((saved / "rpeak_validation.json").read_text(encoding="utf-8"))
    return {
        "validation_id": saved.name,
        "output_dir": str(saved),
        "records_evaluated": len(payload.get("record_ids", [])),
        "aggregate": payload.get("aggregate", {}),
        "artifacts": ["rpeak_validation.json", "rpeak_metrics_per_record.csv"],
        "interpretation": payload.get("interpretation"),
    }


def _run_experiment_job(job_id: str, request: ExperimentRequest, progress) -> dict:
    train_ids, test_ids = _split_records(request.split_protocol, request.quick)
    requested_ids = list(dict.fromkeys(train_ids + test_ids))
    if request.download_missing:
        if not _available_deps()["wfdb"]:
            raise RuntimeError("WFDB is not installed. Run `pip install -r requirements.txt`, then restart the API.")
        data_loader.download_records(
            requested_ids, data_dir=config.DATA_DIR,
            progress=lambda p, m: progress(0.02 + 0.30 * p, m),
        )
    available = set(data_loader.available_records(config.DATA_DIR))
    missing = [rid for rid in requested_ids if rid not in available]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} MIT-BIH records: {', '.join(missing[:12])}. "
            "Enable 'Download missing records' and ensure the internet connection and WFDB dependency are available."
        )
    progress(0.34, "Starting MIT-BIH signal processing and feature extraction")
    output = pipeline.run_experiment(
        train_ids,
        test_ids,
        data_dir=config.DATA_DIR,
        output_root=EXPERIMENTS_DIR,
        progress=progress,
        include_baselines=request.include_baselines,
        save_models=request.save_models,
        run_name=request.run_name,
        peak_source=request.peak_source,
        split_protocol=request.split_protocol,
        allow_patient_overlap=request.split_protocol == "de-chazal-record-wise",
    )
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison_path = output / "model_comparison.csv"
    progress(1.0, "Experiment and artifacts saved")
    return {
        "run_id": output.name,
        "output_dir": str(output),
        "n_train": manifest.get("n_train"),
        "n_test": manifest.get("n_test"),
        "models": manifest.get("models", []),
        "split_protocol": request.split_protocol,
        "patient_disjoint": manifest.get("metadata", {}).get("patient_disjoint"),
        "peak_source": request.peak_source,
        "comparison_available": comparison_path.exists(),
        "artifacts": _list_artifact_names(output),
    }


def _list_artifact_names(run_dir: Path) -> list[str]:
    allowed_suffixes = {".json", ".csv", ".md"}
    return sorted(p.name for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_suffixes)


def _read_experiment_summary(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Experiment run was not found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison_path = run_dir / "model_comparison.csv"
    comparison = []
    if comparison_path.is_file():
        import pandas as pd
        comparison = pd.read_csv(comparison_path).replace({float("nan"): None}).to_dict(orient="records")
    return _json_safe({
        "run_id": manifest.get("run_id", run_dir.name),
        "created_at_utc": manifest.get("created_at_utc"),
        "n_train": manifest.get("n_train"),
        "n_test": manifest.get("n_test"),
        "n_features": manifest.get("n_features"),
        "models": manifest.get("models", []),
        "train_class_distribution": manifest.get("train_class_distribution", {}),
        "test_class_distribution": manifest.get("test_class_distribution", {}),
        "warnings": manifest.get("warnings", []),
        "notes": manifest.get("notes", []),
        "metadata": manifest.get("metadata", {}),
        "metrics": manifest.get("metrics", {}),
        "model_comparison": comparison,
        "artifacts": _list_artifact_names(run_dir),
        "disclaimer": manifest.get("disclaimer", config.DISCLAIMER),
    })


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/web/")


@app.get("/api/health")
def health():
    deps = _available_deps()
    available = data_loader.available_records(config.DATA_DIR)
    return {
        "status": "ok",
        "service": "PulseLab Research API",
        "version": app.version,
        "timestamp_utc": _utc_now(),
        "python_version": __import__("sys").version.split()[0],
        "dependencies": deps,
        "wfdb_ready": deps["wfdb"],
        "dataset_records_available": len(available),
        "saved_inference_models": len(inference.discover_saved_models(EXPERIMENTS_DIR, config.ROOT, config.MODELS_DIR)),
        "inference_upload_ready": all(deps[key] for key in ("numpy", "pandas", "scipy", "scikit_learn", "pywavelets", "python_multipart")),
        "dataset_root": str(config.DATA_DIR),
        "disclaimer": config.DISCLAIMER,
    }


@app.get("/api/dataset/status")
def dataset_status(split_protocol: Literal["patient-disjoint", "de-chazal-record-wise"] = "patient-disjoint", quick: bool = True):
    train_ids, test_ids = _split_records(split_protocol, quick)
    available = set(data_loader.available_records(config.DATA_DIR))
    requested = list(dict.fromkeys(train_ids + test_ids))
    missing = [rid for rid in requested if rid not in available]
    deps = _available_deps()
    return {
        "dataset": "MIT-BIH Arrhythmia Database",
        "dataset_root": str(config.DATA_DIR),
        "split_protocol": split_protocol,
        "quick": quick,
        "train_records_requested": len(train_ids),
        "test_records_requested": len(test_ids),
        "available_records": len(available),
        "available_requested": len(requested) - len(missing),
        "missing_records": missing,
        "ready_for_selected_experiment": not missing,
        "wfdb_installed": deps["wfdb"],
        "download_supported": deps["wfdb"],
        "note": "MIT-BIH files are stored locally and are not bundled with PulseLab.",
    }


@app.post("/api/dataset/audits", status_code=202)
def create_dataset_audit(request: AuditRequest, background_tasks: BackgroundTasks):
    params = request.model_dump()
    job_id = _create_job("dataset_audit", params)
    background_tasks.add_task(_run_job, job_id, lambda progress: _run_dataset_audit(job_id, request, progress))
    return {"job_id": job_id, "status": "queued", "poll_url": f"/api/jobs/{job_id}"}


@app.post("/api/rpeak-validations", status_code=202)
def create_rpeak_validation(request: RPeakValidationRequest, background_tasks: BackgroundTasks):
    params = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    job_id = _create_job("rpeak_validation", params)
    background_tasks.add_task(_run_job, job_id, lambda progress: _run_rpeak_validation_job(job_id, request, progress))
    return {"job_id": job_id, "status": "queued", "poll_url": f"/api/jobs/{job_id}"}


@app.post("/api/experiments", status_code=202)
def create_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks):
    params = request.model_dump()
    job_id = _create_job("experiment", params)
    background_tasks.add_task(_run_job, job_id, lambda progress: _run_experiment_job(job_id, request, progress))
    return {"job_id": job_id, "status": "queued", "poll_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_snapshot(job_id)


@app.get("/api/models")
def list_saved_models():
    """Return saved MIT-BIH model artifacts available for uploaded-ECG inference."""
    return {"models": inference.discover_saved_models(EXPERIMENTS_DIR, config.ROOT, config.MODELS_DIR)}


@app.post("/api/inference", status_code=201)
async def create_uploaded_signal_inference(
    file: UploadFile = File(...),
    run_id: str = Form(...),
    model_name: str = Form(...),
    sampling_rate_hz: float = Form(360.0, ge=100.0, le=2000.0),
):
    """Run a previously saved MIT-BIH-trained model on an uploaded ECG CSV/TXT file."""
    filename = str(file.filename or "uploaded_ecg.csv").replace("\\", "/").split("/")[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise HTTPException(status_code=415, detail="Upload a .csv or .txt ECG signal file.")
    content = await file.read(inference.MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(content) > inference.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"The file exceeds the {inference.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    try:
        return await run_in_threadpool(
            inference.execute_inference,
            content=content,
            filename=filename,
            sampling_rate_hz=sampling_rate_hz,
            run_id=run_id,
            model_name=model_name,
            experiments_dir=EXPERIMENTS_DIR,
            project_root=config.ROOT,
            models_root=config.MODELS_DIR,
        )
    except inference.InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"A required inference dependency is missing: {exc}. Install requirements.txt and restart PulseLab.") from exc
    except (OSError, ValueError, TypeError) as exc:
        logger.exception("Uploaded ECG inference failed validation")
        raise HTTPException(status_code=400, detail=f"Inference could not be completed: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected uploaded ECG inference failure")
        raise HTTPException(status_code=500, detail="Inference failed unexpectedly. Check the local API terminal for details.") from exc


@app.get("/api/inferences/{inference_id}/summary")
def uploaded_inference_summary(inference_id: str):
    try:
        summary, _ = inference.read_inference_summary(inference_id, EXPERIMENTS_DIR)
        return summary
    except inference.InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/inferences/{inference_id}/beats/{beat_index}/explanation")
def uploaded_inference_beat_explanation(inference_id: str, beat_index: int):
    try:
        return inference.explain_inference_beat(
            inference_id, beat_index, EXPERIMENTS_DIR, config.ROOT, config.MODELS_DIR
        )
    except inference.InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/inferences/{inference_id}/artifacts/{artifact_name}")
def download_inference_artifact(inference_id: str, artifact_name: str):
    if not inference.INFERENCE_ID_RE.fullmatch(inference_id) or Path(artifact_name).name != artifact_name:
        raise HTTPException(status_code=400, detail="Invalid inference artifact path.")
    output_dir = (EXPERIMENTS_DIR / inference_id).resolve()
    if output_dir.parent != EXPERIMENTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid inference artifact path.")
    allowed = {"inference_summary.json", "beat_predictions.csv", "beat_features.csv"}
    if artifact_name not in allowed:
        raise HTTPException(status_code=404, detail="Inference artifact not found.")
    path = output_dir / artifact_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Inference artifact not found.")
    return FileResponse(path, filename=artifact_name)


@app.get("/api/experiments")
def list_experiments(limit: int = 20):
    limit = max(1, min(int(limit), 100))
    rows = []
    if EXPERIMENTS_DIR.exists():
        for manifest_path in sorted(EXPERIMENTS_DIR.glob("*/run_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                metrics = manifest.get("metrics", {})
                best = None
                for name, values in metrics.items():
                    candidate = {"model": name, "accuracy": values.get("accuracy"), "macro_f1": values.get("macro_f1"), "balanced_accuracy": values.get("balanced_accuracy")}
                    if best is None or (candidate["macro_f1"] or -1) > (best["macro_f1"] or -1):
                        best = candidate
                rows.append({
                    "run_id": manifest.get("run_id", manifest_path.parent.name),
                    "created_at_utc": manifest.get("created_at_utc"),
                    "n_train": manifest.get("n_train"),
                    "n_test": manifest.get("n_test"),
                    "models": manifest.get("models", []),
                    "peak_source": manifest.get("metadata", {}).get("peak_source"),
                    "split_protocol": manifest.get("metadata", {}).get("split_policy"),
                    "patient_disjoint": manifest.get("metadata", {}).get("patient_disjoint"),
                    "best_model": best,
                })
            except Exception:
                logger.exception("Could not read experiment manifest: %s", manifest_path)
    return {"experiments": rows[:limit], "count": len(rows)}


@app.get("/api/experiments/{run_id}/summary")
def experiment_summary(run_id: str):
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID.")
    run_dir = (EXPERIMENTS_DIR / run_id).resolve()
    if run_dir.parent != EXPERIMENTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid run path.")
    return _read_experiment_summary(run_dir)


@app.get("/api/experiments/{run_id}/artifacts/{artifact_name}")
def download_artifact(run_id: str, artifact_name: str):
    if not _RUN_ID_RE.fullmatch(run_id) or Path(artifact_name).name != artifact_name:
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
    run_dir = (EXPERIMENTS_DIR / run_id).resolve()
    if run_dir.parent != EXPERIMENTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid run path.")
    path = run_dir / artifact_name
    if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".md"}:
        raise HTTPException(status_code=404, detail="Artifact not found or file type is not downloadable.")
    return FileResponse(path, filename=artifact_name)


@app.get("/api/rpeak-validations/{validation_id}/summary")
def rpeak_validation_summary(validation_id: str):
    if not _RUN_ID_RE.fullmatch(validation_id) or not validation_id.startswith("rpeak_validation_"):
        raise HTTPException(status_code=400, detail="Invalid R-peak validation ID.")
    validation_dir = (EXPERIMENTS_DIR / validation_id).resolve()
    if validation_dir.parent != EXPERIMENTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid validation path.")
    path = validation_dir / "rpeak_validation.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="R-peak validation not found.")
    return _json_safe(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/rpeak-validations/{validation_id}/artifacts/{artifact_name}")
def download_rpeak_artifact(validation_id: str, artifact_name: str):
    if not _RUN_ID_RE.fullmatch(validation_id) or not validation_id.startswith("rpeak_validation_") or Path(artifact_name).name != artifact_name:
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
    validation_dir = (EXPERIMENTS_DIR / validation_id).resolve()
    if validation_dir.parent != EXPERIMENTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid validation path.")
    if artifact_name not in {"rpeak_validation.json", "rpeak_metrics_per_record.csv"}:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = validation_dir / artifact_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path, filename=artifact_name)


@app.get("/api/audits/{audit_id}/artifacts/{artifact_name}")
def download_audit_artifact(audit_id: str, artifact_name: str):
    if not _RUN_ID_RE.fullmatch(audit_id) or not audit_id.startswith("dataset_audit_") or Path(artifact_name).name != artifact_name:
        raise HTTPException(status_code=400, detail="Invalid audit artifact path.")
    audit_dir = (EXPERIMENTS_DIR / audit_id).resolve()
    if audit_dir.parent != EXPERIMENTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid audit path.")
    allowed = {"dataset_audit.json", "record_inventory.csv", "class_distribution.csv", "file_sha256_manifest.csv"}
    if artifact_name not in allowed:
        raise HTTPException(status_code=404, detail="Audit artifact not found.")
    path = audit_dir / artifact_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audit artifact not found.")
    return FileResponse(path, filename=artifact_name)


@app.get("/api/audits/{audit_id}/summary")
def audit_summary(audit_id: str):
    if not _RUN_ID_RE.fullmatch(audit_id) or not audit_id.startswith("dataset_audit_"):
        raise HTTPException(status_code=400, detail="Invalid audit ID.")
    audit_dir = (EXPERIMENTS_DIR / audit_id).resolve()
    if audit_dir.parent != EXPERIMENTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid audit path.")
    path = audit_dir / "dataset_audit.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset audit not found.")
    return _json_safe(json.loads(path.read_text(encoding="utf-8")))


# Mount last so the API routes always take precedence.
app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
