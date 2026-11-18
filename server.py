"""PulseLab local API and web server (Phases 4–6).

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
from modules import data_loader, dataset_audit, inference, pipeline, validation, bidmc_pipeline, bidmc_inference, bidmc_identity

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
EXPERIMENTS_DIR = config.EXPERIMENTS_DIR
logger = logging.getLogger("pulselab.api")

# The legacy BIDMC value was calculated after load_record() interpolated missing
# samples. It is therefore a post-interpolation finite-sample fraction, not a
# signal-quality score (and must not be used as a proxy for noise or artefacts).
BIDMC_POST_INTERPOLATION_FRACTION = "post_interpolation_finite_fraction"
BIDMC_SIGNAL_QUALITY_NOTE = (
    "Signal quality was not assessed. The reported post_interpolation_finite_fraction "
    "is calculated after missing samples are interpolated; it describes finite values "
    "after preprocessing only and does not measure original missingness, clipping, "
    "flatlines, noise, or physiological signal quality."
)


def _rename_bidmc_quality_metadata_column(frame):
    """Rename legacy post-interpolation metadata without changing model features."""
    if frame is None or not hasattr(frame, "columns") or "window_quality" not in frame.columns:
        return frame
    return frame.rename(columns={"window_quality": BIDMC_POST_INTERPOLATION_FRACTION})


def _clarify_bidmc_prediction_quality(result: dict) -> dict:
    """Expose the legacy finite fraction honestly in API prediction metadata."""
    for row in result.get("predictions", []):
        if "window_quality" in row:
            row[BIDMC_POST_INTERPOLATION_FRACTION] = row.pop("window_quality")
    result["signal_quality_assessment"] = {
        "status": "not_assessed",
        "note": BIDMC_SIGNAL_QUALITY_NOTE,
    }
    return result


app = FastAPI(
    title="PulseLab Research API",
    description=(
        "Local API for dataset QA, reproducible MIT-BIH ECG classification, saved-model ECG inference, "
        "and BIDMC multimodal respiratory-rate feature extraction and trained Rough-Fuzzy inference. "
        "Research use only; not a medical device."
    ),
    version="0.7.0",
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
_BIDMC_MODEL_REGISTRY: dict | None = None
_BIDMC_MODEL_REGISTRY_LOCK = threading.Lock()


class AuditRequest(BaseModel):
    quick: bool = False
    download_missing: bool = False
    split_protocol: Literal["patient-disjoint", "de-chazal-record-wise"] = "patient-disjoint"


class RPeakValidationRequest(BaseModel):
    quick: bool = False
    download_missing: bool = False
    split_protocol: Literal["patient-disjoint", "de-chazal-record-wise"] = "patient-disjoint"
    tolerance_ms: int = Field(default=75, ge=25, le=150)


class ExperimentRequest(BaseModel):
    quick: bool = False
    download_missing: bool = False
    split_protocol: Literal["patient-disjoint", "de-chazal-record-wise"] = "patient-disjoint"
    peak_source: Literal["detected", "annotations"] = "detected"
    include_baselines: bool = True
    save_models: bool = True
    run_name: str | None = Field(default=None, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")


class BIDMCFeatureExportRequest(BaseModel):
    records: list[str] = Field(default_factory=list)
    window_sec: float = Field(default=32.0, ge=16.0, le=120.0)
    stride_sec: float = Field(default=16.0, ge=4.0, le=120.0)


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
        "n_train_patients": manifest.get("n_train_patients"),
        "n_test_patients": manifest.get("n_test_patients"),
        "n_train_records": len(manifest.get("train_records", [])),
        "n_test_records": len(manifest.get("test_records", [])),
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


def _run_bidmc_feature_export(job_id: str, request: BIDMCFeatureExportRequest, progress) -> dict:
    """Extract windowed multimodal BIDMC features and persist training-ready tables."""
    identity_audit = bidmc_identity.audit_patient_identity(config.BIDMC_DATA_DIR)
    if not identity_audit["all_records_verified"]:
        raise bidmc_identity.BIDMCPatientIdentityError(identity_audit["message"])
    progress(0.03, "Locating BIDMC CSV dataset and verifying patient identities")
    full, rr_safe = bidmc_pipeline.build_feature_dataset(
        config.BIDMC_DATA_DIR,
        record_ids=request.records or None,
        window_sec=request.window_sec,
        stride_sec=request.stride_sec,
        progress=lambda fraction, message: progress(0.05 + 0.82 * fraction, message),
    )
    # This column was computed after load_record() interpolation. Keep its
    # numerical value for backwards traceability, but never label it quality.
    full = _rename_bidmc_quality_metadata_column(full)
    rr_safe = _rename_bidmc_quality_metadata_column(rr_safe)
    export_id = "bidmc_features_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    output_dir = (EXPERIMENTS_DIR / "bidmc_feature_exports" / export_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "features_full_multimodal.csv"
    safe_path = output_dir / "features_rr_estimation_no_resp_input.csv"
    full.to_csv(full_path, index=False)
    rr_safe.to_csv(safe_path, index=False)
    patient_ids = sorted(full["patient_id"].dropna().astype(str).unique().tolist()) if not full.empty else []
    summary = {
        "export_id": export_id,
        "dataset": "BIDMC PPG and Respiration Dataset",
        "records": sorted(full["record_id"].dropna().astype(str).unique().tolist()) if not full.empty else [],
        "n_records": int(full["record_id"].nunique()) if not full.empty else 0,
        "n_patient_groups": len(patient_ids),
        "patient_ids": patient_ids,
        "patient_grouping": {
            "status": identity_audit["status"],
            "verified": identity_audit["all_records_verified"],
            "dataset_records_verified": identity_audit["records_verified"],
            "dataset_records_total": identity_audit["records_total"],
            "dataset_patient_groups": identity_audit["verified_patient_groups"],
            "selected_records": int(full["record_id"].nunique()) if not full.empty else 0,
            "selected_patient_groups": len(patient_ids),
            "identity_source": "MIMIC-II subject IDs parsed from WFDB headers",
        },
        "train_ready_for_patient_disjoint_evaluation": identity_audit["all_records_verified"],
        "window_sec": request.window_sec,
        "stride_sec": request.stride_sec,
        "n_windows": int(len(full)),
        "n_labelled_windows": int(full["ref_rr_mean_bpm"].notna().sum()) if not full.empty else 0,
        "n_features_full_multimodal": int(len([c for c in full.columns if c not in {"record_id", "patient_id", "window_start_sec", "window_end_sec", "ref_rr_ann1_bpm", "ref_rr_ann2_bpm", "ref_rr_mean_bpm", "annotator_difference_bpm", "ann1_breath_count", "ann2_breath_count", BIDMC_POST_INTERPOLATION_FRACTION}])),
        "n_features_rr_estimation": int(len([c for c in rr_safe.columns if c not in {"record_id", "patient_id", "window_start_sec", "window_end_sec", "ref_rr_ann1_bpm", "ref_rr_ann2_bpm", "ref_rr_mean_bpm", "annotator_difference_bpm", "ann1_breath_count", "ann2_breath_count", BIDMC_POST_INTERPOLATION_FRACTION}])),
        "files": [full_path.name, safe_path.name],
        "target_definition": "Median interval-based breaths/min independently derived from ann1 and ann2; ref_rr_mean_bpm is the mean of available annotator rates.",
        "leakage_control": "The RR-estimation table excludes RESP-derived predictors and ECG/PPG-to-RESP correlations. Keep RESP as reference/auxiliary input for descriptive full-fusion analysis, not as a predictor when predicting a target derived from RESP annotations.",
        "patient_grouping_note": "Only verified MIMIC-II subject IDs from WFDB headers are accepted for patient-disjoint feature exports. Record-level fallback identifiers are not valid patient IDs.",
        "model_status": "Trained patient-disjoint BIDMC Rough-Fuzzy models are installed; this feature export does not retrain them.",
        "trained_model_run_id": bidmc_inference.BIDMC_RUN_ID,
        "signal_quality_assessment": {
            "status": "not_assessed",
            "metadata_column": BIDMC_POST_INTERPOLATION_FRACTION,
            "note": BIDMC_SIGNAL_QUALITY_NOTE,
        },

    }
    (output_dir / "feature_export_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    progress(0.95, "Writing export metadata")
    return summary


def _get_bidmc_model_registry() -> dict:
    """Load the versioned BIDMC model bundle once per API process."""
    global _BIDMC_MODEL_REGISTRY
    if _BIDMC_MODEL_REGISTRY is None:
        with _BIDMC_MODEL_REGISTRY_LOCK:
            if _BIDMC_MODEL_REGISTRY is None:
                _BIDMC_MODEL_REGISTRY = bidmc_inference.load_trained_registry(
                    config.ROOT, config.MODELS_DIR, EXPERIMENTS_DIR
                )
    return _BIDMC_MODEL_REGISTRY


def _bidmc_model_status_payload() -> dict:
    try:
        registry = _get_bidmc_model_registry()
        payload = {key: value for key, value in registry.items() if key not in {"models_loaded", "artifact_dir", "experiment_dir"}}
        identity_audit = bidmc_identity.audit_patient_identity(config.BIDMC_DATA_DIR)
        if identity_audit["records_total"]:
            payload["local_patient_split_validation"] = bidmc_identity.validate_patient_disjoint_manifest(identity_audit, registry)
        else:
            payload["local_patient_split_validation"] = {"status": "not_checked", "valid": None, "issues": ["The local BIDMC dataset is not installed."]}
        return payload
    except bidmc_inference.BIDMCArtifactError as exc:
        return {
            "available": False,
            "status": "missing_or_incompatible",
            "run_id": bidmc_inference.BIDMC_RUN_ID,
            "error": str(exc),
            "clinical_status": "Academic research prototype; not clinically validated.",
        }


@app.get("/api/bidmc/status")
def bidmc_status():
    """Inspect local BIDMC dataset availability and verify subject identity from WFDB headers."""
    payload = bidmc_pipeline.inventory(config.BIDMC_DATA_DIR)
    identity_audit = bidmc_identity.audit_patient_identity(config.BIDMC_DATA_DIR)
    identity_by_record = {row["record_id"]: row for row in identity_audit["records"]}
    for record in payload.get("records", []):
        identity = identity_by_record.get(record.get("record_id"), {})
        if not identity.get("patient_id_verified"):
            record["record_group_id"] = record.get("patient_id")
            record["patient_id"] = None
        else:
            record["patient_id"] = identity.get("patient_id")
        record["patient_id_verified"] = bool(identity.get("patient_id_verified"))
        record["identity_status"] = identity.get("identity_status", "not_indexed")
    payload["reported_patient_groups"] = payload.get("patient_groups")
    payload["patient_groups"] = identity_audit["verified_patient_groups"] if identity_audit["all_records_verified"] else None
    payload["patient_grouping_verified"] = identity_audit["all_records_verified"]
    payload["patient_identity"] = {
        key: identity_audit[key] for key in (
            "status", "all_records_verified", "records_total", "records_verified",
            "records_unverified", "verified_patient_groups", "unverified_records", "message"
        )
    }
    model_status = _bidmc_model_status_payload()
    payload["trained_model_available"] = model_status.get("available", False)
    payload["trained_model_patient_split_validation"] = model_status.get("local_patient_split_validation")
    if not identity_audit["all_records_verified"]:
        payload["message"] = identity_audit["message"]
    elif payload.get("ready") and payload["trained_model_available"]:
        payload["message"] = "BIDMC signals, annotations and all 53 patient identities are verified. Trained patient-disjoint Rough-Fuzzy respiratory-rate models are installed."
    else:
        payload["message"] = payload.get("message", "BIDMC dataset is not ready.")
    return _json_safe(payload)


@app.get("/api/bidmc/model/status")
def bidmc_model_status():
    """Return provenance and held-out metrics for the trained BIDMC estimators."""
    return _json_safe(_bidmc_model_status_payload())


@app.get("/api/bidmc/model/artifacts/{filename}")
def download_bidmc_model_artifact(filename: str):
    """Download an allow-listed BIDMC training report or held-out evaluation artifact."""
    if Path(filename).name != filename or filename not in bidmc_inference.ARTIFACT_FILES:
        raise HTTPException(status_code=404, detail="BIDMC model artifact not found.")
    path = (EXPERIMENTS_DIR / bidmc_inference.BIDMC_RUN_ID / filename).resolve()
    if path.parent != (EXPERIMENTS_DIR / bidmc_inference.BIDMC_RUN_ID).resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="BIDMC model artifact not found.")
    return FileResponse(path, filename=filename)


@app.get("/api/bidmc/records/{record_id}/predictions")
def bidmc_record_predictions(record_id: str, model: Literal["ecg_ppg", "ppg_only"] = "ecg_ppg"):
    """Run a saved BIDMC respiratory-rate estimator on actual record windows."""
    try:
        registry = _get_bidmc_model_registry()
    except bidmc_inference.BIDMCArtifactError as exc:
        raise HTTPException(status_code=503, detail=f"Trained BIDMC model is unavailable: {exc}") from exc
    try:
        record = bidmc_pipeline.load_record(record_id, config.BIDMC_DATA_DIR)
        result = bidmc_inference.predict_record(record, registry["models_loaded"], model)
        result = _clarify_bidmc_prediction_quality(result)
        identity_audit = bidmc_identity.audit_patient_identity(config.BIDMC_DATA_DIR)
        verified_patient = (identity_audit.get("record_to_patient") or {}).get(record.record_id)
        result["patient_id"] = verified_patient
        result["patient_identity_verified"] = verified_patient is not None
        result["patient_identity_status"] = "verified" if verified_patient else "unverified"
        if verified_patient is None:
            result["patient_identity_note"] = "The WFDB header did not provide an unambiguous MIMIC-II subject ID. This single-record replay is not evidence of patient-level grouping."
        return _json_safe(result)
    except bidmc_pipeline.BIDMCError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except bidmc_inference.BIDMCArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/bidmc/records/{record_id}/preview")
def bidmc_record_preview(record_id: str, start_sec: float = 0.0, duration_sec: float = 20.0):
    try:
        record = bidmc_pipeline.load_record(record_id, config.BIDMC_DATA_DIR)
        preview = bidmc_pipeline.extract_preview(record, start_sec, duration_sec)
        identity_audit = bidmc_identity.audit_patient_identity(config.BIDMC_DATA_DIR)
        verified_patient = (identity_audit.get("record_to_patient") or {}).get(record.record_id)
        preview["patient_id"] = verified_patient
        preview["patient_identity_verified"] = verified_patient is not None
        preview["patient_identity_status"] = "verified" if verified_patient else "unverified"
        return _json_safe(preview)
    except bidmc_pipeline.BIDMCError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/bidmc/feature-exports", status_code=202)
def create_bidmc_feature_export(request: BIDMCFeatureExportRequest, background_tasks: BackgroundTasks):
    if request.stride_sec > request.window_sec:
        raise HTTPException(status_code=422, detail="Stride cannot exceed the selected window length.")
    params = request.model_dump()
    job_id = _create_job("bidmc_feature_export", params)
    background_tasks.add_task(_run_job, job_id, lambda progress: _run_bidmc_feature_export(job_id, request, progress))
    return {"job_id": job_id, "status": "queued", "poll_url": f"/api/jobs/{job_id}"}


@app.get("/api/bidmc/feature-exports/{export_id}/files/{filename}")
def download_bidmc_feature_export(export_id: str, filename: str):
    if not _RUN_ID_RE.fullmatch(export_id) or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid BIDMC export path.")
    if filename not in {"features_full_multimodal.csv", "features_rr_estimation_no_resp_input.csv", "feature_export_summary.json"}:
        raise HTTPException(status_code=404, detail="BIDMC export file not found.")
    root = (EXPERIMENTS_DIR / "bidmc_feature_exports").resolve()
    path = (root / export_id / filename).resolve()
    if path.parent != (root / export_id).resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="BIDMC export file not found.")
    return FileResponse(path, filename=filename)


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
        "bidmc_records_available": bidmc_pipeline.inventory(config.BIDMC_DATA_DIR)["records_found"],
        "bidmc_trained_models_available": _bidmc_model_status_payload().get("available", False),
        "saved_inference_models": len(inference.discover_saved_models(EXPERIMENTS_DIR, config.ROOT, config.MODELS_DIR)),
        "inference_upload_ready": all(deps[key] for key in ("numpy", "pandas", "scipy", "scikit_learn", "pywavelets", "python_multipart")),
        "dataset_root": str(config.DATA_DIR),
        "disclaimer": config.DISCLAIMER,
    }


@app.get("/api/dataset/status")
def dataset_status(split_protocol: Literal["patient-disjoint", "de-chazal-record-wise"] = "patient-disjoint", quick: bool = False):
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


def _experiment_running() -> bool:
    with _JOBS_LOCK:
        return any(j["kind"] == "experiment" and j["status"] not in _TERMINAL_STATES for j in _JOBS.values())


@app.post("/api/experiments", status_code=202)
def create_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks):
    if _experiment_running():
        raise HTTPException(status_code=409, detail="An experiment is already running. Wait for it to finish before starting another.")
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


@app.get("/api/inferences/{inference_id}/beats/{beat_index}/waveform")
def uploaded_inference_beat_waveform(inference_id: str, beat_index: int):
    try:
        return inference.get_inference_waveform(inference_id, beat_index, EXPERIMENTS_DIR)
    except inference.WaveformPreviewUnavailable as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
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
    allowed = {"inference_summary.json", "beat_predictions.csv", "beat_features.csv", "feature_drift_per_beat.csv", "feature_drift_by_feature.csv"}
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
                    "n_train_records": len(manifest.get("train_records", [])),
                    "n_test_records": len(manifest.get("test_records", [])),
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
