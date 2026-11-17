"""Saved-model inference, explanations and ephemeral beat-waveform previews.

Only locally trained PulseLab artifacts described by a run manifest are loaded.
Uploaded full ECG signals are processed in memory and are never persisted.
Derived beat features, predictions and summaries are saved for reproducibility;
selected-beat waveform windows are held in a bounded in-memory cache only.
"""
from __future__ import annotations

import csv
import hashlib
from collections import OrderedDict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
from threading import RLock
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

import config
from modules import calibration, explainability, feature_extraction
from modules.feature_schema import FEATURE_SCHEMA_VERSION
from modules.pipeline_contract import check_mitbih_inference_compatibility

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
INFERENCE_ID_RE = re.compile(r"^inference_[A-Za-z0-9_.-]{1,120}$")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SIGNAL_SAMPLES = 2_000_000
MAX_DURATION_SECONDS = 10 * 60
MIN_DURATION_SECONDS = 3.0
_WAVEFORM_CACHE_MAX_SESSIONS = 2
_WAVEFORM_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_WAVEFORM_CACHE_LOCK = RLock()
TIME_COLUMN_NAMES = {
    "time", "timestamp", "time_s", "time_sec", "seconds", "sec", "sample",
    "sample_index", "sampleindex", "sample_no", "sample_number", "index", "unnamed: 0", "datetime", "date",
    "times", "time_ms", "timesec", "elapsed_time", "elapsedtime",
    "ms", "millis", "milliseconds", "sample_interval", "sampleinterval", "interval", "t", "no", "n", "id",
    "sample_no", "samplenumber", "sample_num", "idx",
}


def _is_time_like_column(values: np.ndarray) -> bool:
    """True for a strictly increasing, evenly stepped column (a time or sample-index axis).

    ECG leads are never perfectly monotonic with a constant step, so this recognises index/time
    columns even when the file has no header or an unfamiliar header name.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return False
    d = np.diff(v)
    mean_step = float(np.mean(d))
    if mean_step <= 0 or not np.all(d > 0):
        return False
    return float(np.std(d)) <= 0.02 * mean_step



class InferenceError(ValueError):
    """A user-correctable input, model or inference validation error."""


class WaveformPreviewUnavailable(InferenceError):
    """The ephemeral selected-beat waveform is no longer in memory."""


def _cache_inference_waveforms(inference_id: str, waveform_rows: np.ndarray) -> None:
    """Keep a bounded set of derived, normalised beat windows in process memory."""
    values = np.asarray(waveform_rows, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        return
    with _WAVEFORM_CACHE_LOCK:
        _WAVEFORM_CACHE[inference_id] = values.copy()
        _WAVEFORM_CACHE.move_to_end(inference_id)
        while len(_WAVEFORM_CACHE) > _WAVEFORM_CACHE_MAX_SESSIONS:
            _WAVEFORM_CACHE.popitem(last=False)


def get_inference_waveform(
    inference_id: str,
    beat_index: int,
    experiments_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Return one beat's normalised waveform from the process-local cache.

    The full upload and beat waveform arrays are not written to disk. Previews
    are intentionally temporary: they disappear after API restart or LRU eviction.
    """
    if not INFERENCE_ID_RE.fullmatch(str(inference_id)):
        raise InferenceError("Invalid inference ID.")
    summary, _ = read_inference_summary(inference_id, experiments_dir)
    predictions = summary.get("predictions") or []
    if not isinstance(beat_index, int) or beat_index < 0 or beat_index >= len(predictions):
        raise InferenceError("Beat index is outside the inference result.")
    with _WAVEFORM_CACHE_LOCK:
        cached = _WAVEFORM_CACHE.get(inference_id)
        if cached is not None:
            _WAVEFORM_CACHE.move_to_end(inference_id)
            samples = cached[beat_index].astype(float).tolist() if beat_index < len(cached) else None
        else:
            samples = None
    if samples is None:
        raise WaveformPreviewUnavailable(
            "This waveform preview is no longer in memory (the server may have restarted or the preview cache was evicted). Re-run inference to view it. The ECG waveform is not retained on disk."
        )
    pre_samples = int(round(config.PRE_R_SEC * config.FS))
    time_ms = ((np.arange(len(samples), dtype=float) - pre_samples) / float(config.FS) * 1000.0).tolist()
    return {
        "available": True,
        "inference_id": inference_id,
        "beat_index": beat_index,
        "sampling_rate_hz": int(config.FS),
        "r_peak_sample_in_window": pre_samples,
        "time_ms": time_ms,
        "samples": samples,
        "amplitude_units": "R-amplitude-normalised units",
        "retention": "process memory only; not persisted to disk",
        "predicted_class": predictions[beat_index].get("predicted_class"),
    }


def _safe_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise InferenceError("The requested model or inference path is outside its allowed directory.")
    return resolved


def _artifact_path(artifact_value: str, project_root: Path, models_root: Path) -> Path:
    raw = Path(str(artifact_value))
    candidate = raw if raw.is_absolute() else project_root / raw
    candidate = _safe_path(candidate, models_root)
    if candidate.suffix.lower() != ".joblib" or not candidate.is_file():
        raise InferenceError("The saved model artifact is missing. Re-run the experiment with model saving enabled.")
    return candidate


def _verify_model_artifact_hash(path: Path, expected_hash: Any) -> str:
    """Fail closed when a saved MIT-BIH model has no trusted checksum or was changed."""
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise InferenceError(
            "The saved model manifest does not contain a valid model-artifact SHA-256 checksum. "
            "Re-train and re-export the model with the current PulseLab pipeline."
        )
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash.lower() != expected_hash.lower():
        raise InferenceError(
            "The saved MIT-BIH model artifact failed its SHA-256 integrity check. "
            "The file may be corrupted or replaced; restore the original artifact or retrain the model."
        )
    return actual_hash


def _expected_channel_count(feature_columns: list[str]) -> int:
    channel_indexes = []
    for name in feature_columns:
        match = re.match(r"^c(\d+)_", str(name))
        if match:
            channel_indexes.append(int(match.group(1)))
    if not channel_indexes:
        raise InferenceError("This saved model does not contain the PulseLab ECG channel feature schema.")
    expected = max(channel_indexes) + 1
    if expected not in (1, 2):
        raise InferenceError(f"Unsupported saved model channel count: {expected}.")
    return expected


def discover_saved_models(
    experiments_dir: Path | str | None = None,
    project_root: Path | str | None = None,
    models_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List trusted, saved MIT-BIH models whose artifacts still exist."""
    experiments_dir = Path(experiments_dir or config.EXPERIMENTS_DIR)
    project_root = Path(project_root or config.ROOT)
    models_root = Path(models_root or config.MODELS_DIR)
    if not experiments_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    for manifest_path in sorted(experiments_dir.glob("*/run_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_id = str(manifest.get("run_id") or manifest_path.parent.name)
            if not RUN_ID_RE.fullmatch(run_id):
                continue
            metadata = manifest.get("metadata") or {}
            if metadata.get("dataset") != "MIT-BIH Arrhythmia Database":
                continue
            if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                continue
            compatible, _reason = check_mitbih_inference_compatibility(manifest)
            if not compatible:
                continue
            feature_columns = manifest.get("feature_columns") or []
            channel_count = _expected_channel_count(feature_columns)
            metrics = manifest.get("metrics") or {}
            for model_name, model_metrics in metrics.items():
                artifact = (model_metrics or {}).get("model_artifact")
                if not artifact:
                    continue
                try:
                    path = _artifact_path(artifact, project_root, models_root)
                    artifact_hash = _verify_model_artifact_hash(path, (model_metrics or {}).get("model_artifact_sha256"))
                except (InferenceError, OSError):
                    continue
                results.append({
                    "model_id": f"{run_id}::{model_name}",
                    "run_id": run_id,
                    "model_name": str(model_name),
                    "artifact_name": path.name,
                    "created_at_utc": manifest.get("created_at_utc"),
                    "n_features": len(feature_columns),
                    "n_channels": channel_count,
                    "classes": list(config.CLASSES),
                    "peak_source": metadata.get("peak_source"),
                    "split_protocol": metadata.get("split_policy"),
                    "patient_disjoint": metadata.get("patient_disjoint"),
                    "feature_schema_version": manifest.get("feature_schema_version"),
                    "inference_pipeline_contract": (metadata.get("inference_pipeline_contract") or {}).copy(),
                    "score_interpretation": (manifest.get("score_interpretation") or {}).get(model_name, "model score"),
                    "score_temperature": float(((model_metrics or {}).get("score_calibration") or {}).get("temperature", (model_metrics or {}).get("score_temperature", 1.0))),
                    "calibration_status": ((model_metrics or {}).get("score_calibration") or {}).get("status", "not_available"),
                })
        except (OSError, ValueError, TypeError, json.JSONDecodeError, InferenceError):
            # One malformed historical run must not prevent discovery of other models.
            continue
    return results


def get_saved_model_entry(
    run_id: str,
    model_name: str,
    experiments_dir: Path | str | None = None,
    project_root: Path | str | None = None,
    models_root: Path | str | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Resolve a model by run/model name; arbitrary filesystem paths are never accepted."""
    if not RUN_ID_RE.fullmatch(str(run_id)):
        raise InferenceError("Invalid experiment run ID.")
    experiments_dir = Path(experiments_dir or config.EXPERIMENTS_DIR).resolve()
    project_root = Path(project_root or config.ROOT).resolve()
    models_root = Path(models_root or config.MODELS_DIR).resolve()
    run_dir = _safe_path(experiments_dir / run_id, experiments_dir)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise InferenceError("The selected experiment run does not exist or has no manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata") or {}
    if metadata.get("dataset") != "MIT-BIH Arrhythmia Database":
        raise InferenceError("Only models trained by the MIT-BIH research pipeline are available for uploaded-signal inference.")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise InferenceError("The selected model uses an unsupported feature schema. Retrain it with the current PulseLab pipeline.")
    compatible, compatibility_reason = check_mitbih_inference_compatibility(manifest)
    if not compatible:
        raise InferenceError(compatibility_reason or "The selected model is incompatible with the current MIT-BIH inference pipeline. Retrain it before use.")
    model_metrics = (manifest.get("metrics") or {}).get(model_name)
    if not isinstance(model_metrics, dict) or not model_metrics.get("model_artifact"):
        raise InferenceError("No saved artifact was found for this model. Re-run the experiment with model saving enabled.")
    artifact = _artifact_path(model_metrics["model_artifact"], project_root, models_root)
    artifact_hash = _verify_model_artifact_hash(artifact, model_metrics.get("model_artifact_sha256"))
    feature_columns = list(manifest.get("feature_columns") or [])
    if not feature_columns:
        raise InferenceError("The run manifest does not contain the training feature schema.")
    channel_count = _expected_channel_count(feature_columns)
    entry = {
        "run_id": run_id,
        "model_name": model_name,
        "artifact_name": artifact.name,
        "model_artifact_sha256": artifact_hash,
        "n_channels": channel_count,
        "n_features": len(feature_columns),
        "score_interpretation": (manifest.get("score_interpretation") or {}).get(model_name, "model score"),
        "score_temperature": float((model_metrics.get("score_calibration") or {}).get("temperature", model_metrics.get("score_temperature", 1.0))),
        "calibration_status": (model_metrics.get("score_calibration") or {}).get("status", "not_available"),
    }
    return entry, artifact, manifest


def parse_ecg_csv(content: bytes, expected_channels: int, sampling_rate_hz: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse a CSV/TXT upload into finite sample-major ECG channels.

    Accepts headered or headerless comma/semicolon/tab/whitespace delimited files.
    Common time/index columns are ignored when headers identify them. The first
    `expected_channels` remaining numeric columns are treated as ECG leads.
    """
    if not content:
        raise InferenceError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise InferenceError(f"The file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InferenceError("The CSV/TXT file must be UTF-8 encoded.") from exc
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) < 4:
        raise InferenceError("The file must contain a header (optional) and at least three seconds of numeric ECG samples.")
    sample = "\n".join(lines[:8])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
    except csv.Error:
        rows = [re.split(r"\s+", line.strip()) for line in lines]
    rows = [row for row in rows if row]
    if not rows:
        raise InferenceError("No data rows were found in the uploaded file.")
    first = [cell.strip() for cell in rows[0]]
    has_header = any(_to_float(cell) is None for cell in first)
    headers = [cell.strip() for cell in first] if has_header else [f"column_{i}" for i in range(len(first))]
    data_rows = rows[1:] if has_header else rows
    width = len(headers)
    if width < expected_channels:
        raise InferenceError(f"The selected model requires {expected_channels} ECG channel(s), but the file has only {width} column(s).")
    parsed = []
    for row_no, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) != width:
            raise InferenceError(f"Inconsistent CSV column count on data row {row_no}: expected {width}, found {len(row)}.")
        parsed.append([_to_float(cell) for cell in row])
    if not parsed:
        raise InferenceError("No numeric ECG samples were found after the header.")
    matrix = np.asarray([[np.nan if value is None else value for value in row] for row in parsed], dtype=float)
    candidates = []
    skipped_time_columns: list[str] = []
    for index, name in enumerate(headers):
        normalized = re.sub(r"[^a-z0-9_:]+", "", name.lower())
        if has_header and normalized in {re.sub(r"[^a-z0-9_:]+", "", n) for n in TIME_COLUMN_NAMES}:
            skipped_time_columns.append(name)
            continue
        valid_ratio = float(np.isfinite(matrix[:, index]).mean())
        if valid_ratio >= 0.98:
            if _is_time_like_column(matrix[:, index]):
                skipped_time_columns.append(headers[index])
                continue
            candidates.append(index)
    if len(candidates) < expected_channels:
        raise InferenceError(
            f"Could not identify {expected_channels} numeric ECG channel(s) after skipping time/index column(s) "
            f"{skipped_time_columns or 'none'}. Provide {expected_channels} numeric ECG lead columns."
        )
    chosen = candidates[:expected_channels]
    signal = matrix[:, chosen]
    # Permit a small number of isolated missing samples and interpolate only short gaps.
    frame = pd.DataFrame(signal).replace([np.inf, -np.inf], np.nan)
    max_gap = max(1, int(round(sampling_rate_hz * 0.05)))
    frame = frame.interpolate(method="linear", limit=max_gap, limit_area="inside")
    finite_rows = np.isfinite(frame.to_numpy()).all(axis=1)
    dropped_rows = int((~finite_rows).sum())
    if dropped_rows:
        frame = frame.loc[finite_rows]
    signal = frame.to_numpy(dtype=float)
    if len(signal) > MAX_SIGNAL_SAMPLES:
        raise InferenceError(f"The signal exceeds the {MAX_SIGNAL_SAMPLES:,}-sample processing limit.")
    duration = len(signal) / sampling_rate_hz
    if duration < MIN_DURATION_SECONDS:
        raise InferenceError(f"The signal must be at least {MIN_DURATION_SECONDS:g} seconds long after cleaning.")
    if duration > MAX_DURATION_SECONDS:
        raise InferenceError(f"The signal exceeds the {MAX_DURATION_SECONDS // 60}-minute processing limit.")
    if not np.isfinite(signal).all():
        raise InferenceError("The selected ECG channel(s) contain missing or non-numeric values that cannot be safely interpolated.")
    peak_to_peak = np.ptp(signal, axis=0)
    if np.any(peak_to_peak <= 1e-12):
        raise InferenceError("At least one selected ECG channel is constant; choose valid ECG signal columns.")
    metadata = {
        "channel_names": [headers[index] for index in chosen],
        "numeric_columns_detected": len(candidates),
        "time_or_index_columns_skipped": skipped_time_columns,
        "rows_dropped_after_interpolation": dropped_rows,
        "n_samples": int(len(signal)),
        "duration_seconds": float(duration),
        "sampling_rate_input_hz": float(sampling_rate_hz),
    }
    return signal, metadata


def _to_float(value: str) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _normalise_score_matrix(scores: Any, classes: list[str], n_rows: int) -> np.ndarray | None:
    if scores is None:
        return None
    array = np.asarray(scores, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[0] != n_rows or array.shape[1] != len(classes):
        return None
    if not np.isfinite(array).all():
        return None
    if np.any(array < 0):
        shifted = array - array.max(axis=1, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -60, 60))
        return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)
    array = np.clip(array, 0.0, None)
    totals = array.sum(axis=1, keepdims=True)
    zero = totals[:, 0] <= 1e-12
    result = array / np.maximum(totals, 1e-12)
    if zero.any():
        result[zero] = 1.0 / max(len(classes), 1)
    return result


def execute_inference(
    *,
    content: bytes,
    filename: str,
    sampling_rate_hz: float,
    run_id: str,
    model_name: str,
    experiments_dir: Path | str | None = None,
    project_root: Path | str | None = None,
    models_root: Path | str | None = None,
) -> dict[str, Any]:
    """Run a trained MIT-BIH model on a user-uploaded ECG CSV/TXT file."""
    if not (100.0 <= float(sampling_rate_hz) <= 2000.0):
        raise InferenceError("Sampling rate must be between 100 and 2000 Hz.")
    entry, artifact_path, manifest = get_saved_model_entry(
        run_id, model_name, experiments_dir, project_root, models_root
    )
    model = joblib.load(artifact_path)
    signal, input_meta = parse_ecg_csv(content, entry["n_channels"], float(sampling_rate_hz))
    output = feature_extraction.process_signal(signal, float(sampling_rate_hz), rpeaks=None, primary=0)
    feature_frame = output["features"].copy().reset_index(drop=True)
    if feature_frame.empty:
        raise InferenceError("No complete heartbeat segments were detected. Upload a longer ECG segment with clear QRS complexes.")
    if "r_sample" not in feature_frame.columns:
        raise InferenceError("The signal-processing pipeline did not return beat sample locations.")
    feature_columns = list(manifest.get("feature_columns") or getattr(model, "feature_names_", []))
    missing = [column for column in feature_columns if column not in feature_frame.columns]
    if missing:
        raise InferenceError(f"Uploaded signal did not produce the saved model feature schema. Missing features: {missing[:8]}.")
    X = feature_frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = manifest.get("feature_medians") or {}
    for column in feature_columns:
        fallback = medians.get(column)
        if fallback is None:
            if X[column].isna().any():
                raise InferenceError(f"The saved run does not include a training median for non-finite feature '{column}'. Retrain the model with Phase 5 or later.")
        else:
            X[column] = X[column].fillna(float(fallback))
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise InferenceError("Feature extraction produced non-finite values that could not be resolved with training-only medians.")
    drift_summary, drift_per_beat, drift_per_feature = explainability.assess_feature_drift(
        X, manifest.get("feature_reference")
    )
    try:
        predictions = np.asarray(model.predict(X)).astype(str)
    except Exception as exc:
        raise InferenceError(f"The saved model could not predict this feature table: {type(exc).__name__}: {exc}") from exc
    if len(predictions) != len(X):
        raise InferenceError("The saved model returned an unexpected number of predictions.")
    classes = [str(value) for value in getattr(model, "classes_", config.CLASSES)]
    raw_scores = None
    if callable(getattr(model, "scores", None)):
        raw_scores = model.scores(X)
    elif callable(getattr(model, "predict_proba", None)):
        raw_scores = model.predict_proba(X)
    score_matrix = _normalise_score_matrix(raw_scores, classes, len(X))
    temperature = float(entry.get("score_temperature", 1.0))
    if score_matrix is not None:
        score_matrix = calibration.apply_temperature(score_matrix, temperature)
    all_classes = list(config.CLASSES)
    score_rows: list[dict[str, float] | None] = []
    top_scores: list[float | None] = []
    margins: list[float | None] = []
    for row_index in range(len(X)):
        if score_matrix is None:
            score_rows.append(None); top_scores.append(None); margins.append(None)
            continue
        score_map = {label: float(score_matrix[row_index, class_index]) for class_index, label in enumerate(classes)}
        ordered = sorted(score_map.values(), reverse=True)
        score_rows.append({label: score_map[label] for label in all_classes if label in score_map})
        top_scores.append(float(ordered[0]) if ordered else None)
        margins.append(float(ordered[0] - ordered[1]) if len(ordered) > 1 else (float(ordered[0]) if ordered else None))
    r_samples = feature_frame["r_sample"].to_numpy(dtype=int)
    beat_rows = []
    for index, (sample_index, prediction) in enumerate(zip(r_samples, predictions)):
        class_score_values = np.asarray(list((score_rows[index] or {}).values()), dtype=float)
        if len(class_score_values) and class_score_values.sum() > 0:
            p = class_score_values / class_score_values.sum()
            entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))) / np.log(max(len(p), 2)))
        else:
            entropy = None
        drift_count = int(drift_per_beat.iloc[index]["n_features_outside_training_q01_q99"]) if drift_per_beat is not None else None
        row = {
            "beat_index": int(index),
            "sample_index_360hz": int(sample_index),
            "time_seconds": float(sample_index / config.FS),
            "predicted_class": str(prediction),
            "top_model_score": top_scores[index],
            "score_margin": margins[index],
            "normalised_score_entropy": entropy,
            "high_score_entropy": bool(entropy is not None and entropy >= 0.80),
            "n_features_outside_training_range": drift_count,
            "class_scores": score_rows[index],
            "low_score_separation": bool(margins[index] is not None and margins[index] < 0.10),
        }
        beat_rows.append(row)
    inference_id = "inference_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    experiments_dir = Path(experiments_dir or config.EXPERIMENTS_DIR)
    output_dir = experiments_dir / inference_id
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_frame = pd.DataFrame([
        {
            "beat_index": row["beat_index"],
            "sample_index_360hz": row["sample_index_360hz"],
            "time_seconds": row["time_seconds"],
            "predicted_class": row["predicted_class"],
            "top_model_score": row["top_model_score"],
            "score_margin": row["score_margin"],
            "normalised_score_entropy": row["normalised_score_entropy"],
            "high_score_entropy": row["high_score_entropy"],
            "n_features_outside_training_range": row["n_features_outside_training_range"],
            "low_score_separation": row["low_score_separation"],
            **{f"score_{label}": (row["class_scores"] or {}).get(label) for label in all_classes},
        }
        for row in beat_rows
    ])
    prediction_frame.to_csv(output_dir / "beat_predictions.csv", index=False)
    feature_frame.to_csv(output_dir / "beat_features.csv", index=False)
    artifacts = ["beat_predictions.csv", "beat_features.csv", "inference_summary.json"]
    if drift_per_beat is not None and drift_per_feature is not None:
        drift_per_beat.to_csv(output_dir / "feature_drift_per_beat.csv", index=False)
        drift_per_feature.to_csv(output_dir / "feature_drift_by_feature.csv", index=False)
        artifacts.extend(["feature_drift_per_beat.csv", "feature_drift_by_feature.csv"])
    counts = {label: int((predictions == label).sum()) for label in all_classes}
    warnings = [
        "The model was trained on MIT-BIH recordings. Different ECG devices, lead placements, acquisition settings or patient populations may cause domain shift.",
        "This upload has no reference beat annotations, so R-peak sensitivity, positive predictive value and missed-beat counts cannot be measured for this signal.",
    ]
    if input_meta["rows_dropped_after_interpolation"]:
        warnings.append(f"{input_meta['rows_dropped_after_interpolation']} rows with incomplete samples were excluded after short-gap interpolation.")
    if drift_summary and drift_summary.get("high_drift_flag"):
        warnings.append(
            "A notable fraction of extracted beat features falls outside the training 1st–99th percentile ranges. "
            "Review the feature-drift report; this is a descriptive domain-shift warning, not a diagnosis or automatic rejection."
        )
    unexpected_labels = sorted(set(predictions) - set(all_classes))
    if unexpected_labels:
        warnings.append(f"The model returned labels outside the expected AAMI N/S/V/F/Q set: {unexpected_labels}. Review the selected run.")
    if manifest.get("metadata", {}).get("peak_source") == "annotations":
        warnings.append("The selected model was trained with annotation-guided reference beats; uploaded-signal inference uses automatically detected R-peaks, so this is a processing-protocol mismatch.")
    summary = {
        "inference_id": inference_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": Path(filename or "uploaded_ecg.csv").name,
        "input": input_meta,
        "preprocessing": {
            "resampled_sampling_rate_hz": int(config.FS),
            "automatic_r_peak_detection": True,
            "complete_beat_window_seconds": [float(-config.PRE_R_SEC), float(config.POST_R_SEC)],
            "n_rpeaks_detected": int(len(output.get("rpeaks", []))),
            "n_complete_beats": int(len(feature_frame)),
            "n_edge_beats_excluded": int(len(output.get("rpeaks", [])) - len(feature_frame)),
        },
        "model": {
            **entry,
            "training_peak_source": manifest.get("metadata", {}).get("peak_source"),
            "training_split_protocol": manifest.get("metadata", {}).get("split_policy"),
            "patient_disjoint_training": manifest.get("metadata", {}).get("patient_disjoint"),
            "feature_schema_version": manifest.get("feature_schema_version"),
            "score_temperature": temperature,
            "calibration_status": entry.get("calibration_status", "not_available"),
        },
        "n_predictions": len(beat_rows),
        "waveform_preview_available": bool(output.get("beats") and len(output.get("beats", [])) > 0),
        "class_counts": counts,
        "score_interpretation": entry["score_interpretation"],
        "score_note": ("Scores use a scalar temperature fitted from grouped out-of-fold MIT-BIH training records; test/consumer uploads are not used to refit it. This internal calibration is not external or clinical validation." if entry.get("calibration_status") == "fitted" else "Scores are model-specific normalised outputs; a calibration transform was not fitted for this run. They are not clinical probabilities or risk estimates."),
        "feature_drift": drift_summary,
        "explainability": {
            "local_method": "single-feature replacement with training median",
            "global_method": "model-native or fuzzy-rule summary where available",
            "limitations": [
                "Local feature effects are sensitivity analyses, not causal explanations or SHAP values.",
                "Correlated features can make one-at-a-time replacement effects misleading.",
            ],
        },
        "warnings": warnings,
        "artifacts": artifacts,
        "disclaimer": config.DISCLAIMER,
        "predictions": beat_rows,
    }
    # Cache only the selected-channel normalised beat windows. The full uploaded
    # signal and waveform preview arrays are deliberately absent from disk.
    beat_windows = None
    try:
        # process_signal's ``beats`` arrays already contain only complete
        # windows (segment_beats has applied its valid mask).
        beat_windows_all = np.asarray(output.get("beats", [])[0], dtype=float)
        if beat_windows_all.ndim == 2:
            beat_windows = beat_windows_all
    except (IndexError, TypeError, ValueError):
        beat_windows = None
    waveform_ready = beat_windows is not None and len(beat_windows) == len(beat_rows)
    summary["waveform_preview_available"] = bool(waveform_ready)
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if waveform_ready:
        _cache_inference_waveforms(inference_id, beat_windows)
    return summary


def read_inference_summary(inference_id: str, experiments_dir: Path | str | None = None) -> tuple[dict[str, Any], Path]:
    if not INFERENCE_ID_RE.fullmatch(str(inference_id)):
        raise InferenceError("Invalid inference ID.")
    experiments_dir = Path(experiments_dir or config.EXPERIMENTS_DIR).resolve()
    output_dir = _safe_path(experiments_dir / inference_id, experiments_dir)
    summary_path = output_dir / "inference_summary.json"
    if not summary_path.is_file():
        raise InferenceError("Inference result was not found.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, output_dir


def explain_inference_beat(
    inference_id: str,
    beat_index: int,
    experiments_dir: Path | str | None = None,
    project_root: Path | str | None = None,
    models_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return a rule-level explanation for one stored inference beat when supported."""
    summary, output_dir = read_inference_summary(inference_id, experiments_dir)
    predictions = summary.get("predictions") or []
    if not isinstance(beat_index, int) or beat_index < 0 or beat_index >= len(predictions):
        raise InferenceError("Beat index is outside the inference result.")
    model_info = summary.get("model") or {}
    entry, artifact_path, manifest = get_saved_model_entry(
        str(model_info.get("run_id", "")), str(model_info.get("model_name", "")),
        experiments_dir, project_root, models_root,
    )
    features_path = output_dir / "beat_features.csv"
    if not features_path.is_file():
        raise InferenceError("Stored beat feature data is unavailable for explanation.")
    feature_frame = pd.read_csv(features_path)
    if beat_index >= len(feature_frame):
        raise InferenceError("Stored beat feature row is missing.")
    feature_columns = list(manifest.get("feature_columns") or [])
    row = feature_frame.loc[beat_index, feature_columns].copy()
    training_medians = manifest.get("feature_medians") or {}
    for column in feature_columns:
        if pd.isna(row[column]) and column in training_medians:
            row[column] = float(training_medians[column])
    if row.isna().any() or not np.isfinite(pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)).all():
        raise InferenceError("The stored beat feature row contains values that cannot be reconciled with training-time medians.")
    prediction_row = predictions[beat_index]
    explanation: dict[str, Any]
    model = joblib.load(artifact_path)
    details: dict[str, Any] = {}
    explain = getattr(model, "explain", None)
    if callable(explain):
        try:
            # Keep Phase 5's flat fuzzy explanation keys for backward-compatible clients.
            details.update(_json_clean(explain(row)))
        except Exception as exc:
            raise InferenceError(f"The model rule explanation failed: {type(exc).__name__}: {exc}") from exc
    try:
        details["local_feature_ablation"] = _json_clean(explainability.local_reference_ablation(
            model,
            row,
            feature_columns=feature_columns,
            reference_values={name: float(value) for name, value in training_medians.items() if value is not None},
            target_class=str(prediction_row.get("predicted_class", "")),
            temperature=float(entry.get("score_temperature", 1.0)),
            top_k=15,
        ))
    except (ValueError, TypeError, KeyError) as exc:
        details["local_feature_ablation"] = {
            "available": False,
            "method": "single-feature reference ablation",
            "message": f"Local feature sensitivity could not be calculated: {exc}",
            "features": [],
        }
    global_summary = _json_clean(explainability.global_feature_summary(model))
    available = bool(
        callable(explain)
        or (details.get("local_feature_ablation") or {}).get("available")
    )
    explanation = {
        "available": available,
        "method": "fuzzy rule activation and/or single-feature reference ablation",
        "details": details,
        "global_explainability": global_summary,
        "limitations": [
            "The local ablation changes one input feature at a time to its training median; it is not causal and is not SHAP.",
            "Correlated features can make isolated perturbations misleading.",
            "Importance summaries describe model behaviour, not physiological causation.",
        ],
    }
    return {
        "inference_id": inference_id,
        "beat_index": beat_index,
        "predicted_class": prediction_row.get("predicted_class"),
        "class_scores": prediction_row.get("class_scores"),
        "score_interpretation": summary.get("score_interpretation"),
        "explanation": explanation,
    }


def _json_clean(value: Any) -> Any:
    """Convert nested NumPy/Pandas values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value
