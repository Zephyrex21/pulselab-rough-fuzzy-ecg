"""Independent signal-level validation utilities for PulseLab."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd

import config
from modules import data_loader, preprocessing, rpeak_detection


def evaluate_record_rpeaks(
    record,
    *,
    tolerance_seconds: float = config.RPEAK_MATCH_TOLERANCE_SEC,
    primary_channel: int = 0,
    detector: Callable = rpeak_detection.detect_rpeaks,
) -> dict:
    """Evaluate automatic R-peak detection against AAMI-mappable beat refs.

    Counts are matched one-to-one. This is a detection metric, separate from
    beat-classification scores. References are resampled to the detector's
    canonical 360 Hz grid before comparison.
    """
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    signal = np.asarray(record.signal, dtype=float)
    if signal.ndim != 2 or not 0 <= primary_channel < signal.shape[1]:
        raise ValueError("record signal must be (samples, channels) and primary_channel must exist")
    x = signal[:, primary_channel]
    if not np.isclose(record.fs, config.FS):
        x = preprocessing.resample_to(x, record.fs, config.FS)
        reference = np.round(np.asarray(record.ann_samples) * config.FS / record.fs).astype(int)
    else:
        reference = np.asarray(record.ann_samples, dtype=int)
    filtered = preprocessing.preprocess(x, config.FS, powerline=False)["filtered"]
    detected = np.asarray(detector(filtered, config.FS), dtype=int)
    matched = rpeak_detection.match_detections(detected, reference, config.FS, tol=tolerance_seconds)
    errors_ms = np.asarray([(det - ref) * 1000.0 / config.FS for ref, det in matched["matched_pairs"]], dtype=float)
    tp, fp, fn = matched["tp"], matched["fp"], matched["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    return {
        "record": str(record.record_id),
        "patient_id": config.patient_id_for_record(str(record.record_id)),
        "sampling_frequency_hz": float(record.fs),
        "reference_beats": int(len(reference)),
        "detected_peaks": int(len(detected)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "sensitivity": float(sensitivity),
        "ppv": float(precision),
        "f1": float(f1),
        "mean_signed_error_ms": float(errors_ms.mean()) if len(errors_ms) else None,
        "mean_absolute_error_ms": float(np.abs(errors_ms).mean()) if len(errors_ms) else None,
        "max_absolute_error_ms": float(np.abs(errors_ms).max()) if len(errors_ms) else None,
        "tolerance_ms": float(tolerance_seconds * 1000.0),
    }


def aggregate_detection_metrics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    tp = int(sum(int(r["tp"]) for r in rows))
    fp = int(sum(int(r["fp"]) for r in rows))
    fn = int(sum(int(r["fn"]) for r in rows))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) else 0.0
    abs_errors = [r["mean_absolute_error_ms"] for r in rows if r.get("mean_absolute_error_ms") is not None]
    return {
        "n_records": len(rows), "reference_beats": tp + fn,
        "detected_peaks": tp + fp, "tp": tp, "fp": fp, "fn": fn,
        "micro_sensitivity": float(sensitivity), "micro_ppv": float(ppv), "micro_f1": float(f1),
        "mean_record_absolute_timing_error_ms": float(np.mean(abs_errors)) if abs_errors else None,
        "aggregation": "micro counts pooled across records; timing error is mean of per-record means",
    }


def run_rpeak_validation(
    record_ids: Iterable[str],
    *,
    data_dir: Path | str = config.DATA_DIR,
    output_dir: Path | str | None = None,
    tolerance_seconds: float = config.RPEAK_MATCH_TOLERANCE_SEC,
    progress: Callable[[float, str], None] | None = None,
    record_loader: Callable = data_loader.load_record,
) -> Path:
    ids = list(map(str, record_ids))
    if not ids:
        raise ValueError("At least one record ID is required")
    rows = []
    for index, rid in enumerate(ids):
        if progress:
            progress(index / len(ids), f"Validating R-peaks on record {rid}")
        record = record_loader(rid, Path(data_dir))
        rows.append(evaluate_record_rpeaks(record, tolerance_seconds=tolerance_seconds))
    summary = aggregate_detection_metrics(rows)
    output = Path(output_dir) if output_dir is not None else config.EXPERIMENTS_DIR / (
        "rpeak_validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    )
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "rpeak_metrics_per_record.csv", index=False)
    payload = {
        "project": "PulseLab", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MIT-BIH Arrhythmia Database", "record_ids": ids,
        "tolerance_seconds": tolerance_seconds, "aggregate": summary,
        "per_record": rows,
        "interpretation": "R-peak detection performance only; not arrhythmia classification performance.",
        "reference_definition": "AAMI-mappable beat annotations loaded from the WFDB atr annotator.",
        "disclaimer": config.DISCLAIMER,
    }
    (output / "rpeak_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if progress:
        progress(1.0, f"R-peak validation saved to {output}")
    return output
