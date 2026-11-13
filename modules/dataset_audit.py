"""Dataset integrity, annotation coverage, class-balance and provenance audit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import config
from modules import data_loader


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expected_files(record_id: str, data_dir: Path) -> dict[str, Path]:
    base = data_dir / str(record_id)
    return {suffix: base.with_suffix(f".{suffix}") for suffix in ("hea", "dat", "atr")}


def audit_dataset(
    record_ids: Iterable[str] | None = None,
    *,
    data_dir: Path | str = config.DATA_DIR,
    train_record_ids: Iterable[str] | None = None,
    test_record_ids: Iterable[str] | None = None,
    split_protocol: str = "patient-disjoint",
    record_loader: Callable = data_loader.load_record,
    hash_files: bool = True,
) -> dict:
    """Audit local WFDB files and labels without silently skipping bad records.

    The default split is strictly patient-disjoint. The classic de Chazal
    22/22 record list remains available for historical comparability, but
    records 201 and 202 are a same-subject pair and therefore overlap at the
    subject level when placed on opposite sides.
    """
    root = Path(data_dir)
    if split_protocol not in config.SPLIT_PROTOCOLS:
        raise ValueError(f"Unknown split protocol {split_protocol!r}; choose from {config.SPLIT_PROTOCOLS}")
    if train_record_ids is None or test_record_ids is None:
        split_train, split_test = config.get_record_split(split_protocol)
        train_ids = list(split_train if train_record_ids is None else train_record_ids)
        test_ids = list(split_test if test_record_ids is None else test_record_ids)
    else:
        train_ids, test_ids = list(map(str, train_record_ids)), list(map(str, test_record_ids))
    if record_ids is None:
        requested = list(dict.fromkeys(train_ids + test_ids))
    else:
        requested = list(dict.fromkeys(map(str, record_ids)))
    if not requested:
        raise ValueError("At least one record ID is required for the dataset audit")

    record_rows: list[dict] = []
    class_counts = {"train": {c: 0 for c in config.CLASSES}, "test": {c: 0 for c in config.CLASSES}, "other": {c: 0 for c in config.CLASSES}}
    warnings: list[str] = []
    missing_records: list[str] = []
    file_hash_rows: list[dict] = []

    train_set, test_set = set(train_ids), set(test_ids)
    for rid in requested:
        files = _expected_files(rid, root)
        missing_parts = [ext for ext, path in files.items() if not path.is_file()]
        split = "train" if rid in train_set else "test" if rid in test_set else "other"
        row = {
            "record": rid,
            "patient_id": config.patient_id_for_record(rid),
            "split": split,
            "status": "missing_files" if missing_parts else "pending",
            "missing_files": ",".join(missing_parts),
            "sampling_frequency_hz": None,
            "n_samples": None,
            "duration_seconds": None,
            "n_channels": None,
            "channel_names": None,
            "n_aami_beats": 0,
            "n_excluded_annotations": None,
            "annotations_sorted": None,
            "annotation_duplicates": None,
            **{f"class_{c}": 0 for c in config.CLASSES},
        }
        if missing_parts:
            missing_records.append(rid)
            record_rows.append(row)
            continue

        try:
            record = record_loader(rid, root)
            counts = {c: int(np.sum(record.ann_classes == c)) for c in config.CLASSES}
            row.update({
                "status": "ok",
                "sampling_frequency_hz": float(record.fs),
                "n_samples": int(record.signal.shape[0]),
                "duration_seconds": float(record.signal.shape[0] / record.fs),
                "n_channels": int(record.signal.shape[1]),
                "channel_names": "|".join(map(str, record.channel_names)),
                "n_aami_beats": int(len(record.ann_samples)),
                "n_excluded_annotations": int(record.n_excluded),
                "annotations_sorted": bool(np.all(np.diff(record.ann_samples) >= 0)),
                "annotation_duplicates": int(len(record.ann_samples) - len(np.unique(record.ann_samples))),
                **{f"class_{c}": counts[c] for c in config.CLASSES},
            })
            for c, count in counts.items():
                class_counts[split][c] += count
            if not np.isclose(record.fs, config.FS):
                warnings.append(f"Record {rid} has {record.fs:g} Hz; pipeline resamples to {config.FS} Hz.")
            if record.signal.shape[1] < 2:
                warnings.append(f"Record {rid} has fewer than two channels; the configured 47-feature schema expects two.")
            if not row["annotations_sorted"]:
                warnings.append(f"Record {rid} annotations are not sorted.")
            if row["annotation_duplicates"]:
                warnings.append(f"Record {rid} contains duplicate AAMI annotation sample positions.")
            if hash_files:
                for ext, path in files.items():
                    file_hash_rows.append({
                        "record": rid,
                        "file": path.name,
                        "extension": ext,
                        "size_bytes": int(path.stat().st_size),
                        "sha256": _sha256(path),
                    })
        except Exception as exc:
            row["status"] = "read_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            warnings.append(f"Record {rid} could not be read: {type(exc).__name__}: {exc}")
        record_rows.append(row)

    record_overlap = sorted(train_set & test_set)
    duplicate_train_ids = sorted({rid for rid in train_ids if train_ids.count(rid) > 1})
    duplicate_test_ids = sorted({rid for rid in test_ids if test_ids.count(rid) > 1})
    if duplicate_train_ids:
        warnings.append(f"Duplicate record IDs in training list: {duplicate_train_ids}")
    if duplicate_test_ids:
        warnings.append(f"Duplicate record IDs in testing list: {duplicate_test_ids}")
    train_patients = {config.patient_id_for_record(r) for r in train_ids}
    test_patients = {config.patient_id_for_record(r) for r in test_ids}
    patient_overlap = sorted(train_patients & test_patients)
    if record_overlap:
        warnings.append(f"Train/test record overlap: {record_overlap}")
    if patient_overlap:
        warnings.append(f"Train/test patient overlap: {patient_overlap}")
    for split in ("train", "test"):
        absent = [c for c, n in class_counts[split].items() if n == 0]
        if absent:
            warnings.append(f"{split.title()} split has no AAMI examples for classes: {absent}")

    ok_records = sum(row["status"] == "ok" for row in record_rows)
    failed_records = len(record_rows) - ok_records
    report = {
        "project": "PulseLab",
        "audit_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MIT-BIH Arrhythmia Database",
        "dataset_version": "1.0.0",
        "dataset_source": "PhysioNet",
        "data_directory": str(root.resolve()),
        "split_protocol": split_protocol,
        "split_description": (
            "Strict patient-disjoint split; record 202 is excluded because it shares a subject with record 201."
            if split_protocol == "patient-disjoint"
            else ("Historical de Chazal record-wise list. The full 22/22 protocol places records 201 and 202 from one subject across splits; a truncated subset may not include both."
                  if patient_overlap else "Historical de Chazal record-wise list; the specific audited subset contains no patient overlap.")
        ),
        "requested_records": requested,
        "train_records": train_ids,
        "test_records": test_ids,
        "n_records_requested": len(requested),
        "n_records_ok": ok_records,
        "n_records_failed": failed_records,
        "missing_records": missing_records,
        "record_overlap": record_overlap,
        "duplicate_train_record_ids": duplicate_train_ids,
        "duplicate_test_record_ids": duplicate_test_ids,
        "patient_overlap": patient_overlap,
        "patient_disjoint": not patient_overlap and not record_overlap,
        "class_counts": class_counts,
        "warnings": warnings,
        "records": record_rows,
        "file_hashes": file_hash_rows,
        "disclaimer": config.DISCLAIMER,
    }
    return report


def write_audit_report(report: dict, output_dir: Path | str) -> Path:
    """Write the audit summary, per-record inventory, class distribution and hashes."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report.get("records", [])).to_csv(output / "record_inventory.csv", index=False)
    pd.DataFrame(report.get("file_hashes", [])).to_csv(output / "file_sha256_manifest.csv", index=False)
    rows = []
    for split, counts in report.get("class_counts", {}).items():
        for label, count in counts.items():
            rows.append({"split": split, "class": label, "beat_count": int(count)})
    pd.DataFrame(rows).to_csv(output / "class_distribution.csv", index=False)
    return output
