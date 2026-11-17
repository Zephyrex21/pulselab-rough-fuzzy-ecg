"""Strict BIDMC patient-identity auditing without changing model-critical features.

The BIDMC CSV files do not reliably carry a patient identifier. A matching
WFDB header supplies the MIMIC-II subject ID. Record-specific fallbacks are
useful for previews but must never be presented as verified patient groups
for patient-disjoint training or evaluation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from modules import bidmc_pipeline

_PATIENT_ID_RE = re.compile(r"(?:matched/)?(s\\d{4,8})/", flags=re.IGNORECASE)


class BIDMCPatientIdentityError(ValueError):
    """Raised when a patient-level operation lacks verified subject IDs."""


def _header_patient_id(header_path: Path) -> tuple[str | None, str]:
    if not header_path.is_file():
        return None, "missing_wfdb_header"
    try:
        content = header_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, "unreadable_wfdb_header"
    matches = {value.lower() for value in _PATIENT_ID_RE.findall(content)}
    if len(matches) == 1:
        return next(iter(matches)), "verified"
    if not matches:
        return None, "patient_id_not_found_in_header"
    return None, "ambiguous_patient_ids_in_header"


def audit_patient_identity(dataset_root: str | Path) -> dict[str, Any]:
    """Return verified patient IDs for indexed recordings; never invent an ID."""
    inventory = bidmc_pipeline.inventory(dataset_root)
    records = inventory.get("records") or []
    if not records:
        return {
            "status": "dataset_unavailable",
            "all_records_verified": False,
            "records_total": 0,
            "records_verified": 0,
            "records_unverified": 0,
            "verified_patient_groups": 0,
            "unverified_records": [],
            "records": [],
            "record_to_patient": {},
            "patient_record_counts": {},
            "message": inventory.get("message", "BIDMC records are unavailable."),
        }

    csv_dir_value = inventory.get("csv_directory")
    if not csv_dir_value:
        raise BIDMCPatientIdentityError("The BIDMC inventory did not provide a CSV directory.")
    csv_dir = Path(csv_dir_value)
    header_root = csv_dir.parent
    rows: list[dict[str, Any]] = []
    record_to_patient: dict[str, str] = {}
    patient_record_counts: dict[str, int] = {}

    for record in records:
        record_id = str(record.get("record_id", ""))
        match = re.fullmatch(r"bidmc(\\d{2})", record_id, flags=re.IGNORECASE)
        if not match:
            patient_id, status = None, "invalid_record_id"
        else:
            header_path = header_root / f"bidmc{match.group(1)}.hea"
            patient_id, status = _header_patient_id(header_path)
        verified = status == "verified" and patient_id is not None
        if verified:
            record_to_patient[record_id] = patient_id
            patient_record_counts[patient_id] = patient_record_counts.get(patient_id, 0) + 1
        rows.append({
            "record_id": record_id,
            "patient_id": patient_id,
            "patient_id_verified": bool(verified),
            "identity_status": status,
        })

    verified_count = len(record_to_patient)
    unverified = [row for row in rows if not row["patient_id_verified"]]
    all_verified = bool(rows) and not unverified
    return {
        "status": "verified" if all_verified else "incomplete",
        "all_records_verified": all_verified,
        "records_total": len(rows),
        "records_verified": verified_count,
        "records_unverified": len(unverified),
        "verified_patient_groups": len(patient_record_counts),
        "unverified_records": [row["record_id"] for row in unverified],
        "records": rows,
        "record_to_patient": record_to_patient,
        "patient_record_counts": patient_record_counts,
        "message": (
            f"Verified {verified_count}/{len(rows)} recording-to-patient mappings across "
            f"{len(patient_record_counts)} MIMIC-II patient groups."
            if all_verified else
            f"Patient grouping is incomplete: {verified_count}/{len(rows)} recording IDs were verified. "
            "Patient-disjoint feature exports are disabled until every indexed recording has a valid WFDB patient ID."
        ),
    }


def validate_patient_disjoint_manifest(identity_audit: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Check a saved BIDMC split against verified local record-to-patient IDs."""
    issues: list[str] = []
    if not identity_audit.get("all_records_verified"):
        issues.append("The local BIDMC dataset does not have verified patient IDs for every indexed record.")

    record_map = identity_audit.get("record_to_patient") or {}
    actual_patients = set(record_map.values())
    train_patients = set(map(str, manifest.get("train_patient_ids") or []))
    test_patients = set(map(str, manifest.get("test_patient_ids") or []))
    train_records = set(map(str, manifest.get("train_records") or []))
    test_records = set(map(str, manifest.get("test_records") or []))

    if train_patients & test_patients:
        issues.append("The saved training and held-out patient lists overlap.")
    if train_patients | test_patients != actual_patients:
        issues.append("The saved patient lists do not cover exactly the verified local patient IDs.")
    if train_records & test_records:
        issues.append("The saved training and held-out record lists overlap.")
    if train_records | test_records != set(record_map):
        issues.append("The saved record lists do not cover exactly the verified local recordings.")

    for record_id, patient_id in record_map.items():
        expected_split = "train" if patient_id in train_patients else "test" if patient_id in test_patients else None
        actual_split = "train" if record_id in train_records else "test" if record_id in test_records else None
        if expected_split is None or expected_split != actual_split:
            issues.append(
                f"{record_id} ({patient_id}) is not consistently assigned to the split containing its patient."
            )

    expected_groups = manifest.get("patient_groups")
    if expected_groups is not None and int(expected_groups) != len(actual_patients):
        issues.append(
            f"The manifest reports {expected_groups} patient groups, but the local dataset has {len(actual_patients)} verified groups."
        )

    return {
        "status": "verified" if not issues else "failed",
        "valid": not issues,
        "n_local_records": len(record_map),
        "n_local_patient_groups": len(actual_patients),
        "n_train_patients": len(train_patients),
        "n_test_patients": len(test_patients),
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "issues": issues,
    }
