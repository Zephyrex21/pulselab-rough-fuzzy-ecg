from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import config
import server
from modules.bidmc_identity import (
    BIDMCPatientIdentityError,
    audit_patient_identity,
    validate_patient_disjoint_manifest,
)


def _write_record(root: Path, record_number: int, patient_id: str | None):
    csv_dir = root / "bidmc_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    record = f"bidmc{record_number:02d}"
    (csv_dir / f"bidmc_{record_number:02d}_Signals.csv").write_text(
        "Time [s], II, PLETH, RESP\n0, 0.1, 0.2, 0.3\n", encoding="utf-8"
    )
    (csv_dir / f"bidmc_{record_number:02d}_Breaths.csv").write_text(
        "breath_idx,ann1,ann2\n0,0,0\n", encoding="utf-8"
    )
    if patient_id is not None:
        (root / f"{record}.hea").write_text(
            f"{record} 5 125 2\n#<source>: https://physionet.org/physiobank/database/matched/{patient_id}/\n",
            encoding="utf-8",
        )


def test_identity_audit_uses_verified_ids_and_groups_repeated_subjects(tmp_path):
    _write_record(tmp_path, 1, "s01001")
    _write_record(tmp_path, 2, "s01001")
    _write_record(tmp_path, 3, "s02002")

    audit = audit_patient_identity(tmp_path)

    assert audit["all_records_verified"] is True
    assert audit["records_total"] == audit["records_verified"] == 3
    assert audit["verified_patient_groups"] == 2
    assert audit["patient_record_counts"] == {"s01001": 2, "s02002": 1}
    assert audit["record_to_patient"] == {
        "bidmc01": "s01001", "bidmc02": "s01001", "bidmc03": "s02002"
    }


def test_identity_audit_fails_closed_for_missing_or_unparseable_headers(tmp_path):
    _write_record(tmp_path, 1, "s01001")
    _write_record(tmp_path, 2, None)
    _write_record(tmp_path, 3, "not-a-patient-id")

    audit = audit_patient_identity(tmp_path)

    assert audit["all_records_verified"] is False
    assert audit["records_verified"] == 1
    assert audit["records_unverified"] == 2
    assert set(audit["unverified_records"]) == {"bidmc02", "bidmc03"}
    statuses = {row["record_id"]: row["identity_status"] for row in audit["records"]}
    assert statuses["bidmc02"] == "missing_wfdb_header"
    assert statuses["bidmc03"] == "patient_id_not_found_in_header"


def test_identity_audit_rejects_ambiguous_header_patient_ids(tmp_path):
    _write_record(tmp_path, 1, "s01001")
    header = tmp_path / "bidmc01.hea"
    header.write_text(
        "bidmc01 5 125 2\n"
        "# https://physionet.org/physiobank/database/matched/s01001/\n"
        "# https://physionet.org/physiobank/database/matched/s09999/\n",
        encoding="utf-8",
    )
    audit = audit_patient_identity(tmp_path)
    assert audit["all_records_verified"] is False
    assert audit["records"][0]["identity_status"] == "ambiguous_patient_ids_in_header"


def test_saved_patient_disjoint_split_is_checked_against_verified_record_map(tmp_path):
    _write_record(tmp_path, 1, "s01001")
    _write_record(tmp_path, 2, "s01001")
    _write_record(tmp_path, 3, "s02002")
    audit = audit_patient_identity(tmp_path)
    manifest = {
        "patient_groups": 2,
        "train_patient_ids": ["s01001"],
        "test_patient_ids": ["s02002"],
        "train_records": ["bidmc01", "bidmc02"],
        "test_records": ["bidmc03"],
    }
    result = validate_patient_disjoint_manifest(audit, manifest)
    assert result["valid"] is True
    assert result["n_local_patient_groups"] == 2


def test_saved_split_validation_detects_patient_and_record_leakage(tmp_path):
    _write_record(tmp_path, 1, "s01001")
    _write_record(tmp_path, 2, "s01001")
    _write_record(tmp_path, 3, "s02002")
    audit = audit_patient_identity(tmp_path)
    manifest = {
        "patient_groups": 2,
        "train_patient_ids": ["s01001", "s02002"],
        "test_patient_ids": ["s02002"],
        "train_records": ["bidmc01", "bidmc02", "bidmc03"],
        "test_records": ["bidmc03"],
    }
    result = validate_patient_disjoint_manifest(audit, manifest)
    assert result["valid"] is False
    assert any("overlap" in issue for issue in result["issues"])


def test_bidmc_api_does_not_present_record_fallback_as_patient_identity(tmp_path, monkeypatch):
    _write_record(tmp_path, 1, "s01001")
    _write_record(tmp_path, 2, None)
    monkeypatch.setattr(config, "BIDMC_DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        payload = client.get("/api/bidmc/status").json()

    assert payload["ready"] is True
    assert payload["patient_grouping_verified"] is False
    assert payload["patient_groups"] is None
    records = {row["record_id"]: row for row in payload["records"]}
    assert records["bidmc01"]["patient_id"] == "s01001"
    assert records["bidmc01"]["patient_id_verified"] is True
    assert records["bidmc02"]["patient_id"] is None
    assert records["bidmc02"]["patient_id_verified"] is False
    assert payload["patient_identity"]["records_unverified"] == 1


def test_bidmc_feature_export_is_blocked_without_verified_patient_ids(tmp_path, monkeypatch):
    _write_record(tmp_path, 1, None)
    monkeypatch.setattr(config, "BIDMC_DATA_DIR", tmp_path)
    request = server.BIDMCFeatureExportRequest(records=["bidmc01"])

    with pytest.raises(BIDMCPatientIdentityError, match="Patient grouping is incomplete"):
        server._run_bidmc_feature_export("fixture-job", request, lambda *_: None)
