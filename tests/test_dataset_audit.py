from types import SimpleNamespace
import numpy as np

import config
from modules.dataset_audit import audit_dataset, write_audit_report


def _fake_record(record_id, classes=None):
    classes = np.asarray(classes or ["N", "V", "S"])
    return SimpleNamespace(
        record_id=str(record_id), fs=360.0,
        signal=np.zeros((3600, 2), dtype=float),
        channel_names=["MLII", "V1"],
        ann_samples=np.asarray([100, 500, 900], dtype=int),
        ann_symbols=np.asarray(["N", "V", "A"]),
        ann_classes=classes,
        n_excluded=2,
    )


def test_default_split_is_strictly_patient_disjoint():
    train, test = config.get_record_split("patient-disjoint")
    train_patients = {config.patient_id_for_record(x) for x in train}
    test_patients = {config.patient_id_for_record(x) for x in test}
    assert "202" not in test
    assert len(train) == 22
    assert len(test) == 21
    assert not (train_patients & test_patients)


def test_legacy_de_chazal_split_is_flagged_as_same_subject_overlap(tmp_path):
    for rid in ("201", "202"):
        for ext in ("hea", "dat", "atr"):
            (tmp_path / f"{rid}.{ext}").write_bytes(b"fixture")

    def loader(rid, root):
        return _fake_record(rid)

    report = audit_dataset(
        ["201", "202"], data_dir=tmp_path,
        train_record_ids=["201"], test_record_ids=["202"],
        split_protocol="de-chazal-record-wise", record_loader=loader, hash_files=False,
    )
    assert report["patient_disjoint"] is False
    assert report["patient_overlap"] == ["subject_201_202"]
    out = write_audit_report(report, tmp_path / "audit")
    assert (out / "dataset_audit.json").exists()
    assert (out / "record_inventory.csv").exists()
    assert (out / "class_distribution.csv").exists()


def test_audit_reports_missing_wfdb_components(tmp_path):
    report = audit_dataset(
        ["100"], data_dir=tmp_path,
        train_record_ids=["100"], test_record_ids=[],
        record_loader=lambda rid, root: (_ for _ in ()).throw(AssertionError("must not load missing record")),
        hash_files=False,
    )
    assert report["n_records_failed"] == 1
    assert report["missing_records"] == ["100"]
    assert report["records"][0]["status"] == "missing_files"


def test_audit_flags_duplicate_record_ids_in_split_lists(tmp_path):
    report = audit_dataset(
        ["100"], data_dir=tmp_path,
        train_record_ids=["100", "100"], test_record_ids=[],
        record_loader=lambda rid, root: (_ for _ in ()).throw(AssertionError("missing file should skip load")),
        hash_files=False,
    )
    assert report["duplicate_train_record_ids"] == ["100"]
