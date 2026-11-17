import hashlib
import json
from pathlib import Path
import zipfile

import pytest

import config
from modules.pipeline_contract import current_mitbih_inference_contract
from tools.package_mitbih_run import SOURCE_FILES, PackageValidationError, validate_and_package


def _create_fixture_project(root: Path, run_id: str, model_hash_override: str | None = None):
    run_dir = root / "experiments" / run_id
    model_dir = root / "models" / run_id
    data_dir = root / "data" / "mitdb"
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    train_records, test_records = config.get_record_split("patient-disjoint")
    for record in train_records + test_records:
        for ext in ("hea", "dat", "atr"):
            (data_dir / f"{record}.{ext}").write_bytes(f"{record}.{ext}".encode())
    for relative in SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source: {relative}", encoding="utf-8")

    model_path = model_dir / "fixture_model.joblib"
    model_path.write_bytes(b"fixture model bytes")
    model_hash = model_hash_override or hashlib.sha256(model_path.read_bytes()).hexdigest()
    detection_rows = [
        {"record": record, "sensitivity": 0.9, "ppv": 0.99}
        for record in train_records + test_records
    ]
    (run_dir / "rpeak_detection_metrics.json").write_text(
        json.dumps({"per_record": detection_rows}), encoding="utf-8"
    )
    metadata = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "split_policy": "patient-disjoint",
        "patient_disjoint": True,
        "peak_source": "detected",
        "train_record_ids": train_records,
        "test_record_ids": test_records,
        "inference_pipeline_contract": current_mitbih_inference_contract(),
    }
    manifest = {
        "run_id": run_id,
        "feature_schema_version": current_mitbih_inference_contract()["feature_schema_version"],
        "metadata": metadata,
        "train_records": train_records,
        "test_records": test_records,
        "metrics": {
            "Fixture model": {
                "model_artifact": f"models/{run_id}/{model_path.name}",
                "model_artifact_sha256": model_hash,
                "accuracy": 0.9,
                "macro_f1": 0.7,
                "balanced_accuracy": 0.75,
            }
        },
        "n_train": 100,
        "n_test": 50,
        "train_class_distribution": {"N": 80, "S": 5, "V": 10, "F": 3, "Q": 2},
        "test_class_distribution": {"N": 40, "S": 2, "V": 5, "F": 2, "Q": 1},
        "missing_train_classes": [],
        "missing_test_classes": [],
        "warnings": [],
        "configuration": {},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "model_comparison.csv").write_text(
        "model,macro_f1\nFixture model,0.7\n", encoding="utf-8"
    )
    return run_dir, model_dir, train_records, test_records


def test_package_mitbih_run_requires_exact_split_contract_and_checksums(tmp_path):
    root = tmp_path / "project"
    run_id = "fixture_patient_disjoint_run"
    _, _, train_records, test_records = _create_fixture_project(root, run_id)
    archive = validate_and_package(root, run_id, tmp_path / "dist")

    assert archive.is_file()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        assert f"PulseLab_MITBIH_{run_id}/training_environment.json" in names
        assert f"PulseLab_MITBIH_{run_id}/models/{run_id}/fixture_model.joblib" in names
        env_name = f"PulseLab_MITBIH_{run_id}/training_environment.json"
        environment = json.loads(zf.read(env_name))
        assert len(environment["dataset"]["record_file_sha256"]) == (
            len(train_records) + len(test_records)
        ) * 3
        assert environment["dataset"]["training_patient_groups"]
        assert environment["dataset"]["heldout_patient_groups"]
        assert environment["model_artifact_sha256"][f"models/{run_id}/fixture_model.joblib"]


def test_package_mitbih_run_rejects_tampered_model_checksum(tmp_path):
    root = tmp_path / "project"
    run_id = "fixture_bad_hash"
    _create_fixture_project(root, run_id, model_hash_override="0" * 64)

    with pytest.raises(PackageValidationError, match="SHA-256"):
        validate_and_package(root, run_id, tmp_path / "dist")


def test_package_mitbih_run_rejects_annotation_guided_manifest(tmp_path):
    root = tmp_path / "project"
    run_id = "fixture_annotation_guided"
    run_dir, _, _, _ = _create_fixture_project(root, run_id)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["peak_source"] = "annotations"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackageValidationError, match="automatic R-peak"):
        validate_and_package(root, run_id, tmp_path / "dist")
