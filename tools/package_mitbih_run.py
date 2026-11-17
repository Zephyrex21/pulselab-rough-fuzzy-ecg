"""Validate and package a freshly trained, patient-disjoint MIT-BIH run.

The package contains run models/reports and reproducibility metadata. The
WFDB dataset itself is not redistributed; SHA-256 hashes identify the exact
43 selected source records used during this run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import config
from modules.pipeline_contract import check_mitbih_inference_compatibility

SOURCE_FILES = [
    "VERSION", "config.py", "train.py",
    "modules/data_loader.py", "modules/pipeline.py",
    "modules/pipeline_contract.py", "modules/rpeak_detection.py",
    "modules/preprocessing.py", "modules/segmentation.py",
    "modules/feature_extraction.py", "modules/feature_schema.py",
    "modules/hybrid_model.py", "modules/rough_set.py",
    "modules/fuzzy_logic.py", "modules/baseline_models.py",
    "modules/metrics.py", "modules/calibration.py",
    "modules/explainability.py", "modules/validation.py",
]
DISTRIBUTION_NAMES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "scikit-learn": "scikit-learn", "wfdb": "wfdb",
    "PyWavelets": "PyWavelets", "joblib": "joblib",
    "threadpoolctl": "threadpoolctl",
}


class PackageValidationError(RuntimeError):
    """Raised when a run is incomplete, incompatible or not reproducible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackageValidationError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageValidationError(f"{path.name} must contain a JSON object.")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_and_package(project_root: Path, run_id: str, output_dir: Path) -> Path:
    root = project_root.resolve()
    run_dir = (root / "experiments" / run_id).resolve()
    model_dir = (root / "models" / run_id).resolve()
    data_dir = (root / "data" / "mitdb").resolve()

    if not run_dir.is_dir() or not model_dir.is_dir():
        raise PackageValidationError(
            f"Run outputs are incomplete. Expected {run_dir} and {model_dir}."
        )
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise PackageValidationError("The run has no run_manifest.json.")
    manifest = _read_json(manifest_path)
    metadata = manifest.get("metadata") or {}

    if manifest.get("run_id") != run_id:
        raise PackageValidationError("The run ID in the manifest does not match the requested run.")
    if metadata.get("dataset") != "MIT-BIH Arrhythmia Database":
        raise PackageValidationError("The run manifest does not identify the MIT-BIH Arrhythmia Database.")
    if metadata.get("split_policy") != "patient-disjoint" or metadata.get("patient_disjoint") is not True:
        raise PackageValidationError("Only strict patient-disjoint runs can be packaged.")
    if metadata.get("peak_source") != "detected":
        raise PackageValidationError("Only automatic R-peak detection runs can be packaged for uploaded-signal inference.")
    compatible, reason = check_mitbih_inference_compatibility(manifest)
    if not compatible:
        raise PackageValidationError(f"Model/input pipeline compatibility check failed: {reason}")

    train_records, test_records = config.get_record_split("patient-disjoint")
    train_records, test_records = list(map(str, train_records)), list(map(str, test_records))
    expected_records = set(train_records + test_records)
    if set(train_records) & set(test_records):
        raise PackageValidationError("Configured train/test records overlap.")
    recorded_train = set(map(str, metadata.get("train_record_ids") or []))
    recorded_test = set(map(str, metadata.get("test_record_ids") or []))
    if recorded_train != set(train_records) or recorded_test != set(test_records):
        raise PackageValidationError("The run manifest's record split does not match the current patient-disjoint configuration.")
    if set(map(str, manifest.get("train_records") or [])) != set(train_records):
        raise PackageValidationError("Top-level training records in the run manifest are incomplete or unexpected.")
    if set(map(str, manifest.get("test_records") or [])) != set(test_records):
        raise PackageValidationError("Top-level held-out records in the run manifest are incomplete or unexpected.")
    train_patients = {config.patient_id_for_record(record) for record in train_records}
    test_patients = {config.patient_id_for_record(record) for record in test_records}
    if train_patients & test_patients:
        raise PackageValidationError("The current patient mapping has train/test patient overlap.")

    missing_records = []
    data_hashes: dict[str, str] = {}
    for record_id in sorted(expected_records, key=int):
        for extension in ("hea", "dat", "atr"):
            relative = f"{record_id}.{extension}"
            path = data_dir / relative
            if not path.is_file():
                missing_records.append(relative)
            else:
                data_hashes[f"data/mitdb/{relative}"] = sha256_file(path)
    if missing_records:
        raise PackageValidationError(
            "The selected MIT-BIH dataset is incomplete: " + ", ".join(missing_records[:20])
        )

    source_hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise PackageValidationError(f"Required training source file is missing: {relative}")
        source_hashes[relative] = sha256_file(path)

    metrics = manifest.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        raise PackageValidationError("No model metrics were saved.")
    model_hashes: dict[str, str] = {}
    for model_name, model_metrics in metrics.items():
        if not isinstance(model_metrics, dict):
            raise PackageValidationError(f"Metrics for {model_name} are malformed.")
        relative = model_metrics.get("model_artifact")
        expected_hash = model_metrics.get("model_artifact_sha256")
        if not relative or not isinstance(expected_hash, str):
            raise PackageValidationError(f"{model_name} is missing its saved model or SHA-256 checksum.")
        artifact_path = Path(relative)
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        artifact_path = artifact_path.resolve()
        if not _inside(artifact_path, model_dir) or not artifact_path.is_file():
            raise PackageValidationError(f"{model_name} points to a missing or out-of-run model artifact.")
        actual_hash = sha256_file(artifact_path)
        if actual_hash.lower() != expected_hash.lower():
            raise PackageValidationError(f"{model_name} model SHA-256 does not match its run manifest.")
        model_hashes[str(artifact_path.relative_to(root))] = actual_hash

    detector_report = run_dir / "rpeak_detection_metrics.json"
    if not detector_report.is_file():
        raise PackageValidationError("Automatic R-peak detection metrics were not saved.")
    detection = _read_json(detector_report)
    detector_records = {
        str(row.get("record") or row.get("record_id"))
        for row in detection.get("per_record", [])
        if row.get("record") is not None or row.get("record_id") is not None
    }
    if detector_records and detector_records != expected_records:
        raise PackageValidationError("R-peak diagnostics do not cover exactly the 43 selected records.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = f"PulseLab_MITBIH_{run_id}"
    staging = output_dir / package_name
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "experiments").mkdir(parents=True)
    (staging / "models").mkdir(parents=True)
    shutil.copytree(run_dir, staging / "experiments" / run_id)
    shutil.copytree(model_dir, staging / "models" / run_id)

    package_artifact_hashes = {}
    for folder in ("experiments", "models"):
        for path in sorted((staging / folder).rglob("*")):
            if path.is_file():
                package_artifact_hashes[str(path.relative_to(staging)).replace("\\", "/")] = sha256_file(path)

    packages = {}
    for display_name, distribution_name in DISTRIBUTION_NAMES.items():
        try:
            packages[display_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            packages[display_name] = None

    environment = {
        "project": "PulseLab",
        "run_id": run_id,
        "project_version": (root / "VERSION").read_text(encoding="utf-8").strip(),
        "source_commit": _git_commit(root),
        "python_version": platform.python_version(),
        "package_versions": packages,
        "dataset": {
            "name": "MIT-BIH Arrhythmia Database",
            "source": "PhysioNet WFDB",
            "selected_records": len(expected_records),
            "training_records": train_records,
            "heldout_records": test_records,
            "training_patient_groups": sorted(train_patients),
            "heldout_patient_groups": sorted(test_patients),
            "record_file_sha256": data_hashes,
            "note": "The dataset files are not redistributed in this model bundle. Hashes identify the exact local WFDB records used for this run.",
        },
        "inference_pipeline_contract": metadata.get("inference_pipeline_contract"),
        "source_sha256": source_hashes,
        "model_artifact_sha256": model_hashes,
        "bundle_artifact_sha256": package_artifact_hashes,
        "evaluation": {
            "n_train_beats": manifest.get("n_train"),
            "n_heldout_beats": manifest.get("n_test"),
            "training_class_distribution": manifest.get("train_class_distribution"),
            "heldout_class_distribution": manifest.get("test_class_distribution"),
            "missing_training_classes": manifest.get("missing_train_classes"),
            "missing_heldout_classes": manifest.get("missing_test_classes"),
            "warnings": manifest.get("warnings", []),
            "evaluation_note": "Classifier metrics are conditional on automatically detected beats matched to reference annotations. R-peak sensitivity/PPV and false positives/negatives are reported separately. This is not external or clinical validation.",
        },
    }
    (staging / "training_environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )

    readme = f"""# PulseLab MIT-BIH trained model bundle

- Run ID: {run_id}
- Project version: {environment['project_version']}
- Source commit: {environment['source_commit']}
- Split: patient-disjoint, 22 training records / 21 held-out records
- Peak source: automatically detected R-peaks
- Models: {', '.join(metrics.keys())}

## Install

Extract this ZIP at the PulseLab repository root. It restores the matching
experiments/{run_id}/ and models/{run_id}/ directories. Keep the source
repository version that matches source_commit and inference_pipeline_contract.

## Integrity

The run manifest contains per-model SHA-256 checksums. training_environment.json
contains source, dataset-record and packaged-artifact checksums plus the exact
Python/package versions. PulseLab verifies each model checksum and rejects
incompatible input-pipeline contracts before uploaded-signal inference.

## Interpretation

Automatic R-peak detection is evaluated separately. Classification metrics are
conditional on detected beats that were matched to reference annotations.
Inspect rpeak_detection_metrics.json and per-record diagnostics alongside
macro-F1, balanced accuracy, per-class recall and confusion matrices.

This is an offline academic research model, not clinically validated and not
for medical decision-making.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")

    archive = output_dir / f"{package_name}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, Path(package_name) / path.relative_to(staging))
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise PackageValidationError(f"Generated ZIP failed integrity check at {bad}.")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        archive = validate_and_package(
            args.project_root,
            args.run_id,
            args.output_dir or (args.project_root / "dist"),
        )
    except PackageValidationError as exc:
        parser.error(str(exc))
    print(f"Verified MIT-BIH run bundle: {archive}")
    print(f"SHA-256: {sha256_file(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
