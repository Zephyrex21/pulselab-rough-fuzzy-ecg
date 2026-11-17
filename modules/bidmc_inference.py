"""Loading and inference helpers for the trained BIDMC Rough-Fuzzy models.

The shipped estimators were trained in Kaggle on BIDMC v1.0.0 using a
patient-disjoint 36/10 split for evaluation, then refitted on all labelled
windows for later inference. This module verifies the training-source hashes
before loading the trusted local Joblib artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from modules import bidmc_pipeline
from modules.bidmc_model import RoughFuzzyRespiratoryRegressor

BIDMC_RUN_ID = "pulselab_bidmc_patient_disjoint_20261117T144120Z_2489"
MODEL_FILES = {
    "ecg_ppg": "bidmc_rough_fuzzy_ecg_ppg_rr_mean.joblib",
    "ppg_only": "bidmc_rough_fuzzy_ppg_only_rr_mean.joblib",
}
MODEL_LABELS = {
    "ecg_ppg": "Rough-Set + Fuzzy (ECG + PPG)",
    "ppg_only": "Rough-Set + Fuzzy (PPG only)",
}
PREDICTOR_MODALITIES = {
    "ecg_ppg": ["ECG", "PPG"],
    "ppg_only": ["PPG"],
}
ARTIFACT_FILES = {
    "model_comparison.csv",
    "per_patient_metrics.csv",
    "research_report.md",
    "run_manifest.json",
    "annotation_agreement.json",
    "test_predictions.csv",
}


class BIDMCArtifactError(RuntimeError):
    """Raised when trained BIDMC artifacts are missing or incompatible."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BIDMCArtifactError(f"Could not read BIDMC artifact metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise BIDMCArtifactError(f"BIDMC artifact metadata is not an object: {path.name}")
    return value


def load_trained_registry(project_root: str | Path, models_root: str | Path, experiments_root: str | Path) -> dict[str, Any]:
    """Verify source provenance and load both final trained regressors."""
    project_root = Path(project_root).resolve()
    models_root = Path(models_root).resolve()
    experiments_root = Path(experiments_root).resolve()
    model_dir = models_root / BIDMC_RUN_ID
    experiment_dir = experiments_root / BIDMC_RUN_ID
    env_path = project_root / "pulselab_bidmc_training_environment.json"
    manifest_path = experiment_dir / "run_manifest.json"
    comparison_path = experiment_dir / "model_comparison.csv"
    agreement_path = experiment_dir / "annotation_agreement.json"

    required = [env_path, manifest_path, comparison_path, agreement_path]
    required.extend(model_dir / name for name in MODEL_FILES.values())
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise BIDMCArtifactError(
            "Trained BIDMC artifacts are incomplete. Missing: " + ", ".join(missing)
        )

    environment = _read_json(env_path)
    manifest = _read_json(manifest_path)
    if environment.get("run_id") != BIDMC_RUN_ID or manifest.get("run_id") != BIDMC_RUN_ID:
        raise BIDMCArtifactError("BIDMC artifact run ID does not match the configured trained model run.")
    if str(environment.get("project_version")) != str(manifest.get("project_version")):
        raise BIDMCArtifactError("BIDMC training environment and run manifest report different project versions.")

    # Verify only the source files the Kaggle manifest declares as model provenance.
    source_hashes = environment.get("source_sha256") or manifest.get("source_sha256") or {}
    checked_hashes: dict[str, bool] = {}
    for relative, expected in source_hashes.items():
        source_path = project_root / relative
        if not source_path.is_file():
            checked_hashes[relative] = False
            raise BIDMCArtifactError(f"Required BIDMC model source file is missing: {relative}")
        actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
        checked_hashes[relative] = actual == expected
        if actual != expected:
            raise BIDMCArtifactError(
                f"BIDMC model source compatibility check failed for {relative}; "
                "the trained artifact was produced from a different source revision."
            )

    models: dict[str, RoughFuzzyRespiratoryRegressor] = {}
    for key, filename in MODEL_FILES.items():
        path = model_dir / filename
        try:
            model = joblib.load(path)
        except Exception as exc:
            raise BIDMCArtifactError(f"Could not load trained BIDMC model {filename}: {exc}") from exc
        if not isinstance(model, RoughFuzzyRespiratoryRegressor):
            raise BIDMCArtifactError(f"Unexpected Python model type in {filename}.")
        feature_names = list(getattr(model, "feature_names_in_", []))
        if not feature_names or any(str(name).lower().startswith("resp_") for name in feature_names):
            raise BIDMCArtifactError(f"Model {filename} has an invalid or RESP-leaking feature schema.")
        if key == "ppg_only" and any(not str(name).startswith("ppg_") for name in feature_names):
            raise BIDMCArtifactError("PPG-only model contains non-PPG predictor features.")
        if key == "ecg_ppg" and any(not (str(name).startswith("ecg_") or str(name).startswith("ppg_")) for name in feature_names):
            raise BIDMCArtifactError("ECG+PPG model contains an unexpected predictor feature.")
        models[key] = model

    comparison = pd.read_csv(comparison_path)
    mean_rows = comparison.loc[comparison["target"].astype(str) == "annotator_mean"].copy()
    agreement = _read_json(agreement_path)
    labelled_windows = int(manifest.get("labelled_windows_mean", 0))
    heldout_windows = int(mean_rows["n_test_windows"].max()) if not mean_rows.empty and "n_test_windows" in mean_rows else 0
    return {
        "available": True,
        "run_id": BIDMC_RUN_ID,
        "project_version": str(environment.get("project_version", "unknown")),
        "dataset": manifest.get("dataset", "BIDMC PPG and Respiration Dataset"),
        "dataset_version": manifest.get("dataset_version", "1.0.0"),
        "dataset_records": int(manifest.get("dataset_records", 0)),
        "patient_groups": int(manifest.get("patient_groups", 0)),
        "total_windows_full": int(manifest.get("total_windows_full", 0)),
        "labelled_windows_mean": int(manifest.get("labelled_windows_mean", 0)),
        "window_sec": float(manifest.get("window_sec", 32.0)),
        "stride_sec": float(manifest.get("stride_sec", 16.0)),
        "sampling_frequency_hz": float(manifest.get("sampling_frequency_hz", 125.0)),
        "target": manifest.get("target", "ref_rr_mean_bpm"),
        "target_definition": manifest.get("target_definition", "Mean of manual ann1/ann2 breath-interval rates."),
        "predictor_policy": manifest.get("predictor_policy", "PPG-only or ECG+PPG; RESP-derived predictors excluded."),
        "n_train_patients": len(manifest.get("train_patient_ids", [])),
        "n_test_patients": len(manifest.get("test_patient_ids", [])),
        "n_train_windows": max(0, labelled_windows - heldout_windows),
        "n_test_windows": heldout_windows,
        "models": {
            key: {
                "label": MODEL_LABELS[key],
                "predictor_modalities": PREDICTOR_MODALITIES[key],
                "artifact": MODEL_FILES[key],
                "input_feature_count": len(model.feature_names_in_),
                "selected_feature_count": len(model.feature_names_selected_),
                "fuzzy_rule_count": len(getattr(model.fuzzy_, "rules_", [])),
                "target_bin_centres_bpm": [float(x) for x in model.bin_centres_],
            }
            for key, model in models.items()
        },
        "heldout_comparison": [
            {str(k): (None if pd.isna(v) else (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v))) for k, v in row.items()}
            for row in mean_rows.to_dict(orient="records")
        ],
        "annotation_agreement": agreement,
        "evaluation_note": manifest.get("evaluation_note", "Held-out patient results are research estimates, not clinical validation."),
        "clinical_status": manifest.get("clinical_status", "Academic research prototype; not clinically validated."),
        "source_hashes_verified": all(checked_hashes.values()) if checked_hashes else False,
        "training_python_version": environment.get("python_version"),
        "training_package_versions": environment.get("package_versions", {}),
        "models_loaded": models,
        "artifact_dir": str(model_dir),
        "experiment_dir": str(experiment_dir),
    }


def predict_record(record: bidmc_pipeline.BIDMCRecord, models: dict[str, RoughFuzzyRespiratoryRegressor], model_key: str = "ecg_ppg") -> dict[str, Any]:
    """Run a saved estimator on real synchronized windows from one BIDMC record."""
    if model_key not in MODEL_FILES:
        raise BIDMCArtifactError("Unknown BIDMC model. Choose 'ecg_ppg' or 'ppg_only'.")
    if model_key not in models:
        raise BIDMCArtifactError(f"Trained BIDMC model '{model_key}' is not loaded.")

    features = bidmc_pipeline.extract_record_features(
        record,
        window_sec=32.0,
        stride_sec=16.0,
        include_full_fusion=False,
    )
    if features.empty:
        raise BIDMCArtifactError(f"No complete 32-second windows were available for {record.record_id}.")

    model = models[model_key]
    expected = list(model.feature_names_in_)
    missing = [name for name in expected if name not in features.columns]
    if missing:
        raise BIDMCArtifactError(f"Live feature schema is missing trained model features: {missing}")
    prediction = np.asarray(model.predict(features.loc[:, expected]), dtype=float)
    if len(prediction) != len(features) or not np.isfinite(prediction).all():
        raise BIDMCArtifactError("BIDMC model returned an invalid prediction vector.")

    rows = []
    for position, (_, row) in enumerate(features.iterrows()):
        rows.append({
            "window_index": int(position),
            "window_start_sec": float(row["window_start_sec"]),
            "window_end_sec": float(row["window_end_sec"]),
            "predicted_rr_bpm": float(prediction[position]),
            "reference_ann1_bpm": None if pd.isna(row["ref_rr_ann1_bpm"]) else float(row["ref_rr_ann1_bpm"]),
            "reference_ann2_bpm": None if pd.isna(row["ref_rr_ann2_bpm"]) else float(row["ref_rr_ann2_bpm"]),
            "reference_mean_bpm": None if pd.isna(row["ref_rr_mean_bpm"]) else float(row["ref_rr_mean_bpm"]),
            "window_quality": float(row["window_quality"]),
        })

    return {
        "run_id": BIDMC_RUN_ID,
        "record_id": record.record_id,
        "patient_id": record.patient_id,
        "model_key": model_key,
        "model_label": MODEL_LABELS[model_key],
        "predictor_modalities": PREDICTOR_MODALITIES[model_key],
        "window_sec": 32.0,
        "stride_sec": 16.0,
        "sampling_frequency_hz": float(record.fs),
        "n_windows": int(len(rows)),
        "summary": {
            "mean_predicted_rr_bpm": float(np.mean(prediction)),
            "median_predicted_rr_bpm": float(np.median(prediction)),
            "minimum_predicted_rr_bpm": float(np.min(prediction)),
            "maximum_predicted_rr_bpm": float(np.max(prediction)),
        },
        "predictions": rows,
        "interpretation_note": (
            "Retrospective record replay using the final estimator refitted on all labelled BIDMC windows. "
            "This record may have contributed to final fitting; these predictions are not an independent "
            "test result. Use the patient-disjoint held-out comparison for generalization estimates."
        ),
        "clinical_status": "Academic research prototype; not clinically validated.",
    }
