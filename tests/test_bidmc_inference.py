"""BIDMC artifact-loader tests using small deterministic test estimators.

The production Joblib models are distributed separately from the source repo.
These tests build tiny compatible artifacts so a clean checkout can exercise
provenance checks, model loading, prediction plumbing and API routes in CI.
"""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
import server
from modules import bidmc_inference
from modules.bidmc_model import RoughFuzzyRespiratoryRegressor
from modules.fuzzy_logic import FuzzyRuleClassifier
from modules.bidmc_pipeline import BIDMCRecord, extract_record_features


SOURCE_FILES = (
    "VERSION",
    "modules/bidmc_pipeline.py",
    "modules/bidmc_model.py",
    "modules/rough_set.py",
    "modules/fuzzy_logic.py",
)


def _synthetic_record(duration_sec=64.0):
    fs = 125.0
    time = np.arange(int(duration_sec * fs)) / fs
    ecg = np.sin(2 * np.pi * 1.15 * time) + 0.04 * np.sin(2 * np.pi * 17 * time)
    ppg = 0.8 * np.sin(2 * np.pi * 1.15 * time - 0.35) + 0.05 * np.sin(2 * np.pi * 0.25 * time)
    resp = np.sin(2 * np.pi * 0.24 * time)
    breaths = np.arange(0, int(duration_sec * fs), int(fs / 0.24))
    return BIDMCRecord(
        record_id="bidmc01", patient_id="s01182", fs=fs, time_sec=time,
        ecg=ecg, ppg=ppg, resp=resp, ann1_samples=breaths,
        ann2_samples=breaths + 3, monitor=None, channels=["II", "PLETH", "RESP"],
    )


def _make_test_model(feature_names: list[str], selected_count: int):
    """Create a tiny fitted-compatible model without running model training."""
    model = RoughFuzzyRespiratoryRegressor.__new__(RoughFuzzyRespiratoryRegressor)
    selected_names = feature_names[:selected_count]
    model.feature_names_in_ = list(feature_names)
    model.feature_names_selected_ = list(selected_names)
    model.bin_centres_ = np.asarray([15.0, 24.0], dtype=float)

    fuzzy = FuzzyRuleClassifier()
    fuzzy.feature_names_ = list(selected_names)
    fuzzy.medians_ = np.zeros(len(selected_names), dtype=float)
    fuzzy.quantiles_ = np.tile(np.asarray([-0.5, 0.0, 0.5]), (len(selected_names), 1))
    fuzzy.n_features_in_ = len(selected_names)
    fuzzy.classes_ = np.asarray([0, 1], dtype=object)
    fuzzy.rules_by_class_ = {
        0: [{"antecedents": [{"feature": selected_names[0], "term": "medium"}], "weight": 1.0}],
        1: [{"antecedents": [{"feature": selected_names[0], "term": "high"}], "weight": 1.0}],
    }
    fuzzy.rules_ = fuzzy.rules_by_class_[0] + fuzzy.rules_by_class_[1]
    model.fuzzy_ = fuzzy
    return model


@pytest.fixture
def installed_test_artifacts(tmp_path, monkeypatch):
    """Install a self-contained, disposable artifact bundle for tests/API."""
    project_root = tmp_path / "project"
    models_root = project_root / "models"
    experiments_root = project_root / "experiments"
    model_dir = models_root / bidmc_inference.BIDMC_RUN_ID
    experiment_dir = experiments_root / bidmc_inference.BIDMC_RUN_ID
    model_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)

    source_hashes = {}
    for relative in SOURCE_FILES:
        source = config.ROOT / relative
        target = project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        source_hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()

    environment = {
        "project": "PulseLab test fixture",
        "run_id": bidmc_inference.BIDMC_RUN_ID,
        "project_version": (project_root / "VERSION").read_text(encoding="utf-8").strip(),
        "python_version": "test",
        "package_versions": {},
        "source_sha256": source_hashes,
    }
    reproducibility_dir = project_root / "docs" / "reproducibility"
    reproducibility_dir.mkdir(parents=True)
    (reproducibility_dir / "bidmc_training_environment.json").write_text(
        json.dumps(environment), encoding="utf-8"
    )

    record_features = extract_record_features(
        _synthetic_record(), window_sec=32.0, stride_sec=16.0, include_full_fusion=False
    )
    predictor_names = [
        name for name in record_features.columns
        if name.startswith(("ecg_", "ppg_", "ecg_ppg_"))
    ]
    ecg_ppg_names = [name for name in predictor_names if name.startswith(("ecg_", "ppg_", "ecg_ppg_"))]
    ppg_names = [name for name in predictor_names if name.startswith("ppg_")]
    assert len(ecg_ppg_names) == 32
    assert len(ppg_names) == 15

    joblib.dump(_make_test_model(ecg_ppg_names, 19), model_dir / bidmc_inference.MODEL_FILES["ecg_ppg"])
    joblib.dump(_make_test_model(ppg_names, 13), model_dir / bidmc_inference.MODEL_FILES["ppg_only"])

    manifest = {
        "run_id": bidmc_inference.BIDMC_RUN_ID,
        "project_version": environment["project_version"],
        "dataset": "BIDMC test fixture",
        "dataset_version": "1.0.0",
        "dataset_records": 53,
        "patient_groups": 46,
        "total_windows_full": 1537,
        "labelled_windows_mean": 1533,
        "window_sec": 32.0,
        "stride_sec": 16.0,
        "sampling_frequency_hz": 125.0,
        "target": "ref_rr_mean_bpm",
        "target_definition": "Mean of manual ann1/ann2 breath-interval rates.",
        "predictor_policy": "PPG-only or ECG+PPG; RESP-derived predictors excluded.",
        "train_patient_ids": [f"train_{i:02d}" for i in range(36)],
        "test_patient_ids": [f"test_{i:02d}" for i in range(10)],
        "evaluation_note": "Fixture evaluation; not a research result.",
        "clinical_status": "Test fixture; not clinically validated.",
    }
    (experiment_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    comparison = pd.DataFrame([
        {"model": f"fixture_model_{i}", "target": "annotator_mean", "n_test_windows": 286,
         "MAE_bpm": 3.0 + i / 10, "RMSE_bpm": 4.0 + i / 10}
        for i in range(6)
    ])
    comparison.to_csv(experiment_dir / "model_comparison.csv", index=False)
    agreement = {"mean_absolute_annotator_difference_bpm": 1.2, "within_2_bpm_fraction": 0.8}
    (experiment_dir / "annotation_agreement.json").write_text(json.dumps(agreement), encoding="utf-8")

    monkeypatch.setattr(config, "ROOT", project_root)
    monkeypatch.setattr(config, "MODELS_DIR", models_root)
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", experiments_root)
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", experiments_root)
    server._BIDMC_MODEL_REGISTRY = None
    yield project_root, models_root, experiments_root
    server._BIDMC_MODEL_REGISTRY = None


def test_bidmc_trained_registry_verifies_source_and_loads_both_estimators(installed_test_artifacts):
    registry = bidmc_inference.load_trained_registry(
        config.ROOT, config.MODELS_DIR, config.EXPERIMENTS_DIR
    )
    assert registry["available"] is True
    assert registry["source_hashes_verified"] is True
    assert registry["run_id"] == bidmc_inference.BIDMC_RUN_ID
    assert set(registry["models_loaded"]) == {"ecg_ppg", "ppg_only"}
    assert registry["models"]["ecg_ppg"]["input_feature_count"] == 32
    assert registry["models"]["ppg_only"]["input_feature_count"] == 15
    assert len(registry["heldout_comparison"]) == 6


def test_bidmc_saved_models_predict_real_window_feature_frames(installed_test_artifacts):
    registry = bidmc_inference.load_trained_registry(
        config.ROOT, config.MODELS_DIR, config.EXPERIMENTS_DIR
    )
    result = bidmc_inference.predict_record(
        _synthetic_record(), registry["models_loaded"], "ecg_ppg"
    )
    assert result["n_windows"] == 3
    assert result["model_key"] == "ecg_ppg"
    assert len(result["predictions"]) == 3
    assert all(np.isfinite(row["predicted_rr_bpm"]) for row in result["predictions"])
    assert "not an independent test result" in result["interpretation_note"]


def test_bidmc_model_status_api_exposes_trained_patient_disjoint_results(installed_test_artifacts):
    server._BIDMC_MODEL_REGISTRY = None
    with TestClient(server.app) as client:
        response = client.get("/api/bidmc/model/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["n_train_patients"] == 36
    assert payload["n_test_patients"] == 10
    assert payload["models"]["ecg_ppg"]["selected_feature_count"] == 19
    assert payload["models"]["ppg_only"]["selected_feature_count"] == 13


def test_bidmc_prediction_api_uses_saved_python_estimator(installed_test_artifacts, monkeypatch):
    record = _synthetic_record()
    monkeypatch.setattr(server.bidmc_pipeline, "load_record", lambda *args, **kwargs: record)
    server._BIDMC_MODEL_REGISTRY = None
    with TestClient(server.app) as client:
        response = client.get("/api/bidmc/records/bidmc01/predictions?model=ppg_only")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_key"] == "ppg_only"
    assert payload["record_id"] == "bidmc01"
    assert payload["n_windows"] == 3
    assert payload["predictions"][0]["predicted_rr_bpm"] is not None


def test_bidmc_model_report_download_is_allowlisted(installed_test_artifacts):
    with TestClient(server.app) as client:
        okay = client.get("/api/bidmc/model/artifacts/model_comparison.csv")
        manifest = client.get("/api/bidmc/model/artifacts/run_manifest.json")
        agreement = client.get("/api/bidmc/model/artifacts/annotation_agreement.json")
        denied = client.get("/api/bidmc/model/artifacts/../../config.py")
    assert okay.status_code == 200
    assert "MAE_bpm" in okay.text
    assert manifest.status_code == 200 and manifest.json()["run_id"] == bidmc_inference.BIDMC_RUN_ID
    assert agreement.status_code == 200 and "mean_absolute_annotator_difference_bpm" in agreement.json()
    assert denied.status_code == 404


def test_bidmc_registry_fails_closed_if_a_required_model_is_missing(installed_test_artifacts):
    missing_model = config.MODELS_DIR / bidmc_inference.BIDMC_RUN_ID / bidmc_inference.MODEL_FILES["ppg_only"]
    missing_model.unlink()
    with pytest.raises(bidmc_inference.BIDMCArtifactError, match="artifacts are incomplete"):
        bidmc_inference.load_trained_registry(config.ROOT, config.MODELS_DIR, config.EXPERIMENTS_DIR)
