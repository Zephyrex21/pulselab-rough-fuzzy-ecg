from fastapi.testclient import TestClient
import numpy as np

import config
import server
from modules import bidmc_inference
from modules.bidmc_pipeline import BIDMCRecord


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


def test_bidmc_trained_registry_verifies_source_and_loads_both_estimators():
    registry = bidmc_inference.load_trained_registry(
        config.ROOT, config.MODELS_DIR, config.EXPERIMENTS_DIR
    )
    assert registry["available"] is True
    assert registry["source_hashes_verified"] is True
    assert registry["run_id"] == "pulselab_bidmc_patient_disjoint_20261117T144120Z_2489"
    assert set(registry["models_loaded"]) == {"ecg_ppg", "ppg_only"}
    assert registry["models"]["ecg_ppg"]["input_feature_count"] == 32
    assert registry["models"]["ppg_only"]["input_feature_count"] == 15
    assert len(registry["heldout_comparison"]) == 6


def test_bidmc_saved_models_predict_real_window_feature_frames():
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


def test_bidmc_model_status_api_exposes_trained_patient_disjoint_results():
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


def test_bidmc_prediction_api_uses_saved_python_estimator(monkeypatch):
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


def test_bidmc_model_report_download_is_allowlisted():
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
