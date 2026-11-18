"""API integration tests for the Phase 4 local research service."""
import hashlib
from fastapi.testclient import TestClient

import config
import server


def test_health_reports_service_and_dependencies():
    with TestClient(server.app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "PulseLab Research API"
    assert "wfdb" in payload["dependencies"]


def test_dataset_status_uses_quick_patient_disjoint_split():
    with TestClient(server.app) as client:
        response = client.get("/api/dataset/status?quick=true")
    assert response.status_code == 200
    payload = response.json()
    assert payload["split_protocol"] == "patient-disjoint"
    assert payload["train_records_requested"] == 8
    assert payload["test_records_requested"] == 6
    assert isinstance(payload["missing_records"], list)


def test_root_and_web_frontend_are_served():
    with TestClient(server.app) as client:
        root = client.get("/", follow_redirects=False)
        page = client.get("/web/")
    assert root.status_code in (307, 308)
    assert root.headers["location"] == "/web/"
    assert page.status_code == 200
    assert "PulseLab" in page.text


def test_audit_job_persists_a_report_for_missing_local_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "mitdb")
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", tmp_path / "experiments")
    server._JOBS.clear()
    with TestClient(server.app) as client:
        created = client.post("/api/dataset/audits", json={"quick": True, "download_missing": False})
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "completed"
        assert job["result"]["n_records_requested"] == 14
        audit_id = job["result"]["audit_id"]
        summary = client.get(f"/api/audits/{audit_id}/summary")
        assert summary.status_code == 200
        assert summary.json()["n_records_failed"] == 14
        audit_csv = client.get(f"/api/audits/{audit_id}/artifacts/record_inventory.csv")
        assert audit_csv.status_code == 200
        assert "record" in audit_csv.text


def test_experiment_job_reports_missing_records_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "mitdb")
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", tmp_path / "experiments")
    server._JOBS.clear()
    with TestClient(server.app) as client:
        created = client.post("/api/experiments", json={"quick": True, "download_missing": False})
        assert created.status_code == 202
        job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert "Missing 14 MIT-BIH records" in job["error"]


def test_artifact_download_rejects_path_traversal():
    with TestClient(server.app) as client:
        response = client.get("/api/experiments/../../config/artifacts/config.py")
    assert response.status_code in (400, 404)


def test_rpeak_validation_job_reports_missing_records(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "mitdb")
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", tmp_path / "experiments")
    server._JOBS.clear()
    with TestClient(server.app) as client:
        created = client.post("/api/rpeak-validations", json={"quick": True, "download_missing": False})
        assert created.status_code == 202
        job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert "Missing 6 MIT-BIH test records" in job["error"]


def test_rpeak_validation_rejects_out_of_range_tolerance():
    with TestClient(server.app) as client:
        response = client.post("/api/rpeak-validations", json={"tolerance_ms": 500})
    assert response.status_code == 422


def test_api_docs_and_artifact_allowlist_are_available():
    with TestClient(server.app) as client:
        docs = client.get("/api/docs")
        bad_artifact = client.get("/api/audits/dataset_audit_fake/artifacts/secret.txt")
    assert docs.status_code == 200
    assert bad_artifact.status_code == 404


def test_frontend_exposes_two_real_research_engines_without_browser_simulator():
    root = server.WEB_DIR
    index = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    assert 'id="environment-chip"' in index
    assert 'src="app.js"' in index
    assert 'src="core.js"' not in index
    assert "MIT-BIH ECG Engine" in script
    assert "BIDMC Multimodal Engine" in script
    assert "/bidmc/status" in script
    assert "/bidmc/feature-exports" in script
    assert "sample(" not in script
    assert not (root / "core.js").exists()


def test_saved_experiment_summary_and_artifact_download(tmp_path, monkeypatch):
    import json
    import pandas as pd

    experiment_root = tmp_path / "experiments"
    run_dir = experiment_root / "phase4_fixture"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "phase4_fixture",
        "created_at_utc": "2026-11-13T12:00:00+00:00",
        "n_train": 120,
        "n_test": 45,
        "n_features": 47,
        "models": ["Rough-Set + Fuzzy"],
        "train_class_distribution": {"N": 100, "S": 20},
        "test_class_distribution": {"N": 35, "S": 10},
        "warnings": [],
        "notes": ["fixture"],
        "metadata": {"patient_disjoint": True, "peak_source": "detected"},
        "metrics": {"Rough-Set + Fuzzy": {"accuracy": 0.8, "macro_f1": 0.75}},
        "disclaimer": "Research only",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame([{"model": "Rough-Set + Fuzzy", "accuracy": 0.8, "macro_f1": 0.75, "weighted_f1": 0.82}]).to_csv(run_dir / "model_comparison.csv", index=False)
    (run_dir / "research_report.md").write_text("# Fixture report\n", encoding="utf-8")
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", experiment_root)

    with TestClient(server.app) as client:
        listed = client.get("/api/experiments").json()
        summary_response = client.get("/api/experiments/phase4_fixture/summary")
        artifact = client.get("/api/experiments/phase4_fixture/artifacts/research_report.md")

    assert listed["count"] == 1
    assert listed["experiments"][0]["run_id"] == "phase4_fixture"
    assert summary_response.status_code == 200
    assert summary_response.json()["model_comparison"][0]["macro_f1"] == 0.75
    assert artifact.status_code == 200
    assert "Fixture report" in artifact.text


def test_phase5_model_catalog_endpoint(monkeypatch):
    from modules import inference

    expected = [{"model_id": "run-a::model-a", "run_id": "run-a", "model_name": "Model A"}]
    monkeypatch.setattr(inference, "discover_saved_models", lambda *args: expected)
    with TestClient(server.app) as client:
        response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json()["models"] == expected


def test_phase5_uploaded_inference_endpoint_accepts_csv_and_returns_result(monkeypatch, tmp_path):
    from modules import inference

    monkeypatch.setattr(server, "EXPERIMENTS_DIR", tmp_path / "experiments")
    expected = {
        "inference_id": "inference_fixture_1234",
        "n_predictions": 1,
        "predictions": [{"beat_index": 0, "predicted_class": "N"}],
    }
    def fake_execute(**kwargs):
        assert kwargs["content"].startswith(b"ecg\n")
        assert kwargs["filename"] == "uploaded.csv"
        assert kwargs["sampling_rate_hz"] == 360.0
        return expected
    monkeypatch.setattr(inference, "execute_inference", fake_execute)
    with TestClient(server.app) as client:
        response = client.post(
            "/api/inference",
            data={"run_id": "run-a", "model_name": "Model A", "sampling_rate_hz": "360"},
            files={"file": ("uploaded.csv", b"ecg\n0.1\n0.2\n0.3\n", "text/csv")},
        )
    assert response.status_code == 201
    assert response.json() == expected


def test_phase5_uploaded_inference_rejects_unsupported_extension():
    with TestClient(server.app) as client:
        response = client.post(
            "/api/inference",
            data={"run_id": "run-a", "model_name": "Model A", "sampling_rate_hz": "360"},
            files={"file": ("signal.npy", b"not a csv", "application/octet-stream")},
        )
    assert response.status_code == 415


def test_phase5_inference_artifact_allowlist(tmp_path, monkeypatch):
    inference_id = "inference_fixture_1234"
    output = tmp_path / inference_id
    output.mkdir()
    (output / "beat_predictions.csv").write_text("beat_index,predicted_class\n0,N\n")
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", tmp_path)
    with TestClient(server.app) as client:
        okay = client.get(f"/api/inferences/{inference_id}/artifacts/beat_predictions.csv")
        denied = client.get(f"/api/inferences/{inference_id}/artifacts/secrets.txt")
    assert okay.status_code == 200
    assert "predicted_class" in okay.text
    assert denied.status_code == 404


def test_phase5_full_multipart_inference_round_trip(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pandas as pd
    from modules.feature_schema import FEATURE_SCHEMA_VERSION, feature_names
    from modules.pipeline_contract import current_mitbih_inference_contract

    run_id = "api-inference-fixture"
    experiments = tmp_path / "experiments"
    run_dir = experiments / run_id
    models_root = tmp_path / "models"
    model_dir = models_root / run_id
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    artifact = model_dir / "fixture.joblib"
    artifact.write_bytes(b"local test artifact")
    columns = feature_names(1)
    manifest = {
        "run_id": run_id,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": columns,
        "feature_medians": {name: 0.0 for name in columns},
        "metadata": {
            "dataset": "MIT-BIH Arrhythmia Database",
            "peak_source": "detected",
            "split_policy": "patient-disjoint",
            "patient_disjoint": True,
            "inference_pipeline_contract": current_mitbih_inference_contract(),
        },
        "score_interpretation": {"Fixture API model": "test model scores"},
        "metrics": {"Fixture API model": {
            "model_artifact": f"models/{run_id}/{artifact.name}",
            "model_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", experiments)
    monkeypatch.setattr(server.config, "ROOT", tmp_path)
    monkeypatch.setattr(server.config, "MODELS_DIR", models_root)

    class ApiFixtureModel:
        classes_ = np.array(["N", "V"])
        def predict(self, frame):
            return np.array(["N"] * len(frame))
        def scores(self, frame):
            return np.tile(np.array([[0.75, 0.25]]), (len(frame), 1))
        def explain(self, row):
            return {"active_rule_count": 1, "top_two_margin": 0.5, "no_rule_fired": False, "top_rules": []}

    monkeypatch.setattr(server.inference.joblib, "load", lambda _path: ApiFixtureModel())
    def fake_process_signal(signal, fs, rpeaks=None, primary=0):
        frame = pd.DataFrame([{name: 0.25 for name in columns} for _ in range(2)])
        frame["r_sample"] = [360, 720]
        return {"features": frame, "rpeaks": np.array([360, 720]), "fs": 360}
    monkeypatch.setattr(server.inference.feature_extraction, "process_signal", fake_process_signal)

    samples = np.sin(np.linspace(0, 45, 360 * 5))
    csv_content = "\n".join(f"{value:.8f}" for value in samples).encode()
    with TestClient(server.app) as client:
        catalog = client.get("/api/models")
        response = client.post(
            "/api/inference",
            data={"run_id": run_id, "model_name": "Fixture API model", "sampling_rate_hz": "360"},
            files={"file": ("ecg.csv", csv_content, "text/csv")},
        )
        assert catalog.status_code == 200
        assert len(catalog.json()["models"]) == 1
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["n_predictions"] == 2
        assert result["predictions"][0]["predicted_class"] == "N"
        explanation = client.get(
            f"/api/inferences/{result['inference_id']}/beats/0/explanation"
        )
        predictions_artifact = client.get(
            f"/api/inferences/{result['inference_id']}/artifacts/beat_predictions.csv"
        )
    assert explanation.status_code == 200
    assert explanation.json()["explanation"]["available"] is True
    assert predictions_artifact.status_code == 200
    assert "predicted_class" in predictions_artifact.text


def test_frontend_includes_saved_python_mitbih_inference():
    index = (server.WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (server.WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "Saved-model ECG inference" in script
    assert "Run Python ECG inference" in script
    assert "'/inference'" in script or "'/api/inference'" in script or "request('/inference'" in script
    assert "PulseLab" in index


def test_phase6_inference_drift_artifacts_are_allowlisted(tmp_path, monkeypatch):
    inference_id = "inference_phase6_fixture"
    output = tmp_path / inference_id
    output.mkdir()
    (output / "feature_drift_by_feature.csv").write_text("feature,outside_fraction\nf1,0.2\n")
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", tmp_path)
    with TestClient(server.app) as client:
        okay = client.get(f"/api/inferences/{inference_id}/artifacts/feature_drift_by_feature.csv")
        denied = client.get(f"/api/inferences/{inference_id}/artifacts/../../secret.txt")
    assert okay.status_code == 200
    assert "outside_fraction" in okay.text
    assert denied.status_code in (400, 404)


def test_phase6_backend_explainability_endpoints_remain_available():
    paths = {route.path for route in server.app.routes}
    assert "/api/inferences/{inference_id}/beats/{beat_index}/explanation" in paths
    assert "/api/inferences/{inference_id}/beats/{beat_index}/waveform" in paths
    assert "/api/inferences/{inference_id}/artifacts/{artifact_name}" in paths


def test_phase6_beat_waveform_preview_endpoint(monkeypatch):
    from modules import inference

    expected = {
        "available": True, "inference_id": "inference_fixture_1234",
        "beat_index": 0, "sampling_rate_hz": 360,
        "r_peak_sample_in_window": 90, "time_ms": [-250.0, 0.0],
        "samples": [0.1, 1.0], "amplitude_units": "R-amplitude-normalised units",
        "retention": "process memory only; not persisted to disk", "predicted_class": "N",
    }
    monkeypatch.setattr(inference, "get_inference_waveform", lambda *args: expected)
    with TestClient(server.app) as client:
        response = client.get("/api/inferences/inference_fixture_1234/beats/0/waveform")
    assert response.status_code == 200
    assert response.json() == expected


def test_phase6_waveform_endpoint_returns_gone_when_preview_was_evicted(monkeypatch):
    from modules import inference

    def unavailable(*args):
        raise inference.WaveformPreviewUnavailable("Preview cache was evicted.")
    monkeypatch.setattr(inference, "get_inference_waveform", unavailable)
    with TestClient(server.app) as client:
        response = client.get("/api/inferences/inference_fixture_1234/beats/0/waveform")
    assert response.status_code == 410
    assert "evicted" in response.json()["detail"]


def test_bidmc_status_endpoint_reports_missing_local_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BIDMC_DATA_DIR", tmp_path / "bidmc")
    with TestClient(server.app) as client:
        response = client.get("/api/bidmc/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "BIDMC PPG and Respiration Dataset"
    assert payload["records_expected"] == 53
    assert payload["ready"] is False


def test_bidmc_feature_export_requires_a_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BIDMC_DATA_DIR", tmp_path / "bidmc")
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", tmp_path / "experiments")
    server._JOBS.clear()
    with TestClient(server.app) as client:
        created = client.post("/api/bidmc/feature-exports", json={"records": ["bidmc01"]})
        assert created.status_code == 202
        job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert "BIDMC CSV files were not found" in job["error"]


def test_bidmc_preview_rejects_missing_record(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BIDMC_DATA_DIR", tmp_path / "bidmc")
    with TestClient(server.app) as client:
        response = client.get("/api/bidmc/records/bidmc01/preview")
    assert response.status_code == 400
    assert "BIDMC dataset not found" in response.json()["detail"]


def test_frontend_result_summaries_are_in_app_not_raw_json_tabs():
    root = server.WEB_DIR
    index = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    styles = (root / "styles.css").read_text(encoding="utf-8")
    assert 'id="result-dialog"' in index
    assert "renderExperimentSummary" in script
    assert "renderInferenceSummary" in script
    assert "markdownReportToHtml" in script
    assert "state.prediction=null" in script
    external_links = [
        '<a' + part.split('>', 1)[0] + '>'
        for part in script.split('<a')[1:]
        if 'target="_blank"' in part.split('>', 1)[0]
    ]
    assert external_links
    assert all('rel="noopener noreferrer"' in tag for tag in external_links)
    assert 'class="nav-label"' in script
    assert "body.sidebar-collapsed .nav-label" in styles
    assert "result-dialog" in styles



def test_bidmc_post_interpolation_metadata_is_not_labelled_as_signal_quality():
    import pandas as pd

    frame = pd.DataFrame({
        "window_quality": [1.0, 0.95],
        "ppg_mean": [0.1, 0.2],
    })
    renamed = server._rename_bidmc_quality_metadata_column(frame)

    assert "window_quality" not in renamed.columns
    assert "post_interpolation_finite_fraction" in renamed.columns
    assert renamed["post_interpolation_finite_fraction"].tolist() == [1.0, 0.95]
    assert "not measure original missingness" in server.BIDMC_SIGNAL_QUALITY_NOTE


def test_bidmc_prediction_metadata_marks_signal_quality_as_not_assessed():
    payload = {
        "predictions": [
            {"window_index": 0, "window_quality": 1.0},
            {"window_index": 1, "window_quality": 1.0},
        ]
    }
    clarified = server._clarify_bidmc_prediction_quality(payload)

    assert clarified["signal_quality_assessment"]["status"] == "not_assessed"
    assert all("window_quality" not in row for row in clarified["predictions"])
    assert all("post_interpolation_finite_fraction" in row for row in clarified["predictions"])
    assert all(row["post_interpolation_finite_fraction"] == 1.0 for row in clarified["predictions"])


def test_bidmc_ui_explains_post_interpolation_fraction_limitation():
    script = (server.WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "Signal quality is not currently assessed" in script
    assert "post_interpolation_finite_fraction" in script
    assert "does not measure original missingness" in script
