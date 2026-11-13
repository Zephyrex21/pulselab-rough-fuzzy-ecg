"""API integration tests for the Phase 4 local research service."""
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


def test_frontend_exposes_research_backend_and_api_boundary():
    root = server.WEB_DIR
    index = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    assert 'id="environment-chip"' in index
    assert "backend:{label:'Research Backend'" in script
    assert "const API_BASE='/api'" in script
    assert "synthetic browser demonstration" in script


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
