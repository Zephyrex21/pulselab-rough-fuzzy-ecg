import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules import inference
from modules.feature_schema import FEATURE_SCHEMA_VERSION, feature_names


class FixtureSavedModel:
    classes_ = np.array(["N", "V"])

    def predict(self, frame):
        return np.array(["N", "V"][: len(frame)])

    def scores(self, frame):
        return np.array([[0.8, 0.2], [0.25, 0.75]][: len(frame)])

    def explain(self, row):
        return {
            "prediction": "N",
            "active_rule_count": 2,
            "top_two_margin": 0.6,
            "no_rule_fired": False,
            "top_rules": [{
                "weight": 0.7,
                "firing_strength": 0.9,
                "antecedents": [{"feature": "rr_prev", "term": "Low"}],
            }],
        }


def _make_saved_run(tmp_path: Path):
    experiments = tmp_path / "experiments"
    models = tmp_path / "models"
    project = tmp_path
    run_id = "fixture-mitdb-run"
    run_dir = experiments / run_id
    model_dir = models / run_id
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    artifact = model_dir / "fixture_model.joblib"
    artifact.write_bytes(b"fixture placeholder")
    columns = feature_names(1)
    manifest = {
        "run_id": run_id,
        "created_at_utc": "2026-11-13T00:00:00+00:00",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": columns,
        "feature_medians": {name: 0.0 for name in columns},
        "models": ["Fixture model"],
        "metadata": {
            "dataset": "MIT-BIH Arrhythmia Database",
            "peak_source": "detected",
            "split_policy": "patient-disjoint",
            "patient_disjoint": True,
        },
        "score_interpretation": {"Fixture model": "fixture ranking scores"},
        "metrics": {"Fixture model": {"model_artifact": f"models/{run_id}/{artifact.name}"}},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return experiments, project, models, run_id, columns


def test_parse_csv_with_header_skips_time_index_and_keeps_two_channels():
    fs = 360
    lines = ["time_s,MLII,V5"]
    for index in range(fs * 4):
        t = index / fs
        lines.append(f"{t:.6f},{np.sin(2*np.pi*1.2*t):.6f},{np.cos(2*np.pi*1.1*t):.6f}")
    signal, metadata = inference.parse_ecg_csv("\n".join(lines).encode(), 2, fs)
    assert signal.shape == (fs * 4, 2)
    assert metadata["channel_names"] == ["MLII", "V5"]
    assert metadata["duration_seconds"] == pytest.approx(4.0)


def test_parse_headerless_single_channel_csv():
    fs = 360
    samples = np.sin(np.linspace(0, 20, fs * 4))
    content = "\n".join(f"{value:.8f}" for value in samples).encode()
    signal, metadata = inference.parse_ecg_csv(content, 1, fs)
    assert signal.shape == (fs * 4, 1)
    assert metadata["channel_names"] == ["column_0"]


def test_parse_rejects_short_or_wrong_channel_input():
    with pytest.raises(inference.InferenceError, match="at least"):
        inference.parse_ecg_csv(b"ecg\n0.1\n0.2\n0.3\n", 1, 360)
    with pytest.raises(inference.InferenceError, match="requires 2"):
        inference.parse_ecg_csv(("\n".join(["ecg"] + ["0.2"] * 1200)).encode(), 2, 360)


def test_discover_saved_models_lists_only_compatible_existing_artifacts(tmp_path):
    experiments, project, models, run_id, columns = _make_saved_run(tmp_path)
    found = inference.discover_saved_models(experiments, project, models)
    assert len(found) == 1
    assert found[0]["run_id"] == run_id
    assert found[0]["model_name"] == "Fixture model"
    assert found[0]["n_channels"] == 1
    assert found[0]["n_features"] == len(columns)


def test_execute_inference_persists_beat_results_without_raw_signal(tmp_path, monkeypatch):
    experiments, project, models, run_id, columns = _make_saved_run(tmp_path)
    monkeypatch.setattr(inference.joblib, "load", lambda _path: FixtureSavedModel())

    def fake_process_signal(signal, fs, rpeaks=None, primary=0):
        assert signal.shape[1] == 1
        assert fs == 360
        frame = pd.DataFrame([{name: float(i + 1) for name in columns} for i in range(2)])
        frame["r_sample"] = [360, 720]
        return {"features": frame, "rpeaks": np.array([360, 720, 1080]), "fs": 360}

    monkeypatch.setattr(inference.feature_extraction, "process_signal", fake_process_signal)
    samples = np.sin(np.linspace(0, 50, 360 * 5))
    content = "\n".join(f"{value:.8f}" for value in samples).encode()
    result = inference.execute_inference(
        content=content,
        filename="sample.csv",
        sampling_rate_hz=360,
        run_id=run_id,
        model_name="Fixture model",
        experiments_dir=experiments,
        project_root=project,
        models_root=models,
    )
    assert result["n_predictions"] == 2
    assert [row["predicted_class"] for row in result["predictions"]] == ["N", "V"]
    assert result["predictions"][1]["time_seconds"] == pytest.approx(2.0)
    out = experiments / result["inference_id"]
    assert (out / "beat_predictions.csv").is_file()
    assert (out / "beat_features.csv").is_file()
    assert (out / "inference_summary.json").is_file()
    assert not list(out.glob("*raw*"))
    loaded, _ = inference.read_inference_summary(result["inference_id"], experiments)
    assert loaded["model"]["model_name"] == "Fixture model"
    explanation = inference.explain_inference_beat(
        result["inference_id"], 0, experiments, project, models
    )
    assert explanation["explanation"]["available"] is True
    assert explanation["explanation"]["details"]["active_rule_count"] == 2


def test_model_resolution_rejects_arbitrary_model_paths(tmp_path):
    experiments, project, models, run_id, _ = _make_saved_run(tmp_path)
    manifest_path = experiments / run_id / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metrics"]["Fixture model"]["model_artifact"] = "../../secrets.joblib"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(inference.InferenceError):
        inference.get_saved_model_entry(run_id, "Fixture model", experiments, project, models)
