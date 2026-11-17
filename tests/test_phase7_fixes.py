"""Regression tests for the Phase 7 bug-fix pass."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from modules import inference
from modules.rpeak_detection import detect_rpeaks, match_detections


def _synthetic_ecg(n_beats=60, fs=360, ectopic_every=5, normal_amp=0.4, ectopic_amp=1.6, seed=1):
    """Regular rhythm in which a minority of beats are several times larger (as in MIT-BIH 228)."""
    rng = np.random.default_rng(seed)
    n = int((n_beats + 2) * 0.8 * fs)
    x = rng.normal(0, 0.01, n)
    t = np.arange(-int(0.06 * fs), int(0.06 * fs) + 1) / fs
    pulse = np.exp(-(t / 0.012) ** 2) - 0.25 * np.exp(-((t - 0.02) / 0.02) ** 2)
    peaks = []
    for i in range(n_beats):
        centre = int((i + 1) * 0.8 * fs)
        amp = ectopic_amp if i % ectopic_every == 0 else normal_amp
        x[centre - len(t) // 2: centre - len(t) // 2 + len(t)] += amp * pulse
        peaks.append(centre)
    return x, np.asarray(peaks), fs


def test_adaptive_detector_keeps_normal_beats_when_large_ectopics_are_present():
    x, truth, fs = _synthetic_ecg()
    detected = detect_rpeaks(x, fs)
    result = match_detections(detected, truth, fs, tol=0.075)
    assert result["sensitivity"] >= 0.95
    assert result["ppv"] >= 0.95


def test_adaptive_detector_rejects_pure_noise():
    rng = np.random.default_rng(3)
    detected = detect_rpeaks(rng.normal(0, 1, 360 * 20), 360)
    # Noise has no dominant QRS-like peaks; the absolute floor must not flood the output.
    assert len(detected) <= 20 * 360 / (0.25 * 360)


def _csv(columns, header=None, n=360 * 8):
    lines = [header] if header else []
    lines += [",".join(f"{c[i]:.5f}" for c in columns) for i in range(n)]
    return "\n".join(lines).encode()


@pytest.mark.parametrize("header", ["'sample interval','MLII','V5'", "ms,MLII,V5", "Elapsed time,MLII,V5", None])
def test_parser_never_uses_a_time_or_index_column_as_an_ecg_lead(header):
    fs, n = 360, 360 * 8
    t = np.arange(n) / fs
    lead_a = np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.default_rng(0).normal(size=n)
    lead_b = np.cos(2 * np.pi * 1.2 * t)
    axis = {"'sample interval','MLII','V5'": np.arange(n), "ms,MLII,V5": t * 1000}.get(header, t)
    signal, meta = inference.parse_ecg_csv(_csv([axis, lead_a, lead_b], header, n), 2, fs)
    assert signal.shape == (n, 2)
    assert np.corrcoef(signal[:, 0], lead_a)[0, 1] > 0.999
    assert meta["time_or_index_columns_skipped"]


def test_parser_keeps_two_headerless_leads():
    n = 360 * 8
    t = np.arange(n) / 360
    a, b = np.sin(2 * np.pi * 1.1 * t), np.cos(2 * np.pi * 0.9 * t)
    signal, meta = inference.parse_ecg_csv(_csv([a, b], None, n), 2, 360)
    assert signal.shape == (n, 2) and not meta["time_or_index_columns_skipped"]


def test_experiment_summary_exposes_patient_counts(tmp_path):
    run = tmp_path / "run_x"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": "run_x", "n_train": 10, "n_test": 5, "n_train_patients": 22, "n_test_patients": 21,
        "train_records": ["100"], "test_records": ["101", "103"], "models": [], "metrics": {},
    }))
    summary = server._read_experiment_summary(run)
    assert summary["n_test_patients"] == 21 and summary["n_train_patients"] == 22
    assert summary["n_test_records"] == 2


def test_second_experiment_is_rejected_while_one_is_running():
    with server._JOBS_LOCK:
        server._JOBS["fake-running"] = {"job_id": "fake-running", "kind": "experiment", "status": "running"}
    try:
        response = TestClient(server.app).post("/api/experiments", json={"quick": True})
        assert response.status_code == 409
    finally:
        with server._JOBS_LOCK:
            server._JOBS.pop("fake-running", None)


def test_api_requests_default_to_the_full_dataset():
    assert server.AuditRequest().quick is False
    assert server.RPeakValidationRequest().quick is False
    assert server.ExperimentRequest().quick is False
