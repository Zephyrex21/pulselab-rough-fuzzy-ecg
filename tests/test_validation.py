from types import SimpleNamespace
import numpy as np

from modules.validation import evaluate_record_rpeaks, aggregate_detection_metrics
from modules.metrics import record_cluster_bootstrap, per_record_metrics_frame


def test_rpeak_validation_uses_one_to_one_matching_and_reports_detection_metrics():
    reference = np.asarray([300, 700, 1100], dtype=int)
    record = SimpleNamespace(
        record_id="100", fs=360.0,
        signal=np.sin(np.linspace(0, 30, 1800))[:, None],
        ann_samples=reference,
    )
    result = evaluate_record_rpeaks(record, detector=lambda signal, fs: reference.copy())
    assert result["tp"] == 3
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["sensitivity"] == 1.0
    assert result["ppv"] == 1.0
    assert result["f1"] == 1.0
    assert result["mean_absolute_error_ms"] == 0.0


def test_aggregate_detection_metrics_uses_pooled_counts():
    rows = [
        {"tp": 8, "fp": 2, "fn": 2, "mean_absolute_error_ms": 3.0},
        {"tp": 2, "fp": 0, "fn": 1, "mean_absolute_error_ms": 5.0},
    ]
    result = aggregate_detection_metrics(rows)
    assert result["tp"] == 10
    assert result["fp"] == 2
    assert result["fn"] == 3
    assert np.isclose(result["micro_sensitivity"], 10 / 13)
    assert np.isclose(result["micro_ppv"], 10 / 12)


def test_record_cluster_bootstrap_is_repeatable_and_record_grouped():
    truth = np.asarray(["N", "N", "V", "V", "S", "S"])
    pred = np.asarray(["N", "V", "V", "V", "S", "N"])
    records = np.asarray(["100", "100", "101", "101", "102", "102"])
    a = record_cluster_bootstrap(truth, pred, records, n_bootstrap=80, seed=7)
    b = record_cluster_bootstrap(truth, pred, records, n_bootstrap=80, seed=7)
    assert a == b
    assert a["n_records"] == 3
    assert a["status"] == "estimated"
    assert len(a["accuracy_95ci"]) == 2


def test_per_record_metrics_emits_patient_group_ids():
    frame = per_record_metrics_frame(
        "model", ["N", "V", "N", "V"], ["N", "N", "N", "V"], ["201", "201", "202", "202"]
    )
    assert len(frame) == 2
    assert set(frame["patient_id"]) == {"subject_201_202"}
