import numpy as np
from modules.metrics import evaluate_predictions


def test_metrics_include_per_class_sensitivity_specificity_and_auc():
    y = np.array(["N", "N", "S", "S", "V", "V"])
    pred = np.array(["N", "S", "S", "S", "V", "N"])
    scores = np.array([
        [0.8, 0.1, 0.1, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0, 0.0],
        [0.1, 0.8, 0.1, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0, 0.0],
        [0.1, 0.1, 0.8, 0.0, 0.0],
        [0.5, 0.1, 0.4, 0.0, 0.0],
    ])
    result = evaluate_predictions(y, pred, scores)
    assert 0 <= result["accuracy"] <= 1
    assert result["per_class"][0]["specificity"] >= 0
    assert result["macro_roc_auc"] is not None
    assert len(result["confusion_matrix"]) == 5


def test_macro_f1_uses_test_present_classes_and_also_reports_all_aami_classes():
    y = np.array(["N", "N", "S", "S", "V", "V"])
    result = evaluate_predictions(y, y)
    assert result["macro_f1"] == 1.0
    assert result["macro_f1_all_aami"] == 0.6
    assert result["macro_classes_evaluated"] == ["N", "S", "V"]


def test_patient_cluster_bootstrap_and_per_patient_summary_use_patient_units():
    from modules.metrics import patient_cluster_bootstrap, per_patient_metrics_frame

    truth = np.array(["N", "N", "V", "V", "S", "S"])
    pred = np.array(["N", "V", "V", "V", "S", "N"])
    patients = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
    records = np.array(["100", "100", "101", "101", "102", "102"])
    result = patient_cluster_bootstrap(truth, pred, patients, n_bootstrap=25, seed=3)
    assert result["status"] == "estimated"
    assert result["n_patients"] == 3
    assert result["resampling_unit"] == "held-out patient"
    frame = per_patient_metrics_frame("fixture", truth, pred, patients, records)
    assert frame["patient_id"].tolist() == ["p1", "p2", "p3"]
    assert frame["records"].tolist() == ["100", "101", "102"]
    assert frame["n_beats"].tolist() == [2, 2, 2]
