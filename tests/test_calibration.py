import numpy as np
import pytest

from modules.calibration import apply_temperature, fit_grouped_temperature_scaler, score_reliability_diagnostics


def test_score_reliability_reports_perfect_top_label_reliability():
    truth = np.array(["N", "V", "N", "V"])
    scores = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.1, 0.9],
    ])
    report = score_reliability_diagnostics(
        truth, scores, class_order=["N", "V"],
        score_semantics="test class scores", n_bins=5,
    )
    assert report["status"] == "computed"
    assert report["calibration_status"] == "held_out_reliability_diagnostics_only"
    assert report["metrics"]["top_label_expected_calibration_error"] == pytest.approx(0.15)
    assert report["metrics"]["multiclass_brier_score"] == pytest.approx(0.05)
    assert len(report["per_class"]) == 2
    assert report["score_semantics"] == "test class scores"


def test_score_reliability_normalises_non_unit_rows_and_handles_zero_rows():
    truth = ["N", "V"]
    scores = np.array([[8.0, 2.0], [0.0, 0.0]])
    report = score_reliability_diagnostics(truth, scores, class_order=["N", "V"], n_bins=10)
    assert report["metrics"]["mean_top_score"] == pytest.approx(0.65)
    assert report["metrics"]["multiclass_log_loss"] > 0


@pytest.mark.parametrize("scores", [np.array([[0.5, -0.5]]), np.array([[np.nan, 1.0]]), np.array([0.5, 0.5])])
def test_score_reliability_rejects_invalid_score_inputs(scores):
    with pytest.raises(ValueError):
        score_reliability_diagnostics(["N"], scores, class_order=["N", "V"])


class TinyGroupedModel:
    classes_ = None

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(np.asarray(y).astype(str))))
        self.centres_ = {label: float(X.loc[np.asarray(y).astype(str) == label, "x"].mean()) for label in self.classes_}
        return self

    def scores(self, X):
        logits = np.column_stack([-((X["x"].to_numpy(dtype=float) - self.centres_[label]) ** 2) for label in self.classes_])
        logits -= logits.max(axis=1, keepdims=True)
        exp_values = np.exp(logits)
        return exp_values / exp_values.sum(axis=1, keepdims=True)


def test_temperature_scaler_uses_grouped_oof_records_not_test_data():
    rows, labels, groups = [], [], []
    rng = np.random.default_rng(8)
    for label, centre in (("N", -1.0), ("S", 0.0), ("V", 1.0)):
        for record in range(3):
            for _ in range(12):
                rows.append({"x": rng.normal(centre, 0.55)})
                labels.append(label)
                groups.append(f"{label}_{record}")
    model = TinyGroupedModel()
    fitted = fit_grouped_temperature_scaler(
        model, __import__("pandas").DataFrame(rows), np.asarray(labels), np.asarray(groups),
        class_order=["N", "S", "V"], n_splits=3, seed=4,
    )
    assert fitted["status"] == "fitted"
    assert fitted["n_oof_samples"] == len(labels)
    assert fitted["n_record_groups"] == 9
    assert 0.05 <= fitted["temperature"] <= 10.0
    assert "training records only" in fitted["calibration_scope"]


def test_temperature_transform_changes_concentration_and_stays_normalised():
    scores = np.array([[0.8, 0.2], [0.6, 0.4]])
    colder = apply_temperature(scores, 0.5)
    warmer = apply_temperature(scores, 2.0)
    assert np.allclose(colder.sum(axis=1), 1.0)
    assert colder[0, 0] > scores[0, 0]
    assert warmer[0, 0] < scores[0, 0]


def test_grouped_temperature_scaling_skips_when_record_groups_are_missing():
    import pandas as pd
    result = fit_grouped_temperature_scaler(
        TinyGroupedModel(), pd.DataFrame({"x": [0.0, 1.0]}), np.array(["N", "V"]), None,
        class_order=["N", "S", "V"],
    )
    assert result["status"] == "not_fitted"
    assert result["temperature"] == 1.0
