import numpy as np
import pandas as pd
import pytest

from modules import explainability


class FeatureSensitiveModel:
    classes_ = np.array(["N", "V"])
    feature_names_ = ["rr_prev", "qrs_width"]

    def scores(self, frame):
        value = np.clip(frame["rr_prev"].to_numpy(dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - value, value])

    def predict(self, frame):
        return self.classes_[np.argmax(self.scores(frame), axis=1)]


def test_local_reference_ablation_identifies_score_sensitive_feature():
    model = FeatureSensitiveModel()
    explanation = explainability.local_reference_ablation(
        model,
        pd.Series({"rr_prev": 0.9, "qrs_width": 0.4}),
        feature_columns=["rr_prev", "qrs_width"],
        reference_values={"rr_prev": 0.1, "qrs_width": 0.4},
        target_class="V",
    )
    assert explanation["available"] is True
    assert explanation["method"] == "single-feature reference ablation"
    assert explanation["features"][0]["feature"] == "rr_prev"
    assert explanation["features"][0]["score_delta"] == pytest.approx(0.8)
    assert explanation["features"][0]["direction"] == "supports_predicted_class_vs_reference"


def test_feature_reference_is_training_quantile_summary_and_drift_is_descriptive():
    train = pd.DataFrame({"f1": np.arange(100, dtype=float), "f2": np.ones(100)})
    reference = explainability.build_feature_reference(train, ["f1", "f2"])
    assert reference["f1"]["q01"] < reference["f1"]["median"] < reference["f1"]["q99"]
    inference = pd.DataFrame({"f1": [0.0, 50.0, 500.0], "f2": [1.0, 1.0, 1.0]})
    summary, per_beat, per_feature = explainability.assess_feature_drift(inference, reference)
    assert summary["available"] is True
    assert per_beat is not None and len(per_beat) == 3
    assert per_feature.iloc[0]["feature"] == "f1"
    assert summary["outside_value_fraction"] > 0
    assert summary["high_drift_flag"] is True
    assert "not a diagnosis" not in summary["limitations"][0].lower()


def test_global_feature_summary_reports_unavailable_for_plain_model():
    summary = explainability.global_feature_summary(FeatureSensitiveModel())
    assert summary["available"] is False
    assert summary["features"] == []


def test_ablation_requires_reference_features():
    result = explainability.local_reference_ablation(
        FeatureSensitiveModel(), {"rr_prev": 0.4, "qrs_width": 0.8},
        feature_columns=["rr_prev", "qrs_width"], reference_values={},
    )
    assert result["available"] is False
