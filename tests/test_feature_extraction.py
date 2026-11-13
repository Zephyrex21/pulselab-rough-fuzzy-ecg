from types import SimpleNamespace

import numpy as np
import pandas as pd

from modules import feature_extraction


def test_detected_peak_feature_rows_are_matched_one_to_one_and_labelled(monkeypatch):
    detected = np.asarray([102, 498, 750, 1200], dtype=int)
    fake_features = pd.DataFrame({
        "r_sample": detected,
        "rr_prev": [0.8, 0.9, 0.8, 0.9],
        "c0_r_amp": [1.0, 1.1, 0.2, 0.4],
    })
    monkeypatch.setattr(
        feature_extraction,
        "process_signal",
        lambda *args, **kwargs: {"features": fake_features.copy(), "rpeaks": detected.copy()},
    )
    record = SimpleNamespace(
        record_id="100",
        fs=360.0,
        signal=np.zeros((1600, 2), dtype=float),
        ann_samples=np.asarray([100, 500, 900, 1300], dtype=int),
        ann_symbols=np.asarray(["N", "A", "V", "F"]),
        ann_classes=np.asarray(["N", "S", "V", "F"]),
    )

    frame = feature_extraction.build_record_features(record, peak_source="detected")

    assert frame["r_sample"].tolist() == [102, 498]
    assert frame["label"].tolist() == ["N", "S"]
    assert frame["reference_sample"].tolist() == [100, 500]
    metrics = frame.attrs["detection_metrics"]
    assert metrics["tp"] == 2
    assert metrics["fp"] == 2
    assert metrics["fn"] == 2
    assert metrics["n_classifiable_matched"] == 2
    assert metrics["patient_id"] == "subject_100"
