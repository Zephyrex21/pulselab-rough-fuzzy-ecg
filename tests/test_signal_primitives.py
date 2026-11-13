import numpy as np

from modules.feature_extraction import rr_features
from modules.rpeak_detection import match_detections
from modules.segmentation import segment_beats
from modules.feature_schema import feature_names


def test_rpeak_matching_is_one_to_one():
    result = match_detections([100, 105, 300], [102, 302], fs=360, tol=0.02)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert len(result["matched_pairs"]) == 2


def test_segment_beats_returns_fixed_length_windows_and_valid_mask():
    signal = np.arange(1000, dtype=float)
    beats, valid = segment_beats(signal, [9, 500, 995], fs=100, pre=0.1, post=0.2)
    assert valid.tolist() == [False, True, False]
    assert beats.shape == (1, 30)
    assert beats[0, 10] == 500


def test_python_feature_schema_is_47_columns_for_two_channels():
    names = feature_names(2)
    assert len(names) == 47
    assert len(names) == len(set(names))
    assert names[:7] == ["rr_prev", "rr_next", "rr_mean", "rr_std", "rr_ratio_prev", "rr_ratio_next", "heart_rate"]


def test_rr_features_are_finite_and_aligned_to_rpeaks():
    result = rr_features(np.array([100, 460, 820, 1180]), fs=360)
    assert len(result) == 4
    assert np.isfinite(result.to_numpy()).all()


def test_build_record_features_detected_mode_labels_only_one_to_one_matches(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    import modules.feature_extraction as fe

    ann = np.asarray([700, 1300, 1900, 2500, 3100], dtype=int)
    detected = np.asarray([700, 1300, 1900, 2200, 3100], dtype=int)  # one FP and one missed beat
    rng = np.random.default_rng(3)
    signal = rng.normal(0.0, 0.01, size=(4000, 2))
    for sample in ann:
        signal[sample - 2:sample + 3, :] += np.asarray([1.2, 0.8])

    record = SimpleNamespace(
        record_id="100", fs=360.0, signal=signal,
        ann_samples=ann,
        ann_symbols=np.asarray(["N", "V", "A", "N", "F"]),
        ann_classes=np.asarray(["N", "V", "S", "N", "F"]),
    )
    monkeypatch.setattr(fe.rd, "detect_rpeaks", lambda x, fs: detected.copy())
    monkeypatch.setattr(
        fe, "channel_features",
        lambda beats, r0, fs: pd.DataFrame({name: np.arange(len(beats), dtype=float) for name in fe.CH_FEATURES}),
    )

    result = fe.build_record_features(record, peak_source="detected")
    diagnostics = result.attrs["detection_metrics"]
    assert diagnostics["tp"] == 4
    assert diagnostics["fp"] == 1
    assert diagnostics["fn"] == 1
    assert len(result) == 4
    assert set(result["label"]) == {"N", "S", "V", "F"}
    assert set(result["patient_id"]) == {"subject_100"}
    assert result["peak_source"].eq("detected").all()
