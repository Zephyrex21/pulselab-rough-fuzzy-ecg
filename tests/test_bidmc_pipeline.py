import numpy as np

from modules.bidmc_pipeline import _annotation_rr, extract_record_features, BIDMCRecord, inventory


def test_inventory_is_explicit_when_dataset_is_not_installed(tmp_path):
    payload = inventory(tmp_path / "not_installed")
    assert payload["ready"] is False
    assert payload["records_expected"] == 53
    assert payload["records_found"] == 0


def test_annotation_rate_uses_breath_sample_intervals():
    samples = np.array([0, 250, 500, 750, 1000], dtype=int)
    rate, count = _annotation_rr(samples, 0, 1250, 125.0)
    assert count == 5
    assert abs(rate - 30.0) < 1e-6


def test_feature_extraction_keeps_dual_reference_labels_and_patient_group():
    fs = 125.0
    duration = 64.0
    time = np.arange(int(duration * fs)) / fs
    ecg = np.sin(2 * np.pi * 1.2 * time) + 0.05 * np.sin(2 * np.pi * 15 * time)
    ppg = 0.7 * np.sin(2 * np.pi * 1.2 * time - 0.4) + 0.03 * np.sin(2 * np.pi * 0.3 * time)
    resp = np.sin(2 * np.pi * 0.25 * time)
    breaths = np.arange(0, int(duration * fs), int(fs / 0.25))
    record = BIDMCRecord(
        record_id="bidmc01", patient_id="s01182", fs=fs, time_sec=time,
        ecg=ecg, ppg=ppg, resp=resp, ann1_samples=breaths,
        ann2_samples=breaths + 2, monitor=None, channels=["II", "PLETH", "RESP"],
    )
    full = extract_record_features(record, window_sec=32, stride_sec=16, include_full_fusion=True)
    assert len(full) == 3
    assert "ecg_mean" in full.columns and "ppg_mean" in full.columns and "resp_mean" in full.columns
    assert "ref_rr_ann1_bpm" in full.columns and "ref_rr_ann2_bpm" in full.columns
    assert (full["patient_id"] == "s01182").all()
    assert full["ref_rr_mean_bpm"].notna().all()
