"""BIDMC multimodal physiological-signal pipeline.

This module reads the official BIDMC CSV distribution. It keeps ECG, PPG and
impedance-respiration processing separate until window-level feature fusion.
Respiratory targets are derived from the two independent manual breath
annotators; they are never silently replaced by monitor numerics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt
from scipy.stats import kurtosis, skew

BIDMC_RECORD_IDS = [f"{i:02d}" for i in range(1, 54)]
EXPECTED_FS = 125.0
DEFAULT_WINDOW_SEC = 32.0
DEFAULT_STRIDE_SEC = 16.0
SIGNAL_COLUMNS = {"ecg": "II", "ppg": "PLETH", "resp": "RESP"}


class BIDMCError(ValueError):
    """Raised when a BIDMC dataset or record is invalid or incomplete."""


@dataclass
class BIDMCRecord:
    record_id: str
    patient_id: str
    fs: float
    time_sec: np.ndarray
    ecg: np.ndarray
    ppg: np.ndarray
    resp: np.ndarray
    ann1_samples: np.ndarray
    ann2_samples: np.ndarray
    monitor: pd.DataFrame | None
    channels: list[str]

    @property
    def n_samples(self) -> int:
        return int(len(self.time_sec))

    @property
    def duration_sec(self) -> float:
        return float(self.time_sec[-1] - self.time_sec[0]) if self.n_samples else 0.0


def find_csv_dir(dataset_root: str | Path) -> Path | None:
    """Locate bidmc_csv under common PhysioNet ZIP extraction layouts."""
    root = Path(dataset_root).expanduser()
    if not root.exists():
        return None
    candidates = [root, root / "bidmc_csv", root / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("bidmc_*_Signals.csv")):
            return candidate
    try:
        return next((p for p in root.rglob("bidmc_csv") if p.is_dir() and any(p.glob("bidmc_*_Signals.csv"))), None)
    except OSError:
        return None


def _record_paths(csv_dir: Path, record_id: str) -> dict[str, Path]:
    rid = f"{int(record_id):02d}"
    return {
        "signals": csv_dir / f"bidmc_{rid}_Signals.csv",
        "breaths": csv_dir / f"bidmc_{rid}_Breaths.csv",
        "numerics": csv_dir / f"bidmc_{rid}_Numerics.csv",
    }


def _patient_id(csv_dir: Path, record_id: str) -> str:
    header = csv_dir.parent / f"bidmc{int(record_id):02d}.hea"
    if header.is_file():
        try:
            content = header.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"(?:matched/)?(s\d{4,8})/", content, flags=re.IGNORECASE)
            if match:
                return match.group(1).lower()
        except OSError:
            pass
    # The public CSV-only archive does not expose the MIMIC-II subject ID in
    # every layout. Keep a stable record-level group rather than guessing.
    return f"bidmc_record_{int(record_id):02d}"


def inventory(dataset_root: str | Path) -> dict:
    csv_dir = find_csv_dir(dataset_root)
    if csv_dir is None:
        return {
            "dataset": "BIDMC PPG and Respiration Dataset",
            "dataset_version": "1.0.0",
            "dataset_root": str(Path(dataset_root)),
            "csv_directory": None,
            "records_found": 0,
            "records_expected": 53,
            "patient_groups": 0,
            "ready": False,
            "records": [],
            "message": "BIDMC CSV files were not found. Extract the PhysioNet ZIP beneath data/bidmc/ so that bidmc_csv/ is present.",
        }
    rows = []
    for rid in BIDMC_RECORD_IDS:
        paths = _record_paths(csv_dir, rid)
        if not paths["signals"].is_file():
            continue
        missing = [name for name, path in paths.items() if name in {"signals", "breaths"} and not path.is_file()]
        channels = []
        try:
            channels = list(pd.read_csv(paths["signals"], nrows=0).columns)
        except Exception:
            missing.append("unreadable_signals")
        patient = _patient_id(csv_dir, rid)
        rows.append({
            "record_id": f"bidmc{rid}",
            "record_number": int(rid),
            "patient_id": patient,
            "signals_available": paths["signals"].is_file(),
            "breath_annotations_available": paths["breaths"].is_file(),
            "monitor_numerics_available": paths["numerics"].is_file(),
            "channels": channels,
            "missing_files": missing,
        })
    unique_patients = len({r["patient_id"] for r in rows})
    complete = sum(not r["missing_files"] for r in rows)
    return {
        "dataset": "BIDMC PPG and Respiration Dataset",
        "dataset_version": "1.0.0",
        "dataset_root": str(Path(dataset_root)),
        "csv_directory": str(csv_dir),
        "records_found": len(rows),
        "records_expected": 53,
        "records_complete": complete,
        "patient_groups": unique_patients,
        "ready": len(rows) > 0 and complete == len(rows),
        "sampling_frequency_hz": EXPECTED_FS,
        "recording_duration_minutes": 8,
        "signal_channels_expected": ["II", "PLETH", "RESP"],
        "numerical_channels_expected": ["HR", "PULSE", "RESP", "SpO2"],
        "annotation_sources": ["ann1", "ann2"],
        "records": rows,
        "message": "BIDMC signals and manual breath annotations are indexed. No BIDMC model is trained yet.",
    }


def _column_name(columns, required: str) -> str:
    wanted = re.sub(r"[^a-z0-9]", "", required.lower())
    for name in columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
        if normalized == wanted:
            return str(name)
    raise BIDMCError(f"Required BIDMC signal column '{required}' was not found. Available columns: {list(columns)}")


def _read_annotations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.array([], dtype=int), np.array([], dtype=int)
    frame = pd.read_csv(path)
    cols = list(frame.columns)
    c1 = next((c for c in cols if "ann1" in str(c).lower()), None)
    c2 = next((c for c in cols if "ann2" in str(c).lower()), None)
    if c1 is None or c2 is None:
        raise BIDMCError(f"Breath annotation file must contain separate ann1 and ann2 columns: {path.name}")
    a1 = pd.to_numeric(frame[c1], errors="coerce").dropna().to_numpy(dtype=int)
    a2 = pd.to_numeric(frame[c2], errors="coerce").dropna().to_numpy(dtype=int)
    return np.sort(a1), np.sort(a2)


def load_record(record_id: str | int, dataset_root: str | Path) -> BIDMCRecord:
    csv_dir = find_csv_dir(dataset_root)
    if csv_dir is None:
        raise BIDMCError("BIDMC dataset not found. Extract the official ZIP beneath data/bidmc/.")
    try:
        rid_num = int(str(record_id).lower().replace("bidmc", ""))
    except ValueError as exc:
        raise BIDMCError(f"Invalid BIDMC record identifier: {record_id!r}") from exc
    if not 1 <= rid_num <= 53:
        raise BIDMCError("BIDMC record number must be between 01 and 53.")
    rid = f"{rid_num:02d}"
    paths = _record_paths(csv_dir, rid)
    if not paths["signals"].is_file():
        raise BIDMCError(f"Signal CSV for bidmc{rid} is missing: {paths['signals']}")
    frame = pd.read_csv(paths["signals"])
    time_col = next((c for c in frame.columns if str(c).strip().lower().startswith("time")), None)
    if time_col is None:
        raise BIDMCError(f"{paths['signals'].name} has no Time [s] column.")
    time = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=float)
    if len(time) < int(EXPECTED_FS * 30) or not np.isfinite(time).all():
        raise BIDMCError(f"{paths['signals'].name} does not contain a usable signal recording.")
    # The official BIDMC CSV timestamps are rounded to 0.01 s even though
    # the underlying waveforms are sampled at 125 Hz. That rounding creates
    # repeated timestamp values (and a median diff of 0 or 0.01), so sampling
    # rate must be inferred from sample count / total elapsed time, not median
    # timestamp spacing.
    elapsed = float(time[-1] - time[0])
    if elapsed <= 0 or not np.isfinite(elapsed) or np.any(np.diff(time) < 0):
        raise BIDMCError("BIDMC timestamps are invalid or not monotonically ordered.")
    fs = float((len(time) - 1) / elapsed)
    if not 120 <= fs <= 130:
        raise BIDMCError(f"Unexpected BIDMC effective waveform sampling rate: {fs:.2f} Hz (expected approximately 125 Hz).")
    channels = list(map(str, frame.columns))
    normalized = {}
    for key, name in SIGNAL_COLUMNS.items():
        source_col = _column_name(frame.columns, name)
        values = pd.to_numeric(frame[source_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < max(100, int(0.8 * len(values))):
            raise BIDMCError(f"Signal channel {name} contains too many missing or invalid samples.")
        if not finite.all():
            idx = np.arange(len(values))
            values = np.interp(idx, idx[finite], values[finite])
        normalized[key] = values
    ann1, ann2 = _read_annotations(paths["breaths"])
    monitor = pd.read_csv(paths["numerics"]) if paths["numerics"].is_file() else None
    return BIDMCRecord(
        record_id=f"bidmc{rid}", patient_id=_patient_id(csv_dir, rid), fs=fs,
        time_sec=time, ecg=normalized["ecg"], ppg=normalized["ppg"], resp=normalized["resp"],
        ann1_samples=ann1, ann2_samples=ann2, monitor=monitor, channels=channels,
    )


def _filter(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyquist = fs / 2
    high = min(high, nyquist * 0.95)
    low = max(low, 0.001)
    if low >= high or len(x) < 40:
        return np.asarray(x, dtype=float) - float(np.nanmean(x))
    sos = butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, np.asarray(x, dtype=float))
    except ValueError:
        return np.asarray(x, dtype=float) - float(np.nanmean(x))


def _signal_features(x: np.ndarray, fs: float, prefix: str, band: tuple[float, float], peak_min_distance_sec: float) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(32, int(fs * 4)):
        raise BIDMCError(f"Window too short for {prefix} feature extraction.")
    filt = _filter(x, fs, *band)
    q25, q75 = np.quantile(filt, [0.25, 0.75])
    centered = filt - np.mean(filt)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    freqs, power = periodogram(centered, fs=fs, scaling="density")
    valid = (freqs >= band[0]) & (freqs <= band[1])
    peak_freq = float(freqs[valid][np.argmax(power[valid])]) if valid.any() else 0.0
    lower_mask = (freqs >= band[0]) & (freqs < (band[0] + band[1]) / 2)
    upper_mask = (freqs >= (band[0] + band[1]) / 2) & (freqs <= band[1])
    peaks, _ = find_peaks(filt, distance=max(1, int(fs * peak_min_distance_sec)), prominence=max(float(np.std(filt) * 0.18), 1e-6))
    interval = np.diff(peaks) / fs if len(peaks) > 1 else np.array([], dtype=float)
    rate = float(60 / np.median(interval)) if len(interval) else np.nan
    return {
        f"{prefix}_mean": float(np.mean(filt)),
        f"{prefix}_std": float(np.std(filt)),
        f"{prefix}_median": float(np.median(filt)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(np.ptp(filt)),
        f"{prefix}_rms": rms,
        f"{prefix}_line_length": float(np.mean(np.abs(np.diff(filt)))) if len(filt) > 1 else 0.0,
        f"{prefix}_skew": float(skew(filt, bias=False)) if np.std(filt) > 1e-12 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(filt, bias=False)) if np.std(filt) > 1e-12 else 0.0,
        f"{prefix}_dominant_hz": peak_freq,
        f"{prefix}_low_band_power": float(np.trapezoid(power[lower_mask], freqs[lower_mask])) if lower_mask.sum() > 1 else 0.0,
        f"{prefix}_high_band_power": float(np.trapezoid(power[upper_mask], freqs[upper_mask])) if upper_mask.sum() > 1 else 0.0,
        f"{prefix}_peak_rate_bpm": rate,
        f"{prefix}_interval_sd_ms": float(np.std(interval) * 1000) if len(interval) > 1 else np.nan,
        f"{prefix}_peak_count": float(len(peaks)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 10 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _annotation_rr(samples: np.ndarray, start_sample: int, end_sample: int, fs: float) -> tuple[float, int]:
    events = samples[(samples >= start_sample) & (samples < end_sample)]
    if len(events) < 2:
        return np.nan, int(len(events))
    intervals = np.diff(events) / fs
    intervals = intervals[(intervals > 0.5) & (intervals < 10.0)]
    if not len(intervals):
        return np.nan, int(len(events))
    return float(60.0 / np.median(intervals)), int(len(events))


def extract_record_features(
    record: BIDMCRecord,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    include_full_fusion: bool = True,
) -> pd.DataFrame:
    """Extract synchronized ECG/PPG/RESP features and two-annotator RR targets."""
    if not 16 <= float(window_sec) <= 120:
        raise BIDMCError("Window length must be between 16 and 120 seconds.")
    if not 4 <= float(stride_sec) <= float(window_sec):
        raise BIDMCError("Stride must be between 4 seconds and the window length.")
    samples_per_window = int(round(window_sec * record.fs))
    samples_per_stride = int(round(stride_sec * record.fs))
    rows = []
    if record.n_samples < samples_per_window:
        raise BIDMCError(f"{record.record_id} is shorter than the selected window.")
    for start in range(0, record.n_samples - samples_per_window + 1, samples_per_stride):
        end = start + samples_per_window
        ecg = record.ecg[start:end]
        ppg = record.ppg[start:end]
        resp = record.resp[start:end]
        feature = {}
        feature.update(_signal_features(ecg, record.fs, "ecg", (0.5, 40.0), 0.30))
        feature.update(_signal_features(ppg, record.fs, "ppg", (0.5, 8.0), 0.30))
        if include_full_fusion:
            feature.update(_signal_features(resp, record.fs, "resp", (0.05, 1.0), 1.0))
            feature["ppg_resp_corr"] = _corr(_filter(ppg, record.fs, 0.05, 0.7), _filter(resp, record.fs, 0.05, 0.7))
            feature["ecg_resp_corr"] = _corr(_filter(ecg, record.fs, 0.05, 0.7), _filter(resp, record.fs, 0.05, 0.7))
        feature["ecg_ppg_rate_delta_bpm"] = (
            float(feature["ecg_peak_rate_bpm"] - feature["ppg_peak_rate_bpm"])
            if np.isfinite(feature["ecg_peak_rate_bpm"]) and np.isfinite(feature["ppg_peak_rate_bpm"]) else np.nan
        )
        feature["ecg_ppg_corr"] = _corr(_filter(ecg, record.fs, 0.5, 8.0), _filter(ppg, record.fs, 0.5, 8.0))
        rr1, n1 = _annotation_rr(record.ann1_samples, start, end, record.fs)
        rr2, n2 = _annotation_rr(record.ann2_samples, start, end, record.fs)
        valid = [v for v in (rr1, rr2) if np.isfinite(v)]
        start_sec = float(record.time_sec[start])
        end_sec = float(record.time_sec[end - 1])
        feature.update({
            "record_id": record.record_id,
            "patient_id": record.patient_id,
            "window_start_sec": start_sec,
            "window_end_sec": end_sec,
            "ref_rr_ann1_bpm": rr1,
            "ref_rr_ann2_bpm": rr2,
            "ref_rr_mean_bpm": float(np.mean(valid)) if valid else np.nan,
            "annotator_difference_bpm": abs(rr1 - rr2) if np.isfinite(rr1) and np.isfinite(rr2) else np.nan,
            "ann1_breath_count": n1,
            "ann2_breath_count": n2,
            "window_quality": float(np.mean(np.isfinite(ecg) & np.isfinite(ppg) & np.isfinite(resp))),
        })
        rows.append(feature)
    return pd.DataFrame(rows)


def extract_preview(record: BIDMCRecord, start_sec: float = 0.0, duration_sec: float = 20.0, max_points: int = 1600) -> dict:
    start_sec = max(0.0, min(float(start_sec), max(record.duration_sec, 0.0)))
    duration_sec = max(2.0, min(float(duration_sec), 60.0))
    start = min(record.n_samples - 1, int(start_sec * record.fs))
    end = min(record.n_samples, int((start_sec + duration_sec) * record.fs))
    indices = np.arange(start, end)
    if len(indices) > max_points:
        indices = indices[:: int(np.ceil(len(indices) / max_points))]
    return {
        "record_id": record.record_id,
        "patient_id": record.patient_id,
        "sampling_frequency_hz": record.fs,
        "start_sec": float(record.time_sec[start]),
        "duration_sec": float(record.time_sec[min(end - 1, record.n_samples - 1)] - record.time_sec[start]),
        "time_sec": record.time_sec[indices].round(4).tolist(),
        "ecg": record.ecg[indices].round(6).tolist(),
        "ppg": record.ppg[indices].round(6).tolist(),
        "resp": record.resp[indices].round(6).tolist(),
        "channel_names": record.channels,
        "annotations": {
            "ann1_sec": (record.ann1_samples[(record.ann1_samples >= start) & (record.ann1_samples < end)] / record.fs).round(4).tolist(),
            "ann2_sec": (record.ann2_samples[(record.ann2_samples >= start) & (record.ann2_samples < end)] / record.fs).round(4).tolist(),
        },
    }


def build_feature_dataset(
    dataset_root: str | Path,
    record_ids: list[str] | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create full multimodal and non-leaky RR-estimation feature tables."""
    inv = inventory(dataset_root)
    if not inv["records_found"]:
        raise BIDMCError(inv["message"])
    available = {r["record_id"] for r in inv["records"] if not r["missing_files"]}
    selected = record_ids or sorted(available)
    selected = [f"bidmc{int(str(r).lower().replace('bidmc', '')):02d}" for r in selected]
    missing = [r for r in selected if r not in available]
    if missing:
        raise BIDMCError(f"Requested BIDMC records are missing required files: {missing}")
    full_tables = []
    for i, record_id in enumerate(selected):
        record = load_record(record_id, dataset_root)
        table = extract_record_features(record, window_sec, stride_sec, include_full_fusion=True)
        full_tables.append(table)
        if progress:
            progress((i + 1) / len(selected), f"Extracted {record_id} ({len(table)} synchronized windows)")
    full = pd.concat(full_tables, ignore_index=True) if full_tables else pd.DataFrame()
    # For respiratory-rate estimation, RESP-derived predictors (and any
    # correlations involving RESP) are excluded to prevent target leakage.
    nonleaky_drop = [c for c in full.columns if c.startswith("resp_") or c in {"ppg_resp_corr", "ecg_resp_corr"}]
    nonleaky = full.drop(columns=nonleaky_drop, errors="ignore")
    return full, nonleaky
