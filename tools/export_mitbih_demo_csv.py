"""Export held-out MIT-BIH records as CSV files for saved-model upload demos.

Run from the repository root:
    python tools/export_mitbih_demo_csv.py --max-records 5

The script exports up to five minutes from the first records in the configured
patient-disjoint test split, preserves the two source ECG leads and native
sampling rate, and writes only derived demo CSVs under data/demo_ecg_csv/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def export_record(record_id: str, data_dir: Path, output_dir: Path, duration_seconds: float) -> Path:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "WFDB is required. Install project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    header = data_dir / f"{record_id}.hea"
    if not header.is_file():
        raise FileNotFoundError(f"Missing MIT-BIH header: {header}")

    record = wfdb.rdrecord(str(data_dir / record_id), physical=True)
    signal = np.asarray(record.p_signal, dtype=float)
    lead_names = list(record.sig_name or [])
    if signal.ndim != 2 or signal.shape[1] < 2:
        raise ValueError(f"Record {record_id} does not contain two ECG leads.")
    if len(lead_names) < 2:
        lead_names = [f"ECG_{i + 1}" for i in range(signal.shape[1])]

    # The upload API caps a signal at 10 minutes; export a bounded segment.
    max_samples = min(len(signal), int(round(duration_seconds * float(record.fs))))
    selected = signal[:max_samples, :2]
    if not np.isfinite(selected).all():
        raise ValueError(f"Record {record_id} contains non-finite samples in its first two leads.")

    output_dir.mkdir(parents=True, exist_ok=True)
    time_seconds = np.arange(len(selected), dtype=float) / float(record.fs)
    frame = pd.DataFrame({
        "time_s": time_seconds,
        str(lead_names[0]).strip().replace(" ", "_"): selected[:, 0],
        str(lead_names[1]).strip().replace(" ", "_"): selected[:, 1],
    })
    output_path = output_dir / f"mitdb_{record_id}_ecg.csv"
    frame.to_csv(output_path, index=False, float_format="%.8f")
    print(
        f"Exported {record_id}: {len(frame):,} samples, "
        f"{record.fs:g} Hz, leads {lead_names[0]} + {lead_names[1]} -> {output_path}"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", nargs="+",
        help="Optional record IDs. Every requested ID must belong to the configured patient-disjoint test split.",
    )
    parser.add_argument(
        "--max-records", type=int, default=5,
        help="Number of available held-out records to export when --records is omitted (default: 5).",
    )
    parser.add_argument("--duration-seconds", type=float, default=300.0, help="Duration exported per record (default: 300 seconds; API maximum: 600).")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "demo_ecg_csv")
    args = parser.parse_args()

    if args.max_records < 1:
        parser.error("--max-records must be at least 1.")
    if not 3.0 <= args.duration_seconds <= 600.0:
        parser.error("--duration-seconds must be between 3 and 600.")

    _, heldout_records = config.get_record_split("patient-disjoint")
    selected = args.records or heldout_records[:args.max_records]
    invalid = [record_id for record_id in selected if record_id not in heldout_records]
    if invalid:
        parser.error(
            "Only patient-disjoint held-out test records are accepted. "
            f"Not in the configured test split: {', '.join(invalid)}"
        )

    exported = []
    for record_id in selected:
        try:
            exported.append(export_record(record_id, args.data_dir, args.output_dir, args.duration_seconds))
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            print(f"Skipped {record_id}: {exc}", file=sys.stderr)

    if not exported:
        print(
            "No CSV files were exported. Check data/mitdb and the WFDB dependency.",
            file=sys.stderr,
        )
        return 1

    print(f"\nDone: {len(exported)} held-out CSV file(s) saved to {args.output_dir}")
    print("Use 360 Hz in the PulseLab upload form.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
