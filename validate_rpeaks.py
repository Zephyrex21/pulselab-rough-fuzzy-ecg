"""Run separate one-to-one R-peak detector validation on MIT-BIH records."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import config
from modules import data_loader
from modules.validation import run_rpeak_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download missing validation records")
    parser.add_argument("--quick", action="store_true", help="validate the first 6 strict test records")
    parser.add_argument("--split-protocol", choices=config.SPLIT_PROTOCOLS, default="patient-disjoint")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    args = parser.parse_args()
    if args.tolerance_ms < 0:
        parser.error("--tolerance-ms must be non-negative")

    _, test_records = config.get_record_split(args.split_protocol)
    record_ids = test_records[:6] if args.quick else test_records
    if args.download:
        try:
            data_loader.download_records(record_ids, data_dir=args.data_dir,
                                         progress=lambda p, m: print(f"[{p:6.1%}] {m}", flush=True))
        except ImportError as exc:
            parser.error(f"WFDB is required to download MIT-BIH records: {exc}")

    missing = sorted(set(record_ids) - set(data_loader.available_records(args.data_dir)))
    if missing:
        print("Missing records:", ", ".join(missing))
        print("Run: py validate_rpeaks.py --download --quick")
        return 2

    output_dir = args.output_dir or config.EXPERIMENTS_DIR / (
        "rpeak_validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    )
    result = run_rpeak_validation(
        record_ids, data_dir=args.data_dir, output_dir=output_dir,
        tolerance_seconds=args.tolerance_ms / 1000.0,
        progress=lambda p, m: print(f"[{p:6.1%}] {m}", flush=True),
    )
    print(f"\nR-peak validation saved to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
