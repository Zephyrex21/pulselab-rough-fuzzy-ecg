"""Audit MIT-BIH files, AAMI labels, class balance and split leakage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import config
from modules import data_loader
from modules.dataset_audit import audit_dataset, write_audit_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download missing records before auditing")
    parser.add_argument("--quick", action="store_true", help="audit the 8/6-record engineering subset")
    parser.add_argument("--split-protocol", choices=config.SPLIT_PROTOCOLS, default="patient-disjoint")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 hashing of local WFDB files")
    args = parser.parse_args()

    train, test = config.get_record_split(args.split_protocol)
    if args.quick:
        train, test = train[:8], test[:6]
    requested = train + test

    if args.download:
        try:
            data_loader.download_records(requested, data_dir=args.data_dir,
                                         progress=lambda p, m: print(f"[{p:6.1%}] {m}", flush=True))
        except ImportError as exc:
            parser.error(f"WFDB is required to download MIT-BIH records: {exc}")

    report = audit_dataset(
        requested, data_dir=args.data_dir, train_record_ids=train, test_record_ids=test,
        split_protocol=args.split_protocol, hash_files=not args.no_hash,
    )
    output = args.output_dir or config.EXPERIMENTS_DIR / (
        "dataset_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    )
    output = write_audit_report(report, output)
    print(f"\nAudit saved to: {output}")
    print(f"Records OK: {report['n_records_ok']}/{report['n_records_requested']}")
    print(f"Missing/failed: {report['n_records_failed']}")
    print(f"Patient-disjoint: {report['patient_disjoint']}")
    print("Class counts:")
    for split, counts in report["class_counts"].items():
        if split != "other":
            print(f"  {split}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for warning in report["warnings"]:
        print("WARNING:", warning)
    return 0 if report["n_records_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
