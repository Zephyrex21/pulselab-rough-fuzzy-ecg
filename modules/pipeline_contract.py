"""Compatibility contract for saved MIT-BIH models and live ECG inference.

A model may only be used with an input pipeline whose transformation contract
matches the one recorded at training time. Bump the individual component
versions when their transformation semantics change.
"""
from __future__ import annotations

from typing import Any

import config
from modules.feature_schema import FEATURE_SCHEMA_VERSION
from modules.feature_extraction import FEATURE_EXTRACTION_VERSION
from modules.preprocessing import PREPROCESSING_VERSION
from modules.rpeak_detection import RPEAK_DETECTOR_VERSION
from modules.segmentation import SEGMENTATION_VERSION

MITBIH_INFERENCE_CONTRACT_VERSION = "mitdb-input-pipeline-v1"


def current_mitbih_inference_contract() -> dict[str, Any]:
    """Return the exact input-transformation contract used by this source tree."""
    return {
        "contract_version": MITBIH_INFERENCE_CONTRACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "rpeak_detector_version": RPEAK_DETECTOR_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "segmentation_version": SEGMENTATION_VERSION,
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
        "sampling_frequency_hz": float(config.FS),
        "bandpass_hz": [float(value) for value in config.BANDPASS],
        "baseline_median_windows_sec": [float(value) for value in config.BASELINE_MEDIAN_WINDOWS],
        "notch_hz": float(config.NOTCH_HZ),
        "pre_r_seconds": float(config.PRE_R_SEC),
        "post_r_seconds": float(config.POST_R_SEC),
    }


def check_mitbih_inference_compatibility(manifest: dict[str, Any]) -> tuple[bool, str | None]:
    """Fail closed for legacy, annotation-guided or pipeline-mismatched runs."""
    metadata = manifest.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False, "The saved run manifest has invalid MIT-BIH metadata."
    if metadata.get("peak_source") != "detected":
        return False, (
            "This model was trained with reference annotations rather than automatic R-peak detection; "
            "annotation-guided runs are benchmark-only and cannot be used for uploaded-signal inference."
        )

    recorded = metadata.get("inference_pipeline_contract")
    if not isinstance(recorded, dict):
        return False, (
            "This historical model does not record an inference-pipeline compatibility contract. "
            "Retrain it with the current PulseLab pipeline before using uploaded-signal inference."
        )

    current = current_mitbih_inference_contract()
    mismatches = [
        key for key, expected in current.items()
        if recorded.get(key) != expected
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: trained={recorded.get(key, 'missing')!r}, current={current[key]!r}"
            for key in mismatches
        )
        return False, (
            "The saved model was trained with a different MIT-BIH input pipeline and is blocked from "
            f"uploaded-signal inference ({details}). Retrain the model with the current PulseLab pipeline."
        )
    return True, None
