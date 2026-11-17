"""Reproducible end-to-end MIT-BIH experiment pipeline for PulseLab."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

import config
from modules import baseline_models, calibration, data_loader, explainability, feature_extraction
from modules.feature_schema import FEATURE_SCHEMA_VERSION, feature_names as canonical_feature_names
from modules.pipeline_contract import current_mitbih_inference_contract
from modules.hybrid_model import FuzzyOnlyModel, HybridRoughFuzzy
from modules.metrics import (
    align_class_scores, evaluate_predictions, per_class_frame,
    per_record_metrics_frame, per_patient_metrics_frame,
    record_cluster_bootstrap, patient_cluster_bootstrap,
)

Progress = Callable[[float, str], None] | None


def _notify(progress: Progress, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(np.clip(fraction, 0.0, 1.0)), message)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "model"


def _write_research_report(run_dir: Path, summary: pd.DataFrame, manifest: dict) -> None:
    """Create a human-readable report alongside machine-readable artifacts."""
    metadata = manifest.get("metadata", {})
    lines = [
        "# PulseLab experiment report",
        "",
        f"- **Run ID:** `{manifest['run_id']}`",
        f"- **Created (UTC):** {manifest['created_at_utc']}",
        f"- **Dataset:** {metadata.get('dataset', 'Provided feature tables')}",
        f"- **Split protocol:** `{metadata.get('split_policy', 'custom')}`",
        f"- **Patient-disjoint:** `{metadata.get('patient_disjoint', 'not specified')}`",
        f"- **Peak source:** `{metadata.get('peak_source', 'pre-extracted features')}`",
        f"- **Training beats:** {manifest['n_train']:,}",
        f"- **Testing beats:** {manifest['n_test']:,}",
        f"- **Held-out patients:** {manifest.get('n_test_patients', 'not available')}",
        f"- **Input features:** {manifest['n_features']}",
        "",
        "> Research prototype only. Not clinically validated. Do not use for diagnosis or treatment decisions.",
        "",
        "## Model comparison",
        "",
        "| " + " | ".join(map(str, summary.columns)) + " |",
        "| " + " | ".join(["---"] * len(summary.columns)) + " |",
        *["| " + " | ".join(
            (f"{value:.4f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value))
            for value in row
        ) + " |" for row in summary.itertuples(index=False, name=None)],
        "",
        "## Train / test class distribution",
        "",
        "| AAMI class | Train | Test |",
        "|---|---:|---:|",
    ]
    for label in config.CLASSES:
        lines.append(
            f"| {label} | {manifest['train_class_distribution'].get(label, 0):,} | "
            f"{manifest['test_class_distribution'].get(label, 0):,} |"
        )
    lines.extend(["", "## Score calibration and reliability", ""])
    lines.append("| Model | Calibration status | Temperature | Held-out Brier | Held-out log loss | Top-label ECE |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for model_name, model_metrics in (manifest.get("metrics") or {}).items():
        calibration_result = model_metrics.get("score_calibration") or {}
        reliability = (model_metrics.get("score_reliability") or {}).get("metrics") or {}
        temperature = calibration_result.get("temperature")
        lines.append(
            f"| {model_name} | {calibration_result.get('status', 'not available')} | "
            f"{float(temperature):.4f} | "
            f"{float(reliability.get('multiclass_brier_score', float('nan'))):.4f} | "
            f"{float(reliability.get('multiclass_log_loss', float('nan'))):.4f} | "
            f"{float(reliability.get('top_label_expected_calibration_error', float('nan'))):.4f} |"
        )
    lines.extend(["", "## Patient-level robustness", ""])
    lines.append("| Model | Held-out patients | Accuracy 95% interval | Macro-F1 (all AAMI) 95% interval |")
    lines.append("|---|---:|---:|---:|")
    for model_name, model_metrics in (manifest.get("metrics") or {}).items():
        cluster = model_metrics.get("patient_cluster_bootstrap") or {}
        acc_ci = cluster.get("accuracy_95ci") or [None, None]
        f1_ci = cluster.get("macro_f1_all_aami_95ci") or [None, None]
        acc_text = f"{float(acc_ci[0]):.4f}–{float(acc_ci[1]):.4f}" if all(value is not None for value in acc_ci) else "not estimated"
        f1_text = f"{float(f1_ci[0]):.4f}–{float(f1_ci[1]):.4f}" if all(value is not None for value in f1_ci) else "not estimated"
        lines.append(f"| {model_name} | {cluster.get('n_patients', '—')} | {acc_text} | {f1_text} |")
    lines.extend([
        "",
        "Patient-cluster bootstrap samples whole held-out patients rather than individual beats. The intervals describe variation among patients in this split; they do not establish external generalisation.",
        "",
        "Temperature scaling, when fitted, uses grouped out-of-fold training-record predictions and fold-specific training-only imputation. Held-out labels are used only for the reliability diagnostics above.",
        "These are internal score diagnostics; they do not establish external or clinical calibration. Fuzzy and SVM score vectors retain model-specific semantics.",
        "",
        "## Protocol notes", ""])
    lines.extend(f"- {note}" for note in manifest.get("notes", []))
    if manifest.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in manifest["warnings"])
    if metadata.get("detection_metrics"):
        detector = metadata["detection_metrics"]
        test_detection = detector.get("testing_records", {})
        lines.extend([
            "", "## Test-set R-peak detection summary", "",
            f"- Micro sensitivity: {test_detection.get('micro_sensitivity', float('nan')):.4f}",
            f"- Micro PPV: {test_detection.get('micro_ppv', float('nan')):.4f}",
            f"- Micro F1: {test_detection.get('micro_f1', float('nan')):.4f}",
            "- These are detection metrics, not classification metrics.",
        ])
    (run_dir / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _feature_columns(frame: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    if requested is not None:
        columns = list(requested)
    else:
        reserved = {"record", "label", "symbol", "r_sample", "record_id", "patient_id",
                    "peak_source", "reference_sample", "detection_error_ms"}
        columns = [c for c in frame.columns if c not in reserved and pd.api.types.is_numeric_dtype(frame[c])]
    if not columns:
        raise ValueError("No numeric feature columns were found")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns are missing from the table: {missing}")
    return columns


def _clean_feature_tables(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]):
    for label, frame in (("training", train_df), ("testing", test_df)):
        if "label" not in frame.columns:
            raise ValueError(f"The {label} feature table must contain a 'label' column")
        if frame.empty:
            raise ValueError(f"The {label} feature table is empty")
    y_train = train_df["label"].astype(str).to_numpy()
    y_test = test_df["label"].astype(str).to_numpy()
    invalid = sorted((set(y_train) | set(y_test)) - set(config.CLASSES))
    if invalid:
        raise ValueError(f"Unexpected AAMI labels in feature tables: {invalid}")

    X_train = train_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    X_test = test_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    medians = X_train.median(axis=0).fillna(0.0)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Feature cleaning failed to produce finite numeric values")
    return X_train, y_train, X_test, y_test


def _model_suite(include_baselines: bool = True):
    models = [
        FuzzyOnlyModel(),
        HybridRoughFuzzy(),
    ]
    if include_baselines:
        models.extend([baseline_models.RandomForestModel(), baseline_models.SVMModel()])
    return models


def run_feature_experiment(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
    output_root: Path | str | None = None,
    progress: Progress = None,
    include_baselines: bool = True,
    save_models: bool = True,
    run_name: str | None = None,
    metadata: dict | None = None,
    model_instances=None,
    allow_patient_overlap: bool = False,
) -> Path:
    """Fit/evaluate the model suite on already-extracted, record-wise feature tables.

    This lower-level entry point is also used by tests, allowing the complete
    training/evaluation path to be exercised without downloading MIT-BIH data.
    """
    train_df = train_features.copy()
    test_df = test_features.copy()
    for identity_column, description in (("patient_id", "patient"), ("subject_id", "subject"), ("record", "record")):
        if identity_column in train_df.columns and identity_column in test_df.columns:
            train_ids = set(train_df[identity_column].astype(str))
            test_ids = set(test_df[identity_column].astype(str))
            overlap = sorted(train_ids & test_ids)
            legacy_pair_only = identity_column == "patient_id" and allow_patient_overlap and overlap == ["subject_201_202"]
            if overlap and not legacy_pair_only:
                raise ValueError(f"{description.title()}-wise leakage detected; IDs occur in both splits: {overlap}")
    columns = _feature_columns(train_df, feature_columns)
    test_missing = [c for c in columns if c not in test_df.columns]
    if test_missing:
        raise ValueError(f"Testing feature table is missing training columns: {test_missing}")
    X_train, y_train, X_test, y_test = _clean_feature_tables(train_df, test_df, columns)

    run_id = run_name or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
    if Path(run_id).name != run_id or run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_name may contain only letters, numbers, dot, underscore and hyphen")
    root = Path(output_root) if output_root is not None else config.EXPERIMENTS_DIR
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir = config.MODELS_DIR / run_id
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=False)

    # Keep an un-imputed copy for grouped OOF calibration. The calibrator fits
    # its own imputation medians inside each training fold; final model fitting
    # below continues to use the outer-training medians from _clean_feature_tables.
    X_train_calibration = train_features.loc[:, columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    all_models = list(model_instances) if model_instances is not None else _model_suite(include_baselines)
    results: dict[str, dict] = {}
    model_reports: dict[str, dict] = {}
    summary_rows = []
    per_class_tables = []
    per_record_tables = []
    per_patient_tables = []
    prediction_tables = []
    score_kinds = {}
    n_models = len(all_models)

    for model_idx, model in enumerate(all_models):
        model_name = getattr(model, "name", model.__class__.__name__)
        slug = _safe_slug(model_name)
        start = time.perf_counter()
        _notify(progress, model_idx / max(n_models, 1), f"Training {model_name}: grouped out-of-fold score calibration")
        calibration_info = calibration.fit_grouped_temperature_scaler(
            model,
            X_train_calibration,
            y_train,
            (train_df["patient_id"].astype(str).to_numpy() if "patient_id" in train_df.columns else
             train_df["record"].astype(str).to_numpy() if "record" in train_df.columns else None),
            class_order=config.CLASSES,
            n_splits=3,
            seed=config.SEED,
        )
        _notify(progress, model_idx / max(n_models, 1), f"Training final {model_name}")
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        predict_start = time.perf_counter()
        predictions = np.asarray(model.predict(X_test)).astype(str)
        predict_seconds = time.perf_counter() - predict_start
        score_matrix = None
        score_kind = "predicted labels only"
        if hasattr(model, "scores"):
            score_raw = model.scores(X_test)
            score_matrix = align_class_scores(score_raw, model.classes_, config.CLASSES)
            if "Fuzzy" in model_name or "Rough" in model_name:
                score_kind = "normalised fuzzy rule scores (not calibrated probabilities)"
            elif "SVM" in model_name:
                score_kind = "softmax-transformed SVM decision ranking scores (not calibrated probabilities)"
            elif "Random Forest" in model_name:
                score_kind = "Random Forest class probability estimates (not post-hoc calibrated)"
            else:
                score_kind = "model-specific normalised ranking scores"
        elif hasattr(model, "predict_proba"):
            score_matrix = align_class_scores(model.predict_proba(X_test), model.classes_, config.CLASSES)
            score_kind = "estimator predict_proba output (not independently calibrated by PulseLab)"

        if score_matrix is not None and calibration_info.get("status") == "fitted":
            score_matrix = calibration.apply_temperature(score_matrix, calibration_info["temperature"])
            score_kind = f"temperature-scaled {score_kind}; scalar temperature fitted from grouped OOF training scores"
        metrics = evaluate_predictions(y_test, predictions, score_matrix, config.CLASSES)
        metrics["score_calibration"] = calibration_info
        (run_dir / f"{slug}_score_calibration.json").write_text(
            json.dumps(calibration_info, indent=2, default=_json_default), encoding="utf-8"
        )
        if score_matrix is not None:
            reliability = calibration.score_reliability_diagnostics(
                y_test, score_matrix, class_order=config.CLASSES,
                score_semantics=score_kind, n_bins=10,
            )
            metrics["score_reliability"] = reliability
            (run_dir / f"{slug}_score_reliability.json").write_text(
                json.dumps(reliability, indent=2, default=_json_default), encoding="utf-8"
            )
        if "record" in test_df.columns:
            test_records = test_df["record"].astype(str).to_numpy()
            bootstrap = record_cluster_bootstrap(y_test, predictions, test_records, n_bootstrap=500)
            metrics["record_cluster_bootstrap"] = bootstrap
            per_record_tables.append(per_record_metrics_frame(model_name, y_test, predictions, test_records))
            test_patients = (test_df["patient_id"].astype(str).to_numpy() if "patient_id" in test_df.columns
                             else np.asarray([config.patient_id_for_record(record) for record in test_records], dtype=str))
            patient_bootstrap = patient_cluster_bootstrap(y_test, predictions, test_patients, n_bootstrap=500)
            metrics["patient_cluster_bootstrap"] = patient_bootstrap
            per_patient_tables.append(per_patient_metrics_frame(model_name, y_test, predictions, test_patients, test_records))
            prediction_rows = pd.DataFrame({
                "record": test_records,
                "patient_id": test_patients,
                "y_true": y_test,
                "model": model_name,
                "y_pred": predictions,
            })
            if score_matrix is not None:
                for score_index, class_label in enumerate(config.CLASSES):
                    prediction_rows[f"score_{class_label}"] = score_matrix[:, score_index]
            prediction_tables.append(prediction_rows)
        else:
            metrics["record_cluster_bootstrap"] = {"status": "not_available", "reason": "No record IDs in test feature table."}
            if "patient_id" in test_df.columns:
                test_patients = test_df["patient_id"].astype(str).to_numpy()
                metrics["patient_cluster_bootstrap"] = patient_cluster_bootstrap(y_test, predictions, test_patients, n_bootstrap=500)
                per_patient_tables.append(per_patient_metrics_frame(model_name, y_test, predictions, test_patients))
            else:
                metrics["patient_cluster_bootstrap"] = {"status": "not_available", "reason": "No patient or record identifiers in test feature table."}
        metrics["train_seconds"] = float(train_seconds)
        metrics["predict_seconds"] = float(predict_seconds)
        metrics["n_features_input"] = int(X_train.shape[1])
        metrics["n_features_used"] = int(len(getattr(model, "selected_", columns)))
        metrics["score_interpretation"] = score_kind
        metrics["score_temperature"] = float(calibration_info.get("temperature", 1.0))
        results[model_name] = metrics
        score_kinds[model_name] = score_kind
        summary_rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_f1_all_aami": metrics["macro_f1_all_aami"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_roc_auc": metrics["macro_roc_auc"],
            "n_features_input": metrics["n_features_input"],
            "n_features_used": metrics["n_features_used"],
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "accuracy_95ci_low": (metrics.get("record_cluster_bootstrap", {}).get("accuracy_95ci") or [None, None])[0],
            "accuracy_95ci_high": (metrics.get("record_cluster_bootstrap", {}).get("accuracy_95ci") or [None, None])[1],
            "macro_f1_all_aami_95ci_low": (metrics.get("record_cluster_bootstrap", {}).get("macro_f1_all_aami_95ci") or [None, None])[0],
            "macro_f1_all_aami_95ci_high": (metrics.get("record_cluster_bootstrap", {}).get("macro_f1_all_aami_95ci") or [None, None])[1],
            "patient_accuracy_95ci_low": (metrics.get("patient_cluster_bootstrap", {}).get("accuracy_95ci") or [None, None])[0],
            "patient_accuracy_95ci_high": (metrics.get("patient_cluster_bootstrap", {}).get("accuracy_95ci") or [None, None])[1],
            "patient_macro_f1_all_aami_95ci_low": (metrics.get("patient_cluster_bootstrap", {}).get("macro_f1_all_aami_95ci") or [None, None])[0],
            "patient_macro_f1_all_aami_95ci_high": (metrics.get("patient_cluster_bootstrap", {}).get("macro_f1_all_aami_95ci") or [None, None])[1],
        })
        per_class_tables.append(per_class_frame(model_name, metrics))

        if hasattr(model, "fuzzy"):
            fuzzy_report = model.fuzzy.report()
            model_report = {"fuzzy": fuzzy_report}
            rule_payload = {"model": model_name, "fuzzy_report": fuzzy_report, "rules": model.fuzzy.rules_}
            if hasattr(model, "rough"):
                model_report["rough_set"] = model.rough.report()
                model_report["selected_features"] = list(model.selected_)
                model_report["removed_features"] = list(model.removed_)
                rule_payload["selected_features"] = list(model.selected_)
                rule_payload["rough_set_report"] = model.rough.report()
            model_reports[model_name] = model_report
            (run_dir / f"{slug}_rule_base.json").write_text(
                json.dumps(rule_payload, indent=2, default=_json_default), encoding="utf-8"
            )
        if hasattr(model, "m") and hasattr(model.m, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": columns,
                "importance": model.m.feature_importances_,
            }).sort_values("importance", ascending=False)
            importance.to_csv(run_dir / f"{slug}_feature_importance.csv", index=False)
        global_explanation = explainability.global_feature_summary(model)
        model_reports.setdefault(model_name, {})["global_explainability"] = global_explanation
        (run_dir / f"{slug}_global_explainability.json").write_text(
            json.dumps(global_explanation, indent=2, default=_json_default), encoding="utf-8"
        )
        if save_models:
            model_path = model_dir / f"{slug}.joblib"
            joblib.dump(model, model_path, compress=3)
            metrics["model_artifact"] = str(model_path.relative_to(config.ROOT)) if model_path.is_relative_to(config.ROOT) else str(model_path)
        cm_frame = pd.DataFrame(metrics["confusion_matrix"], index=config.CLASSES, columns=config.CLASSES)
        cm_frame.index.name = "actual\\predicted"
        cm_frame.to_csv(run_dir / f"{slug}_confusion_matrix.csv")
        _notify(progress, (model_idx + 1) / max(n_models, 1), f"Completed {model_name}")

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(run_dir / "model_comparison.csv", index=False)
    pd.concat(per_class_tables, ignore_index=True).to_csv(run_dir / "per_class_metrics.csv", index=False)
    if per_record_tables:
        pd.concat(per_record_tables, ignore_index=True).to_csv(run_dir / "per_record_model_metrics.csv", index=False)
    if per_patient_tables:
        pd.concat(per_patient_tables, ignore_index=True).to_csv(run_dir / "per_patient_model_metrics.csv", index=False)
    if prediction_tables:
        pd.concat(prediction_tables, ignore_index=True).to_csv(run_dir / "test_predictions.csv", index=False)

    train_distribution = {str(c): int((y_train == c).sum()) for c in config.CLASSES}
    test_distribution = {str(c): int((y_test == c).sum()) for c in config.CLASSES}
    train_missing = [c for c in config.CLASSES if train_distribution[c] == 0]
    test_missing = [c for c in config.CLASSES if test_distribution[c] == 0]
    low_train_support = {c: count for c, count in train_distribution.items() if 0 < count < config.MIN_TRAIN_SAMPLES_PER_CLASS}
    config_snapshot = {
        "seed": config.SEED,
        "sampling_frequency_hz": config.FS,
        "bandpass_hz": list(config.BANDPASS),
        "baseline_median_windows_seconds": list(config.BASELINE_MEDIAN_WINDOWS),
        "notch_hz": config.NOTCH_HZ,
        "pre_r_seconds": config.PRE_R_SEC,
        "post_r_seconds": config.POST_R_SEC,
        "rough_bins": config.ROUGH_BINS,
        "rough_max_per_class": config.ROUGH_MAX_PER_CLASS,
        "rough_tolerance": config.ROUGH_TOLERANCE,
        "fuzzy_max_per_class": config.FUZZY_MAX_PER_CLASS,
        "fuzzy_rules_per_class": config.FUZZY_RULES_PER_CLASS,
        "fuzzy_max_antecedents": config.FUZZY_MAX_ANTECEDENTS,
        "fuzzy_beam_width": config.FUZZY_BEAM,
        "fuzzy_min_support": config.FUZZY_MIN_SUPPORT,
        "rpeak_match_tolerance_seconds": config.RPEAK_MATCH_TOLERANCE_SEC,
    }
    run_warnings = []
    if train_missing:
        run_warnings.append(f"No training examples for AAMI classes: {train_missing}")
    if test_missing:
        run_warnings.append(f"No testing examples for AAMI classes: {test_missing}; those classes cannot be evaluated on this split")
    if low_train_support:
        run_warnings.append(f"Low training support (below {config.MIN_TRAIN_SAMPLES_PER_CLASS} beats): {low_train_support}")
    if (metadata or {}).get("patient_disjoint") is False:
        run_warnings.append("This historical record-wise split places records 201 and 202 from the same subject across train/test; interpret metrics as non-patient-disjoint.")
    run_manifest = {
        "project": "PulseLab",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": (metadata or {}).get("feature_schema_version", FEATURE_SCHEMA_VERSION),
        "feature_columns": columns,
        # Persist training-only medians so uploaded-signal inference applies the
        # exact same missing-value policy without fitting on inference samples.
        "feature_medians": {str(name): float(value) for name, value in X_train.median(axis=0).fillna(0.0).items()},
        "feature_reference": explainability.build_feature_reference(X_train, columns),
        "n_features": len(columns),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_patients": int(train_df["patient_id"].astype(str).nunique()) if "patient_id" in train_df.columns else None,
        "n_test_patients": int(test_df["patient_id"].astype(str).nunique()) if "patient_id" in test_df.columns else (int(pd.Series([config.patient_id_for_record(r) for r in test_df["record"].astype(str)]).nunique()) if "record" in test_df.columns else None),
        "train_class_distribution": train_distribution,
        "test_class_distribution": test_distribution,
        "missing_train_classes": train_missing,
        "missing_test_classes": test_missing,
        "warnings": run_warnings,
        "configuration": config_snapshot,
        "train_records": sorted(map(str, set(train_df["record"]))) if "record" in train_df else [],
        "test_records": sorted(map(str, set(test_df["record"]))) if "record" in test_df else [],
        "models": list(results.keys()),
        "score_interpretation": score_kinds,
        "metrics": results,
        "model_reports": model_reports,
        "metadata": metadata or {},
        "disclaimer": config.DISCLAIMER,
        "notes": [
            "Strict patient-disjoint runs exclude record 202 because records 201 and 202 belong to the same subject; historical compatibility mode explicitly allows that overlap.",
            "In automatic-detection mode, classifier metrics are conditional on detected beats matched one-to-one to reference beats; detector FP/FN are reported separately.",
            "Fuzzy normalised scores and SVM ranking scores are not calibrated clinical probabilities.",
            "Record-cluster bootstrap intervals resample held-out records, not individual beats; they are descriptive uncertainty summaries.",
            "Patient-cluster bootstrap intervals and per-patient metrics are also reported when patient identifiers are available (record identifiers are used as a documented fallback); these summaries describe held-out patient heterogeneity and are not external validation.",
            "The feature pipeline is offline / retrospective: zero-phase filtering, record-level R-amplitude scaling and next-RR features use full-record or future-beat context; it is not a validated streaming detector.",
            "A single MIT-BIH split is not external validation or clinical validation.",
            "When grouped training records support it, scalar temperature scaling is fitted only from grouped out-of-fold training scores; fold-specific imputation is fitted within each fold. The untouched test split is used only for evaluation. This is internal calibration, not external or clinical validation.",
            "Held-out score reliability diagnostics (Brier, log loss and ECE) are computed after the training-only transform. Fuzzy and SVM outputs remain model score estimates and must not be interpreted as clinical risk probabilities.",
            "Global feature summaries are model-native importance or fuzzy-rule usage summaries; they are not causal effects."
        ],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, default=_json_default), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    _write_research_report(run_dir, summary, run_manifest)
    detector_report = (metadata or {}).get("detection_metrics")
    if detector_report:
        (run_dir / "rpeak_detection_metrics.json").write_text(
            json.dumps(detector_report, indent=2, default=_json_default), encoding="utf-8"
        )
        pd.DataFrame(detector_report.get("per_record", [])).to_csv(
            run_dir / "rpeak_detection_per_record.csv", index=False
        )
    if save_models:
        (model_dir / "README.txt").write_text(
            "PulseLab research model artifacts. Load only trusted joblib files. "
            "These models are for academic research and are not clinically validated.\n",
            encoding="utf-8",
        )
    _notify(progress, 1.0, f"Experiment saved to {run_dir}")
    return run_dir


def collect_record_features(
    record_ids: Iterable[str],
    *,
    data_dir: Path | str = config.DATA_DIR,
    progress: Progress = None,
    record_loader: Callable = data_loader.load_record,
    feature_builder: Callable = feature_extraction.build_record_features,
    peak_source: str = "annotations",
    diagnostics_out: list | None = None,
) -> pd.DataFrame:
    """Load and extract features from each requested annotated MIT-BIH record."""
    ids = [str(r) for r in record_ids]
    if not ids:
        raise ValueError("At least one record ID is required")
    frames = []
    failures = []
    for idx, record_id in enumerate(ids):
        _notify(progress, idx / len(ids), f"Processing MIT-BIH record {record_id}")
        try:
            record = record_loader(record_id, Path(data_dir))
            frame = feature_builder(record, peak_source=peak_source)
            if diagnostics_out is not None and frame is not None:
                diagnostic = frame.attrs.get("detection_metrics")
                if diagnostic is not None:
                    diagnostics_out.append(dict(diagnostic))
            if frame is None or frame.empty:
                raise ValueError("feature extraction returned no heartbeat rows")
            if "record" not in frame:
                frame.insert(0, "record", record_id)
            if "label" not in frame:
                raise ValueError("feature table does not contain AAMI labels")
            frames.append(frame)
        except Exception as exc:
            failures.append(f"{record_id}: {type(exc).__name__}: {exc}")
        _notify(progress, (idx + 1) / len(ids), f"Finished record {record_id}")
    if failures:
        details = "\n".join(failures)
        raise RuntimeError(
            "Could not build a complete record feature table. No partial experiment was run.\n" + details
        )
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("No annotated heartbeat features were extracted")
    return combined


def run_experiment(
    train_record_ids: Iterable[str] | None = None,
    test_record_ids: Iterable[str] | None = None,
    *,
    data_dir: Path | str = config.DATA_DIR,
    output_root: Path | str | None = None,
    progress: Progress = None,
    include_baselines: bool = True,
    save_models: bool = True,
    run_name: str | None = None,
    peak_source: str = "detected",
    split_protocol: str = "patient-disjoint",
    allow_patient_overlap: bool | None = None,
    record_loader: Callable = data_loader.load_record,
    feature_builder: Callable = feature_extraction.build_record_features,
) -> Path:
    """Extract record-wise MIT-BIH features and run the full model comparison.

    The default uses automatically detected R peaks on both splits. The
    classifier scores are therefore conditional on detected peaks that can be
    matched to reference beats; detector sensitivity/PPV are reported
    separately. ``peak_source='annotations'`` is an explicitly annotation-
    guided benchmark, not an end-to-end inference estimate.
    """
    if train_record_ids is None or test_record_ids is None:
        default_train, default_test = config.get_record_split(split_protocol)
        train_ids = [str(r) for r in (default_train if train_record_ids is None else train_record_ids)]
        test_ids = [str(r) for r in (default_test if test_record_ids is None else test_record_ids)]
    else:
        train_ids = [str(r) for r in train_record_ids]
        test_ids = [str(r) for r in test_record_ids]
    if not train_ids or not test_ids:
        raise ValueError("Both training and testing record lists must be non-empty")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Training and testing record lists must not contain duplicate record IDs")
    if split_protocol not in config.SPLIT_PROTOCOLS:
        raise ValueError(f"Unknown split protocol {split_protocol!r}; choose from {config.SPLIT_PROTOCOLS}")
    if allow_patient_overlap is None:
        allow_patient_overlap = split_protocol == "de-chazal-record-wise"
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Patient/record leakage detected before feature extraction: {overlap}")
    train_patients = {config.patient_id_for_record(r) for r in train_ids}
    test_patients = {config.patient_id_for_record(r) for r in test_ids}
    patient_overlap = sorted(train_patients & test_patients)
    if patient_overlap and not allow_patient_overlap:
        raise ValueError(f"Patient-wise leakage detected before feature extraction: {patient_overlap}")
    if peak_source not in {"detected", "annotations"}:
        raise ValueError("peak_source must be 'detected' or 'annotations'")

    train_detection_rows: list[dict] = []
    test_detection_rows: list[dict] = []
    train = collect_record_features(
        train_ids, data_dir=data_dir,
        progress=lambda p, m: _notify(progress, 0.45 * p, "Training: " + m),
        peak_source=peak_source, diagnostics_out=train_detection_rows,
        record_loader=record_loader, feature_builder=feature_builder,
    )
    test = collect_record_features(
        test_ids, data_dir=data_dir,
        progress=lambda p, m: _notify(progress, 0.45 + 0.45 * p, "Testing: " + m),
        peak_source=peak_source, diagnostics_out=test_detection_rows,
        record_loader=record_loader, feature_builder=feature_builder,
    )
    for row in train_detection_rows:
        row["split"] = "train"
    for row in test_detection_rows:
        row["split"] = "test"
    feature_cols = [c for c in canonical_feature_names(2) if c in train.columns]
    if len(feature_cols) != len(canonical_feature_names(2)):
        missing = sorted(set(canonical_feature_names(2)) - set(feature_cols))
        raise ValueError(f"Extracted data does not match the two-channel feature schema; missing={missing}")
    from modules.validation import aggregate_detection_metrics
    detection_summary = None
    if peak_source == "detected":
        detection_summary = {
            "training_records": aggregate_detection_metrics(train_detection_rows),
            "testing_records": aggregate_detection_metrics(test_detection_rows),
            "per_record": train_detection_rows + test_detection_rows,
            "evaluation_note": "Classification metrics use only detected beats matched one-to-one to reference annotations; FP/FN are reported separately here.",
        }
    metadata = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "dataset_source": "PhysioNet WFDB records",
        "train_record_ids": train_ids,
        "test_record_ids": test_ids,
        "sampling_frequency_hz": config.FS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "inference_pipeline_contract": (
            current_mitbih_inference_contract()
            if feature_builder is feature_extraction.build_record_features else None
        ),
        "inference_pipeline_contract_note": (
            None if feature_builder is feature_extraction.build_record_features
            else "A custom feature builder was supplied; uploaded-signal inference compatibility was not asserted."
        ),
        "split_policy": split_protocol,
        "patient_disjoint": len({config.patient_id_for_record(r) for r in train_ids} & {config.patient_id_for_record(r) for r in test_ids}) == 0,
        "peak_source": peak_source,
        "classification_evaluation": ("conditional_on_matched_detected_beats" if peak_source == "detected" else "annotation_guided_reference_beats"),
        "detection_metrics": detection_summary,
    }
    _notify(progress, 0.9, "Training and evaluating model suite")
    return run_feature_experiment(
        train, test,
        feature_columns=feature_cols,
        output_root=output_root,
        progress=lambda p, m: _notify(progress, 0.9 + 0.1 * p, m),
        include_baselines=include_baselines,
        save_models=save_models,
        run_name=run_name,
        metadata=metadata,
        allow_patient_overlap=bool(allow_patient_overlap),
    )
