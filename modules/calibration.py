"""Training-only score calibration and held-out reliability diagnostics.

Temperature scaling is fitted from grouped out-of-fold predictions within the
training partition. A separate function then evaluates the transformed score
vectors on untouched held-out labels. Score semantics remain model-specific;
this internal transform is not external or clinical validation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def _bin_summary(confidence: np.ndarray, outcomes: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, Any]] = []
    for index in range(n_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        selected = (confidence >= lower) & ((confidence < upper) if index < n_bins - 1 else (confidence <= upper))
        count = int(selected.sum())
        rows.append({
            "bin": index,
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_score": float(confidence[selected].mean()) if count else None,
            "observed_frequency": float(outcomes[selected].mean()) if count else None,
        })
    return rows


def apply_temperature(scores, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to non-negative normalised score rows."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("scores must be a finite non-negative two-dimensional matrix")
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0:
        raise ValueError("temperature must be finite and positive")
    totals = values.sum(axis=1, keepdims=True)
    probs = values / np.maximum(totals, 1e-12)
    zero = totals[:, 0] <= 1e-12
    if zero.any():
        probs[zero] = 1.0 / max(values.shape[1], 1)
    if np.isclose(temp, 1.0):
        return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temp
    logits -= logits.max(axis=1, keepdims=True)
    exp_values = np.exp(np.clip(logits, -60, 60))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def fit_grouped_temperature_scaler(
    model,
    X,
    y,
    groups,
    *,
    class_order: list[str],
    n_splits: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a scalar temperature from out-of-fold predictions grouped by patient/record.

    The held-out test split is never touched. The caller should pass patient IDs
    where available, falling back to record IDs only when patient IDs are absent.
    If the feature table has too few
    groups, missing record IDs, or a class disappears from a fold's training
    partition, calibration is skipped and the model keeps temperature 1.0.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from modules.metrics import align_class_scores

    labels = list(map(str, class_order))
    target = np.asarray(y).astype(str)
    group_values = np.asarray(groups).astype(str) if groups is not None else np.array([], dtype=str)
    base = {
        "method": "out-of-fold patient/record-grouped scalar temperature scaling",
        "status": "not_fitted",
        "temperature": 1.0,
        "n_oof_samples": 0,
        "n_record_groups": int(len(np.unique(group_values))) if len(group_values) else 0,
        "n_splits": 0,
        "calibration_scope": "training records only; held-out test records are not used to fit temperature",
    }
    if len(target) == 0 or len(group_values) != len(target):
        base["reason"] = "Record IDs are unavailable or not aligned with training samples."
        return base
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 3:
        base["reason"] = "At least three distinct training records are required for grouped out-of-fold calibration."
        return base
    observed_classes = set(target)
    splits = min(max(2, int(n_splits)), len(unique_groups))
    try:
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=int(seed))
        oof = np.full((len(target), len(labels)), np.nan, dtype=float)
        feature_frame = X.copy() if hasattr(X, "iloc") else __import__("pandas").DataFrame(X)
        if len(feature_frame) != len(target):
            base["reason"] = "The feature table and labels are not aligned."
            return base
        for fit_indices, validation_indices in splitter.split(np.zeros(len(target)), target, group_values):
            if not observed_classes.issubset(set(target[fit_indices])):
                base["reason"] = "A training fold loses one or more observed classes; calibration was skipped rather than estimating unstable scores."
                return base
            # Fit missing-value handling inside each fold. OOF validation
            # records must not influence even unsupervised imputation statistics.
            import pandas as pd
            x_fit = feature_frame.iloc[fit_indices].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            x_valid = feature_frame.iloc[validation_indices].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            fold_medians = x_fit.median(axis=0).fillna(0.0)
            x_fit = x_fit.fillna(fold_medians).fillna(0.0)
            x_valid = x_valid.fillna(fold_medians).fillna(0.0)
            fold_model = deepcopy(model)
            fold_model.fit(x_fit, target[fit_indices])
            classes = getattr(fold_model, "classes_", None)
            if classes is None:
                base["reason"] = "The model does not expose classes_ for score alignment."
                return base
            if callable(getattr(fold_model, "scores", None)):
                raw = fold_model.scores(x_valid)
            elif callable(getattr(fold_model, "predict_proba", None)):
                raw = fold_model.predict_proba(x_valid)
            else:
                base["reason"] = "The model does not expose class score vectors."
                return base
            aligned = align_class_scores(raw, classes, labels)
            row_sum = aligned.sum(axis=1, keepdims=True)
            fold_scores = aligned / np.maximum(row_sum, 1e-12)
            zero = row_sum[:, 0] <= 1e-12
            if zero.any():
                fold_scores[zero] = 1.0 / len(labels)
            oof[validation_indices] = fold_scores
        if not np.isfinite(oof).all():
            base["reason"] = "The grouped folds did not produce a complete out-of-fold score matrix."
            return base
        class_index = {label: index for index, label in enumerate(labels)}
        true_indices = np.asarray([class_index[label] for label in target], dtype=int)
        from scipy.optimize import minimize_scalar

        def objective(temperature: float) -> float:
            calibrated = apply_temperature(oof, temperature)
            return float(-np.mean(np.log(np.clip(calibrated[np.arange(len(target)), true_indices], 1e-12, 1.0))))

        optimum = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded", options={"xatol": 1e-4})
        if not optimum.success or not np.isfinite(optimum.fun):
            base["reason"] = "The temperature optimisation did not converge; unscaled model scores were retained."
            return base
        base.update({
            "status": "fitted",
            "temperature": float(optimum.x),
            "n_oof_samples": int(len(target)),
            "n_record_groups": int(len(unique_groups)),
            "n_splits": int(splits),
            "oof_imputation": "fold-specific medians fitted only on each fold's training records",
            "oof_log_loss_after_scaling": float(optimum.fun),
            "oof_log_loss_before_scaling": objective(1.0),
            "class_support": {label: int((target == label).sum()) for label in labels},
            "limitations": [
                "Temperature was fitted to grouped out-of-fold training scores and then applied to the final model fit on all training records.",
                "Held-out test reliability must still be inspected; this is internal calibration, not external or clinical validation.",
                "Calibration can be unstable when a class is represented by only a few training records or beats.",
            ],
        })
        return base
    except Exception as exc:
        base["reason"] = f"Grouped calibration could not be completed: {type(exc).__name__}: {exc}"
        return base


def score_reliability_diagnostics(
    y_true,
    scores,
    *,
    class_order: list[str],
    score_semantics: str = "unspecified model scores",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute multiclass Brier/log loss and top-label / classwise ECE.

    Inputs are row-normalised to sum to one for comparability. This function
    never fits a transform: its labels are for untouched held-out evaluation.
    The caller may pass scores already transformed using training-only grouped
    out-of-fold calibration. The supplied score semantics are retained.
    """
    truth = np.asarray(y_true).astype(str)
    values = np.asarray(scores, dtype=float)
    labels = list(map(str, class_order))
    if n_bins < 2 or n_bins > 50:
        raise ValueError("n_bins must be between 2 and 50")
    if values.ndim != 2 or values.shape != (len(truth), len(labels)):
        raise ValueError("scores must have shape (n_samples, n_classes)")
    if not len(truth):
        raise ValueError("At least one sample is required for score reliability diagnostics")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    if np.any(values < -1e-9):
        raise ValueError("scores must be non-negative before normalisation")
    unknown = sorted(set(truth) - set(labels))
    if unknown:
        raise ValueError(f"Unknown labels in y_true: {unknown}")
    values = np.clip(values, 0.0, None)
    row_totals = values.sum(axis=1, keepdims=True)
    zero_rows = row_totals[:, 0] <= 1e-12
    probs = values / np.maximum(row_totals, 1e-12)
    if zero_rows.any():
        probs[zero_rows] = 1.0 / len(labels)
    indices = np.asarray([labels.index(label) for label in truth], dtype=int)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(truth)), indices] = 1.0
    clipped = np.clip(probs, 1e-12, 1.0)
    multiclass_brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    log_loss = float(-np.mean(np.log(clipped[np.arange(len(truth)), indices])))

    predicted = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correctness = (predicted == indices).astype(float)
    top_bins = _bin_summary(confidence, correctness, n_bins)
    top_ece = float(sum(
        (row["count"] / len(truth)) * abs(row["mean_score"] - row["observed_frequency"])
        for row in top_bins if row["count"] and row["mean_score"] is not None
    ))
    per_class = []
    for class_index, label in enumerate(labels):
        class_score = probs[:, class_index]
        class_truth = one_hot[:, class_index]
        bins = _bin_summary(class_score, class_truth, n_bins)
        ece = float(sum(
            (row["count"] / len(truth)) * abs(row["mean_score"] - row["observed_frequency"])
            for row in bins if row["count"] and row["mean_score"] is not None
        ))
        per_class.append({
            "class": label,
            "brier_score": float(np.mean((class_score - class_truth) ** 2)),
            "expected_calibration_error": ece,
            "reliability_bins": bins,
            "positive_support": int(class_truth.sum()),
        })
    return {
        "status": "computed",
        "n_samples": int(len(truth)),
        "n_bins": int(n_bins),
        "score_semantics": str(score_semantics),
        "normalisation": "each score row divided by its non-negative row sum; all-zero rows replaced by uniform scores",
        "metrics": {
            "multiclass_brier_score": multiclass_brier,
            "multiclass_log_loss": log_loss,
            "top_label_expected_calibration_error": top_ece,
            "top_label_accuracy": float(correctness.mean()),
            "mean_top_score": float(confidence.mean()),
        },
        "top_label_reliability_bins": top_bins,
        "per_class": per_class,
        "calibration_status": "held_out_reliability_diagnostics_only",
        "interpretation": (
            "These diagnostics evaluate the supplied score vectors on untouched held-out labels; no transform is fitted here. "
            "When available, the upstream temperature was fitted only from grouped out-of-fold training records. "
            "Internal calibration is not external or clinical validation, and score semantics remain model-specific."
        ),
    }
