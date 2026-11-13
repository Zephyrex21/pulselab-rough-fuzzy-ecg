"""Model-agnostic explainability and input-range diagnostics for PulseLab.

Local explanations use single-feature reference ablation: one observed feature
is replaced with its training-set median while the other values are held fixed.
This is an exploratory sensitivity analysis, not a causal explanation or SHAP
value. Correlated features can make one-at-a-time effects misleading.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modules.calibration import apply_temperature

_EPS = 1e-12


def _normalise_rows(scores: Any, n_rows: int, n_classes: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape != (n_rows, n_classes):
        raise ValueError("Model score output has an unexpected shape")
    if not np.isfinite(values).all():
        raise ValueError("Model score output contains non-finite values")
    # Model wrappers in PulseLab expose non-negative score vectors. For an
    # estimator returning signed decision values, convert them with softmax.
    if np.any(values < 0):
        shifted = values - values.max(axis=1, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -60, 60))
        return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), _EPS)
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    zero = totals[:, 0] <= _EPS
    result = values / np.maximum(totals, _EPS)
    if zero.any():
        result[zero] = 1.0 / max(n_classes, 1)
    return result


def model_score_matrix(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return row-normalised scores and the model's declared class order."""
    classes = [str(value) for value in getattr(model, "classes_", [])]
    if not classes:
        raise ValueError("The selected model does not expose its fitted class labels")
    scorer = getattr(model, "scores", None)
    if callable(scorer):
        raw = scorer(frame)
    else:
        predict_proba = getattr(model, "predict_proba", None)
        if not callable(predict_proba):
            raise ValueError("The selected model does not expose class scores")
        raw = predict_proba(frame)
    return _normalise_rows(raw, len(frame), len(classes)), classes


def local_reference_ablation(
    model: Any,
    row: pd.Series | dict[str, Any],
    *,
    feature_columns: list[str],
    reference_values: dict[str, float],
    target_class: str | None = None,
    temperature: float = 1.0,
    top_k: int = 15,
) -> dict[str, Any]:
    """Measure one-feature-at-a-time score changes against training medians."""
    if isinstance(row, pd.Series):
        base = row.copy()
    elif isinstance(row, dict):
        base = pd.Series(row)
    else:
        raise TypeError("row must be a pandas Series or dict")

    used_features = list(getattr(model, "selected_", None) or getattr(model, "feature_names_", None) or feature_columns)
    used_features = [name for name in used_features if name in feature_columns and name in base.index and name in reference_values]
    if not used_features:
        return {
            "available": False,
            "method": "single-feature reference ablation",
            "message": "No model input features have matching training reference values.",
            "features": [],
        }

    base_frame = pd.DataFrame([{name: float(base[name]) for name in feature_columns}], columns=feature_columns)
    original_scores, classes = model_score_matrix(model, base_frame)
    original_scores = apply_temperature(original_scores, temperature)
    original_pred = str(classes[int(np.argmax(original_scores[0]))])
    target = str(target_class or original_pred)
    if target not in classes:
        target = original_pred
    target_idx = classes.index(target)

    # Vectorise all one-at-a-time perturbations into one estimator call.
    perturbed = pd.DataFrame([base_frame.iloc[0].to_dict() for _ in used_features], columns=feature_columns)
    values = []
    for index, name in enumerate(used_features):
        original_value = float(base[name])
        reference_value = float(reference_values[name])
        perturbed.iat[index, perturbed.columns.get_loc(name)] = reference_value
        values.append((name, original_value, reference_value))
    changed_scores, _ = model_score_matrix(model, perturbed)
    changed_scores = apply_temperature(changed_scores, temperature)

    attribution_rows = []
    original_target_score = float(original_scores[0, target_idx])
    for index, (name, observed, reference) in enumerate(values):
        ablated_score = float(changed_scores[index, target_idx])
        delta = original_target_score - ablated_score
        if delta > 1e-8:
            direction = "supports_predicted_class_vs_reference"
        elif delta < -1e-8:
            direction = "reduces_predicted_class_score_vs_reference"
        else:
            direction = "little_change"
        attribution_rows.append({
            "feature": str(name),
            "observed_value": observed,
            "training_median_reference": reference,
            "score_with_observed_value": original_target_score,
            "score_after_reference_replacement": ablated_score,
            "score_delta": float(delta),
            "direction": direction,
            "absolute_effect": float(abs(delta)),
        })
    attribution_rows.sort(key=lambda item: item["absolute_effect"], reverse=True)
    limit = max(1, min(int(top_k), len(attribution_rows)))
    return {
        "available": True,
        "method": "single-feature reference ablation",
        "target_class": target,
        "original_prediction": original_pred,
        "original_target_score": original_target_score,
        "temperature": float(temperature),
        "n_model_features": len(used_features),
        "n_features_returned": min(limit, len(attribution_rows)),
        "features": attribution_rows[:limit],
        "limitations": [
            "Each feature is replaced independently with its training-set median; this is not a causal effect or a SHAP value.",
            "Correlated ECG features are not varied jointly, so attribution magnitudes can be redistributed or misleading.",
            "Effects describe changes in the model's own normalised score, not calibrated probabilities or clinical risk.",
        ],
    }


def global_feature_summary(model: Any) -> dict[str, Any]:
    """Extract native model importance or aggregate fuzzy rule antecedents."""
    native = getattr(getattr(model, "m", None), "feature_importances_", None)
    names = list(getattr(model, "feature_names_", []))
    if native is not None and len(names) == len(native):
        rows = [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(names, native)
        ]
        rows.sort(key=lambda item: item["importance"], reverse=True)
        return {
            "available": True,
            "method": "random forest impurity-based feature importance",
            "features": rows,
            "limitations": ["Impurity-based importance can favour variables with many split opportunities and is not a causal effect."],
        }

    fuzzy = getattr(model, "fuzzy", None)
    rules = getattr(fuzzy, "rules_", None)
    if rules:
        totals: dict[str, float] = {}
        for rule in rules:
            strength = float(rule.get("weight", 0.0)) * float(rule.get("support", 0.0))
            for antecedent in rule.get("antecedents", []):
                feature = str(antecedent.get("feature", ""))
                if feature:
                    totals[feature] = totals.get(feature, 0.0) + strength
        total = sum(totals.values())
        rows = [
            {"feature": name, "importance": float(value / total) if total > 0 else 0.0}
            for name, value in totals.items()
        ]
        rows.sort(key=lambda item: item["importance"], reverse=True)
        return {
            "available": bool(rows),
            "method": "normalised fuzzy-rule antecedent weight × support",
            "features": rows,
            "limitations": ["This aggregates rule usage, not causal importance or held-out permutation importance."],
        }
    return {
        "available": False,
        "method": "not available",
        "message": "This model has no native global feature-importance measure. Use per-beat reference ablation for a local sensitivity view.",
        "features": [],
    }


def build_feature_reference(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    """Describe training feature ranges using quantiles (training data only)."""
    reference: dict[str, dict[str, float]] = {}
    for name in feature_columns:
        values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
        reference[str(name)] = {
            "q01": float(quantiles.loc[0.01]),
            "q05": float(quantiles.loc[0.05]),
            "median": float(quantiles.loc[0.50]),
            "q95": float(quantiles.loc[0.95]),
            "q99": float(quantiles.loc[0.99]),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return reference


def assess_feature_drift(
    frame: pd.DataFrame,
    reference: dict[str, dict[str, float]] | None,
) -> tuple[dict[str, Any] | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Flag input beat features outside the training 1st–99th percentile range.

    This is a descriptive covariate-shift warning, not a statistical hypothesis
    test and not a determination that an ECG is invalid.
    """
    if not reference:
        return None, None, None
    valid = [name for name in reference if name in frame.columns]
    if not valid or frame.empty:
        return None, None, None
    mask = np.zeros((len(frame), len(valid)), dtype=bool)
    feature_rows = []
    for column_index, name in enumerate(valid):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        bounds = reference[name]
        low, high = float(bounds["q01"]), float(bounds["q99"])
        outside = np.isfinite(values) & ((values < low) | (values > high))
        mask[:, column_index] = outside
        feature_rows.append({
            "feature": name,
            "training_q01": low,
            "training_median": float(bounds.get("median", np.nan)),
            "training_q99": high,
            "inference_min": float(np.nanmin(values)) if np.isfinite(values).any() else None,
            "inference_max": float(np.nanmax(values)) if np.isfinite(values).any() else None,
            "outside_count": int(outside.sum()),
            "outside_fraction": float(outside.mean()) if len(outside) else 0.0,
        })
    per_beat = pd.DataFrame({
        "beat_index": np.arange(len(frame), dtype=int),
        "n_features_outside_training_q01_q99": mask.sum(axis=1).astype(int),
        "fraction_features_outside_training_q01_q99": mask.mean(axis=1).astype(float),
    })
    per_feature = pd.DataFrame(feature_rows).sort_values("outside_fraction", ascending=False).reset_index(drop=True)
    total_fraction = float(mask.mean()) if mask.size else 0.0
    affected_beats = int((mask.sum(axis=1) > 0).sum())
    summary = {
        "available": True,
        "method": "training 1st–99th percentile range comparison",
        "n_beats": int(len(frame)),
        "n_features_compared": int(len(valid)),
        "outside_value_fraction": total_fraction,
        "beats_with_any_out_of_range_feature": affected_beats,
        "beats_with_any_out_of_range_feature_fraction": float(affected_beats / len(frame)),
        "high_drift_flag": bool(total_fraction >= 0.10 or affected_beats / len(frame) >= 0.25),
        "top_features": feature_rows[:10],
        "limitations": [
            "Quantile-range flags are descriptive and do not prove a different clinical population or device.",
            "Feature correlations and signal quality are not represented by this univariate range check.",
            "A lack of range flags does not establish that the uploaded ECG is in-domain or clinically valid.",
        ],
    }
    return summary, per_beat, per_feature
