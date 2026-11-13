"""Evaluation helpers for the five-class AAMI heartbeat task."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


def align_class_scores(scores: np.ndarray, model_classes, class_order=None) -> np.ndarray:
    """Place model score columns into a fixed class order, filling absent classes with zero."""
    class_order = list(class_order or config.CLASSES)
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("Model scores must be a two-dimensional array")
    if values.shape[1] != len(model_classes):
        raise ValueError("Number of score columns does not match model classes")
    aligned = np.zeros((values.shape[0], len(class_order)), dtype=float)
    for source_idx, label in enumerate(model_classes):
        if label in class_order:
            aligned[:, class_order.index(label)] = values[:, source_idx]
    return aligned


def evaluate_predictions(y_true, y_pred, scores=None, class_order=None) -> dict:
    """Return accuracy, balanced accuracy, macro/per-class metrics and OvR AUCs."""
    labels = list(class_order or config.CLASSES)
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.ndim != 1 or pred.ndim != 1 or len(truth) != len(pred):
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    if len(truth) == 0:
        raise ValueError("Cannot evaluate an empty test set")

    cm = confusion_matrix(truth, pred, labels=labels)
    precision = precision_score(truth, pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(truth, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(truth, pred, labels=labels, average=None, zero_division=0)
    support = np.asarray([(truth == label).sum() for label in labels], dtype=int)
    observed_labels = [label for label, count in zip(labels, support) if count > 0]
    per_class = []
    for idx, label in enumerate(labels):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        specificity = tn / (tn + fp) if tn + fp else 0.0
        per_class.append({
            "class": label,
            "precision": float(precision[idx]),
            "recall_sensitivity": float(recall[idx]),
            "specificity": float(specificity),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        })

    metrics = {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(np.mean(recall[ [labels.index(label) for label in observed_labels] ])) if observed_labels else 0.0,
        "macro_precision": float(precision_score(truth, pred, labels=observed_labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, labels=observed_labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, labels=observed_labels, average="macro", zero_division=0)),
        "macro_precision_all_aami": float(precision_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall_all_aami": float(recall_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_all_aami": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_classes_evaluated": observed_labels,
        "weighted_f1": float(f1_score(truth, pred, labels=labels, average="weighted", zero_division=0)),
        "n_test": int(len(truth)),
        "class_order": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

    if scores is not None:
        score_values = np.asarray(scores, dtype=float)
        if score_values.shape != (len(truth), len(labels)):
            raise ValueError("scores must have shape (n_samples, n_classes)")
        auc_rows = []
        for idx, label in enumerate(labels):
            binary = (truth == label).astype(int)
            if binary.min() == binary.max():
                continue
            auc_rows.append({
                "class": label,
                "roc_auc_ovr": float(roc_auc_score(binary, score_values[:, idx])),
                "average_precision_ovr": float(average_precision_score(binary, score_values[:, idx])),
            })
        metrics["ovr_auc"] = auc_rows
        metrics["macro_roc_auc"] = float(np.mean([r["roc_auc_ovr"] for r in auc_rows])) if auc_rows else None
        metrics["macro_average_precision"] = float(np.mean([r["average_precision_ovr"] for r in auc_rows])) if auc_rows else None
    else:
        metrics["ovr_auc"] = []
        metrics["macro_roc_auc"] = None
        metrics["macro_average_precision"] = None

    return metrics


def per_class_frame(model_name: str, metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([{"model": model_name, **row} for row in metrics["per_class"]])


def record_cluster_bootstrap(
    y_true,
    y_pred,
    record_ids,
    *,
    n_bootstrap: int = 500,
    seed: int = config.SEED,
    class_order=None,
) -> dict:
    """Cluster bootstrap confidence intervals by held-out record, not by beat.

    Sampling individual beats as if they were independent can produce
    unrealistically narrow intervals because beats from one recording are
    correlated. This resamples complete record clusters and aggregates their
    fixed-order confusion matrices. It is an uncertainty summary, not clinical
    validation and not a substitute for external datasets.
    """
    labels = list(class_order or config.CLASSES)
    truth, pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(record_ids).astype(str)
    if truth.ndim != 1 or pred.ndim != 1 or groups.ndim != 1 or not (len(truth) == len(pred) == len(groups)):
        raise ValueError("y_true, y_pred and record_ids must be aligned one-dimensional arrays")
    if len(truth) == 0:
        raise ValueError("Cannot bootstrap an empty test set")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {
            "method": "record-cluster percentile bootstrap",
            "n_records": int(len(unique_groups)), "n_bootstrap": int(n_bootstrap),
            "status": "not_estimated", "reason": "At least two held-out records are required.",
        }
    matrices = []
    for group in unique_groups:
        mask = groups == group
        matrices.append(confusion_matrix(truth[mask], pred[mask], labels=labels))
    matrices = np.asarray(matrices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    acc_values, f1_values = [], []
    n_clusters = len(unique_groups)
    for _ in range(int(n_bootstrap)):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        cm = matrices[sampled].sum(axis=0)
        total = int(cm.sum())
        accuracy = float(np.trace(cm) / total) if total else 0.0
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        class_f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        acc_values.append(accuracy)
        f1_values.append(float(np.mean(class_f1)))
    return {
        "method": "record-cluster percentile bootstrap",
        "n_records": int(n_clusters), "n_bootstrap": int(n_bootstrap),
        "status": "estimated",
        "accuracy_95ci": [float(np.percentile(acc_values, 2.5)), float(np.percentile(acc_values, 97.5))],
        "macro_f1_all_aami_95ci": [float(np.percentile(f1_values, 2.5)), float(np.percentile(f1_values, 97.5))],
        "seed": int(seed),
        "resampling_unit": "held-out record / subject cluster",
    }


def per_record_metrics_frame(model_name: str, y_true, y_pred, record_ids) -> pd.DataFrame:
    """Summarise classification results separately for every held-out record."""
    truth, pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(record_ids).astype(str)
    if not (len(truth) == len(pred) == len(groups)):
        raise ValueError("y_true, y_pred and record_ids must have equal lengths")
    rows = []
    for record in sorted(np.unique(groups), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        mask = groups == record
        result = evaluate_predictions(truth[mask], pred[mask], class_order=config.CLASSES)
        rows.append({
            "model": model_name,
            "record": record,
            "patient_id": config.patient_id_for_record(record),
            "n_beats": int(mask.sum()),
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "macro_f1_observed_classes": result["macro_f1"],
            "macro_f1_all_aami": result["macro_f1_all_aami"],
            "classes_present": "".join(result["macro_classes_evaluated"]),
        })
    return pd.DataFrame(rows)


def patient_cluster_bootstrap(
    y_true,
    y_pred,
    patient_ids,
    *,
    n_bootstrap: int = 500,
    seed: int = config.SEED,
    class_order=None,
) -> dict:
    """Bootstrap held-out patients as the resampling unit.

    Unlike beat bootstrap, this samples complete patient clusters and sums their
    fixed-order confusion matrices. When only one patient is present, intervals
    are not estimated and that limitation is reported explicitly.
    """
    result = record_cluster_bootstrap(
        y_true, y_pred, patient_ids, n_bootstrap=n_bootstrap, seed=seed, class_order=class_order
    )
    result["method"] = "patient-cluster percentile bootstrap"
    result["n_patients"] = int(result.pop("n_records", 0))
    result["resampling_unit"] = "held-out patient"
    if result.get("status") == "not_estimated":
        result["reason"] = "At least two distinct held-out patients are required for a patient-cluster interval."
    return result


def per_patient_metrics_frame(model_name: str, y_true, y_pred, patient_ids, record_ids=None) -> pd.DataFrame:
    """Summarise fixed-order class metrics separately for each held-out patient."""
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    patients = np.asarray(patient_ids).astype(str)
    records = np.asarray(record_ids).astype(str) if record_ids is not None else np.full(len(truth), "", dtype=str)
    if not (len(truth) == len(pred) == len(patients) == len(records)):
        raise ValueError("y_true, y_pred, patient_ids and record_ids must have equal lengths")
    rows = []
    for patient in sorted(np.unique(patients)):
        mask = patients == patient
        result = evaluate_predictions(truth[mask], pred[mask], class_order=config.CLASSES)
        patient_records = sorted(set(records[mask]) - {""})
        rows.append({
            "model": model_name,
            "patient_id": patient,
            "records": ";".join(patient_records),
            "n_records": int(len(patient_records)),
            "n_beats": int(mask.sum()),
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "macro_f1_observed_classes": result["macro_f1"],
            "macro_f1_all_aami": result["macro_f1_all_aami"],
            "classes_present": "".join(result["macro_classes_evaluated"]),
        })
    return pd.DataFrame(rows)
