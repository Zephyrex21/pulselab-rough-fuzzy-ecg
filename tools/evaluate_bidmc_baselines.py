"""Patient-grouped baseline screen for the supplied BIDMC feature export.

This script selects hyperparameters using GroupKFold over the training patients
only, then evaluates each selected estimator once on the manifest's held-out
patients. It is a research comparison utility, not a production model trainer.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR


def metric_summary(y_true, y_pred, patient_ids):
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    absolute_errors = np.abs(errors)
    patient_mae = pd.DataFrame({"patient_id": patient_ids, "absolute_error": absolute_errors}).groupby("patient_id")["absolute_error"].mean()
    return {
        "MAE_bpm": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_bpm": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "bias_pred_minus_reference_bpm": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
        "mean_patient_MAE_bpm": float(patient_mae.mean()),
        "median_patient_MAE_bpm": float(patient_mae.median()),
    }


def candidate_models(seed: int):
    return [
        ("Median baseline", make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="median")), {}),
        ("Ridge", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()),
         {"ridge__alpha": [1, 10, 100]}),
        ("ElasticNet", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNet(max_iter=10000)),
         {"elasticnet__alpha": [0.01, 0.1, 0.5], "elasticnet__l1_ratio": [0.5, 0.9]}),
        ("SVR-RBF", make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), SVR(kernel="rbf")),
         {"svr__C": [0.3, 1, 3], "svr__epsilon": [0.1, 0.5], "svr__gamma": ["scale", 0.01]}),
        ("Extra Trees", make_pipeline(SimpleImputer(strategy="median"),
                                      ExtraTreesRegressor(random_state=seed, n_estimators=120, n_jobs=1)),
         {"extratreesregressor__min_samples_leaf": [5, 10],
          "extratreesregressor__max_features": [0.5, 1.0]}),
        ("HistGradientBoosting", make_pipeline(SimpleImputer(strategy="median"),
                                               HistGradientBoostingRegressor(random_state=seed)),
         {"histgradientboostingregressor__learning_rate": [0.03, 0.07],
          "histgradientboostingregressor__max_iter": [80, 150],
          "histgradientboostingregressor__max_leaf_nodes": [7, 15]}),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="features_rr_estimation_no_resp_input.csv")
    parser.add_argument("--manifest", type=Path, required=True, help="BIDMC run_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for candidate metrics")
    parser.add_argument("--cv-folds", type=int, default=3, help="GroupKFold folds over training patients only (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.features)
    required_columns = {"patient_id", "record_id", "ref_rr_mean_bpm"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        parser.error(f"Feature export is missing required columns: {missing}")

    frame = frame.loc[frame["ref_rr_mean_bpm"].notna()].copy()
    train_patients = set(map(str, manifest.get("train_patient_ids", [])))
    test_patients = set(map(str, manifest.get("test_patient_ids", [])))
    if not train_patients or not test_patients or train_patients & test_patients:
        parser.error("The manifest must contain non-empty, disjoint train/test patient ID lists.")
    dataset_patients = set(frame["patient_id"].astype(str))
    if dataset_patients != train_patients | test_patients:
        parser.error("The feature export patient IDs do not match the manifest's train/test patient lists.")

    train = frame.loc[frame["patient_id"].astype(str).isin(train_patients)].copy()
    test = frame.loc[frame["patient_id"].astype(str).isin(test_patients)].copy()
    n_train_patients = train["patient_id"].nunique()
    folds = min(args.cv_folds, n_train_patients)
    if folds < 2:
        parser.error("At least two training patient groups are required for grouped cross-validation.")

    results = []
    for feature_set, columns in (
        ("PPG-only", [name for name in frame.columns if name.startswith("ppg_")]),
        ("ECG+PPG", [name for name in frame.columns if name.startswith(("ecg_", "ppg_", "ecg_ppg_"))]),
    ):
        if not columns:
            parser.error(f"No predictor columns found for {feature_set}.")
        X_train, y_train = train[columns], train["ref_rr_mean_bpm"].astype(float)
        X_test, y_test = test[columns], test["ref_rr_mean_bpm"].astype(float)
        groups = train["patient_id"].astype(str).to_numpy()
        cv = GroupKFold(n_splits=folds)

        for model_name, estimator, grid in candidate_models(args.seed):
            if grid:
                search = GridSearchCV(
                    estimator, grid, scoring="neg_mean_absolute_error",
                    cv=cv, n_jobs=1, refit=True,
                )
                search.fit(X_train, y_train, groups=groups)
                fitted = search.best_estimator_
                cv_group_mae = float(-search.best_score_)
                best_params = search.best_params_
            else:
                fitted = estimator.fit(X_train, y_train)
                cv_group_mae = None
                best_params = {}

            prediction = fitted.predict(X_test)
            metrics = metric_summary(
                y_test, prediction, test["patient_id"].astype(str).to_numpy()
            )
            results.append({
                "feature_set": feature_set,
                "model": model_name,
                "predictor_count": len(columns),
                "n_train_patients": n_train_patients,
                "n_test_patients": int(test["patient_id"].nunique()),
                "n_test_windows": len(test),
                "train_group_cv_MAE_bpm": cv_group_mae,
                "best_params": best_params,
                **metrics,
            })
            print(
                f"{feature_set:9s} {model_name:22s} "
                f"train_group_CV_MAE={cv_group_mae if cv_group_mae is not None else 'n/a'} "
                f"heldout_MAE={metrics['MAE_bpm']:.4f} R2={metrics['R2']:.4f}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).sort_values(["feature_set", "MAE_bpm"])
    table.to_csv(args.output / "baseline_comparison.csv", index=False)
    metadata = {
        "dataset": manifest.get("dataset"),
        "run_id": manifest.get("run_id"),
        "train_patient_ids": sorted(train_patients),
        "test_patient_ids": sorted(test_patients),
        "cv": f"GroupKFold(n_splits={folds}) over training patient IDs only",
        "heldout_policy": "The held-out 10-patient split is evaluated after all estimator/parameter selection.",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "target": "ref_rr_mean_bpm",
        "feature_policy": "PPG-only or ECG+PPG prefixes; no RESP-derived predictors.",
        "clinical_status": "Research baseline comparison; not clinically validated.",
        "warning": "A single held-out split is not external validation. Do not select a production model solely from this comparison.",
    }
    (args.output / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
