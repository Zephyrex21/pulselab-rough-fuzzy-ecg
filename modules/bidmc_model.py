"""Rough-set + fuzzy respiratory-rate regressor for the BIDMC pipeline.

The model is intentionally not trained during project startup. It is trained
later from patient-disjoint feature exports in the Kaggle workflow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.rough_set import RoughSetSelector
from modules.fuzzy_logic import FuzzyRuleClassifier


class BIDMCModelError(ValueError):
    pass


class RoughFuzzyRespiratoryRegressor:
    """Continuous RR regressor built from rough-set-selected fuzzy class rules.

    The continuous target is quantile-binned during training. Fuzzy rule scores
    are then defuzzified as a weighted sum of training-only bin-centre rates.
    Patient splitting and all preprocessing must be performed before fitting.
    """

    def __init__(self, n_target_bins: int = 5, seed: int = 42):
        if n_target_bins < 3:
            raise ValueError("At least three respiratory-rate bins are required.")
        self.n_target_bins = int(n_target_bins)
        self.seed = int(seed)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        target = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
        values = X.replace([np.inf, -np.inf], np.nan)
        valid = np.isfinite(target)
        values = values.loc[valid].reset_index(drop=True)
        target = target[valid]
        if len(target) < 30:
            raise BIDMCModelError("At least 30 labelled windows are required to fit the BIDMC regressor.")
        if values.empty or values.shape[1] == 0:
            raise BIDMCModelError("No predictor features were provided.")
        if not np.isfinite(target).all():
            raise BIDMCModelError("Respiratory-rate target contains invalid values.")
        edges = np.unique(np.quantile(target, np.linspace(0, 1, self.n_target_bins + 1)))
        if len(edges) < 4:
            raise BIDMCModelError("The training target has insufficient variation for fuzzy regression bins.")
        labels = np.digitize(target, edges[1:-1], right=False).astype(int)
        self.bin_centres_ = np.array([float(np.mean(target[labels == i])) for i in range(len(edges) - 1) if np.any(labels == i)])
        # Remove empty quantile bins and remap labels so the selector and fuzzy classifier agree.
        occupied = sorted(np.unique(labels).tolist())
        remap = {old: new for new, old in enumerate(occupied)}
        labels = np.array([remap[x] for x in labels], dtype=int)
        self.bin_centres_ = np.array([float(np.mean(target[labels == i])) for i in range(len(occupied))])
        self.target_edges_ = edges.tolist()
        self.selector_ = RoughSetSelector(n_bins=4, max_per_class=2000, seed=self.seed).fit(values, labels)
        selected = values.loc[:, self.selector_.selected_]
        self.fuzzy_ = FuzzyRuleClassifier(
            n_rules_per_class=12, max_antecedents=3, beam_width=30,
            min_support=0.03, max_samples_per_class=2000, seed=self.seed,
        ).fit(selected, labels)
        self.feature_names_in_ = list(values.columns)
        self.feature_names_selected_ = list(self.selector_.selected_)
        self.n_training_windows_ = len(values)
        self.target_summary_ = {
            "minimum_bpm": float(np.min(target)), "maximum_bpm": float(np.max(target)),
            "median_bpm": float(np.median(target)), "mean_bpm": float(np.mean(target)),
        }
        return self

    def _selected(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fuzzy_"):
            raise RuntimeError("BIDMC respiratory-rate model has not been fitted.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        missing = [f for f in self.feature_names_in_ if f not in X.columns]
        if missing:
            raise BIDMCModelError(f"Missing model input features: {missing}")
        return X.loc[:, self.feature_names_selected_]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        selected = self._selected(X)
        scores = self.fuzzy_.predict_proba(selected)
        classes = self.fuzzy_.classes_.astype(int)
        centres = self.bin_centres_[classes]
        return scores @ centres

    def explain(self, row: pd.Series | dict) -> dict:
        frame = pd.DataFrame([row]) if isinstance(row, dict) else row.to_frame().T if isinstance(row, pd.Series) else row
        selected = self._selected(frame)
        fuzzy_explanation = self.fuzzy_.explain(selected.iloc[0])
        scores = fuzzy_explanation["class_scores"]
        predicted = float(sum(float(scores.get(str(i), 0.0)) * self.bin_centres_[i] for i in range(len(self.bin_centres_))))
        return {
            "predicted_respiratory_rate_bpm": predicted,
            "target_bin_centres_bpm": self.bin_centres_.tolist(),
            "fuzzy_class_scores": scores,
            "rough_set": self.selector_.report(),
            "fuzzy_rules": fuzzy_explanation["top_rules"],
        }
