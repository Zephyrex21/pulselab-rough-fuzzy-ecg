# PulseLab methodology

## MIT-BIH ECG classification

The MIT-BIH workflow uses ECG records and AAMI-mappable beat annotations. It preprocesses ECG, locates R-peaks, extracts beat-centred windows and computes the existing 47-feature schema. Rough Set Theory selects a reduced feature subset, and fuzzy rules produce interpretable AAMI beat classes. Random Forest and RBF-SVM provide baselines. Evaluation defaults to patient-disjoint record groups and reports per-class as well as aggregate metrics.

## BIDMC multimodal respiratory-rate research

The BIDMC loader reads ECG Lead II, PPG (`PLETH`) and impedance respiration (`RESP`) from synchronized 125 Hz recordings. Window-level features are calculated independently for each modality and then fused. Breath reference events remain separate for manual annotators ann1 and ann2. Respiratory-rate reference values are derived from median intervals between annotated breath samples.

Two exports are produced:

- **Full multimodal:** ECG, PPG and RESP features, suitable for exploratory and descriptive multimodal analysis.
- **RR estimation / no RESP predictors:** ECG + PPG and their pairwise features, excluding RESP-derived predictors and cross-correlations to avoid using the signal from which the manual reference labels were annotated as an input to its own estimator.

Use patient-grouped splits before any overlapping-window training. Report the two annotators separately and characterize their disagreement; do not collapse all label uncertainty into one supposedly perfect target. The model class included in `modules/bidmc_model.py` converts continuous respiratory-rate values into training-only quantile bins, applies Rough Set reduction to those ordinal decisions, induces fuzzy rules, and defuzzifies fuzzy class scores into a continuous rate. This model must be fitted and evaluated on the Kaggle training split before inference is enabled.

## Safety

PulseLab is an academic prototype. Neither model output nor monitor reference values are clinically validated by this application. No output should be used for diagnosis or treatment.
