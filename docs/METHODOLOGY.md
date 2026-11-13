# Research methodology

**Data.** MIT-BIH Arrhythmia Database, 48 records, 360 Hz, 2 leads (typically MLII + V1/V2/V4/V5). Beat annotations mapped to AAMI classes
(N: N,L,R,e,j; S: A,a,J,S; V: V,E; F: F; Q: /,f,Q). Non-beat symbols are excluded (listed in `config.EXCLUDED_SYMBOLS`).

**Preprocessing.** Artefact clipping -> two-stage median baseline estimate (200 ms, 600 ms) subtracted -> optional IIR notch -> 4th-order zero-phase
Butterworth band-pass 0.5-40 Hz -> record-level scaling by median |R amplitude| -> beat window [-250 ms, +400 ms].
The default Phase 3 experiment uses the automatic detector on both training and test records: 5-15 Hz detector band-pass, derivative, squaring, 120 ms moving integration, a global threshold of 30% of the 99.5th percentile of the integrated signal, a 250 ms refractory period, then local maximum-absolute-amplitude refinement. Detected peaks are matched one-to-one to AAMI-mappable reference annotations within 75 ms. Only matched detections with complete beat windows enter the supervised classifier; detector false positives and missed reference beats are reported separately. `--peak-source annotations` is an explicit annotation-guided benchmark, not an automatic inference estimate.

**Features.** RR: previous, next, local mean/std (10 intervals), ratios, heart rate. Per channel: R amplitude, QRS duration (width of region with
|slope| > 20% of max), energy, variance, skewness, kurtosis, peak-to-peak, area, pre/post-R means, T-wave amplitude, dominant frequency,
PSD fraction in 0.5-5 / 5-15 / 15-40 Hz, spectral entropy, relative db4 level-4 wavelet energies (A4, D4, D3, D2).

**Rough sets.** On a class-balanced train sample, each feature is discretised into 4 equal-frequency bins (cuts from train). For attribute subset B,
IND(B) partitions objects; the lower approximation of class X is the union of equivalence classes fully inside X, the upper approximation those intersecting X;
POS_B(d) is the union of lower approximations of all decision classes; gamma_B(d) = |POS_B(d)|/|U|. Attribute significance = gamma(B+{a}) - gamma(B).
The reduct is built greedily (QuickReduct) until gamma(R) = gamma(C) (tolerance configurable) and pruned by backward elimination. Core = attributes whose removal lowers gamma(C).
Limitation: with many features equal-frequency bins make most equivalence classes small, so gamma(C) approaches 1 and the reduct is a consistency-preserving, not accuracy-optimised, subset.

**Fuzzy classifier.** Each feature has Low/Medium/High terms (left-shoulder, triangular and right-shoulder memberships meeting at training quantiles 15/50/85%). Membership quantiles are fitted on a deterministic class-balanced training sample only; constant features receive a small non-zero shoulder span. Rules are conjunctions of up to 3 distinct features. For each class, a beam search scores antecedents by class-normalised fuzzy support and fuzzy confidence. A confidence-weighted marginal-coverage step selects up to 12 rules/class; rule weight = confidence. Firing = product t-norm; class score = 1 - prod(1 - weight*firing); row-normalised scores are used for comparison and prediction = argmax. These scores are not calibrated probabilities. Explanations include feature memberships, fired rules, top-two score margin, active-rule count and "no rule fired" status.

**Experiments.** Non-finite extracted feature values are replaced with training-set medians; the same training medians are applied to test rows. Models A (fuzzy, all features), B (rough set + fuzzy), C1 (Random Forest, 200 trees), C2 (RBF-SVM) share the same record-wise partitions.
Metrics: accuracy, balanced accuracy, per-class precision / recall(sensitivity) / specificity / F1, weighted F1, macro-F1 over classes represented in the test set, macro-F1 over all five AAMI classes, confusion matrices, one-vs-rest ROC-AUC and average precision where both positive and negative examples are present, and train/predict time. Missing classes are retained in the fixed-order confusion matrix and explicitly reported in the run manifest. For each model, PulseLab also writes per-record metrics and 95% percentile intervals from a record-cluster bootstrap that resamples whole held-out records, not individual beats.

**Split caveat and strict default.** Records 201 and 202 belong to the same subject. The common historical 22/22 de Chazal record-wise lists place them on opposite sides. PulseLab's default strict patient-disjoint split therefore uses all 22 DS1 training records and 21 DS2 test records, excluding 202. The historical list remains available via `--split-protocol de-chazal-record-wise` and is marked as not patient-disjoint.

**Offline / retrospective scope.** This is a batch-processing experiment, not a real-time streaming pipeline. Zero-phase filtering uses future samples, record-level R-amplitude normalisation uses detections across the record, and the `rr_next` feature requires the next detected beat. A streaming implementation would need causal filters, online normalisation and a latency-aware feature design, then a separate evaluation.

**Statistical caveat.** A cluster bootstrap on one 21-record test split is a descriptive uncertainty estimate, not independent external validation. The data are not a clinical cohort validation and the detector/classifier are not clinically validated.
