# PulseLab architecture — dual real-data pipelines

## Design principle

The browser contains no signal simulator or model implementation. It renders data returned by the local Python API and provides controls for the two research engines. Dataset loaders, preprocessing, feature schemas, model artifacts and evaluation remain separate for MIT-BIH and BIDMC.

## Pipeline 1 — MIT-BIH

`WFDB records → integrity audit → ECG preprocessing → R-peak detection → beat segmentation → 47 features → Rough Set reduct → fuzzy AAMI classifier / baselines → patient-disjoint evaluation → saved artifacts → Python inference`

Core modules: `modules/data_loader.py`, `preprocessing.py`, `rpeak_detection.py`, `segmentation.py`, `feature_extraction.py`, `rough_set.py`, `fuzzy_logic.py`, `hybrid_model.py`, `pipeline.py`, `inference.py`.

## Pipeline 2 — BIDMC

`BIDMC Signals/Breaths/Numerics CSVs → record and patient metadata discovery → ECG/PPG/RESP conditioning → synchronized windows → per-modality features → multimodal fusion → dual-annotator respiratory references → leakage-controlled feature export → patient-grouped Kaggle training → Rough-Fuzzy respiratory-rate regression`

Core modules: `modules/bidmc_pipeline.py` (dataset discovery, record loading, preview, feature extraction and exports) and `modules/bidmc_model.py` (continuous regression using rough-set-selected features and fuzzy bin rules). The BIDMC model is not loaded or advertised as trained until a training artifact is supplied.

## API and interface

`server.py` exposes shared job polling plus separate MIT-BIH and BIDMC routes. `web/app.js` is an API client only; it contains no ECG generator, in-browser classifier or synthetic benchmark. `web/core.js` and the former browser demo smoke test have been removed.

## Separation guarantees

- MIT-BIH AAMI labels and BIDMC breath annotations are never interchanged.
- Each pipeline owns its dataset configuration, feature schema and model artifacts.
- BIDMC patient grouping is read from WFDB source metadata when available; CSV-only fallbacks are explicit and must be checked before patient-disjoint claims.
- For respiratory-rate prediction from RESP-derived manual annotations, the provided RR-estimation feature table excludes RESP-derived inputs and RESP cross-correlations. The full-fusion table is preserved for descriptive/multimodal work and must not be used to claim a non-leaky PPG-only or ECG+PPG respiratory estimate.
- Fitted imputers, rough-set feature selection, fuzzy membership functions and rule induction must use training patients only.
