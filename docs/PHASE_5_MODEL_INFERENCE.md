# Phase 5 — Saved-model ECG inference

## Purpose

Phase 5 closes the Phase 4 gap between saved Python research models and the browser interface. It introduces a separate Model Inference workspace and a versioned-by-feature-schema upload contract. The older in-browser ECG demonstration remains synthetic and is not used by this path.

## Inference data flow

```text
CSV / TXT ECG upload
        |
        v
Upload validation (size, extension, duration, sampling rate)
        |
        v
Numeric channel parsing (1 or 2 leads; optional header)
        |
        v
Saved MIT-BIH model lookup (run manifest + artifact allowlist)
        |
        v
Python preprocessing and resampling to 360 Hz
        |
        v
Automatic R-peak detection and complete beat segmentation
        |
        v
Canonical feature extraction (python-mitdb-1.0.0)
        |
        v
Training-only median imputation from the selected run manifest
        |
        v
Saved model prediction and model-specific scores
        |
        v
Beat-level predictions + on-demand fuzzy explanations
```

## Model registry rules

The API does not accept an arbitrary model path or a model file upload. The model catalog is built by reading local `experiments/*/run_manifest.json` files and only returns a model when:

- The run identifies `MIT-BIH Arrhythmia Database` as its dataset.
- `feature_schema_version` equals the current canonical Python schema.
- The run manifest includes ordered feature columns.
- The metrics entry identifies a saved `.joblib` artifact.
- The resolved artifact is an existing file inside the project `models/` directory.

This prevents a browser request from selecting a path outside the local model store. Joblib files are trusted local artifacts; do not copy untrusted `.joblib` files into `models/`.

## CSV / TXT upload contract

- File type: `.csv` or `.txt` only.
- Maximum upload: 20 MB.
- Signal duration after cleaning: 3 seconds to 10 minutes.
- Sampling rate: 100 to 2000 Hz; signals are resampled to 360 Hz by the existing preprocessing pipeline.
- Channel count: the model's feature schema determines whether one or two numeric ECG signal columns are required.
- Header: optional. When present, common time/index columns such as `time`, `timestamp`, `sample` and `index` are excluded. Remaining numeric columns are selected from left to right.
- Headerless files: the first one or two numeric columns are treated as ECG leads. Do not prepend a numeric sample counter unless it is intentionally a signal channel.
- Short internal missing-value gaps may be interpolated; rows still incomplete after that limited interpolation are excluded. Constant channels are rejected.

The first selected signal channel is used as the primary channel for R-peak detection, matching training's `primary=0` convention. Keep lead order consistent with the model's training data.

## API routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/models` | Discover compatible saved MIT-BIH model artifacts |
| `POST` | `/api/inference` | Multipart CSV/TXT upload and saved-model prediction |
| `GET` | `/api/inferences/{id}/summary` | Read saved inference summary |
| `GET` | `/api/inferences/{id}/beats/{index}/explanation` | On-demand explanation for one beat |
| `GET` | `/api/inferences/{id}/artifacts/{name}` | Download allow-listed inference artifacts |

## Persisted outputs and privacy boundary

Each successful inference creates `experiments/inference_<id>/` containing:

- `inference_summary.json`
- `beat_predictions.csv`
- `beat_features.csv`

The raw uploaded signal bytes are processed in memory and are not written to disk. Derived features can still be sensitive; keep the local project folder private and remove inference folders when they are no longer needed.

## Scores and explanations

The UI labels the returned values as **model scores**, not probabilities. Fuzzy-only and Rough-Set + fuzzy models expose their fitted rule explanation, including active rules, firing strengths, membership values and class-score breakdown. Random Forest and SVM wrappers do not implement equivalent rule explanations; the UI states this instead of fabricating feature attributions.

Models trained using annotation-guided reference peaks are listed but flagged in inference results because uploaded signals use automatic R-peak detection. Their inference protocol therefore differs from their training feature-generation protocol.

## Validation

The Phase 5 test suite covers CSV/TXT parsing, time-column exclusion, short and incompatible files, saved model discovery, model path containment, beat prediction persistence, no raw-signal persistence, fuzzy explanation retrieval, multipart API upload and the full model-catalog → upload → prediction → explanation → artifact round trip.

The automated tests use deterministic fixtures. A full real-data validation requires installing `requirements.txt`, running a MIT-BIH experiment with model saving enabled, and testing with a separate ECG signal. Phase 5 does not claim clinical validation, streaming inference or external generalisation.

## Local security

Run the service only on loopback:

```powershell
py -m uvicorn server:app --host 127.0.0.1 --port 8000
```

The API has no authentication. Do not bind it to `0.0.0.0`, port-forward it, or expose it to a public network.
