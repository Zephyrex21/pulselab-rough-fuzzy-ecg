PulseLab supplied MIT-BIH training artifacts
Run ID: pulselab_patient_disjoint_detected_20261114T053618Z_5383
Source repository commit recorded by the run: 0bb5d33755c90616fa1b89a0a4835dae68a541da

CONTENTS IN THIS PROJECT
- models/pulselab_patient_disjoint_detected_20261114T053618Z_5383/          Four saved Joblib model files
- experiments/pulselab_patient_disjoint_detected_20261114T053618Z_5383/    Run manifest, metrics, predictions, reports and diagnostics
- requirements/mitbih-model-runtime.txt                                  Key package versions used in Kaggle
- docs/reproducibility/mitbih_training_environment.json                                      Python and package version record

CURRENT PROJECT
These files are already placed in PulseLab 0.7.0. No separate import or copy step is needed.

RUN LOCALLY
1. Activate the project virtual environment and install requirements.txt.
2. Start from the project root:
   python -m uvicorn server:app --host 127.0.0.1 --port 8000
3. Open http://127.0.0.1:8000/ and select the MIT-BIH ECG engine.

COMPATIBILITY NOTE
The run manifest records the source commit and 0.6.0 training source. The supplied project uses the adaptive R-peak detector introduced in 0.6.1. The detector change affects the feature-generation path, so treat the supplied metrics as historical results for that training run. Re-train MIT-BIH models with the current source before relying on inference as a matched model/preprocessing pair.

Uploaded-signal inference now requires an explicit `inference_pipeline_contract` in the run manifest. The contract records the feature-schema, preprocessing, segmentation, feature-extraction, sampling/filter settings and R-peak detector versions used for training. Legacy exports without this contract, exports with a different contract, and annotation-guided benchmark runs are not offered for uploaded-signal inference. Their historical metrics and research reports remain available. Re-run the patient-disjoint experiment with automatic R-peak detection to generate an eligible model export.

IMPORTANT
- Do not rename the run folder; the manifest contains relative paths into it.
- Only load Joblib files from trusted sources.
- The model is for academic research and is not clinically validated.
