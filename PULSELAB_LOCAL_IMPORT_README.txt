PulseLab Kaggle model export
Run ID: pulselab_patient_disjoint_detected_20261114T053618Z_5383
Source repository commit: 0bb5d33755c90616fa1b89a0a4835dae68a541da

CONTENTS
- models/pulselab_patient_disjoint_detected_20261114T053618Z_5383/          Saved Joblib model files
- experiments/pulselab_patient_disjoint_detected_20261114T053618Z_5383/    Run manifest, metrics, predictions, reports and diagnostics
- pulselab_model_runtime_requirements.txt  Exact key package versions used in Kaggle
- pulselab_training_environment.json       Python and package version record

LOCAL IMPORT
1. Close the running PulseLab server.
2. Extract this ZIP directly into the root folder of your existing PulseLab repository.
   The ZIP intentionally starts with the folders models/ and experiments/.
3. Merge/overwrite only if Windows asks about existing files. This run has a unique run ID.
4. Activate your local project virtual environment and install the pinned runtime versions:
   python -m pip install -r pulselab_model_runtime_requirements.txt
5. Start the local app from the repository root:
   python -m uvicorn server:app --host 127.0.0.1 --port 8000
6. Open http://127.0.0.1:8000/web/ and visit Model Inference. The run should be listed.

IMPORTANT
- Keep the repository's modules/ source code unchanged; Joblib artifacts refer to these Python classes.
- Use the PulseLab source snapshot from commit 0bb5d33755c90616fa1b89a0a4835dae68a541da; do not mix artifacts with a different code version.
- The run manifest contains relative model paths such as models/pulselab_patient_disjoint_detected_20261114T053618Z_5383/....
- Do not rename the run folder.
- Only load Joblib files that you created and trust.
- Academic research prototype only; not clinically validated.
