"""PulseLab Phase 3 research console: dataset QA, detector validation and classification."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

import config
from modules import data_loader, pipeline
from modules.dataset_audit import audit_dataset, write_audit_report
from modules.validation import run_rpeak_validation

st.set_page_config(page_title="PulseLab Research Console", page_icon="📈", layout="wide")
st.title("PulseLab Research Console")
st.caption("Phase 3 · Real MIT-BIH data validation and reproducible experiments")
st.warning(config.DISCLAIMER, icon="⚠️")
st.info("Offline / retrospective research workflow. Zero-phase filtering, record-level scaling and next-RR features use full-record or future-beat context; this is not a streaming or real-time diagnostic system.")

with st.sidebar:
    st.header("Study protocol")
    split_label = st.selectbox(
        "Train / test split",
        ["Strict patient-disjoint (recommended)", "Historical de Chazal 22 / 22 (record-wise)"],
        index=0,
    )
    split_protocol = "patient-disjoint" if split_label.startswith("Strict") else "de-chazal-record-wise"
    run_size = st.radio("Experiment size", ["Quick engineering subset", "Complete configured split"], index=0)
    quick = run_size.startswith("Quick")
    peak_source_label = st.radio(
        "Beat location source",
        ["Automatic R-peak detector (recommended)", "Reference-annotation guided benchmark"],
        index=0,
        help=("Automatic mode runs the detector on every record and evaluates classification only on detected beats "
              "matched one-to-one to reference annotations. Annotation-guided mode is an easier benchmark and "
              "must not be described as end-to-end performance."),
    )
    peak_source = "detected" if peak_source_label.startswith("Automatic") else "annotations"
    include_baselines = st.checkbox("Include Random Forest and RBF-SVM", value=True)
    save_models = st.checkbox("Save model artifacts", value=True)
    st.divider()
    st.caption(f"WFDB data directory: `{config.DATA_DIR}`")

train_all, test_all = config.get_record_split(split_protocol)
train_ids = train_all[:8] if quick else train_all
test_ids = test_all[:6] if quick else test_all
requested = train_ids + test_ids
available = set(data_loader.available_records(config.DATA_DIR))
missing = [rid for rid in requested if rid not in available]
patient_overlap = sorted(
    {config.patient_id_for_record(r) for r in train_ids}
    & {config.patient_id_for_record(r) for r in test_ids}
)

metric_cols = st.columns(4)
metric_cols[0].metric("Training records", len(train_ids))
metric_cols[1].metric("Testing records", len(test_ids))
metric_cols[2].metric("Available records", len(available))
metric_cols[3].metric("Missing selected", len(missing))

if patient_overlap:
    st.error(f"Patient overlap detected: {patient_overlap}. This split is not patient-disjoint.")
elif split_protocol == "patient-disjoint":
    st.success("The selected split is patient-disjoint. Record 202 is excluded because it shares a subject with record 201.")
else:
    st.info("Historical de Chazal compatibility mode. The complete 22/22 lists place records 201 and 202 from the same subject on opposite sides; a quick truncated subset may omit the pair.")

if missing:
    st.info("The selected real MIT-BIH records are not all present locally. The database is not bundled with PulseLab.")
    if st.button("Download selected MIT-BIH records", type="primary"):
        bar = st.progress(0.0)
        status = st.empty()
        try:
            data_loader.download_records(
                requested,
                progress=lambda fraction, message: (bar.progress(fraction), status.write(message)),
            )
            st.success("Download request completed. Refresh the page to re-check local files.")
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset download failed: {type(exc).__name__}: {exc}")


tabs = st.tabs(["Overview", "Dataset quality", "R-peak validation", "Beat classification", "Results"])

with tabs[0]:
    st.subheader("Research workflow")
    st.markdown(
        "**MIT-BIH records → WFDB integrity audit → ECG preprocessing → R-peak detection → beat windows → "
        "47-feature extraction → Rough Set reduction → fuzzy rules → AAMI classification.**"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Model comparison")
        st.markdown("- A: Fuzzy classifier using all Python features\n- B: Rough Set reduct + fuzzy classifier\n- C1: Random Forest\n- C2: RBF-SVM")
    with c2:
        st.markdown("#### Evaluation safeguards")
        st.markdown(
            "- Strict patient-disjoint split by default\n- Training-only imputation and feature selection\n"
            "- Automatic detector validation reported separately\n- Record-cluster confidence intervals\n"
            "- No clinical probability or diagnostic claims"
        )
    st.caption("Browser simulation remains separate from the real-data Python research engine.")

with tabs[1]:
    st.subheader("Dataset quality and provenance")
    st.write("Inspect every selected record's WFDB files, sampling rate, channels, annotation counts, AAMI class distribution and file hashes.")
    if st.button("Run dataset integrity audit", type="primary"):
        with st.spinner("Reading WFDB records and computing file hashes…"):
            try:
                report = audit_dataset(
                    requested,
                    train_record_ids=train_ids,
                    test_record_ids=test_ids,
                    split_protocol=split_protocol,
                    hash_files=True,
                )
                audit_dir = config.EXPERIMENTS_DIR / ("dataset_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
                write_audit_report(report, audit_dir)
                st.session_state["last_audit_dir"] = str(audit_dir)
                st.session_state["last_audit"] = report
                st.success(f"Audit saved to {audit_dir}")
            except Exception as exc:
                st.error(f"Dataset audit failed: {type(exc).__name__}: {exc}")
    audit = st.session_state.get("last_audit")
    if audit:
        a, b, c = st.columns(3)
        a.metric("Records readable", f"{audit['n_records_ok']}/{audit['n_records_requested']}")
        b.metric("Patient-disjoint", "Yes" if audit["patient_disjoint"] else "No")
        c.metric("Missing / failed", audit["n_records_failed"])
        st.markdown("#### AAMI class counts")
        class_rows = [
            {"Split": split.title(), "Class": label, "Beats": count}
            for split, values in audit["class_counts"].items() if split in {"train", "test"}
            for label, count in values.items()
        ]
        st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Record inventory")
        st.dataframe(pd.DataFrame(audit["records"]), use_container_width=True, hide_index=True)
        if audit["warnings"]:
            for warning in audit["warnings"]:
                st.warning(warning)
        file = Path(st.session_state["last_audit_dir"]) / "dataset_audit.json"
        if file.exists():
            st.download_button("Download dataset audit JSON", file.read_bytes(), file_name="dataset_audit.json", mime="application/json")

with tabs[2]:
    st.subheader("Automatic R-peak detector validation")
    st.write("The detector is tested independently against AAMI-mappable reference annotations using one-to-one matching.")
    tolerance_ms = st.slider("Matching tolerance (milliseconds)", min_value=25, max_value=150, value=75, step=5)
    if st.button("Validate detector on selected test records", disabled=bool(missing), type="primary"):
        bar = st.progress(0.0)
        status = st.empty()
        try:
            output = run_rpeak_validation(
                test_ids,
                tolerance_seconds=tolerance_ms / 1000.0,
                progress=lambda fraction, message: (bar.progress(fraction), status.write(message)),
            )
            st.session_state["last_detection_dir"] = str(output)
            st.success(f"R-peak validation saved to {output}")
        except Exception as exc:
            st.error(f"R-peak validation failed: {type(exc).__name__}: {exc}")
    detection_dir = st.session_state.get("last_detection_dir")
    if detection_dir and Path(detection_dir, "rpeak_validation.json").exists():
        payload = json.loads(Path(detection_dir, "rpeak_validation.json").read_text(encoding="utf-8"))
        aggregate = payload["aggregate"]
        d1, d2, d3 = st.columns(3)
        d1.metric("Sensitivity", f"{aggregate['micro_sensitivity']:.3%}")
        d2.metric("PPV", f"{aggregate['micro_ppv']:.3%}")
        d3.metric("F1", f"{aggregate['micro_f1']:.3%}")
        st.caption("These numbers measure R-peak detection only; they are not arrhythmia-classification accuracy.")
        st.dataframe(pd.DataFrame(payload["per_record"]), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("AAMI heartbeat classification experiment")
    st.write("The default automatic mode extracts classifier features from detected beats. Classification scores are conditional on detections matched to the reference beat annotations; false positives and missed beats are reported separately.")
    if peak_source == "annotations":
        st.warning("Annotation-guided mode uses reference beat locations. Use it only as a controlled classifier benchmark, not as end-to-end performance.")
    if st.button("Run selected experiment", disabled=bool(missing) or (bool(patient_overlap) and split_protocol == "patient-disjoint"), type="primary"):
        bar = st.progress(0.0)
        status = st.empty()
        try:
            result_path = pipeline.run_experiment(
                train_ids,
                test_ids,
                progress=lambda fraction, message: (bar.progress(fraction), status.write(message)),
                include_baselines=include_baselines,
                save_models=save_models,
                peak_source=peak_source,
                split_protocol=split_protocol,
                allow_patient_overlap=(split_protocol == "de-chazal-record-wise"),
            )
            st.session_state["last_run_path"] = str(result_path)
            st.success(f"Experiment completed: {result_path}")
        except Exception as exc:
            st.error(f"Experiment failed: {type(exc).__name__}: {exc}")

with tabs[4]:
    st.subheader("Saved experiment results")
    last_run = st.session_state.get("last_run_path")
    if last_run and Path(last_run, "run_manifest.json").exists():
        run_path = Path(last_run)
        manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
        st.caption(f"Run {manifest['run_id']} · {manifest['created_at_utc']} · {manifest.get('metadata', {}).get('split_policy', 'unspecified')}")
        comparison_file = run_path / "model_comparison.csv"
        if comparison_file.exists():
            comparison = pd.read_csv(comparison_file)
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.bar_chart(comparison.set_index("model")[["macro_f1"]])
        per_record_file = run_path / "per_record_model_metrics.csv"
        if per_record_file.exists():
            with st.expander("Per-record classification metrics"):
                st.dataframe(pd.read_csv(per_record_file), use_container_width=True, hide_index=True)
        detection_file = run_path / "rpeak_detection_per_record.csv"
        if detection_file.exists():
            with st.expander("R-peak detector results captured during feature extraction"):
                st.dataframe(pd.read_csv(detection_file), use_container_width=True, hide_index=True)
        with st.expander("Full run manifest"):
            st.json(manifest)
        if comparison_file.exists():
            st.download_button("Download model comparison CSV", comparison_file.read_bytes(), file_name="model_comparison.csv", mime="text/csv")
    else:
        st.info("No experiment has been run in this Streamlit session yet. Run one from the Beat classification tab.")

st.divider()
st.caption("PulseLab Phase 3 · MIT-BIH data must be downloaded separately from PhysioNet. Academic research only; not clinically validated.")
