import json
import numpy as np
import pandas as pd
import pytest

from modules.pipeline import run_feature_experiment


def feature_tables():
    rng = np.random.default_rng(19)
    train_rows, test_rows = [], []
    centres = {"N": -1.0, "S": 0.0, "V": 1.0}
    for split, target in (("train", train_rows), ("test", test_rows)):
        for label, centre in centres.items():
            for idx in range(18):
                target.append({
                    "record": f"{split}_{label}_{idx // 6}",
                    "label": label,
                    "f1": rng.normal(centre, 0.25),
                    "f2": rng.normal(centre * 0.8, 0.3),
                    "f3": rng.normal(-centre, 0.35),
                })
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def test_pipeline_runs_python_fuzzy_and_hybrid_models(tmp_path):
    train, test = feature_tables()
    output = run_feature_experiment(
        train, test, feature_columns=["f1", "f2", "f3"],
        output_root=tmp_path, include_baselines=False, save_models=False,
        run_name="test-run",
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert set(manifest["models"]) == {"A: Fuzzy (all features)", "B: Rough Set + Fuzzy"}
    assert (output / "model_comparison.csv").exists()
    assert (output / "per_class_metrics.csv").exists()
    assert (output / "per_record_model_metrics.csv").exists()
    assert (output / "test_predictions.csv").exists()
    assert (output / "research_report.md").exists()
    assert (output / "metrics.json").exists()
    assert (output / "a_fuzzy_all_features_rule_base.json").exists()
    assert (output / "b_rough_set_fuzzy_rule_base.json").exists()
    assert "model_reports" in manifest


def test_pipeline_rejects_record_overlap(tmp_path):
    train, test = feature_tables()
    test.loc[test.index[0], "record"] = train.loc[train.index[0], "record"]
    with pytest.raises(ValueError, match="Record-wise leakage"):
        run_feature_experiment(
            train, test, feature_columns=["f1", "f2", "f3"],
            output_root=tmp_path, include_baselines=False, save_models=False,
        )


def test_pipeline_can_train_all_four_models_and_report_baselines(tmp_path):
    train, test = feature_tables()
    output = run_feature_experiment(
        train, test, feature_columns=["f1", "f2", "f3"],
        output_root=tmp_path, include_baselines=True, save_models=False,
        run_name="four-model-check",
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert len(manifest["models"]) == 4
    assert any("Random Forest" in name for name in manifest["models"])
    assert any("SVM" in name for name in manifest["models"])
    comparison = pd.read_csv(output / "model_comparison.csv")
    assert len(comparison) == 4
    assert "macro_f1_all_aami" in comparison.columns


def test_pipeline_rejects_same_subject_even_when_record_ids_differ(tmp_path):
    train, test = feature_tables()
    train["patient_id"] = "subject_201_202"
    test["patient_id"] = "subject_201_202"
    with pytest.raises(ValueError, match="Patient-wise leakage"):
        run_feature_experiment(
            train, test, feature_columns=["f1", "f2", "f3"],
            output_root=tmp_path, include_baselines=False, save_models=False,
        )


def test_record_level_runner_rejects_201_202_subject_overlap_before_io(tmp_path):
    from modules.pipeline import run_experiment
    with pytest.raises(ValueError, match="Patient-wise leakage"):
        run_experiment(
            ["201"], ["202"], data_dir=tmp_path,
            output_root=tmp_path / "out", save_models=False,
        )


def test_experiment_rejects_duplicate_record_ids(tmp_path):
    from modules.pipeline import run_experiment
    with pytest.raises(ValueError, match="duplicate record IDs"):
        run_experiment(["100", "100"], ["101"], data_dir=tmp_path, output_root=tmp_path / "out")


def test_full_record_pipeline_wires_detected_peak_diagnostics_and_reports(tmp_path):
    from types import SimpleNamespace

    from modules.feature_schema import feature_names
    from modules.pipeline import run_experiment

    rng = np.random.default_rng(1234)
    schema = feature_names(2)
    class_values = ["N", "S", "V", "F", "Q"]
    per_class_counts = [8, 8, 8, 7, 7]  # 38 matched, classifiable beats per record

    def fake_loader(record_id, data_dir):
        return SimpleNamespace(record_id=str(record_id), fs=360.0)

    def fake_feature_builder(record, peak_source="detected"):
        assert peak_source == "detected"
        rows = []
        record_offset = int(record.record_id) / 10000.0
        for class_index, (label, count) in enumerate(zip(class_values, per_class_counts)):
            for _ in range(count):
                row = {
                    name: float(rng.normal(class_index * 1.5 + record_offset, 0.2))
                    for name in schema
                }
                row.update({
                    "record": str(record.record_id),
                    "patient_id": f"subject_{record.record_id}",
                    "label": label,
                    "symbol": label,
                    "peak_source": "detected",
                    "r_sample": len(rows) * 300 + 200,
                    "reference_sample": len(rows) * 300 + 201,
                    "detection_error_ms": 1000.0 / 360.0,
                })
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame.attrs["peak_source"] = "detected"
        frame.attrs["detection_metrics"] = {
            "record": str(record.record_id),
            "patient_id": f"subject_{record.record_id}",
            "peak_source": "detected",
            "n_reference": 40,
            "n_detected": 40,
            "n_matched": 38,
            "n_classifiable_matched": 38,
            "matched_windows_excluded_at_record_edges": 0,
            "tp": 38,
            "fp": 2,
            "fn": 2,
            "sensitivity": 0.95,
            "ppv": 0.95,
            "f1": 0.95,
            "mean_signed_error_ms": 2.78,
            "mean_absolute_error_ms": 2.78,
        }
        return frame

    output = run_experiment(
        ["100", "101"], ["102", "103"],
        output_root=tmp_path / "experiments",
        include_baselines=False,
        save_models=False,
        run_name="detected-pipeline-fixture",
        record_loader=fake_loader,
        feature_builder=fake_feature_builder,
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    detection = json.loads((output / "rpeak_detection_metrics.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["peak_source"] == "detected"
    assert manifest["metadata"]["classification_evaluation"] == "conditional_on_matched_detected_beats"
    assert manifest["metadata"]["patient_disjoint"] is True
    assert len(manifest["train_records"]) == 2
    assert len(manifest["test_records"]) == 2
    assert detection["testing_records"]["tp"] == 76
    assert detection["testing_records"]["fp"] == 4
    assert detection["testing_records"]["fn"] == 4
    assert (output / "rpeak_detection_per_record.csv").exists()
    assert (output / "test_predictions.csv").exists()
    assert "conditional on detected" in (output / "research_report.md").read_text(encoding="utf-8").lower()
