"""Unit tests for src.models.metrics_report.save_metrics_report."""

import json
from pathlib import Path

from src.models.metrics_report import save_metrics_report


def _config_at(tmp_path: Path) -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  metrics_dir: "{(tmp_path / "metrics").as_posix()}"\n')
    return str(config_path)


# 1. Writes valid JSON with the expected structure ---------------------------------


def test_save_metrics_report_writes_expected_structure(tmp_path):
    config_path = _config_at(tmp_path)
    metrics = {
        "linear_regression": {"mae": 18.3, "rmse": 29.4, "r2": 0.95},
        "random_forest": {"mae": 17.9, "rmse": 28.8, "r2": 0.956},
    }

    saved_path = save_metrics_report(
        metrics, "random_forest", "rmse", config_path=config_path
    )

    assert saved_path == str(tmp_path / "metrics" / "model_metrics.json")
    report = json.loads(Path(saved_path).read_text(encoding="utf-8"))

    assert report["models"] == metrics
    assert report["best_model"] == {
        "name": "random_forest",
        "selection_metric": "rmse",
        "selection_value": 28.8,
    }
    assert "generated_at" in report


# 2. dataset_summary is passed through verbatim -------------------------------------


def test_save_metrics_report_includes_dataset_summary(tmp_path):
    config_path = _config_at(tmp_path)
    metrics = {"linear_regression": {"mae": 1.0, "rmse": 1.5, "r2": 0.9}}

    saved_path = save_metrics_report(
        metrics,
        "linear_regression",
        "rmse",
        dataset_summary={"train_rows": 100, "test_rows": 25, "feature_count": 10},
        config_path=config_path,
    )

    report = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert report["dataset"] == {"train_rows": 100, "test_rows": 25, "feature_count": 10}


# 3. dataset_summary defaults to an empty dict when omitted -------------------------


def test_save_metrics_report_defaults_dataset_summary_to_empty_dict(tmp_path):
    config_path = _config_at(tmp_path)
    metrics = {"linear_regression": {"mae": 1.0, "rmse": 1.5, "r2": 0.9}}

    saved_path = save_metrics_report(metrics, "linear_regression", "rmse", config_path=config_path)

    report = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert report["dataset"] == {}


# 4. Output directory is created if missing -----------------------------------------


def test_save_metrics_report_creates_metrics_dir_if_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    nested_dir = tmp_path / "nested" / "metrics"
    config_path.write_text(f'paths:\n  metrics_dir: "{nested_dir.as_posix()}"\n')
    metrics = {"linear_regression": {"mae": 1.0, "rmse": 1.5, "r2": 0.9}}

    saved_path = save_metrics_report(
        metrics, "linear_regression", "rmse", config_path=str(config_path)
    )

    assert Path(saved_path).exists()
    assert Path(saved_path).parent == nested_dir
