"""Tests for scripts/train.py's orchestration.

These tests never call model.fit() for real: the non-dry-run path is
exercised with ModelTrainingPipeline mocked out, so this file trains no
model at all -- it only proves scripts/train.py wires the existing,
already-tested components together correctly and in the right order.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from scripts.train import parse_args, run_training


def _write_synthetic_raw_data(tmp_path: Path, n_days: int = 50):
    """n_days of one (vegetable, market) group with complete daily weather + fuel data.

    n_days=50 matches the scale tests/feature_engineering/test_dataset_builder.py already
    uses for drop_incomplete_rows=True coverage -- large enough that price_lag_30d (the
    most restrictive engineered feature) has real, non-NaN rows in both the train and test
    splits, rather than every row being dropped as incomplete.
    """
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D").strftime("%Y-%m-%d")

    market_df = pd.DataFrame(
        {
            "date": dates,
            "vegetable": ["Carrot"] * n_days,
            "market": ["Dambulla"] * n_days,
            "price": [100.0 + i for i in range(n_days)],  # rises monotonically with date
        }
    )
    weather_df = pd.DataFrame(
        {
            "date": dates,
            "rainfall_mm": [5.0] * n_days,
            "temperature_c": [28.0] * n_days,
            "humidity_pct": [70.0] * n_days,
        }
    )
    fuel_df = pd.DataFrame(
        {
            "date": dates,
            "diesel_price": [300.0 + i for i in range(n_days)],
            "petrol_price": [350.0 + i for i in range(n_days)],
        }
    )

    market_path = tmp_path / "market.csv"
    weather_path = tmp_path / "weather.csv"
    fuel_path = tmp_path / "fuel.csv"
    market_df.to_csv(market_path, index=False)
    weather_df.to_csv(weather_path, index=False)
    fuel_df.to_csv(fuel_path, index=False)
    return market_path, weather_path, fuel_path


def _write_config(
    tmp_path: Path,
    market_path: Path,
    weather_path: Path,
    fuel_path: Path | None = None,
    models_to_train: list[str] | None = None,
) -> Path:
    config = {
        "project": {"random_seed": 42},
        "paths": {
            "models_dir": str(tmp_path / "models"),
            "metrics_dir": str(tmp_path / "reports" / "metrics"),
            "raw_files": {
                "harti_prices": str(market_path),
                "dambulla_prices": None,
                "weather_data": str(weather_path),
                # Deliberately nonexistent: keeps this synthetic test hermetic --
                # apply_production_region_weather() treats a missing file as "leave
                # weather unchanged" (not fatal), so it must never resolve to the
                # real project's own region-weather file/mapping.
                "weather_data_by_region": str(tmp_path / "nonexistent_weather_data_by_region.csv"),
                "fuel_prices": str(fuel_path) if fuel_path else None,
            },
            "crop_production_regions_mapping": str(tmp_path / "nonexistent_crop_production_regions.yaml"),
        },
        "data_schema": {
            "date_column": "date",
            "vegetable_column": "vegetable",
            "market_column": "market",
            "price_column": "price",
            "weather_columns": ["rainfall_mm", "temperature_c", "humidity_pct"],
            "fuel_price_columns": ["diesel_price", "petrol_price"],
        },
        "feature_engineering": {
            "lag_periods_days": [1, 7, 14, 30],
            "rolling_windows_days": [7, 14, 30],
            "weather_lag_days": 14,
            "fuel_price_lag_days": 1,
        },
        "model": {
            "target_column": "price",
            "test_size": 0.2,
            "models_to_train": models_to_train or ["linear_regression"],
            "random_forest": {"n_estimators": 5, "max_depth": 3},
            "xgboost": {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 3},
        },
        "evaluation": {"metrics": ["mae", "rmse", "r2"], "best_model_metric": "rmse"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


# 1. CLI argument parsing -------------------------------------------------------


def test_parse_args_accepts_config_and_dry_run():
    args = parse_args(["--config", "some/path.yaml", "--dry-run"])
    assert args.config == "some/path.yaml"
    assert args.dry_run is True


def test_parse_args_dry_run_defaults_to_false():
    args = parse_args(["--config", "some/path.yaml"])
    assert args.dry_run is False


def test_parse_args_config_defaults_to_project_config():
    args = parse_args([])
    assert args.config == "config/config.yaml"


# 2. Dry-run against the REAL project config/data -------------------------------


def test_dry_run_completes_successfully_against_real_config():
    summary = run_training("config/config.yaml", dry_run=True)

    assert summary["dry_run"] is True
    assert summary["fuel_configured"] is True
    assert summary["fuel_rows"] > 0
    assert summary["market_rows"] > 0
    assert summary["weather_rows"] > 0
    assert summary["train_rows"] > 0
    assert summary["test_rows"] > 0

    # fuel lag features reach X; raw same-day fuel columns never do.
    assert "diesel_price_lag_1d" in summary["feature_names"]
    assert "petrol_price_lag_1d" in summary["feature_names"]
    assert "diesel_price" not in summary["feature_names"]
    assert "petrol_price" not in summary["feature_names"]

    # dry-run never reaches the training/persistence stage.
    assert "models_trained" not in summary
    assert "best_model_name" not in summary


def test_dry_run_against_real_config_does_not_touch_model_artifacts():
    model_path = Path("models/best_model.joblib")
    artifact_path = Path("models/feature_pipeline_state.joblib")
    mtimes_before = (model_path.stat().st_mtime, artifact_path.stat().st_mtime)

    run_training("config/config.yaml", dry_run=True)

    mtimes_after = (model_path.stat().st_mtime, artifact_path.stat().st_mtime)
    assert mtimes_before == mtimes_after


# 3. Dry-run against a small synthetic config (chronological split + fuel) ------


def test_dry_run_uses_chronological_split_with_no_train_test_overlap(tmp_path):
    market_path, weather_path, fuel_path = _write_synthetic_raw_data(tmp_path)
    config_path = _write_config(tmp_path, market_path, weather_path, fuel_path)

    summary = run_training(str(config_path), dry_run=True)

    assert summary["merged_rows"] == 50
    assert summary["train_rows"] > 0
    assert summary["test_rows"] > 0

    # Cross-check ordering directly against the same underlying components
    # run_training() itself calls -- proves the split really is chronological,
    # not just "some split."
    from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
    from src.preprocessing.pipeline import PreprocessingPipeline

    preprocessing = PreprocessingPipeline(config_path=str(config_path))
    preprocessing.load_data()
    preprocessing.clean_data()
    preprocessing.transform_data()
    merged = preprocessing.merge_data()
    _, _, y_train, y_test = TrainTestDatasetBuilder(config_path=str(config_path)).build(merged)

    # price rises monotonically with date in this fixture, so this proves
    # every train date is strictly earlier than every test date.
    assert y_train.max() < y_test.min()


def test_dry_run_never_constructs_model_training_pipeline(tmp_path):
    market_path, weather_path, fuel_path = _write_synthetic_raw_data(tmp_path)
    config_path = _write_config(tmp_path, market_path, weather_path, fuel_path)

    with patch("scripts.train.ModelTrainingPipeline") as mock_pipeline_cls, \
         patch("scripts.train.save_inference_artifact") as mock_save_artifact:
        run_training(str(config_path), dry_run=True)

    mock_pipeline_cls.assert_not_called()
    mock_save_artifact.assert_not_called()


def test_dry_run_without_fuel_configured_matches_prior_behavior(tmp_path):
    """fuel_prices left null -- pipeline must behave exactly as it did before fuel support."""
    market_path, weather_path, _ = _write_synthetic_raw_data(tmp_path)
    config_path = _write_config(tmp_path, market_path, weather_path, fuel_path=None)

    summary = run_training(str(config_path), dry_run=True)

    assert summary["fuel_configured"] is False
    assert summary["fuel_rows"] == 0
    assert "diesel_price_lag_1d" not in summary["feature_names"]
    assert "petrol_price_lag_1d" not in summary["feature_names"]
    assert summary["train_rows"] > 0
    assert summary["test_rows"] > 0


# 4. Full (non-dry-run) orchestration -- ModelTrainingPipeline fully mocked, ----
#    so no real model.fit() is ever called anywhere in this test file. ---------


def test_full_run_discovers_models_from_config_and_never_hardcodes_them(tmp_path):
    """Proves scripts/train.py passes config_path through to ModelTrainingPipeline (which
    reads config.model.models_to_train itself) rather than hardcoding any model name --
    and that fuel lag features reach training input while raw fuel columns are excluded."""
    market_path, weather_path, fuel_path = _write_synthetic_raw_data(tmp_path)
    config_path = _write_config(
        tmp_path, market_path, weather_path, fuel_path, models_to_train=["linear_regression"]
    )

    fake_result = SimpleNamespace(
        models={"linear_regression": object()},
        metrics={"linear_regression": {"mae": 1.0, "rmse": 1.5, "r2": 0.9}},
        best_model_name="linear_regression",
        best_model=object(),
        saved_model_path=str(tmp_path / "models" / "best_model.joblib"),
    )

    with patch("scripts.train.ModelTrainingPipeline") as mock_pipeline_cls, patch(
        "scripts.train.save_inference_artifact", return_value=str(tmp_path / "models" / "feature_pipeline_state.joblib")
    ) as mock_save_artifact:
        mock_pipeline_cls.return_value.run.return_value = fake_result

        summary = run_training(str(config_path), dry_run=False)

    # config_path propagated -- ModelTrainingPipeline itself decides which
    # models to build by reading config.model.models_to_train; scripts/train.py
    # never passes a model list of its own.
    mock_pipeline_cls.assert_called_once_with(config_path=str(config_path))
    mock_pipeline_cls.return_value.run.assert_called_once()

    call_args = mock_pipeline_cls.return_value.run.call_args.args
    X_train, X_test, y_train, y_test = call_args
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert "diesel_price_lag_1d" in X_train.columns
    assert "petrol_price_lag_1d" in X_train.columns
    assert "diesel_price" not in X_train.columns
    assert "petrol_price" not in X_train.columns
    assert "diesel_price" not in X_test.columns

    mock_save_artifact.assert_called_once()

    assert summary["models_trained"] == ["linear_regression"]
    assert summary["best_model_name"] == "linear_regression"
    assert summary["best_model_metric"] == "rmse"

    # metrics report is written under the config's (tmp_path) metrics_dir, never the
    # real project's reports/metrics/ -- proves this test leaves no artifact behind.
    metrics_report_path = Path(summary["metrics_report_path"])
    assert metrics_report_path == tmp_path / "reports" / "metrics" / "model_metrics.json"
    assert metrics_report_path.exists()
    report = json.loads(metrics_report_path.read_text(encoding="utf-8"))
    assert report["models"] == fake_result.metrics
    assert report["best_model"] == {
        "name": "linear_regression",
        "selection_metric": "rmse",
        "selection_value": fake_result.metrics["linear_regression"]["rmse"],
    }


# 5. Artifact metadata ties the model, encoder, and feature list to one run -----


def test_full_run_persists_artifact_metadata_for_the_same_training_run(tmp_path):
    """The model, the InferenceArtifact save_inference_artifact() is called with, and the
    persisted ArtifactMetadata must all share one training_run_id -- proves scripts/train.py
    (the real production entry point) actually wires that identifier through, not just that
    metadata is written at all."""
    market_path, weather_path, fuel_path = _write_synthetic_raw_data(tmp_path)
    config_path = _write_config(
        tmp_path, market_path, weather_path, fuel_path, models_to_train=["linear_regression"]
    )

    fake_best_model = SimpleNamespace()  # type(fake_best_model).__name__ == "SimpleNamespace"
    fake_result = SimpleNamespace(
        models={"linear_regression": object()},
        metrics={"linear_regression": {"mae": 1.0, "rmse": 1.5, "r2": 0.9}},
        best_model_name="linear_regression",
        best_model=fake_best_model,
        saved_model_path=str(tmp_path / "models" / "best_model.joblib"),
    )

    with patch("scripts.train.ModelTrainingPipeline") as mock_pipeline_cls, patch(
        "scripts.train.save_inference_artifact",
        return_value=str(tmp_path / "models" / "feature_pipeline_state.joblib"),
    ) as mock_save_artifact:
        mock_pipeline_cls.return_value.run.return_value = fake_result
        summary = run_training(str(config_path), dry_run=False)

    metadata_path = Path(summary["artifact_metadata_path"])
    assert metadata_path == tmp_path / "models" / "artifact_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    saved_inference_artifact = mock_save_artifact.call_args.args[0]
    assert saved_inference_artifact.training_run_id == summary["training_run_id"]
    assert metadata["training_run_id"] == summary["training_run_id"]
    assert metadata["model_type"] == "SimpleNamespace"
    assert metadata["model_name"] == "linear_regression"
    assert metadata["feature_count"] == len(metadata["feature_columns"])
    assert metadata["feature_columns"] == summary["feature_names"]
    assert metadata["metrics"] == fake_result.metrics["linear_regression"]
    assert metadata["training_data"]["row_count"] == summary["merged_rows"]
    assert "HARTI" in metadata["training_data"]["source"]


def test_full_run_never_writes_artifact_metadata_under_dry_run(tmp_path):
    market_path, weather_path, fuel_path = _write_synthetic_raw_data(tmp_path)
    config_path = _write_config(tmp_path, market_path, weather_path, fuel_path)

    run_training(str(config_path), dry_run=True)

    assert not (tmp_path / "models" / "artifact_metadata.json").exists()
