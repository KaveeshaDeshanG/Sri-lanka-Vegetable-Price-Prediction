"""Unit tests for app.services (the Streamlit-free logic backing the Streamlit app)."""

import json

import pandas as pd

from app.services import (
    build_prediction_input,
    load_sample_history,
    resolve_demo_models_dir,
    train_demo_model,
)
from src.models.pipeline import ModelTrainingResult
from src.prediction.predictor import Predictor

REQUIRED_COLUMNS = {"date", "vegetable", "market", "price", "rainfall_mm", "temperature_c", "humidity_pct"}


# 1. Bundled sample data ----------------------------------------------------------


def test_load_sample_history_has_expected_schema():
    df = load_sample_history()
    assert REQUIRED_COLUMNS <= set(df.columns)
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_sample_history_has_no_missing_values():
    df = load_sample_history()
    assert not df.isna().any().any()


def test_load_sample_history_is_labeled_as_synthetic_not_real_data():
    df = load_sample_history()
    assert set(df["source"].unique()) == {"SAMPLE"}


# 2. build_prediction_input --------------------------------------------------------


def test_build_prediction_input_appends_one_row_with_nan_price():
    history = load_sample_history()
    target_date = history["date"].max() + pd.Timedelta(days=1)

    result = build_prediction_input(history, target_date, "Carrot", "Dambulla")

    assert len(result) == len(history) + 1
    new_row = result.iloc[-1]
    assert new_row["date"] == target_date
    assert new_row["vegetable"] == "Carrot"
    assert new_row["market"] == "Dambulla"
    assert pd.isna(new_row["price"])
    assert pd.isna(new_row["rainfall_mm"])  # no forecast needed -- see build_prediction_input's docstring


def test_build_prediction_input_does_not_mutate_history():
    history = load_sample_history()
    original = history.copy()
    build_prediction_input(history, history["date"].max() + pd.Timedelta(days=1), "Carrot", "Dambulla")
    pd.testing.assert_frame_equal(history, original)


def test_build_prediction_input_omits_fuel_columns_when_history_has_none():
    """The bundled sample data has no fuel columns -- unchanged behavior, no columns added."""
    history = load_sample_history()
    target_date = history["date"].max() + pd.Timedelta(days=1)

    result = build_prediction_input(history, target_date, "Carrot", "Dambulla")

    assert "diesel_price" not in result.columns
    assert "petrol_price" not in result.columns


def test_build_prediction_input_carries_fuel_price_columns_when_present():
    """A model trained with fuel_prices configured needs diesel_price_lag_1d/
    petrol_price_lag_1d -- FuelPriceFeatureGenerator computes those from an
    EARLIER row's fuel price, never the target row's own (unknown) value, so
    the target row's own diesel_price/petrol_price must be NaN, same as price/
    weather, while historical rows are left untouched."""
    history = load_sample_history().assign(diesel_price=300.0, petrol_price=350.0)
    target_date = history["date"].max() + pd.Timedelta(days=1)

    result = build_prediction_input(history, target_date, "Carrot", "Dambulla")

    assert "diesel_price" in result.columns
    assert "petrol_price" in result.columns
    new_row = result.iloc[-1]
    assert pd.isna(new_row["diesel_price"])
    assert pd.isna(new_row["petrol_price"])
    assert not result["diesel_price"].iloc[:-1].isna().any()
    assert not result["petrol_price"].iloc[:-1].isna().any()


# 3. train_demo_model: the first real end-to-end wiring of the training chain ----


def test_train_demo_model_produces_a_loadable_predictor(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )
    history = load_sample_history()

    result = train_demo_model(history, str(config_path))

    assert isinstance(result, ModelTrainingResult)
    assert result.best_model_name == "linear_regression"
    demo_dir = tmp_path / "demo"
    assert (demo_dir / "best_model.joblib").exists()
    assert (demo_dir / "feature_pipeline_state.joblib").exists()

    # the whole point: a Predictor can be loaded from what this just persisted
    predictor = Predictor.load(config_path=str(config_path), models_dir=str(demo_dir))
    assert isinstance(predictor, Predictor)


def test_train_demo_model_trains_every_configured_model(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression", "random_forest", "xgboost"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )
    history = load_sample_history()

    result = train_demo_model(history, str(config_path))
    assert set(result.models) == {"linear_regression", "random_forest", "xgboost"}


# 4. train_demo_model must never write into the configured production models_dir ---


def test_train_demo_model_never_writes_into_the_configured_production_models_dir(tmp_path):
    """Regression test for the incident this fixed: a sandbox/demo retrain from the
    Predict page once silently overwrote the real production best_model.joblib
    because train_demo_model() saved straight to config's paths.models_dir. It must
    now always land under a separate directory instead."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )
    history = load_sample_history()

    train_demo_model(history, str(config_path))

    assert not (tmp_path / "best_model.joblib").exists()
    assert not (tmp_path / "feature_pipeline_state.joblib").exists()
    assert (tmp_path / "demo" / "best_model.joblib").exists()


def test_train_demo_model_respects_an_explicit_demo_models_dir_config(tmp_path):
    demo_dir = tmp_path / "sandbox_models"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n  demo_models_dir: "{demo_dir.as_posix()}"\n'
    )
    history = load_sample_history()

    train_demo_model(history, str(config_path))

    assert (demo_dir / "best_model.joblib").exists()
    assert not (tmp_path / "best_model.joblib").exists()


def test_train_demo_model_output_dir_argument_overrides_the_config_default(tmp_path):
    explicit_dir = tmp_path / "wherever"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )
    history = load_sample_history()

    train_demo_model(history, str(config_path), output_dir=str(explicit_dir))

    assert (explicit_dir / "best_model.joblib").exists()
    assert not (tmp_path / "demo" / "best_model.joblib").exists()


def test_train_demo_model_writes_artifact_metadata_for_the_same_run(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )
    history = load_sample_history()

    result = train_demo_model(history, str(config_path))

    metadata = json.loads((tmp_path / "demo" / "artifact_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_type"] == type(result.best_model).__name__
    assert metadata["model_name"] == result.best_model_name
    assert metadata["feature_count"] == len(metadata["feature_columns"])
    assert metadata["metrics"] == result.metrics[result.best_model_name]
    assert "NOT the real production HARTI dataset" in metadata["training_data"]["source"]


# 5. resolve_demo_models_dir --------------------------------------------------------


def test_resolve_demo_models_dir_defaults_to_a_demo_subdir_of_models_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n')

    assert resolve_demo_models_dir(str(config_path)) == str(tmp_path / "demo")


def test_resolve_demo_models_dir_honors_explicit_config_value(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n  demo_models_dir: "somewhere/else"\n'
    )

    assert resolve_demo_models_dir(str(config_path)) == "somewhere/else"
