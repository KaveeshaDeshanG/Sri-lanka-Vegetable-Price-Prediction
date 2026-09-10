"""Unit tests for src.models.trainer.ModelTrainer."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
from src.models.trainer import ModelTrainer, ModelTrainingError


def _daily_group_df(start, n_days, vegetable, market, price_start, price_step=1.0, source="HARTI") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "vegetable": [vegetable] * n_days,
            "market": [market] * n_days,
            "source": [source] * n_days,
            "price": [price_start + price_step * i for i in range(n_days)],
            "rainfall_mm": [5.0] * n_days,
            "temperature_c": [28.0] * n_days,
            "humidity_pct": [70.0] * n_days,
        }
    )


@pytest.fixture
def train_test_data():
    group_a = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    group_b = _daily_group_df("2024-01-01", 50, "Tomato", "Pettah", 80.0)
    merged = pd.concat([group_a, group_b], ignore_index=True)
    builder = TrainTestDatasetBuilder(config_path="config/config.yaml")
    return builder.build(merged, test_size=0.2)  # (X_train, X_test, y_train, y_test)


# 1. Trains every model configured in config.model.models_to_train --------------


def test_train_all_fits_every_configured_model(train_test_data):
    X_train, _, y_train, _ = train_test_data
    trainer = ModelTrainer(config_path="config/config.yaml")
    models = trainer.train_all(X_train, y_train)

    assert set(models) == {"linear_regression", "random_forest", "xgboost"}
    assert isinstance(models["linear_regression"], LinearRegression)
    assert isinstance(models["random_forest"], RandomForestRegressor)
    assert isinstance(models["xgboost"], XGBRegressor)


def test_train_all_models_are_fitted_and_can_predict(train_test_data):
    X_train, X_test, y_train, _ = train_test_data
    trainer = ModelTrainer(config_path="config/config.yaml")
    models = trainer.train_all(X_train, y_train)

    for model in models.values():
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


# 2. Configured hyperparameters are actually applied, not hardcoded -------------


def test_random_forest_uses_configured_hyperparameters():
    trainer = ModelTrainer(config_path="tests/fixtures/model_test_config.yaml")
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([1.0, 2.0, 3.0, 4.0])
    models = trainer.train_all(X_train, y_train)

    assert models["random_forest"].n_estimators == 5
    assert models["random_forest"].max_depth == 3


def test_xgboost_uses_configured_hyperparameters():
    trainer = ModelTrainer(config_path="tests/fixtures/model_test_config.yaml")
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([1.0, 2.0, 3.0, 4.0])
    models = trainer.train_all(X_train, y_train)

    assert models["xgboost"].n_estimators == 6
    assert models["xgboost"].learning_rate == 0.3
    assert models["xgboost"].max_depth == 2


# 3. Configured random seed drives reproducibility -------------------------------


def test_random_forest_and_xgboost_are_deterministic_given_the_same_config():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y_train = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    X_check = pd.DataFrame({"a": [7.0, 8.0]})

    models_a = ModelTrainer(config_path="config/config.yaml").train_all(X_train, y_train)
    models_b = ModelTrainer(config_path="config/config.yaml").train_all(X_train, y_train)

    assert list(models_a["random_forest"].predict(X_check)) == list(models_b["random_forest"].predict(X_check))
    assert list(models_a["xgboost"].predict(X_check)) == list(models_b["xgboost"].predict(X_check))


# 4. Unknown model name in config raises rather than being silently skipped -----


def test_unknown_model_name_raises():
    trainer = ModelTrainer(config_path="tests/fixtures/model_test_config_invalid_model.yaml")
    with pytest.raises(ModelTrainingError, match="not_a_real_model"):
        trainer.train_all(pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([1.0, 2.0]))
