"""Unit tests for src.models.model_selection.select_best_model."""

import pytest

from src.models.model_selection import ModelSelectionError, select_best_model


# 1. Default config metric (rmse, lower is better) --------------------------------


def test_select_best_model_picks_lowest_rmse_by_default():
    metrics_by_model = {
        "linear_regression": {"mae": 5.0, "rmse": 6.0, "r2": 0.7},
        "random_forest": {"mae": 3.0, "rmse": 4.0, "r2": 0.85},
        "xgboost": {"mae": 4.0, "rmse": 4.5, "r2": 0.8},
    }
    # config/config.yaml's evaluation.best_model_metric is "rmse"
    best = select_best_model(metrics_by_model, config_path="config/config.yaml")
    assert best == "random_forest"


# 2. Lower-is-better metric (mae) --------------------------------------------------


def test_select_best_model_lower_is_better_mae():
    metrics_by_model = {"model_a": {"mae": 5.0}, "model_b": {"mae": 3.0}}
    # tests/fixtures/model_test_config.yaml's evaluation.best_model_metric is "mae"
    best = select_best_model(metrics_by_model, config_path="tests/fixtures/model_test_config.yaml")
    assert best == "model_b"


# 3. Higher-is-better metric (r2) is not treated as "lower is better" -------------


def test_select_best_model_higher_is_better_r2():
    metrics_by_model = {"model_a": {"r2": 0.6}, "model_b": {"r2": 0.9}}
    best = select_best_model(metrics_by_model, config_path="tests/fixtures/model_selection_r2_config.yaml")
    assert best == "model_b"


# 4. Error cases --------------------------------------------------------------------


def test_select_best_model_raises_on_empty_metrics():
    with pytest.raises(ModelSelectionError, match="empty"):
        select_best_model({}, config_path="config/config.yaml")


def test_select_best_model_raises_when_metric_missing_from_a_model():
    metrics_by_model = {"model_a": {"rmse": 5.0}, "model_b": {"mae": 5.0}}  # model_b has no rmse
    with pytest.raises(ModelSelectionError, match="model_b"):
        select_best_model(metrics_by_model, config_path="config/config.yaml")
