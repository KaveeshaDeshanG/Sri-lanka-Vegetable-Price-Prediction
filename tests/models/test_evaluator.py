"""Unit tests for src.models.evaluator.ModelEvaluator."""

import math

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.evaluator import EvaluationError, ModelEvaluator


class _StubModel:
    """A fake fitted model with a fixed, known predict() output, to test evaluation math in isolation."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions


# 1. Configured metrics are computed correctly -----------------------------------


def test_evaluate_computes_configured_metrics_correctly():
    y_test = pd.Series([100.0, 110.0, 120.0, 130.0])
    predictions = [98.0, 115.0, 118.0, 135.0]
    model = _StubModel(predictions)

    evaluator = ModelEvaluator(config_path="config/config.yaml")  # evaluation.metrics: mae, rmse, r2
    results = evaluator.evaluate(model, pd.DataFrame(index=range(4)), y_test)

    assert results["mae"] == pytest.approx(mean_absolute_error(y_test, predictions))
    assert results["rmse"] == pytest.approx(math.sqrt(mean_squared_error(y_test, predictions)))
    assert results["r2"] == pytest.approx(r2_score(y_test, predictions))


def test_evaluate_perfect_predictions_yield_zero_error_and_r2_of_one():
    y_test = pd.Series([10.0, 20.0, 30.0])
    model = _StubModel([10.0, 20.0, 30.0])

    evaluator = ModelEvaluator(config_path="config/config.yaml")
    results = evaluator.evaluate(model, pd.DataFrame(index=range(3)), y_test)

    assert results["mae"] == pytest.approx(0.0)
    assert results["rmse"] == pytest.approx(0.0)
    assert results["r2"] == pytest.approx(1.0)


# 2. Only the metrics named in config are computed -------------------------------


def test_evaluate_only_computes_configured_metrics():
    y_test = pd.Series([1.0, 2.0, 3.0])
    model = _StubModel([1.0, 2.0, 3.0])

    evaluator = ModelEvaluator(config_path="tests/fixtures/model_test_config.yaml")  # metrics: mae, rmse only
    results = evaluator.evaluate(model, pd.DataFrame(index=range(3)), y_test)

    assert set(results) == {"mae", "rmse"}
    assert "r2" not in results


# 3. evaluate_all evaluates every model against the same test set ----------------


def test_evaluate_all_evaluates_every_model():
    y_test = pd.Series([1.0, 2.0, 3.0])
    models = {"model_a": _StubModel([1.0, 2.0, 3.0]), "model_b": _StubModel([2.0, 2.0, 2.0])}

    evaluator = ModelEvaluator(config_path="config/config.yaml")
    results = evaluator.evaluate_all(models, pd.DataFrame(index=range(3)), y_test)

    assert set(results) == {"model_a", "model_b"}
    assert results["model_a"]["mae"] == pytest.approx(0.0)
    assert results["model_b"]["mae"] > 0.0


# 4. Unknown metric name in config raises rather than being silently skipped -----


def test_evaluate_raises_on_unknown_metric():
    y_test = pd.Series([1.0, 2.0])
    model = _StubModel([1.0, 2.0])
    evaluator = ModelEvaluator(config_path="tests/fixtures/model_test_config_invalid_metric.yaml")
    with pytest.raises(EvaluationError, match="not_a_real_metric"):
        evaluator.evaluate(model, pd.DataFrame(index=range(2)), y_test)
