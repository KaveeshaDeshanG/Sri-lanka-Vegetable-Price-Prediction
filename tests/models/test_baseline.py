"""Unit tests for src.models.baseline."""

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.baseline import build_baseline_model, train_baseline_model


def _simple_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0, 5.0], "feature_b": [5.0, 4.0, 3.0, 2.0, 1.0]})
    y_train = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0], name="price")
    return X_train, y_train


def test_build_baseline_model_returns_linear_regression():
    model = build_baseline_model()
    assert isinstance(model, LinearRegression)


def test_train_baseline_model_fits_on_training_data():
    X_train, y_train = _simple_training_data()
    model = train_baseline_model(X_train, y_train)
    assert hasattr(model, "coef_")  # fitted state now exists


def test_train_baseline_model_is_deterministic():
    """OLS has no random_state; the same input must always give the same fit."""
    X_train, y_train = _simple_training_data()
    model_a = train_baseline_model(X_train, y_train)
    model_b = train_baseline_model(X_train, y_train)
    assert list(model_a.coef_) == list(model_b.coef_)
    assert model_a.intercept_ == model_b.intercept_


def test_train_baseline_model_predicts_correct_shape():
    X_train, y_train = _simple_training_data()
    model = train_baseline_model(X_train, y_train)
    X_test = pd.DataFrame({"feature_a": [6.0, 7.0], "feature_b": [0.0, -1.0]})
    predictions = model.predict(X_test)
    assert len(predictions) == 2
