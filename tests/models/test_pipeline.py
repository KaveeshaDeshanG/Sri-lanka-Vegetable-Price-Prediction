"""Integration tests for src.models.pipeline.ModelTrainingPipeline, composed with TrainTestDatasetBuilder."""

import pandas as pd
import pytest

from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
from src.models.persistence import load_model
from src.models.pipeline import ModelTrainingPipeline, ModelTrainingResult


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


# 1. End-to-end run, consuming TrainTestDatasetBuilder's output directly ---------


def test_pipeline_runs_end_to_end_and_selects_best_model_by_configured_metric(train_test_data, tmp_path):
    X_train, X_test, y_train, y_test = train_test_data
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project:\n  random_seed: 42\n"
        'model:\n  models_to_train: ["linear_regression", "random_forest", "xgboost"]\n'
        '  random_forest: {n_estimators: 50, max_depth: 5}\n'
        '  xgboost: {n_estimators: 50, learning_rate: 0.1, max_depth: 4}\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )
    pipeline = ModelTrainingPipeline(config_path=str(config_path))
    result = pipeline.run(X_train, X_test, y_train, y_test)

    assert isinstance(result, ModelTrainingResult)
    assert set(result.models) == {"linear_regression", "random_forest", "xgboost"}
    assert set(result.metrics) == {"linear_regression", "random_forest", "xgboost"}
    assert result.best_model_name in result.models

    # best model must genuinely have the lowest rmse among the candidates
    best_rmse = result.metrics[result.best_model_name]["rmse"]
    assert all(best_rmse <= result.metrics[name]["rmse"] for name in result.models)


# 2. Evaluation happens on the held-out test set, not the training set -----------


def test_pipeline_evaluation_uses_the_held_out_test_set(train_test_data, tmp_path):
    X_train, X_test, y_train, y_test = train_test_data
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )
    pipeline = ModelTrainingPipeline(config_path=str(config_path))
    result = pipeline.run(X_train, X_test, y_train, y_test)

    fitted_model = result.models["linear_regression"]
    # recompute the metric independently against X_test/y_test to confirm
    # the reported metrics genuinely come from the test set, not train
    from sklearn.metrics import mean_squared_error

    expected_rmse = mean_squared_error(y_test, fitted_model.predict(X_test)) ** 0.5
    assert result.metrics["linear_regression"]["rmse"] == pytest.approx(expected_rmse)


# 3. The best model is persisted and reloadable -----------------------------------


def test_pipeline_persists_the_best_model_and_it_is_reloadable(train_test_data, tmp_path):
    X_train, X_test, y_train, y_test = train_test_data
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )
    pipeline = ModelTrainingPipeline(config_path=str(config_path))
    result = pipeline.run(X_train, X_test, y_train, y_test, best_model_filename="best_model.joblib")

    assert (tmp_path / "best_model.joblib").exists()
    assert result.saved_model_path == str(tmp_path / "best_model.joblib")

    loaded = load_model(filename="best_model.joblib", config_path=str(config_path))
    assert list(loaded.predict(X_test)) == list(result.best_model.predict(X_test))


# 3b. output_dir overrides config's paths.models_dir entirely ---------------------


def test_pipeline_output_dir_overrides_configured_models_dir(train_test_data, tmp_path):
    """Lets a caller (e.g. a demo/sandbox training run) redirect persistence away
    from the configured production models_dir without editing config."""
    X_train, X_test, y_train, y_test = train_test_data
    config_path = tmp_path / "config.yaml"
    configured_dir = tmp_path / "configured"
    override_dir = tmp_path / "override"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{configured_dir.as_posix()}"\n'
    )
    pipeline = ModelTrainingPipeline(config_path=str(config_path))
    result = pipeline.run(X_train, X_test, y_train, y_test, output_dir=str(override_dir))

    assert (override_dir / "best_model.joblib").exists()
    assert not configured_dir.exists()
    assert result.saved_model_path == str(override_dir / "best_model.joblib")


# 4. Training never touches X_test/y_test -----------------------------------------


def test_pipeline_training_is_unaffected_by_test_split_contents(train_test_data, tmp_path):
    X_train, X_test, y_train, y_test = train_test_data
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'model:\n  models_to_train: ["linear_regression", "random_forest", "xgboost"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n'
        f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n'
    )

    perturbed_X_test = X_test.copy()
    perturbed_y_test = y_test + 999_999.0

    result_a = ModelTrainingPipeline(config_path=str(config_path)).run(X_train, X_test, y_train, y_test)
    result_b = ModelTrainingPipeline(config_path=str(config_path)).run(
        X_train, perturbed_X_test, y_train, perturbed_y_test
    )

    for name in result_a.models:
        predictions_a = result_a.models[name].predict(X_train)
        predictions_b = result_b.models[name].predict(X_train)
        assert list(predictions_a) == pytest.approx(list(predictions_b))
