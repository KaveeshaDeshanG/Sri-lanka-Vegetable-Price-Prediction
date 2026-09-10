"""Integration tests for src.prediction.pipeline.PredictionPipeline, end to end."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
from src.feature_engineering.feature_engineer import FeatureEngineeringError
from src.models.baseline import train_baseline_model
from src.models.persistence import save_model
from src.models.pipeline import ModelTrainingPipeline
from src.prediction.pipeline import PredictionInputError, PredictionPipeline
from src.prediction.predictor import InferenceArtifact, save_inference_artifact


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


def _config_at(tmp_path, extra_yaml: str = "") -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n{extra_yaml}')
    return str(config_path)


def _train_and_persist(config_path, merged_df, test_size=0.2, use_full_pipeline=False):
    """Build+train+persist via the real production pieces."""
    builder = TrainTestDatasetBuilder(config_path=config_path)
    X_train, X_test, y_train, y_test = builder.build(merged_df, test_size=test_size)

    if use_full_pipeline:
        result = ModelTrainingPipeline(config_path=config_path).run(X_train, X_test, y_train, y_test)
        model = result.best_model
    else:
        model = train_baseline_model(X_train, y_train)
        save_model(model, filename="best_model.joblib", config_path=config_path)

    save_inference_artifact(
        InferenceArtifact(builder.fitted_encoder, list(X_train.columns)), config_path=config_path
    )
    return model


def _with_target_row(history_df: pd.DataFrame, date: str, vegetable: str, market: str) -> pd.DataFrame:
    """Append one row for `date` with price left NaN -- the row PredictionPipeline will predict."""
    target_row = pd.DataFrame(
        {
            "date": [pd.Timestamp(date)],
            "vegetable": [vegetable],
            "market": [market],
            "source": ["HARTI"],
            "price": [float("nan")],
            "rainfall_mm": [float("nan")],
            "temperature_c": [float("nan")],
            "humidity_pct": [float("nan")],
        }
    )
    return pd.concat([history_df, target_row], ignore_index=True)


# 1. Successful end-to-end prediction ---------------------------------------------


def test_predict_end_to_end_with_full_production_training_path(tmp_path):
    config_path = _config_at(
        tmp_path,
        'model:\n  models_to_train: ["linear_regression"]\n'
        'evaluation:\n  metrics: ["mae", "rmse", "r2"]\n  best_model_metric: "rmse"\n',
    )
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history, use_full_pipeline=True)

    input_df = _with_target_row(history, "2024-02-20", "Carrot", "Dambulla")
    pipeline = PredictionPipeline(config_path=config_path)
    result = pipeline.predict(input_df)

    assert len(result) == 1
    assert result.loc[0, "vegetable"] == "Carrot"
    assert result.loc[0, "market"] == "Dambulla"
    assert np.isfinite(result.loc[0, "predicted_price"])


# 2. Deterministic across repeated calls ------------------------------------------


def test_predict_is_deterministic_across_repeated_calls(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    input_df = _with_target_row(history, "2024-02-20", "Carrot", "Dambulla")
    pipeline = PredictionPipeline(config_path=config_path)

    first = pipeline.predict(input_df)
    second = pipeline.predict(input_df)
    assert first["predicted_price"].tolist() == second["predicted_price"].tolist()


# 3. Batch prediction across multiple groups/dates --------------------------------


def test_predict_handles_multiple_target_rows_at_once(tmp_path):
    config_path = _config_at(tmp_path)
    group_a = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    group_b = _daily_group_df("2024-01-01", 50, "Tomato", "Pettah", 80.0)
    history = pd.concat([group_a, group_b], ignore_index=True)
    _train_and_persist(config_path, history)

    input_df = _with_target_row(history, "2024-02-20", "Carrot", "Dambulla")
    input_df = _with_target_row(input_df, "2024-02-20", "Tomato", "Pettah")

    pipeline = PredictionPipeline(config_path=config_path)
    result = pipeline.predict(input_df)

    assert len(result) == 2
    assert set(result["vegetable"]) == {"Carrot", "Tomato"}


# 4. Category normalization matches training (casing/whitespace) ----------------


def test_predict_normalizes_category_casing_to_match_training(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    # lowercase + stray whitespace, as a real caller (e.g. a form field) might send
    messy_input = _with_target_row(history, "2024-02-20", "  carrot  ", "dambulla")
    pipeline = PredictionPipeline(config_path=config_path)
    result = pipeline.predict(messy_input)

    assert result.loc[0, "vegetable"] == "Carrot"
    assert np.isfinite(result.loc[0, "predicted_price"])


# 5. Missing required column reuses FeatureEngineer's existing error -------------


def test_predict_raises_when_a_required_column_is_missing(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    input_df = _with_target_row(history, "2024-02-20", "Carrot", "Dambulla").drop(columns=["market"])
    pipeline = PredictionPipeline(config_path=config_path)
    with pytest.raises(FeatureEngineeringError, match="market"):
        pipeline.predict(input_df)


# 6. Malformed dates raise a clear, prediction-specific error --------------------


def test_predict_raises_on_malformed_date():
    pipeline = PredictionPipeline(config_path="config/config.yaml")
    input_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date"],
            "vegetable": ["Carrot", "Carrot"],
            "market": ["Dambulla", "Dambulla"],
            "price": [100.0, float("nan")],
            "rainfall_mm": [5.0, float("nan")],
            "temperature_c": [28.0, float("nan")],
            "humidity_pct": [70.0, float("nan")],
        }
    )
    with pytest.raises(PredictionInputError, match="date"):
        pipeline.predict(input_df)


# 7. get_engineered_features exposes the same values predict() scores on ---------


def test_get_engineered_features_returns_the_row_predict_actually_scores(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    input_df = _with_target_row(history, "2024-02-20", "Carrot", "Dambulla")
    pipeline = PredictionPipeline(config_path=config_path)

    row = pipeline.get_engineered_features(
        input_df, target_date="2024-02-20", vegetable="Carrot", market="Dambulla"
    )
    assert row["rainfall_mm_lag14d"] == pytest.approx(5.0)
    assert row["temperature_c_lag14d"] == pytest.approx(28.0)
    assert row["humidity_pct_lag14d"] == pytest.approx(70.0)

    # Never calls the model -- doesn't require predict() to have been called first,
    # and doesn't affect a subsequent predict() call on the same input.
    result = pipeline.predict(input_df, target_date="2024-02-20", vegetable="Carrot", market="Dambulla")
    assert np.isfinite(result.loc[0, "predicted_price"])


def test_get_engineered_features_raises_for_no_matching_row(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    input_df = _with_target_row(history, "2024-02-20", "Carrot", "Dambulla")
    pipeline = PredictionPipeline(config_path=config_path)
    with pytest.raises(PredictionInputError, match="no row found"):
        pipeline.get_engineered_features(
            input_df, target_date="2024-02-20", vegetable="Tomato", market="Pettah"
        )


# 7. Nothing to predict (no NaN price) --------------------------------------------


def test_predict_raises_when_no_row_has_an_unknown_price(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    pipeline = PredictionPipeline(config_path=config_path)
    with pytest.raises(PredictionInputError, match="nothing to predict"):
        pipeline.predict(history)  # every row has a known price


# 8. Insufficient historical context raises rather than fabricating a value -----


def test_predict_raises_when_historical_context_is_insufficient(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    short_history = _daily_group_df("2024-06-01", 5, "Carrot", "Dambulla", 100.0)  # only 5 days
    input_df = _with_target_row(short_history, "2024-06-06", "Carrot", "Dambulla")

    pipeline = PredictionPipeline(config_path=config_path)
    from src.prediction.predictor import FeatureAlignmentError

    with pytest.raises(FeatureAlignmentError):
        pipeline.predict(input_df)


# 9. Model / transformation artifact missing --------------------------------------


def test_predict_raises_when_model_artifact_missing(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    input_df = _with_target_row(history, "2024-02-20", "Carrot", "Dambulla")

    from src.prediction.predictor import ModelArtifactNotFoundError

    pipeline = PredictionPipeline(config_path=config_path)
    with pytest.raises(ModelArtifactNotFoundError):
        pipeline.predict(input_df)


# 10. THE key regression test: unseen category does not refit / does not change schema


def test_unseen_category_does_not_refit_encoder_or_change_trained_feature_schema(tmp_path):
    config_path = _config_at(tmp_path)
    training_history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, training_history)

    pipeline = PredictionPipeline(config_path=config_path)
    predictor_before = pipeline._require_predictor()
    feature_columns_before = list(predictor_before.feature_columns)
    assert not any(c.startswith("vegetable_Cabbage") for c in feature_columns_before)

    # a vegetable/market never seen during training, but with its OWN full
    # 31+ days of history so this exercises the *categorical* unseen-value
    # path specifically, not the insufficient-history path
    unseen_history = _daily_group_df("2024-01-01", 31, "Cabbage", "Jaffna", 50.0)
    input_df = _with_target_row(unseen_history, "2024-02-01", "Cabbage", "Jaffna")

    result = pipeline.predict(input_df)  # must not raise

    assert len(result) == 1
    assert result.loc[0, "vegetable"] == "Cabbage"
    assert np.isfinite(result.loc[0, "predicted_price"])

    # the schema training fixed must be byte-identical after handling an unseen category
    predictor_after = pipeline._require_predictor()
    assert predictor_after.feature_columns == feature_columns_before
    assert not any(c.startswith("vegetable_Cabbage") for c in predictor_after.feature_columns)

    # and a subsequent prediction for the ORIGINAL trained category must be unaffected
    # (2024-02-20 is the day immediately after training_history's last row, so price_lag_1d
    # still has an exact match -- an arbitrary later date would fail for lack of history,
    # not because of anything this assertion is testing)
    followup = _with_target_row(training_history, "2024-02-20", "Carrot", "Dambulla")
    followup_result = pipeline.predict(followup)
    assert np.isfinite(followup_result.loc[0, "predicted_price"])
