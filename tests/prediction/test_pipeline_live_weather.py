"""Tests for PredictionPipeline.predict()'s explicit target_date/vegetable/market selection.

Exercises the mechanism the live-weather integration relies on: inserting extra
NaN-price weather-context rows into input_df (to satisfy WeatherFeatureGenerator's
exact-date lookup for *_lag14d) must never cause those rows to be scored, and must
never change the model's saved schema or require refitting anything.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
from src.models.baseline import train_baseline_model
from src.models.persistence import save_model
from src.prediction.pipeline import PredictionInputError, PredictionPipeline
from src.prediction.predictor import FeatureAlignmentError, InferenceArtifact, save_inference_artifact


def _daily_group_df(
    start, n_days, vegetable, market, price_start, price_step=1.0, source="HARTI", nan_weather_dates=()
):
    """Dense daily price history. `nan_weather_dates` simulates a weather source that
    didn't cover a specific date even though price data exists for it -- the realistic
    scenario live weather backfills (price and weather come from independent feeds)."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    nan_dates = {pd.Timestamp(d) for d in nan_weather_dates}
    rainfall = [float("nan") if d in nan_dates else 5.0 for d in dates]
    temperature = [float("nan") if d in nan_dates else 28.0 for d in dates]
    humidity = [float("nan") if d in nan_dates else 70.0 for d in dates]
    return pd.DataFrame(
        {
            "date": dates,
            "vegetable": [vegetable] * n_days,
            "market": [market] * n_days,
            "source": [source] * n_days,
            "price": [price_start + price_step * i for i in range(n_days)],
            "rainfall_mm": rainfall,
            "temperature_c": temperature,
            "humidity_pct": humidity,
        }
    )


def _config_at(tmp_path, extra_yaml: str = "") -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n{extra_yaml}')
    return str(config_path)


def _train_and_persist(config_path, merged_df):
    builder = TrainTestDatasetBuilder(config_path=config_path)
    X_train, X_test, y_train, y_test = builder.build(merged_df)
    model = train_baseline_model(X_train, y_train)
    save_model(model, filename="best_model.joblib", config_path=config_path)
    save_inference_artifact(
        InferenceArtifact(builder.fitted_encoder, list(X_train.columns)), config_path=config_path
    )
    return model


def _weather_context_row(date, vegetable, market, rainfall, temperature, humidity):
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date)],
            "vegetable": [vegetable],
            "market": [market],
            "source": ["LIVE_WEATHER"],
            "price": [float("nan")],
            "rainfall_mm": [rainfall],
            "temperature_c": [temperature],
            "humidity_pct": [humidity],
        }
    )


def _target_row(date, vegetable, market):
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date)],
            "vegetable": [vegetable],
            "market": [market],
            "source": ["USER_REQUEST"],
            "price": [float("nan")],
            "rainfall_mm": [float("nan")],
            "temperature_c": [float("nan")],
            "humidity_pct": [float("nan")],
        }
    )


# 1. Backward compatibility: default behavior is completely unchanged ------------


def test_predict_default_behavior_unchanged_without_identity_kwargs(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    input_df = pd.concat([history, _target_row("2024-02-20", "Carrot", "Dambulla")], ignore_index=True)
    pipeline = PredictionPipeline(config_path=config_path)

    result = pipeline.predict(input_df)  # no kwargs -- identical to pre-existing behavior

    assert len(result) == 1
    assert result.loc[0, "vegetable"] == "Carrot"
    assert np.isfinite(result.loc[0, "predicted_price"])


def test_predict_raises_if_only_some_identity_args_given(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    input_df = pd.concat([history, _target_row("2024-02-20", "Carrot", "Dambulla")], ignore_index=True)
    pipeline = PredictionPipeline(config_path=config_path)

    with pytest.raises(PredictionInputError, match="target_date, vegetable, and market"):
        pipeline.predict(input_df, target_date="2024-02-20", vegetable="Carrot")


# 2. Explicit identity narrows scoring to exactly one row -------------------------


def test_predict_with_explicit_identity_ignores_other_nan_price_rows(tmp_path):
    """Weather-context rows (NaN price, some other date) must contribute to features
    but never appear in the output or be sent to the model."""
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    # a context row far short of a full feature set for its own date -- if this were
    # ever scored, it would raise FeatureAlignmentError and break the whole call.
    context_row = _weather_context_row("2024-02-10", "Carrot", "Dambulla", 3.0, 29.0, 80.0)
    target_row = _target_row("2024-02-20", "Carrot", "Dambulla")
    input_df = pd.concat([history, context_row, target_row], ignore_index=True)

    pipeline = PredictionPipeline(config_path=config_path)
    result = pipeline.predict(
        input_df, target_date="2024-02-20", vegetable="Carrot", market="Dambulla"
    )

    assert len(result) == 1
    assert result.loc[0, "date"] == pd.Timestamp("2024-02-20")


def test_predict_raises_nothing_to_predict_when_identity_matches_no_row(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    _train_and_persist(config_path, history)

    input_df = pd.concat([history, _target_row("2024-02-20", "Carrot", "Dambulla")], ignore_index=True)
    pipeline = PredictionPipeline(config_path=config_path)

    with pytest.raises(PredictionInputError, match="nothing to predict"):
        pipeline.predict(input_df, target_date="2024-03-01", vegetable="Carrot", market="Dambulla")


# 3. THE key semantic test: correct 14-day lag using context rows, not a fabricated value ---


def test_weather_context_rows_produce_correct_14_day_lag_not_a_fabricated_value(tmp_path):
    """50 dense days (2024-01-01 to 2024-02-19, so price_lag/rolling all resolve for the
    very next day, 2024-02-20 -- the only date the current architecture can validly
    predict, since price_lag_1d needs 2024-02-19's real price). 2024-02-06's WEATHER is
    deliberately missing even though its price exists (independent feeds falling out of
    sync -- the realistic case live weather backfills). Predicting 2024-02-20 needs
    rainfall_mm_lag14d, looked up from exactly 2024-02-06 (2024-02-20 - 14d). Adding a
    same-date, price-NaN weather-context row proves the resulting lag feature reflects
    THAT context row's value at the correct earlier date -- not today's/target's own
    weather smuggled into a *_lag14d column, and not a value fabricated out of thin air."""
    config_path = _config_at(tmp_path)
    history = _daily_group_df(
        "2024-01-01", 50, "Carrot", "Dambulla", 100.0, nan_weather_dates=["2024-02-06"]
    )
    _train_and_persist(config_path, history)

    # Distinctive weather value, easy to tell apart from history's constant 5.0/28.0/70.0.
    distinctive_rainfall, distinctive_temp, distinctive_humidity = 42.0, 33.3, 91.0
    context_row = _weather_context_row(
        "2024-02-06", "Carrot", "Dambulla", distinctive_rainfall, distinctive_temp, distinctive_humidity
    )
    target_row = _target_row("2024-02-20", "Carrot", "Dambulla")
    input_df = pd.concat([history, context_row, target_row], ignore_index=True)

    pipeline = PredictionPipeline(config_path=config_path)

    # Reach into the same feature-engineering path predict() uses, to assert the
    # actual lag value computed for the target row (predict() itself only returns
    # a price, not the intermediate feature) -- using the SAME FeatureEngineer
    # instance predict() holds, never a separate/bypassed implementation.
    normalized = pipeline._normalize(input_df)
    featured = pipeline._feature_engineer.build_features(normalized)
    target_features = featured.loc[featured["date"] == pd.Timestamp("2024-02-20")].iloc[0]

    assert target_features["rainfall_mm_lag14d"] == pytest.approx(distinctive_rainfall)
    assert target_features["temperature_c_lag14d"] == pytest.approx(distinctive_temp)
    assert target_features["humidity_pct_lag14d"] == pytest.approx(distinctive_humidity)

    # The real price at 2024-02-06 (used by price_lag_14d) must be unaffected by the
    # extra same-date row -- proves the context row only supplies the missing weather.
    assert target_features["price_lag_14d"] == pytest.approx(100.0 + 36)  # day index 36 = Feb 6

    # And the full predict() call succeeds end to end using that same feature.
    result = pipeline.predict(input_df, target_date="2024-02-20", vegetable="Carrot", market="Dambulla")
    assert len(result) == 1
    assert np.isfinite(result.loc[0, "predicted_price"])


def test_missing_context_row_at_exact_lag_date_still_raises_feature_alignment_error(tmp_path):
    """Without a context row at exactly target_date - 14d, this must fail loudly
    (FeatureAlignmentError) rather than silently predicting from incomplete features --
    confirms the safety net inserting context rows relies on is still intact."""
    config_path = _config_at(tmp_path)
    history = _daily_group_df(
        "2024-01-01", 50, "Carrot", "Dambulla", 100.0, nan_weather_dates=["2024-02-06"]
    )
    _train_and_persist(config_path, history)

    target_row = _target_row("2024-02-20", "Carrot", "Dambulla")  # no context row at 2024-02-06
    input_df = pd.concat([history, target_row], ignore_index=True)

    pipeline = PredictionPipeline(config_path=config_path)
    with pytest.raises(FeatureAlignmentError):
        pipeline.predict(input_df, target_date="2024-02-20", vegetable="Carrot", market="Dambulla")


def test_context_rows_do_not_change_saved_feature_schema_or_refit_encoder(tmp_path):
    config_path = _config_at(tmp_path)
    history = _daily_group_df(
        "2024-01-01", 50, "Carrot", "Dambulla", 100.0, nan_weather_dates=["2024-02-06"]
    )
    _train_and_persist(config_path, history)

    pipeline = PredictionPipeline(config_path=config_path)
    predictor = pipeline._require_predictor()
    feature_columns_before = list(predictor.feature_columns)

    context_row = _weather_context_row("2024-02-06", "Carrot", "Dambulla", 42.0, 33.3, 91.0)
    target_row = _target_row("2024-02-20", "Carrot", "Dambulla")
    input_df = pd.concat([history, context_row, target_row], ignore_index=True)

    pipeline.predict(input_df, target_date="2024-02-20", vegetable="Carrot", market="Dambulla")

    assert list(pipeline._require_predictor().feature_columns) == feature_columns_before
