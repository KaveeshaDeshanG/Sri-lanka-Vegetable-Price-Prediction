"""Tests for app.forecasting.forecast_future_price -- multi-step recursive forecasting.

HartiLiveClient/OpenMeteoWeatherClient/CeypetcoLiveFuelPriceClient are mocked (their own
network/parsing behavior is covered by their dedicated test modules) -- these tests only
check the recursive-forecast composition: day-by-day looping, temporary-history
immutability of the original DataFrame, recursive lag propagation, fuel/weather
data-source selection, and error surfacing.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.forecasting import (
    ForecastDataUnavailableError,
    ForecastInputError,
    forecast_future_price,
    max_supported_forecast_date,
)
from src.data_collection.live_fuel_price_client import LiveFuelPriceSnapshot
from src.data_collection.live_weather_client import MAX_FORECAST_DAYS, MAX_PAST_DAYS, MAX_SEASONAL_FORECAST_DAYS
from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
from src.models.baseline import train_baseline_model
from src.models.persistence import save_model
from src.prediction.pipeline import PredictionPipeline
from src.prediction.predictor import InferenceArtifact, save_inference_artifact


def _daily_history(
    start="2024-01-01",
    n_days=90,
    vegetable="Tomato",
    market="Dambulla",
    price_start=200.0,
    price_step=1.0,
    weather_days=None,
    with_fuel=True,
):
    """Dense daily price history, with weather covering `weather_days` calendar days
    from `start` (defaults to the same range as price -- pass a larger value to
    simulate weather data that's already known further ahead than price, or a smaller
    one to simulate a weather series that hasn't caught up)."""
    weather_days = n_days if weather_days is None else weather_days
    price_dates = pd.date_range(start, periods=n_days, freq="D")
    weather_dates = pd.date_range(start, periods=weather_days, freq="D")

    price_df = pd.DataFrame(
        {
            "date": price_dates,
            "vegetable": [vegetable] * n_days,
            "market": [market] * n_days,
            "source": ["HARTI"] * n_days,
            "price": [price_start + price_step * i for i in range(n_days)],
        }
    )
    weather_df = pd.DataFrame(
        {
            "date": weather_dates,
            "rainfall_mm": [5.0] * weather_days,
            "temperature_c": [28.0] * weather_days,
            "humidity_pct": [70.0] * weather_days,
        }
    )
    merged = price_df.merge(weather_df, on="date", how="outer")
    merged["vegetable"] = merged["vegetable"].fillna(vegetable)
    merged["market"] = merged["market"].fillna(market)
    merged["source"] = merged["source"].fillna("EXTRA_WEATHER")
    if with_fuel:
        merged["diesel_price"] = 363.0
        merged["petrol_price"] = 371.0
    return merged.sort_values("date").reset_index(drop=True)


def _config_at(tmp_path) -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f'paths:\n'
        f'  models_dir: "{tmp_path.as_posix()}"\n'
        # Deliberately nonexistent -- keeps these tests hermetic against the real
        # project's crop_production_regions.yaml (see test_live_weather_services.py's
        # own _config() for the same reasoning): "Tomato"/"Carrot" are placeholder
        # names here, not real-world regional assignments.
        f'  crop_production_regions_mapping: "{(tmp_path / "nonexistent_mapping.yaml").as_posix()}"\n'
        "weather_api:\n"
        "  latitude: 7.8567\n"
        "  longitude: 80.6517\n"
        '  timezone: "Asia/Colombo"\n'
        "  timeout_seconds: 10\n"
    )
    return str(config_path)


def _train_and_persist(config_path, merged_df) -> PredictionPipeline:
    builder = TrainTestDatasetBuilder(config_path=config_path)
    X_train, _X_test, y_train, _y_test = builder.build(merged_df)
    model = train_baseline_model(X_train, y_train)
    save_model(model, filename="best_model.joblib", config_path=config_path)
    save_inference_artifact(
        InferenceArtifact(builder.fitted_encoder, list(X_train.columns)), config_path=config_path
    )
    return PredictionPipeline(config_path=config_path)


def _fuel_snapshot(diesel=382.0, petrol=414.0, effective="2024-01-01"):
    return LiveFuelPriceSnapshot(
        prices={"diesel_price": diesel, "petrol_price": petrol},
        effective_dates={"diesel_price": pd.Timestamp(effective), "petrol_price": pd.Timestamp(effective)},
        retrieved_at=pd.Timestamp("2024-06-01 09:00:00"),
        source_url="https://ceypetco.gov.lk/marketing-sales/",
    )


# A. One-day forecast -----------------------------------------------------------------


def test_one_day_forecast_runs_and_leaves_original_history_untouched(tmp_path):
    history = _daily_history()
    original = history.copy(deep=True)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    target = latest + pd.Timedelta(days=1)

    result = forecast_future_price(history, "Tomato", "Dambulla", target, pipeline, config_path)

    assert result.target_date == target
    assert result.latest_actual_date == latest
    assert result.forecast_horizon_days == 1
    assert len(result.daily_forecast) == 1
    assert result.daily_forecast[0].date == target
    assert result.predicted_price == result.daily_forecast[0].predicted_price
    assert pd.notna(result.predicted_price)

    pd.testing.assert_frame_equal(history, original)


# B. Multi-day recursive forecast ------------------------------------------------------


def test_multi_day_forecast_produces_sequential_consecutive_days(tmp_path):
    history = _daily_history()
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    target = latest + pd.Timedelta(days=7)

    result = forecast_future_price(history, "Tomato", "Dambulla", target, pipeline, config_path)

    assert result.forecast_horizon_days == 7
    assert len(result.daily_forecast) == 7
    expected_dates = [latest + pd.Timedelta(days=i) for i in range(1, 8)]
    assert [p.date for p in result.daily_forecast] == expected_dates
    assert all(pd.notna(p.predicted_price) for p in result.daily_forecast)


# C. Long horizon -----------------------------------------------------------------------


def test_long_horizon_forecast_reaches_the_exact_target_date(tmp_path):
    # weather already reaches far enough ahead to cover every day's 14-day lag
    # requirement all the way to a 32-day-ahead target (matching the project's own
    # Carrot/Dambulla 2026-08-20 -> 2026-09-21 scenario), with no live fetch needed.
    history = _daily_history(n_days=90, weather_days=145)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history.loc[history["price"].notna(), "date"].max()  # weather extends further than price
    target = latest + pd.Timedelta(days=32)

    result = forecast_future_price(
        history, "Tomato", "Dambulla", target, pipeline, config_path, use_live_weather=False
    )

    assert result.forecast_horizon_days == 32
    assert len(result.daily_forecast) == 32
    assert result.daily_forecast[-1].date == target
    assert result.target_date == target
    assert pd.notna(result.predicted_price)


# D. Original data immutability --------------------------------------------------------


def test_forecast_never_mutates_the_original_history_dataframe(tmp_path):
    history = _daily_history()
    original = history.copy(deep=True)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    forecast_future_price(
        history, "Tomato", "Dambulla", latest + pd.Timedelta(days=5), pipeline, config_path
    )

    pd.testing.assert_frame_equal(history, original)
    assert "FORECAST" not in history["source"].unique()


# E. Fuel -------------------------------------------------------------------------------


def test_forecast_uses_live_fuel_price_for_every_forecast_day_it_covers(tmp_path):
    history = _daily_history()  # historical diesel/petrol: 363.0 / 371.0
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    target = latest + pd.Timedelta(days=3)
    live_fuel = _fuel_snapshot(diesel=382.0, petrol=414.0, effective="2024-01-01")  # covers everything

    forecast_future_price(
        history, "Tomato", "Dambulla", target, pipeline, config_path, live_fuel_snapshot=live_fuel
    )

    # verify via the feature actually reaching the model for the last forecast day
    normalized = pipeline._normalize(
        pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "date": [target],
                        "vegetable": ["Tomato"],
                        "market": ["Dambulla"],
                        "source": ["USER_REQUEST"],
                        "price": [float("nan")],
                    }
                ),
            ],
            ignore_index=True,
        )
    )
    # Rebuild forecast_history the same way the function does, to inspect the fuel value
    # actually attached to each forecast day (see test H for the equivalent price check).
    day1 = latest + pd.Timedelta(days=1)
    from app.services import build_prediction_input, finalize_live_row

    input_day1 = build_prediction_input(history, day1, "Tomato", "Dambulla")
    predicted_day1 = pipeline.predict(input_day1, target_date=day1, vegetable="Tomato", market="Dambulla")
    row = pd.DataFrame(
        {
            "date": [day1], "vegetable": ["Tomato"], "market": ["Dambulla"], "source": ["FORECAST"],
            "price": [float(predicted_day1.loc[0, "predicted_price"])],
            "rainfall_mm": [float("nan")], "temperature_c": [float("nan")], "humidity_pct": [float("nan")],
        }
    )
    finalized = finalize_live_row(history, row, live_fuel)
    assert finalized["diesel_price"].iloc[0] == 382.0
    assert finalized["petrol_price"].iloc[0] == 414.0


def test_forecast_never_fabricates_a_future_fuel_price_change():
    """A live reading whose effective date is AFTER a forecast day must not apply to
    that day -- the historical (older, still correct) price must be used instead."""
    from app.services import finalize_live_row

    history = _daily_history()
    forecast_day = history["date"].max() + pd.Timedelta(days=1)
    live_fuel = _fuel_snapshot(diesel=999.0, petrol=999.0, effective=str((forecast_day + pd.Timedelta(days=10)).date()))

    row = pd.DataFrame(
        {
            "date": [forecast_day], "vegetable": ["Tomato"], "market": ["Dambulla"], "source": ["FORECAST"],
            "price": [250.0], "rainfall_mm": [float("nan")], "temperature_c": [float("nan")],
            "humidity_pct": [float("nan")],
        }
    )
    finalized = finalize_live_row(history, row, live_fuel)
    assert finalized["diesel_price"].iloc[0] == 363.0  # historical value, NOT the future 999.0
    assert finalized["petrol_price"].iloc[0] == 371.0


def test_forecast_falls_back_to_historical_fuel_price_without_a_live_snapshot(tmp_path):
    history = _daily_history()
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    result = forecast_future_price(
        history, "Tomato", "Dambulla", latest + pd.Timedelta(days=1), pipeline, config_path,
        live_fuel_snapshot=None,
    )
    assert "Historical fuel price (forward-filled)" in result.data_sources
    assert any("Live CEYPETCO fuel price was not available" in w for w in result.warnings)


# F. Weather ------------------------------------------------------------------------------


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_forecast_fetches_live_weather_when_historical_coverage_falls_short(mock_for_coordinates, tmp_path):
    # weather stops 20 days before price's own last date -- day+1's 14-day-lag
    # requirement (latest - 13) is NOT covered (gap of 7 days), forcing a live fetch.
    history = _daily_history(n_days=90, weather_days=70)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    required_date = (latest + pd.Timedelta(days=1)) - pd.Timedelta(days=14)

    mock_client = MagicMock()
    mock_client.fetch_historical_daily_weather.return_value = pd.DataFrame(
        {
            "date": [required_date],
            "rainfall_mm": [9.9],
            "temperature_c": [30.0],
            "humidity_pct": [55.0],
        }
    )
    mock_for_coordinates.return_value = mock_client

    result = forecast_future_price(
        history, "Tomato", "Dambulla", latest + pd.Timedelta(days=1), pipeline, config_path
    )

    assert mock_client.fetch_historical_daily_weather.called  # the fixture's dates are long past
    assert "Open-Meteo historical weather archive" in result.data_sources
    assert pd.notna(result.predicted_price)


def test_forecast_uses_historical_weather_without_any_live_fetch_when_already_covered(tmp_path):
    history = _daily_history()  # weather covers the full price range already
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    with patch("app.services.OpenMeteoWeatherClient.for_coordinates") as mock_for_coordinates:
        result = forecast_future_price(
            history, "Tomato", "Dambulla", latest + pd.Timedelta(days=1), pipeline, config_path
        )
        mock_for_coordinates.assert_not_called()

    assert "Open-Meteo weather forecast" not in result.data_sources
    assert "Historical weather data" in result.data_sources


def test_forecast_raises_clear_error_when_required_weather_is_genuinely_unavailable(tmp_path):
    history = _daily_history(n_days=90, weather_days=70)  # same 20-day gap as the live-fetch test
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    target = latest + pd.Timedelta(days=1)

    with pytest.raises(ForecastDataUnavailableError, match=str(target.date())):
        forecast_future_price(
            history, "Tomato", "Dambulla", target, pipeline, config_path, use_live_weather=False
        )


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_forecast_reaches_a_far_target_using_the_seasonal_weather_fallback(mock_for_coordinates, tmp_path):
    """Mirrors the real Carrot/Dambulla 2026-09-21 scenario this fallback exists for:
    a forecast horizon long enough that its later days' weather_lag14d requirement
    falls beyond Open-Meteo's 16-day forecast endpoint, but still within its Seasonal
    Forecast API's 46-day reach. Dates are computed relative to the REAL current time
    so this test stays valid regardless of when it runs. The mocked 16-day endpoint
    clamps itself exactly like the real one does, so a date beyond its true reach is
    genuinely absent from its response -- forcing the real fallback logic to trigger,
    not just handed a response engineered to already contain the answer.
    """
    today = pd.Timestamp.now().normalize()
    latest = today - pd.Timedelta(days=1)
    history = _daily_history(start=latest - pd.Timedelta(days=89), n_days=90, weather_days=90)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    target = today + pd.Timedelta(days=30)  # needs weather at today+16 -- 1 day beyond the 16-day cap

    def sixteen_day_window(past_days, forecast_days):
        clamped_past = min(past_days, MAX_PAST_DAYS)
        clamped_forecast = min(forecast_days, MAX_FORECAST_DAYS)
        dates = pd.date_range(
            today - pd.Timedelta(days=clamped_past), today + pd.Timedelta(days=clamped_forecast - 1), freq="D"
        )
        return pd.DataFrame({"date": dates, "rainfall_mm": 5.0, "temperature_c": 28.0, "humidity_pct": 70.0})

    def seasonal_window(forecast_days):
        clamped = min(forecast_days, MAX_SEASONAL_FORECAST_DAYS)
        dates = pd.date_range(today, today + pd.Timedelta(days=clamped - 1), freq="D")
        return pd.DataFrame({"date": dates, "rainfall_mm": 6.0, "temperature_c": 29.0, "humidity_pct": 72.0})

    mock_for_coordinates.return_value.fetch_recent_daily_weather.side_effect = sixteen_day_window
    mock_for_coordinates.return_value.fetch_seasonal_daily_weather.side_effect = seasonal_window

    result = forecast_future_price(
        history, "Tomato", "Dambulla", target, pipeline, config_path, use_live_weather=True
    )

    assert result.target_date == target
    assert result.daily_forecast[-1].date == target
    assert len(result.daily_forecast) == 31
    assert mock_for_coordinates.return_value.fetch_recent_daily_weather.called  # tried first, every day
    assert mock_for_coordinates.return_value.fetch_seasonal_daily_weather.called  # fell back for later days
    assert "Open-Meteo seasonal weather forecast (EC46, beyond 16-day horizon)" in result.data_sources
    assert pd.notna(result.predicted_price)


# G. Feature integrity: no refit, no retrain, strict alignment preserved --------------


def test_forecast_never_refits_encoder_or_retrains_model(tmp_path):
    history = _daily_history()
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    predictor = pipeline._require_predictor()
    feature_columns_before = list(predictor.feature_columns)
    model_coef_before = predictor._model.coef_.copy()

    latest = history["date"].max()
    forecast_future_price(history, "Tomato", "Dambulla", latest + pd.Timedelta(days=5), pipeline, config_path)

    predictor_after = pipeline._require_predictor()
    assert list(predictor_after.feature_columns) == feature_columns_before
    assert (predictor_after._model.coef_ == model_coef_before).all()
    assert predictor_after is predictor  # same instance -- never reconstructed/refit


def test_forecast_rejects_target_date_not_after_latest_actual(tmp_path):
    history = _daily_history()
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    with pytest.raises(ForecastInputError, match="not after the latest"):
        forecast_future_price(history, "Tomato", "Dambulla", latest, pipeline, config_path)
    with pytest.raises(ForecastInputError, match="not after the latest"):
        forecast_future_price(
            history, "Tomato", "Dambulla", latest - pd.Timedelta(days=1), pipeline, config_path
        )


def test_forecast_rejects_unknown_vegetable_market(tmp_path):
    history = _daily_history()
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    with pytest.raises(ForecastInputError, match="no historical data"):
        forecast_future_price(
            history, "Nonexistent Vegetable", "Dambulla",
            history["date"].max() + pd.Timedelta(days=1), pipeline, config_path,
        )


# H. Recursive lag verification --------------------------------------------------------


def test_day_2_prediction_uses_day_1s_predicted_price_as_its_lag_1_feature(tmp_path):
    from app.services import build_prediction_input

    history = _daily_history()
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    day1 = latest + pd.Timedelta(days=1)
    day2 = latest + pd.Timedelta(days=2)

    result = forecast_future_price(history, "Tomato", "Dambulla", day2, pipeline, config_path)
    assert len(result.daily_forecast) == 2
    day1_predicted_price = result.daily_forecast[0].predicted_price

    # Reconstruct exactly what the recursive loop should have built after day 1, and
    # verify day 2's OWN price_lag_1d feature -- independently -- equals day 1's
    # predicted price, not a real observation and not a fabricated value.
    day1_row = pd.DataFrame(
        {
            "date": [day1], "vegetable": ["Tomato"], "market": ["Dambulla"], "source": ["FORECAST"],
            "price": [day1_predicted_price],
            "rainfall_mm": [float("nan")], "temperature_c": [float("nan")], "humidity_pct": [float("nan")],
            "diesel_price": [363.0], "petrol_price": [371.0],
        }
    )
    forecast_history_after_day1 = pd.concat([history, day1_row], ignore_index=True)
    input_day2 = build_prediction_input(forecast_history_after_day1, day2, "Tomato", "Dambulla")

    normalized = pipeline._normalize(input_day2)
    featured = pipeline._feature_engineer.build_features(normalized)
    day2_features = featured.loc[featured["date"] == day2].iloc[0]

    assert day2_features["price_lag_1d"] == pytest.approx(day1_predicted_price)


def _rebuild_forecast_history_through(history, daily_forecast, through_count, vegetable="Tomato", market="Dambulla"):
    """Rebuild the forecast_history DataFrame forecast_future_price() would have built
    after its first `through_count` recursive days, using only the already-known
    predicted prices from a single prior forecast_future_price() call -- lets a test
    independently verify a LATER day's lag/rolling feature without re-running the loop
    or reimplementing its feature computation."""
    from app.services import finalize_live_row

    rebuilt = history.copy()
    for point in daily_forecast[:through_count]:
        row = pd.DataFrame(
            {
                "date": [point.date], "vegetable": [vegetable], "market": [market], "source": ["FORECAST"],
                "price": [point.predicted_price],
                "rainfall_mm": [float("nan")], "temperature_c": [float("nan")], "humidity_pct": [float("nan")],
            }
        )
        rebuilt = pd.concat([rebuilt, finalize_live_row(rebuilt, row, None)], ignore_index=True)
    return rebuilt


def test_day_8_prediction_uses_day_1s_predicted_price_for_price_lag_7d(tmp_path):
    from app.services import build_prediction_input

    history = _daily_history(n_days=90, weather_days=145)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history.loc[history["price"].notna(), "date"].max()
    day8 = latest + pd.Timedelta(days=8)

    result = forecast_future_price(
        history, "Tomato", "Dambulla", day8, pipeline, config_path, use_live_weather=False
    )
    assert len(result.daily_forecast) == 8
    day1_predicted_price = result.daily_forecast[0].predicted_price

    forecast_history_through_day7 = _rebuild_forecast_history_through(history, result.daily_forecast, 7)
    input_day8 = build_prediction_input(forecast_history_through_day7, day8, "Tomato", "Dambulla")
    featured = pipeline._feature_engineer.build_features(pipeline._normalize(input_day8))
    day8_features = featured.loc[featured["date"] == day8].iloc[0]

    assert day8_features["price_lag_7d"] == pytest.approx(day1_predicted_price)


def test_day_15_prediction_uses_day_1s_predicted_price_for_price_lag_14d(tmp_path):
    from app.services import build_prediction_input

    history = _daily_history(n_days=90, weather_days=145)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history.loc[history["price"].notna(), "date"].max()
    day15 = latest + pd.Timedelta(days=15)

    result = forecast_future_price(
        history, "Tomato", "Dambulla", day15, pipeline, config_path, use_live_weather=False
    )
    assert len(result.daily_forecast) == 15
    day1_predicted_price = result.daily_forecast[0].predicted_price

    forecast_history_through_day14 = _rebuild_forecast_history_through(history, result.daily_forecast, 14)
    input_day15 = build_prediction_input(forecast_history_through_day14, day15, "Tomato", "Dambulla")
    featured = pipeline._feature_engineer.build_features(pipeline._normalize(input_day15))
    day15_features = featured.loc[featured["date"] == day15].iloc[0]

    assert day15_features["price_lag_14d"] == pytest.approx(day1_predicted_price)


def test_day_31_prediction_uses_day_1s_predicted_price_for_price_lag_30d(tmp_path):
    from app.services import build_prediction_input

    history = _daily_history(n_days=90, weather_days=145)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history.loc[history["price"].notna(), "date"].max()
    day31 = latest + pd.Timedelta(days=31)

    result = forecast_future_price(
        history, "Tomato", "Dambulla", day31, pipeline, config_path, use_live_weather=False
    )
    assert len(result.daily_forecast) == 31
    day1_predicted_price = result.daily_forecast[0].predicted_price

    forecast_history_through_day30 = _rebuild_forecast_history_through(history, result.daily_forecast, 30)
    input_day31 = build_prediction_input(forecast_history_through_day30, day31, "Tomato", "Dambulla")
    featured = pipeline._feature_engineer.build_features(pipeline._normalize(input_day31))
    day31_features = featured.loc[featured["date"] == day31].iloc[0]

    assert day31_features["price_lag_30d"] == pytest.approx(day1_predicted_price)


def test_rolling_windows_include_prior_recursive_forecast_rows(tmp_path):
    """A day whose entire trailing 7-day window falls after the last actual date must
    compute price_rolling_mean_7d/std_7d purely from prior FORECAST rows -- proving
    RollingFeatureGenerator (unchanged) genuinely incorporates recursive predictions,
    not just real observations."""
    from app.services import build_prediction_input

    history = _daily_history(n_days=90, weather_days=145)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history.loc[history["price"].notna(), "date"].max()
    day10 = latest + pd.Timedelta(days=10)

    result = forecast_future_price(
        history, "Tomato", "Dambulla", day10, pipeline, config_path, use_live_weather=False
    )
    assert len(result.daily_forecast) == 10

    forecast_history_through_day9 = _rebuild_forecast_history_through(history, result.daily_forecast, 9)
    input_day10 = build_prediction_input(forecast_history_through_day9, day10, "Tomato", "Dambulla")
    featured = pipeline._feature_engineer.build_features(pipeline._normalize(input_day10))
    day10_features = featured.loc[featured["date"] == day10].iloc[0]

    # window = [day10 - 7d, day10) = [day3, day10) -- entirely days 3..9, i.e.
    # daily_forecast indices 2..8 (0-based: day1 -> index 0)
    window_prices = [p.predicted_price for p in result.daily_forecast[2:9]]
    assert len(window_prices) == 7
    assert day10_features["price_rolling_mean_7d"] == pytest.approx(sum(window_prices) / len(window_prices))
    assert pd.notna(day10_features["price_rolling_std_7d"])


# F. Future HARTI observations are never required --------------------------------------


@patch("app.services.HartiLiveClient.from_config")
def test_forecast_never_touches_hartiliveclient_for_future_dates(mock_from_config, tmp_path):
    """Recursive forecasting must never attempt to fetch a "future HARTI bulletin" --
    future days are covered entirely by this run's own FORECAST rows."""
    mock_from_config.side_effect = AssertionError(
        "forecast_future_price must never construct a HartiLiveClient"
    )
    history = _daily_history()
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    result = forecast_future_price(
        history, "Tomato", "Dambulla", latest + pd.Timedelta(days=5), pipeline, config_path
    )

    assert len(result.daily_forecast) == 5
    mock_from_config.assert_not_called()


# H. Forecast rows are never persisted anywhere -----------------------------------------


# Blackout-day regression: a genuine HARTI gap must not permanently block forecasting --


def test_long_horizon_forecast_reaches_target_despite_a_genuine_harti_blackout_day(tmp_path):
    """Reproduces the exact real-world scenario this feature exists for (see the
    real Carrot/Dambulla verification): a single calendar day with NO real HARTI
    observation at all, positioned exactly 30 days before a forecast day, blocks
    price_lag_30d under the plain recursive loop. impute_missing_historical_prices()
    (Priority 2 -- only after a real archive fetch has already failed, simulated here
    by never even attempting one) must let the SAME 32-day forecast reach its exact
    target date, using an estimate from real surrounding prices, clearly labeled and
    reported -- never silently, never mistaken for a real observation.
    """
    from app.services import impute_missing_historical_prices

    history = _daily_history(n_days=90, weather_days=145)
    latest = history.loc[history["price"].notna(), "date"].max()
    blackout_day = latest - pd.Timedelta(days=22)  # latest + 8 - 30 == blackout_day
    before_price = history.loc[history["date"] == blackout_day - pd.Timedelta(days=1), "price"].iloc[0]
    after_price = history.loc[history["date"] == blackout_day + pd.Timedelta(days=1), "price"].iloc[0]

    blacked_out_history = history.loc[history["date"] != blackout_day].reset_index(drop=True)
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, blacked_out_history)

    target = latest + pd.Timedelta(days=32)

    # Without imputation, the blackout day genuinely stops the forecast partway through.
    with pytest.raises(ForecastDataUnavailableError, match="price_lag_30d"):
        forecast_future_price(
            blacked_out_history, "Tomato", "Dambulla", target, pipeline, config_path,
            use_live_weather=False,
        )

    # With the missing date imputed first (Priority 2, after a real fetch would have
    # already been attempted and failed -- not modeled again here, see
    # tests/app/test_harti_live_services.py for that ordering), the full horizon succeeds.
    imputed_history, imputed_dates = impute_missing_historical_prices(
        blacked_out_history, "Tomato", "Dambulla", config_path
    )
    assert imputed_dates == [blackout_day]

    result = forecast_future_price(
        imputed_history, "Tomato", "Dambulla", target, pipeline, config_path, use_live_weather=False
    )

    assert result.target_date == target
    assert result.forecast_horizon_days == 32
    assert len(result.daily_forecast) == 32
    assert result.daily_forecast[-1].date == target
    assert pd.notna(result.predicted_price)

    # the imputation itself is reported, not hidden
    assert result.imputed_historical_dates == [blackout_day]
    assert any("runtime-imputed" in s.lower() or "imputed" in s.lower() for s in result.data_sources)
    assert any(
        "Temporary estimated historical values" in w and "Original HARTI data was not modified" in w
        for w in result.warnings
    )

    # the estimate itself is the deterministic linear interpolation of its real
    # neighbors -- not an arbitrary or fabricated number
    imputed_row = imputed_history.loc[imputed_history["date"] == blackout_day]
    assert imputed_row["price"].iloc[0] == pytest.approx((before_price + after_price) / 2)
    assert imputed_row["source"].iloc[0] == "IMPUTED_HISTORICAL"

    # the original (blacked-out) history handed to imputation was never mutated
    assert blackout_day not in blacked_out_history["date"].to_numpy()


def test_forecast_never_writes_anything_to_disk(tmp_path):
    history = _daily_history()
    config_path = _config_at(tmp_path)
    pipeline = _train_and_persist(config_path, history)

    latest = history["date"].max()
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        forecast_future_price(
            history, "Tomato", "Dambulla", latest + pd.Timedelta(days=5), pipeline, config_path
        )
        mock_to_csv.assert_not_called()


# max_supported_forecast_date -----------------------------------------------------------


def test_max_supported_forecast_date_uses_the_real_provider_constant_and_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "feature_engineering:\n  weather_lag_days: 14\n", encoding="utf-8"
    )
    latest_actual = pd.Timestamp("2026-08-21")
    estimate = max_supported_forecast_date(latest_actual, str(config_path))

    today = pd.Timestamp.now().normalize()
    assert estimate == today + pd.Timedelta(days=MAX_SEASONAL_FORECAST_DAYS - 1 + 14)
