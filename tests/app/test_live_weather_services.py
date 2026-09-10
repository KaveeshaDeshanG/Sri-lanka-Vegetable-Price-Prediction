"""Unit tests for app.services.build_live_prediction_input.

The Open-Meteo HTTP call is mocked out at the OpenMeteoWeatherClient level (never
the real network) -- these tests only check the "is required_date's weather already
covered?" / row-assembly logic that sits between the history DataFrame and
PredictionPipeline.

required_date = target_date - feature_engineering.weather_lag_days (14 by default):
see src/feature_engineering/weather_features.py -- that is the ONLY date whose
weather actually feeds rainfall_mm_lag14d/temperature_c_lag14d/humidity_pct_lag14d
for the row being predicted.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from app.services import build_live_prediction_input, build_prediction_input
from src.data_collection.live_weather_client import LiveWeatherError


def _history(last_date="2024-01-31", n_days=31, vegetable="Carrot", market="Dambulla", nan_weather_dates=()):
    dates = pd.date_range(end=last_date, periods=n_days, freq="D")
    nan_dates = {pd.Timestamp(d) for d in nan_weather_dates}
    rainfall = [float("nan") if d in nan_dates else 5.0 for d in dates]
    temperature = [float("nan") if d in nan_dates else 28.0 for d in dates]
    humidity = [float("nan") if d in nan_dates else 70.0 for d in dates]
    return pd.DataFrame(
        {
            "date": dates,
            "vegetable": [vegetable] * n_days,
            "market": [market] * n_days,
            "source": ["HARTI"] * n_days,
            "price": [100.0 + i for i in range(n_days)],
            "rainfall_mm": rainfall,
            "temperature_c": temperature,
            "humidity_pct": humidity,
        }
    )


def _config(tmp_path, enabled=True) -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weather_api:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        "  latitude: 7.8567\n"
        "  longitude: 80.6517\n"
        '  timezone: "Asia/Colombo"\n'
        "  timeout_seconds: 10\n"
        "paths:\n"
        # Deliberately nonexistent: keeps this test hermetic against the real
        # project's crop_production_regions.yaml -- "Carrot" (used throughout
        # this file) IS mapped there in reality, which would otherwise resolve
        # to different coordinates than this test's own weather_api section.
        f'  crop_production_regions_mapping: "{(tmp_path / "nonexistent_mapping.yaml").as_posix()}"\n',
        encoding="utf-8",
    )
    return str(config_path)


def _live_weather_covering(dates) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(list(dates))})
    df["rainfall_mm"] = 9.0
    df["temperature_c"] = 31.0
    df["humidity_pct"] = 85.0
    return df


# 1. required_date already covered: no API call, identical to build_prediction_input --


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_no_missing_weather_never_calls_the_live_weather_client(mock_for_coordinates, tmp_path):
    # dense 31-day history, weather fully populated everywhere
    history = _history(last_date="2024-01-31", n_days=31)
    target_date = pd.Timestamp("2024-02-01")  # required_date = 2024-01-18, already in history

    result = build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))

    mock_for_coordinates.assert_not_called()
    expected = build_prediction_input(history, target_date, "Carrot", "Dambulla")
    pd.testing.assert_frame_equal(result, expected)


# 2. required_date's weather missing (beyond history's range): fetched + inserted -----


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_missing_weather_reading_is_filled_with_one_live_weather_context_row(mock_for_coordinates, tmp_path):
    history = _history(last_date="2024-01-31", n_days=31)
    target_date = pd.Timestamp("2024-02-20")  # required_date = 2024-02-06, not in history at all
    required_date = pd.Timestamp("2024-02-06")

    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _live_weather_covering(
        pd.date_range("2024-02-01", "2024-02-25", freq="D")
    )

    result = build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))

    context_rows = result[result["source"] == "LIVE_WEATHER"]
    assert len(context_rows) == 1
    row = context_rows.iloc[0]
    assert row["date"] == required_date
    assert pd.isna(row["price"])
    assert row["rainfall_mm"] == 9.0
    assert row["temperature_c"] == 31.0
    assert row["humidity_pct"] == 85.0
    assert row["vegetable"] == "Carrot"
    assert row["market"] == "Dambulla"

    target_row = result[result["date"] == target_date]
    assert len(target_row) == 1
    assert pd.isna(target_row.iloc[0]["price"])


# 3. required_date's weather missing WITHIN existing history (price present, weather NaN) --


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_missing_weather_within_existing_history_is_backfilled_not_replaced(mock_for_coordinates, tmp_path):
    """Realistic case: price data is current, but the weather feed has a hole on one
    specific date. The existing price row for that date must be left untouched -- the
    fix is an ADDITIONAL row, never an edit to the real historical row."""
    history = _history(last_date="2024-02-19", n_days=50, nan_weather_dates=["2024-02-06"])
    target_date = pd.Timestamp("2024-02-20")  # required_date = 2024-02-06 -- price exists, weather NaN
    required_date = pd.Timestamp("2024-02-06")

    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _live_weather_covering(
        pd.date_range("2024-02-01", "2024-02-25", freq="D")
    )

    result = build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))

    rows_at_required_date = result[result["date"] == required_date]
    assert len(rows_at_required_date) == 2  # original (price, NaN weather) + new context row
    original_row = rows_at_required_date[rows_at_required_date["source"] == "HARTI"].iloc[0]
    assert not pd.isna(original_row["price"])  # untouched
    assert pd.isna(original_row["rainfall_mm"])  # untouched -- still NaN, not silently patched

    context_row = rows_at_required_date[rows_at_required_date["source"] == "LIVE_WEATHER"].iloc[0]
    assert pd.isna(context_row["price"])
    assert context_row["rainfall_mm"] == 9.0


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_fetch_uses_past_days_and_forecast_days_sized_to_required_date(mock_for_coordinates, tmp_path):
    history = _history(last_date="2024-01-31", n_days=31)
    target_date = pd.Timestamp("2024-02-20")

    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _live_weather_covering(
        pd.date_range("2024-02-01", "2024-02-25", freq="D")
    )

    build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))

    assert mock_for_coordinates.return_value.fetch_recent_daily_weather.called
    _, kwargs = mock_for_coordinates.return_value.fetch_recent_daily_weather.call_args
    assert kwargs["past_days"] >= 0
    assert kwargs["forecast_days"] >= 1


# 4. Disabled config: fails clearly instead of silently skipping the missing weather --


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_disabled_weather_api_raises_when_required_date_is_missing(mock_for_coordinates, tmp_path):
    history = _history(last_date="2024-01-31", n_days=31)
    target_date = pd.Timestamp("2024-02-20")

    with pytest.raises(LiveWeatherError, match="disabled"):
        build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path, enabled=False))

    mock_for_coordinates.assert_not_called()


# 5. Client failures propagate, are never swallowed or substituted -------------------


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_live_weather_client_error_propagates_unchanged(mock_for_coordinates, tmp_path):
    history = _history(last_date="2024-01-31", n_days=31)
    target_date = pd.Timestamp("2024-02-20")
    mock_for_coordinates.return_value.fetch_recent_daily_weather.side_effect = LiveWeatherError(
        "Open-Meteo request timed out after 10s"
    )

    with pytest.raises(LiveWeatherError, match="timed out"):
        build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_response_not_covering_required_date_raises_instead_of_predicting_incorrectly(mock_for_coordinates, tmp_path):
    history = _history(last_date="2024-01-31", n_days=31)
    target_date = pd.Timestamp("2024-02-20")  # required_date = 2024-02-06

    # API returned data, but not for the one date that's actually required.
    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _live_weather_covering(
        pd.date_range("2024-03-01", "2024-03-05", freq="D")
    )

    with pytest.raises(LiveWeatherError, match="returned a weather reading for 2024-02-06"):
        build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))


# 6. No mutation of the caller's history_df ------------------------------------------


# 7. required_date beyond the 16-day forecast horizon: Seasonal Forecast API fallback --


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_seasonal_fallback_used_when_required_date_is_beyond_the_16_day_forecast(mock_for_coordinates, tmp_path):
    """required_date is genuinely in the future and beyond fetch_recent_daily_weather()'s
    own 16-day reach -- computed relative to the REAL current time so this test stays
    valid regardless of when it actually runs, exactly mirroring the real-world
    Carrot/Dambulla scenario this fallback exists for."""
    today = pd.Timestamp.now().normalize()
    required_date = today + pd.Timedelta(days=20)  # beyond MAX_FORECAST_DAYS (16)
    target_date = required_date + pd.Timedelta(days=14)  # weather_lag_days default
    history = _history(last_date=(today - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), n_days=31)

    # the primary 16-day endpoint genuinely has nothing for a date this far out
    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _live_weather_covering(
        pd.date_range(today, today + pd.Timedelta(days=15), freq="D")
    )
    mock_for_coordinates.return_value.fetch_seasonal_daily_weather.return_value = _live_weather_covering(
        [required_date]
    )

    result = build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))

    assert mock_for_coordinates.return_value.fetch_recent_daily_weather.called  # tried first
    assert mock_for_coordinates.return_value.fetch_seasonal_daily_weather.called  # fell back

    context_rows = result[result["source"] == "LIVE_WEATHER_SEASONAL"]
    assert len(context_rows) == 1
    row = context_rows.iloc[0]
    assert row["date"] == required_date
    assert pd.isna(row["price"])
    assert row["rainfall_mm"] == 9.0  # the mocked value returned by the seasonal endpoint
    assert row["temperature_c"] == 31.0
    assert row["humidity_pct"] == 85.0


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_seasonal_fallback_not_attempted_for_a_required_date_in_the_past(mock_for_coordinates, tmp_path):
    """A required_date the 16-day endpoint doesn't cover because it's genuinely in the
    PAST (not because it's too far in the future) must never trigger the seasonal
    fallback -- a seasonal forecast is only meaningful going forward from today."""
    history = _history(last_date="2024-01-31", n_days=31)
    target_date = pd.Timestamp("2024-02-20")  # required_date = 2024-02-06, in the past

    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _live_weather_covering(
        pd.date_range("2024-03-01", "2024-03-05", freq="D")
    )

    with pytest.raises(LiveWeatherError, match="2024-02-06"):
        build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))

    mock_for_coordinates.return_value.fetch_seasonal_daily_weather.assert_not_called()


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_seasonal_fallback_never_fabricates_when_it_also_has_no_reading(mock_for_coordinates, tmp_path):
    today = pd.Timestamp.now().normalize()
    required_date = today + pd.Timedelta(days=20)
    target_date = required_date + pd.Timedelta(days=14)
    history = _history(last_date=(today - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), n_days=31)

    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _live_weather_covering(
        pd.date_range(today, today + pd.Timedelta(days=15), freq="D")
    )
    # the seasonal endpoint ALSO doesn't cover the exact required date -- must raise,
    # never invent a value just because a fallback was attempted
    mock_for_coordinates.return_value.fetch_seasonal_daily_weather.return_value = _live_weather_covering(
        pd.date_range(today, today + pd.Timedelta(days=10), freq="D")
    )

    with pytest.raises(LiveWeatherError, match=str(required_date.date())):
        build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_build_live_prediction_input_does_not_mutate_history(mock_for_coordinates, tmp_path):
    history = _history(last_date="2024-01-31", n_days=31)
    original = history.copy()
    target_date = pd.Timestamp("2024-02-20")

    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _live_weather_covering(
        pd.date_range("2024-02-01", "2024-02-25", freq="D")
    )

    build_live_prediction_input(history, target_date, "Carrot", "Dambulla", _config(tmp_path))

    pd.testing.assert_frame_equal(history, original)
