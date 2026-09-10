"""Unit tests for src.data_collection.live_weather_client.OpenMeteoWeatherClient.

All HTTP calls are mocked -- these tests never touch the real Open-Meteo service.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_collection.live_weather_client import (
    DEFAULT_ARCHIVE_BASE_URL,
    DEFAULT_BASE_URL,
    DEFAULT_SEASONAL_BASE_URL,
    MAX_FORECAST_DAYS,
    MAX_PAST_DAYS,
    MAX_SEASONAL_FORECAST_DAYS,
    LiveWeatherError,
    OpenMeteoWeatherClient,
)


def _client(**overrides) -> OpenMeteoWeatherClient:
    kwargs = dict(latitude=7.8567, longitude=80.6517, timezone="Asia/Colombo", timeout_seconds=5.0)
    kwargs.update(overrides)
    return OpenMeteoWeatherClient(**kwargs)


def _daily_payload(
    dates=("2026-08-13", "2026-08-14", "2026-08-15"),
    rainfall=(0.0, 1.5, 1.5),
    temperature=(27.8, 27.3, 27.0),
    humidity=(70, 73, 74),
) -> dict:
    return {
        "latitude": 7.84,
        "longitude": 80.64,
        "daily_units": {
            "time": "iso8601",
            "temperature_2m_mean": "°C",
            "relative_humidity_2m_mean": "%",
            "precipitation_sum": "mm",
        },
        "daily": {
            "time": list(dates),
            "temperature_2m_mean": list(temperature),
            "relative_humidity_2m_mean": list(humidity),
            "precipitation_sum": list(rainfall),
        },
    }


def _fake_response(status_code=200, json_data=None, text="") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    if json_data is not None:
        response.json.return_value = json_data
    return response


# 1. Successful response + field mapping -------------------------------------------


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_success_returns_expected_schema(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    client = _client()

    df = client.fetch_recent_daily_weather(past_days=2, forecast_days=1)

    assert list(df.columns) == ["date", "rainfall_mm", "temperature_c", "humidity_pct"]
    assert len(df) == 3


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_maps_temperature_from_temperature_2m_mean(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload(temperature=(27.8, 27.3, 27.0)))
    df = _client().fetch_recent_daily_weather(past_days=2)
    assert df["temperature_c"].tolist() == pytest.approx([27.8, 27.3, 27.0])


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_maps_humidity_from_relative_humidity_2m_mean(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload(humidity=(70, 73, 74)))
    df = _client().fetch_recent_daily_weather(past_days=2)
    assert df["humidity_pct"].tolist() == pytest.approx([70, 73, 74])


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_maps_rainfall_from_precipitation_sum(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload(rainfall=(0.0, 1.8, 1.5)))
    df = _client().fetch_recent_daily_weather(past_days=2)
    assert df["rainfall_mm"].tolist() == pytest.approx([0.0, 1.8, 1.5])


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_converts_dates_to_timestamps(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    df = _client().fetch_recent_daily_weather(past_days=2)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].tolist() == list(pd.to_datetime(["2026-08-13", "2026-08-14", "2026-08-15"]))


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_sends_expected_request_params(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    client = _client(base_url=DEFAULT_BASE_URL)

    client.fetch_recent_daily_weather(past_days=5, forecast_days=2)

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["latitude"] == 7.8567
    assert kwargs["params"]["longitude"] == 80.6517
    assert kwargs["params"]["timezone"] == "Asia/Colombo"
    assert kwargs["params"]["past_days"] == 5
    assert kwargs["params"]["forecast_days"] == 2
    assert "precipitation_sum" in kwargs["params"]["daily"]
    assert "temperature_2m_mean" in kwargs["params"]["daily"]
    assert "relative_humidity_2m_mean" in kwargs["params"]["daily"]
    assert kwargs["timeout"] == 5.0


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_clamps_past_and_forecast_days_to_documented_maximums(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    client = _client()

    client.fetch_recent_daily_weather(past_days=99999, forecast_days=99999)

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["past_days"] == MAX_PAST_DAYS
    assert kwargs["params"]["forecast_days"] == MAX_FORECAST_DAYS


# 2. Invalid / error responses ------------------------------------------------------


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_open_meteo_error_payload(mock_get):
    mock_get.return_value = _fake_response(
        json_data={"error": True, "reason": "Latitude must be in range of -90 to 90°."}
    )
    with pytest.raises(LiveWeatherError, match="Latitude must be in range"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_missing_daily_block(mock_get):
    mock_get.return_value = _fake_response(json_data={"latitude": 7.8, "longitude": 80.6})
    with pytest.raises(LiveWeatherError, match="daily"):
        _client().fetch_recent_daily_weather(past_days=1)


@pytest.mark.parametrize("missing_field", ["time", "precipitation_sum", "temperature_2m_mean", "relative_humidity_2m_mean"])
@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_missing_required_field(mock_get, missing_field):
    payload = _daily_payload()
    del payload["daily"][missing_field]
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="missing required daily field"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_network_failure(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("no route to host")
    with pytest.raises(LiveWeatherError, match="request failed"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_timeout(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("timed out")
    with pytest.raises(LiveWeatherError, match="timed out"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_http_error_status(mock_get):
    mock_get.return_value = _fake_response(status_code=400, text="Bad Request")
    with pytest.raises(LiveWeatherError, match="HTTP 400"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_malformed_json(mock_get):
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    mock_get.return_value = response
    with pytest.raises(LiveWeatherError, match="not valid JSON"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_non_numeric_value(mock_get):
    payload = _daily_payload()
    payload["daily"]["precipitation_sum"] = [0.0, "not-a-number", 1.5]
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="non-numeric"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_invalid_date(mock_get):
    payload = _daily_payload(dates=("2026-08-13", "not-a-date", "2026-08-15"))
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="unparseable date"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_implausible_temperature(mock_get):
    payload = _daily_payload(temperature=(27.8, 250.0, 27.0))  # 250C is not physically plausible
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="implausible"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_negative_rainfall(mock_get):
    payload = _daily_payload(rainfall=(0.0, -5.0, 1.5))
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="implausible"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_null_value_in_response(mock_get):
    payload = _daily_payload()
    payload["daily"]["temperature_2m_mean"] = [27.8, None, 27.0]
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="missing \\(null\\)"):
        _client().fetch_recent_daily_weather(past_days=1)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_recent_daily_weather_raises_on_empty_daily_rows(mock_get):
    payload = _daily_payload(dates=(), rainfall=(), temperature=(), humidity=())
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="no daily rows"):
        _client().fetch_recent_daily_weather(past_days=1)


# 3. from_config ----------------------------------------------------------------


def test_from_config_reads_weather_api_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weather_api:\n"
        "  latitude: 7.8567\n"
        "  longitude: 80.6517\n"
        '  timezone: "Asia/Colombo"\n'
        "  timeout_seconds: 15\n",
        encoding="utf-8",
    )
    client = OpenMeteoWeatherClient.from_config(str(config_path))
    assert client._latitude == 7.8567
    assert client._longitude == 80.6517
    assert client._timezone == "Asia/Colombo"
    assert client._timeout_seconds == 15


def test_from_config_raises_when_coordinates_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weather_api:\n  timezone: UTC\n", encoding="utf-8")
    from config.config_loader import ConfigKeyError

    with pytest.raises(ConfigKeyError):
        OpenMeteoWeatherClient.from_config(str(config_path))


def test_real_config_still_resolves_historical_weather_csv_path_unchanged():
    """Adding weather_api: to config/config.yaml must not disturb the historical CSV path."""
    from config.config_loader import ConfigLoader

    config = ConfigLoader("config/config.yaml")
    assert config.get("paths.raw_files.weather_data") == "data/raw/weather_data.csv"
    assert config.get_required("weather_api.latitude") is not None


# 4. fetch_historical_daily_weather() -- Historical Weather API, same client -----------


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_historical_daily_weather_success_returns_expected_schema(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    df = _client().fetch_historical_daily_weather(start_date="2026-08-13", end_date="2026-08-15")

    assert list(df.columns) == ["date", "rainfall_mm", "temperature_c", "humidity_pct"]
    assert len(df) == 3


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_historical_daily_weather_sends_start_end_date_to_archive_endpoint(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    client = _client(archive_base_url=DEFAULT_ARCHIVE_BASE_URL)

    client.fetch_historical_daily_weather(start_date="2024-04-11", end_date="2026-08-13")

    args, kwargs = mock_get.call_args
    assert args[0] == DEFAULT_ARCHIVE_BASE_URL
    assert kwargs["params"]["start_date"] == "2024-04-11"
    assert kwargs["params"]["end_date"] == "2026-08-13"
    assert "past_days" not in kwargs["params"]
    assert "forecast_days" not in kwargs["params"]
    assert "precipitation_sum" in kwargs["params"]["daily"]


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_historical_daily_weather_reuses_same_error_handling_as_forecast_endpoint(mock_get):
    """Proves the archive method reuses _send()/_parse_daily_payload() rather than
    duplicating error handling -- one representative failure mode (network error) is
    enough to prove reuse; the full failure-mode matrix is already covered for the
    shared parse/validate code by fetch_recent_daily_weather()'s tests above."""
    mock_get.side_effect = requests.exceptions.ConnectionError("no route to host")
    with pytest.raises(LiveWeatherError, match="request failed"):
        _client().fetch_historical_daily_weather(start_date="2026-08-13", end_date="2026-08-15")


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_historical_daily_weather_raises_on_implausible_temperature(mock_get):
    payload = _daily_payload(temperature=(27.8, 250.0, 27.0))
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="implausible"):
        _client().fetch_historical_daily_weather(start_date="2026-08-13", end_date="2026-08-15")


# 5. for_coordinates() -- client for an arbitrary (production-region) location --------


def test_for_coordinates_uses_given_coordinates_not_config_market_location(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weather_api:\n"
        "  latitude: 7.8567\n"
        "  longitude: 80.6517\n"
        '  timezone: "Asia/Colombo"\n'
        "  timeout_seconds: 15\n",
        encoding="utf-8",
    )
    client = OpenMeteoWeatherClient.for_coordinates(6.97078, 80.78286, config_path=str(config_path))

    assert client._latitude == 6.97078
    assert client._longitude == 80.78286
    # everything else still comes from config, same as from_config()
    assert client._timezone == "Asia/Colombo"
    assert client._timeout_seconds == 15


@patch("src.data_collection.live_weather_client.requests.get")
def test_for_coordinates_client_sends_the_given_coordinates_in_requests(mock_get, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weather_api:\n  latitude: 7.8567\n  longitude: 80.6517\n", encoding="utf-8")
    mock_get.return_value = _fake_response(json_data=_daily_payload())

    client = OpenMeteoWeatherClient.for_coordinates(9.66845, 80.00742, config_path=str(config_path))
    client.fetch_recent_daily_weather(past_days=2)

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["latitude"] == 9.66845
    assert kwargs["params"]["longitude"] == 80.00742


# 6. fetch_seasonal_daily_weather() -- Seasonal Forecast API, beyond the 16-day horizon --


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_seasonal_daily_weather_success_returns_expected_schema(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    df = _client().fetch_seasonal_daily_weather(forecast_days=3)

    assert list(df.columns) == ["date", "rainfall_mm", "temperature_c", "humidity_pct"]
    assert len(df) == 3


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_seasonal_daily_weather_sends_forecast_days_to_seasonal_endpoint(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    client = _client(seasonal_base_url=DEFAULT_SEASONAL_BASE_URL)

    client.fetch_seasonal_daily_weather(forecast_days=20)

    args, kwargs = mock_get.call_args
    assert args[0] == DEFAULT_SEASONAL_BASE_URL
    assert kwargs["params"]["forecast_days"] == 20
    assert "past_days" not in kwargs["params"]
    assert "precipitation_sum" in kwargs["params"]["daily"]
    assert "temperature_2m_mean" in kwargs["params"]["daily"]
    assert "relative_humidity_2m_mean" in kwargs["params"]["daily"]
    # only the plain (ensemble-mean) field names are requested -- never a specific
    # ensemble member, and no request for fields this client doesn't parse
    assert "member" not in kwargs["params"]["daily"]


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_seasonal_daily_weather_clamps_forecast_days_to_the_ec46_boundary(mock_get):
    mock_get.return_value = _fake_response(json_data=_daily_payload())
    client = _client()

    client.fetch_seasonal_daily_weather(forecast_days=99999)

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["forecast_days"] == MAX_SEASONAL_FORECAST_DAYS


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_seasonal_daily_weather_reuses_same_error_handling_as_forecast_endpoint(mock_get):
    """Proves the seasonal method reuses _send()/_parse_daily_payload()/_validate()
    rather than duplicating error handling -- one representative failure mode is
    enough; the full failure-mode matrix is already covered by
    fetch_recent_daily_weather()'s tests above for the shared parse/validate code."""
    mock_get.side_effect = requests.exceptions.ConnectionError("no route to host")
    with pytest.raises(LiveWeatherError, match="request failed"):
        _client().fetch_seasonal_daily_weather(forecast_days=20)


@patch("src.data_collection.live_weather_client.requests.get")
def test_fetch_seasonal_daily_weather_raises_on_implausible_temperature(mock_get):
    payload = _daily_payload(temperature=(27.8, 250.0, 27.0))
    mock_get.return_value = _fake_response(json_data=payload)
    with pytest.raises(LiveWeatherError, match="implausible"):
        _client().fetch_seasonal_daily_weather(forecast_days=3)


def test_from_config_reads_seasonal_base_url(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weather_api:\n"
        "  latitude: 7.8567\n"
        "  longitude: 80.6517\n"
        '  seasonal_base_url: "https://example.test/seasonal"\n',
        encoding="utf-8",
    )
    client = OpenMeteoWeatherClient.from_config(str(config_path))
    assert client._seasonal_base_url == "https://example.test/seasonal"


def test_from_config_defaults_seasonal_base_url_when_absent(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weather_api:\n  latitude: 7.8567\n  longitude: 80.6517\n", encoding="utf-8")
    client = OpenMeteoWeatherClient.from_config(str(config_path))
    assert client._seasonal_base_url == DEFAULT_SEASONAL_BASE_URL
