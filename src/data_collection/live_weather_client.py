"""Live weather client: fetches recent/forecast daily weather from Open-Meteo and
normalizes it into the project's existing weather schema (date, rainfall_mm,
temperature_c, humidity_pct).

This is the ONLY module that builds an Open-Meteo request URL or reads its JSON
field names -- everything downstream (merging, feature engineering, prediction)
consumes the same schema WeatherDataLoader already produces from the CSV, so a
live-fetched DataFrame is a drop-in alternative weather *source*, never a
different shape.

Mirrors WeatherDataLoader's "one exception type" design (see its module
docstring): every failure mode here -- network error, timeout, HTTP error,
malformed JSON, missing field, invalid date, implausible value -- is
normalized to LiveWeatherError, so callers only need to catch one thing.
"""

import logging
from typing import Any

import pandas as pd
import requests

from config.config_loader import ConfigLoader

from .data_validator import DataValidationError, DataValidator

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.open-meteo.com/v1/forecast"
# Open-Meteo's Historical Weather API -- same daily variable names and JSON response
# shape as the forecast endpoint above (verified against a real, unmocked request),
# just keyed by start_date/end_date instead of past_days/forecast_days, and covering
# back to 1940 rather than the forecast endpoint's 92-day past window. Used for
# rebuilding training-time weather for a production region, not for live prediction.
DEFAULT_ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo's separate Seasonal Forecast API -- a genuinely distinct endpoint (not a
# parameter on the forecast endpoint above), backed by ECMWF's EC46 sub-seasonal
# ensemble model for its first ~46 days and the coarser, longer-range SEAS5 model
# beyond that (verified against the real, unmocked endpoint). Used only as a fallback
# when a required weather-lag date falls beyond the forecast endpoint's own
# MAX_FORECAST_DAYS reach -- see fetch_seasonal_daily_weather() and
# app/services.py::build_live_prediction_input()'s call order.
DEFAULT_SEASONAL_BASE_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"
# Open-Meteo's own documented ceilings for these parameters (open-meteo.com/en/docs).
MAX_PAST_DAYS = 92
MAX_FORECAST_DAYS = 16
# The API itself accepts forecast_days up to 217 for the seasonal endpoint (SEAS5's
# own ~7-month reach), but only the first 46 days come from EC46's daily-resolution
# sub-seasonal model -- days beyond that are SEAS5's coarser, lower-confidence
# monthly-blended product. This client deliberately caps itself at the EC46 boundary,
# not the API's outer limit, so every day it returns is the higher-confidence product.
MAX_SEASONAL_FORECAST_DAYS = 46

# Open-Meteo daily variable -> the project's existing weather_columns schema.
# temperature_2m_mean (a daily mean) is used, not temperature_2m_max/min, to match
# weather_data.csv's single-reading-per-day semantics. precipitation_sum is Open-Meteo's
# standard daily rainfall total; relative_humidity_2m_mean is its daily humidity aggregate.
RAINFALL_FIELD = "precipitation_sum"
TEMPERATURE_FIELD = "temperature_2m_mean"
HUMIDITY_FIELD = "relative_humidity_2m_mean"
_REQUIRED_DAILY_FIELDS = ["time", RAINFALL_FIELD, TEMPERATURE_FIELD, HUMIDITY_FIELD]


class LiveWeatherError(Exception):
    """Raised when live weather cannot be fetched, parsed, or validated.

    Covers network failure, timeout, non-200 HTTP status, an Open-Meteo-level
    error payload, malformed JSON, a missing required field, an unparseable
    date, a non-numeric reading, or a reading outside a plausible range.
    Never partially returns a result -- either a fully validated DataFrame
    comes back, or this is raised.
    """


class OpenMeteoWeatherClient:
    """Fetches recent/forecast daily weather from Open-Meteo for one configured location.

    Returns data in exactly WeatherDataLoader's output schema (date,
    rainfall_mm, temperature_c, humidity_pct), so callers can treat this as an
    alternative weather *source* without any downstream branching.

    Uses Open-Meteo's forecast endpoint's `past_days`/`forecast_days`
    parameters, which return recent-past and forecast days in a single
    response. A date at or before today is an actual observation; a date
    after today is a genuine weather *forecast*, not an observation -- that
    distinction is inherent to predicting far enough ahead that a required
    lag feature's source date hasn't happened yet, and is not hidden by this
    client (see build_prediction_input's caller for how the gap is sized).
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        timezone: str = "UTC",
        timeout_seconds: float = 10.0,
        base_url: str = DEFAULT_BASE_URL,
        archive_base_url: str = DEFAULT_ARCHIVE_BASE_URL,
        seasonal_base_url: str = DEFAULT_SEASONAL_BASE_URL,
        date_column: str = "date",
        temperature_min_c: float = 0.0,
        temperature_max_c: float = 45.0,
        humidity_min_pct: float = 0.0,
        humidity_max_pct: float = 100.0,
    ) -> None:
        self._latitude = latitude
        self._longitude = longitude
        self._timezone = timezone
        self._timeout_seconds = timeout_seconds
        self._base_url = base_url
        self._archive_base_url = archive_base_url
        self._seasonal_base_url = seasonal_base_url
        self._date_column = date_column
        self._temperature_min_c = temperature_min_c
        self._temperature_max_c = temperature_max_c
        self._humidity_min_pct = humidity_min_pct
        self._humidity_max_pct = humidity_max_pct
        self._validator = DataValidator()

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "OpenMeteoWeatherClient":
        """Build a client from config.yaml's weather_api section.

        Raises:
            ConfigKeyError: if weather_api.latitude/longitude aren't
                configured -- there is no safe default location to guess.
        """
        config = ConfigLoader(config_path)
        return cls(
            latitude=config.get_required("weather_api.latitude"),
            longitude=config.get_required("weather_api.longitude"),
            timezone=config.get("weather_api.timezone", "UTC"),
            timeout_seconds=config.get("weather_api.timeout_seconds", 10.0),
            base_url=config.get("weather_api.base_url", DEFAULT_BASE_URL),
            archive_base_url=config.get("weather_api.archive_base_url", DEFAULT_ARCHIVE_BASE_URL),
            seasonal_base_url=config.get("weather_api.seasonal_base_url", DEFAULT_SEASONAL_BASE_URL),
            date_column=config.get("data_schema.date_column", "date"),
            temperature_min_c=config.get("preprocessing.temperature_min_c", 0.0),
            temperature_max_c=config.get("preprocessing.temperature_max_c", 45.0),
            humidity_min_pct=config.get("preprocessing.humidity_min_pct", 0.0),
            humidity_max_pct=config.get("preprocessing.humidity_max_pct", 100.0),
        )

    @classmethod
    def for_coordinates(
        cls,
        latitude: float,
        longitude: float,
        config_path: str = "config/config.yaml",
    ) -> "OpenMeteoWeatherClient":
        """Build a client for an arbitrary (latitude, longitude), reusing every other
        setting (timezone, timeout, base URLs, validation ranges) from config.yaml's
        weather_api section.

        Used for production-region weather: the coordinates come from
        config/crop_production_regions.yaml (resolved per vegetable), not from
        weather_api.latitude/longitude, but everything else about the request
        (timezone, timeout, validation thresholds) is still config-driven and
        identical to the market-location client.
        """
        config = ConfigLoader(config_path)
        return cls(
            latitude=latitude,
            longitude=longitude,
            timezone=config.get("weather_api.timezone", "UTC"),
            timeout_seconds=config.get("weather_api.timeout_seconds", 10.0),
            base_url=config.get("weather_api.base_url", DEFAULT_BASE_URL),
            archive_base_url=config.get("weather_api.archive_base_url", DEFAULT_ARCHIVE_BASE_URL),
            seasonal_base_url=config.get("weather_api.seasonal_base_url", DEFAULT_SEASONAL_BASE_URL),
            date_column=config.get("data_schema.date_column", "date"),
            temperature_min_c=config.get("preprocessing.temperature_min_c", 0.0),
            temperature_max_c=config.get("preprocessing.temperature_max_c", 45.0),
            humidity_min_pct=config.get("preprocessing.humidity_min_pct", 0.0),
            humidity_max_pct=config.get("preprocessing.humidity_max_pct", 100.0),
        )

    def fetch_recent_daily_weather(self, past_days: int, forecast_days: int = 1) -> pd.DataFrame:
        """Fetch one row per day of rainfall/temperature/humidity for the requested window.

        The window is [today - past_days, today + forecast_days - 1] in the
        configured timezone, matching Open-Meteo's own past_days/forecast_days
        semantics. Both bounds are clamped to Open-Meteo's documented maximums
        (92 / 16) rather than sent as-is and left to fail unpredictably.

        Returns:
            A DataFrame with columns [date_column, "rainfall_mm",
            "temperature_c", "humidity_pct"], validated the same way
            WeatherDataLoader validates the CSV (non-negative rainfall,
            temperature/humidity within a plausible range).

        Raises:
            LiveWeatherError: on any network failure, timeout, non-200 HTTP
                response, Open-Meteo-reported error, malformed JSON, missing
                required field, unparseable date, non-numeric value, or a
                value outside its plausible range.
        """
        clamped_past_days = max(0, min(int(past_days), MAX_PAST_DAYS))
        clamped_forecast_days = max(1, min(int(forecast_days), MAX_FORECAST_DAYS))

        payload = self._request(clamped_past_days, clamped_forecast_days)
        df = self._parse_daily_payload(payload)
        self._validate(df)

        logger.info(
            "LiveWeather: fetched %d day(s) of weather (%s to %s) from Open-Meteo",
            len(df),
            df[self._date_column].min().date(),
            df[self._date_column].max().date(),
        )
        return df.reset_index(drop=True)

    def fetch_historical_daily_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch one row per day of rainfall/temperature/humidity for [start_date, end_date].

        Uses Open-Meteo's Historical Weather API (archive_base_url), not the
        forecast endpoint fetch_recent_daily_weather() uses -- the forecast
        endpoint's past_days is capped at MAX_PAST_DAYS (92), far too short for
        rebuilding a multi-year training-time weather series for a production
        region. The archive endpoint takes the same daily variable names and
        returns the identical JSON shape (verified against a real request), so
        _parse_daily_payload()/_validate() below are reused unchanged -- this
        is the same client, not a second one.

        Args:
            start_date / end_date: "YYYY-MM-DD" strings, inclusive.

        Returns:
            Same schema as fetch_recent_daily_weather(): [date_column,
            "rainfall_mm", "temperature_c", "humidity_pct"].

        Raises:
            LiveWeatherError: same failure modes as fetch_recent_daily_weather().
        """
        payload = self._send(
            self._archive_base_url,
            {
                "latitude": self._latitude,
                "longitude": self._longitude,
                "timezone": self._timezone,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ",".join([RAINFALL_FIELD, TEMPERATURE_FIELD, HUMIDITY_FIELD]),
            },
        )
        df = self._parse_daily_payload(payload)
        self._validate(df)

        logger.info(
            "LiveWeather: fetched %d day(s) of historical weather (%s to %s) from Open-Meteo archive",
            len(df),
            df[self._date_column].min().date(),
            df[self._date_column].max().date(),
        )
        return df.reset_index(drop=True)

    def fetch_seasonal_daily_weather(self, forecast_days: int = MAX_SEASONAL_FORECAST_DAYS) -> pd.DataFrame:
        """Fetch one row per day of rainfall/temperature/humidity, starting today, from
        Open-Meteo's separate Seasonal Forecast API (seasonal_base_url) -- for a
        required date beyond fetch_recent_daily_weather()'s own MAX_FORECAST_DAYS
        reach (16 days), never for a date in the past (see this client's module
        docstring: only fetch_historical_daily_weather()/the archive endpoint serve
        past dates).

        Backed by ECMWF's EC46 sub-seasonal ensemble model (see MAX_SEASONAL_FORECAST_DAYS'
        own docstring for why this client caps itself there rather than at the API's
        outer ~7-month SEAS5 limit). The API returns each requested variable as both an
        ensemble MEAN (the plain field name, e.g. "precipitation_sum") and 51 individual
        ensemble members (e.g. "precipitation_sum_member01"); this requests and reads
        only the plain (mean) field, the genuine published forecast value for that day
        -- never one arbitrarily-chosen member, and never an invented value.

        Returns:
            Same schema as fetch_recent_daily_weather(): [date_column, "rainfall_mm",
            "temperature_c", "humidity_pct"].

        Raises:
            LiveWeatherError: same failure modes as fetch_recent_daily_weather().
        """
        clamped_forecast_days = max(1, min(int(forecast_days), MAX_SEASONAL_FORECAST_DAYS))
        payload = self._send(
            self._seasonal_base_url,
            {
                "latitude": self._latitude,
                "longitude": self._longitude,
                "timezone": self._timezone,
                "forecast_days": clamped_forecast_days,
                "daily": ",".join([RAINFALL_FIELD, TEMPERATURE_FIELD, HUMIDITY_FIELD]),
            },
        )
        df = self._parse_daily_payload(payload)
        self._validate(df)

        logger.info(
            "LiveWeather: fetched %d day(s) of seasonal forecast weather (%s to %s) from "
            "Open-Meteo's seasonal API",
            len(df),
            df[self._date_column].min().date(),
            df[self._date_column].max().date(),
        )
        return df.reset_index(drop=True)

    def _request(self, past_days: int, forecast_days: int) -> dict[str, Any]:
        params = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "timezone": self._timezone,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "daily": ",".join([RAINFALL_FIELD, TEMPERATURE_FIELD, HUMIDITY_FIELD]),
        }
        return self._send(self._base_url, params)

    def _send(self, base_url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.get(base_url, params=params, timeout=self._timeout_seconds)
        except requests.exceptions.Timeout as error:
            message = f"Open-Meteo request timed out after {self._timeout_seconds}s"
            logger.error(message)
            raise LiveWeatherError(message) from error
        except requests.exceptions.RequestException as error:
            message = f"Open-Meteo request failed: {error}"
            logger.error(message)
            raise LiveWeatherError(message) from error

        if response.status_code != 200:
            message = f"Open-Meteo returned HTTP {response.status_code}: {response.text[:300]}"
            logger.error(message)
            raise LiveWeatherError(message)

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as error:
            message = f"Open-Meteo response was not valid JSON: {error}"
            logger.error(message)
            raise LiveWeatherError(message) from error

        if payload.get("error"):
            message = f"Open-Meteo reported an error: {payload.get('reason', 'unknown')}"
            logger.error(message)
            raise LiveWeatherError(message)

        return payload

    def _parse_daily_payload(self, payload: dict[str, Any]) -> pd.DataFrame:
        daily = payload.get("daily")
        if not isinstance(daily, dict):
            message = "Open-Meteo response is missing the 'daily' block"
            logger.error(message)
            raise LiveWeatherError(message)

        missing_fields = [field for field in _REQUIRED_DAILY_FIELDS if field not in daily]
        if missing_fields:
            message = f"Open-Meteo response is missing required daily field(s): {missing_fields}"
            logger.error(message)
            raise LiveWeatherError(message)

        try:
            df = pd.DataFrame(
                {
                    self._date_column: daily["time"],
                    "rainfall_mm": daily[RAINFALL_FIELD],
                    "temperature_c": daily[TEMPERATURE_FIELD],
                    "humidity_pct": daily[HUMIDITY_FIELD],
                }
            )
        except ValueError as error:
            message = f"Open-Meteo daily field(s) have mismatched lengths: {error}"
            logger.error(message)
            raise LiveWeatherError(message) from error

        if df.empty:
            message = "Open-Meteo response contained no daily rows"
            logger.error(message)
            raise LiveWeatherError(message)

        parsed_dates = pd.to_datetime(df[self._date_column], format="%Y-%m-%d", errors="coerce")
        if parsed_dates.isna().any():
            bad = df.loc[parsed_dates.isna(), self._date_column].tolist()
            message = f"Open-Meteo returned unparseable date(s): {bad[:5]}"
            logger.error(message)
            raise LiveWeatherError(message)
        df[self._date_column] = parsed_dates

        for column in ("rainfall_mm", "temperature_c", "humidity_pct"):
            numeric = pd.to_numeric(df[column], errors="coerce")
            invalid_mask = numeric.isna() & df[column].notna()
            if invalid_mask.any():
                bad = df.loc[invalid_mask, column].tolist()
                message = f"Open-Meteo returned non-numeric value(s) for '{column}': {bad[:5]}"
                logger.error(message)
                raise LiveWeatherError(message)
            df[column] = numeric

        if df[["rainfall_mm", "temperature_c", "humidity_pct"]].isna().any().any():
            message = "Open-Meteo response has missing (null) weather value(s) for at least one day"
            logger.error(message)
            raise LiveWeatherError(message)

        return df

    def _validate(self, df: pd.DataFrame) -> None:
        try:
            self._validator.validate_non_negative(df, "rainfall_mm", "LiveWeather")
            self._validator.validate_range(
                df, "temperature_c", "LiveWeather", self._temperature_min_c, self._temperature_max_c
            )
            self._validator.validate_range(
                df, "humidity_pct", "LiveWeather", self._humidity_min_pct, self._humidity_max_pct
            )
        except DataValidationError as error:
            message = f"Open-Meteo returned implausible weather value(s): {error}"
            logger.error(message)
            raise LiveWeatherError(message) from error
