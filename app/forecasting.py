"""Multi-step recursive future-date price forecasting.

Streamlit-free, like app/services.py (which this module composes) -- testable with
plain pytest, kept in its own module because it's a distinct concern from that file's
single-row prediction-input assembly.

Why this lives in app/, not src/prediction/: every live-data building block this needs
(build_prediction_input, build_live_prediction_input, merge_harti_live_price,
backfill_harti_price_gap, finalize_live_row, fetch_harti_latest_snapshot,
fetch_live_fuel_price) already lives in app/services.py, which is deliberately the
layer that composes src/'s live-data-agnostic pieces for the app's own use cases.
src/prediction/pipeline.py is deliberately kept free of any HARTI/Open-Meteo/CEYPETCO
awareness (see its own module docstring) -- forecasting inherently needs all three, so
it belongs alongside app/services.py's other live-data orchestration, not inside
src/prediction/.

Design: the model was trained to predict exactly one day ahead of its most recent known
price (see tests/prediction/test_pipeline_live_weather.py's own docstring: "the only
date the current architecture can validly predict... since price_lag_1d needs
[yesterday]'s real price"). A genuinely distant target date is reached by predicting
one day at a time and feeding each day's own prediction back in as that day's "known"
price before predicting the next -- recursive multi-step forecasting. This calls the
existing, unmodified PredictionPipeline.predict() once per forecast day; no feature
engineering, encoding, or model logic is reimplemented here.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from config.config_loader import ConfigLoader
from src.data_collection.live_fuel_price_client import LiveFuelPriceSnapshot
from src.data_collection.live_weather_client import MAX_SEASONAL_FORECAST_DAYS, LiveWeatherError
from src.models.direct_forecaster import MAX_HORIZON_DAYS, DirectForecaster
from src.prediction.pipeline import PredictionPipeline
from src.prediction.predictor import FeatureAlignmentError

from .services import (
    IMPUTED_HISTORICAL_SOURCE_LABEL,
    LOCAL_WEATHER_SOURCE_LABEL,
    build_live_prediction_input,
    build_prediction_input,
    finalize_live_row,
)

logger = logging.getLogger(__name__)

FORECAST_SOURCE_LABEL = "FORECAST"


class ForecastInputError(Exception):
    """Raised when a forecast request itself is invalid -- not a data-availability problem.

    Covers an unparseable target_date, a (vegetable, market) with no historical rows
    at all, or a target_date that isn't actually in the future relative to the latest
    known real price (callers should use the existing single-step prediction flow for
    that, not recursive forecasting -- see ForecastResult's caller guidance).
    """


class ForecastDataUnavailableError(Exception):
    """Raised when a forecast can't continue because a required day's real/forecast
    data (weather, or any other model feature) is genuinely unavailable.

    Never raised in place of fabricating a value -- this is that refusal, surfaced with
    forecast-specific context (which day, why, how far the forecast got before failing)
    instead of a bare FeatureAlignmentError/LiveWeatherError.
    """


@dataclass(frozen=True)
class DailyForecastPoint:
    """One recursively-predicted day on the way to the requested target_date."""

    date: pd.Timestamp
    predicted_price: float


@dataclass(frozen=True)
class ForecastResult:
    """The outcome of a multi-step recursive forecast.

    predicted_price is the target_date's own forecast -- the last entry of
    daily_forecast. daily_forecast is returned in full (not just the final value) so a
    caller (the UI) can show the day-by-day path without re-running anything.

    data_sources: human-readable labels for what actually fed this forecast (e.g.
    "Latest live HARTI bulletin", "Open-Meteo weather forecast") -- for display only,
    not machine-parsed.

    warnings: non-fatal notices (e.g. "live fuel price unavailable -- used the last
    known historical fuel price instead"). An empty list means nothing degraded.

    imputed_historical_dates: dates the caller's history_df already contained a
    source="IMPUTED_HISTORICAL" (runtime-only, non-HARTI) row for -- i.e. a date real
    HARTI archive data genuinely doesn't cover for this (vegetable, market), estimated
    by app.services.impute_missing_historical_prices() before this call, purely so a
    required lag/rolling feature could be computed. Empty means every date this
    forecast's features needed was backed by a real observation. This field is
    read-only reporting -- forecast_future_price() never imputes anything itself.
    """

    vegetable: str
    market: str
    latest_actual_date: pd.Timestamp
    target_date: pd.Timestamp
    forecast_horizon_days: int
    predicted_price: float
    daily_forecast: list[DailyForecastPoint] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    imputed_historical_dates: list[pd.Timestamp] = field(default_factory=list)


def direct_forecast(
    forecaster: DirectForecaster,
    origin_rows: pd.DataFrame,
    vegetable: str,
    market: str,
    target_date: "pd.Timestamp | str",
    latest_actual_date: pd.Timestamp,
    data_sources: list[str],
) -> ForecastResult:
    """Forecast target_date, and each day up to it, using the direct forecaster.

    Every day's model uses only what was known on latest_actual_date, so no day's forecast
    depends on an earlier forecast or on future weather.

    Raises:
        ForecastInputError: target_date is unparseable or not after latest_actual_date.
        ForecastDataUnavailableError: target_date is too far ahead, or there's no usable
            current price to forecast from.
    """
    try:
        target_date = pd.Timestamp(target_date)
    except (ValueError, TypeError) as error:
        raise ForecastInputError(f"Could not read the date {target_date!r}.") from error
    latest_actual_date = pd.Timestamp(latest_actual_date)
    horizon = (target_date - latest_actual_date).days
    if horizon < 1:
        raise ForecastInputError(
            f"{target_date.date()} is not after the latest known price ({latest_actual_date.date()})."
        )
    if horizon > MAX_HORIZON_DAYS:
        raise ForecastDataUnavailableError(
            f"{target_date.date()} is {horizon} days after the latest known price ({latest_actual_date.date()}). "
            f"The forecaster reaches at most {MAX_HORIZON_DAYS} days ahead."
        )
    try:
        path = forecaster.forecast(origin_rows, vegetable, latest_actual_date, list(range(1, horizon + 1)))
    except ValueError as error:
        raise ForecastDataUnavailableError(str(error)) from error

    return ForecastResult(
        vegetable=vegetable,
        market=market,
        latest_actual_date=latest_actual_date,
        target_date=target_date,
        forecast_horizon_days=horizon,
        predicted_price=float(path["predicted_price"].iloc[-1]),
        daily_forecast=[
            DailyForecastPoint(date=row.target_date, predicted_price=row.predicted_price)
            for row in path.itertuples(index=False)
        ],
        data_sources=data_sources,
    )


def max_supported_forecast_date(
    latest_actual_date: "pd.Timestamp | str",
    config_path: str = "config/config.yaml",
) -> pd.Timestamp:
    """An ESTIMATE of the furthest date forecast_future_price() can currently reach.

    For display only (e.g. a UI hint) -- not a hard cap this module enforces itself:
    the actual limit on any given call depends on how far back the loaded historical
    weather already reaches, which this function does not know (it only knows the
    live weather provider's own forward limit). If historical weather coverage is
    generous, forecast_future_price() can succeed well past this estimate; if the live
    provider is temporarily degraded, it can fail before reaching it. The loop's own
    per-day check is always the authority, never this estimate.

    Computed from MAX_SEASONAL_FORECAST_DAYS (the furthest Open-Meteo endpoint this project
    uses) and the configured feature_engineering.weather_lag_days: a forecast day D needs
    weather at D - weather_lag_days, so the furthest D reachable from a live weather fetch
    is today + (MAX_SEASONAL_FORECAST_DAYS - 1) + weather_lag_days.
    """
    config = ConfigLoader(config_path)
    weather_lag_days = config.get("feature_engineering.weather_lag_days", 14)
    today = pd.Timestamp.now().normalize()
    return today + pd.Timedelta(days=MAX_SEASONAL_FORECAST_DAYS - 1 + weather_lag_days)


def forecast_future_price(
    history_df: pd.DataFrame,
    vegetable: str,
    market: str,
    target_date: "pd.Timestamp | str",
    pipeline: PredictionPipeline,
    config_path: str = "config/config.yaml",
    use_live_weather: bool = True,
    live_fuel_snapshot: LiveFuelPriceSnapshot | None = None,
) -> ForecastResult:
    """Recursively forecast (vegetable, market)'s price on target_date, one day at a time.

    history_df must already be the fully-prepared actual-price history the caller
    wants to forecast from -- e.g. for "Latest HARTI Data", the result of
    backfill_harti_price_gap() + merge_harti_live_price(), exactly as the existing
    single-step prediction flow already builds it. This function does none of that
    preparation itself, and never mutates history_df: every forecast row is appended
    to a local copy (see the module docstring's design note), so the real historical
    dataset this DataFrame originated from is never touched.

    For each day from (latest actual date + 1) through target_date, in order: builds
    that day's inference row with build_live_prediction_input()/build_prediction_input()
    (unchanged -- same weather-gap logic every other prediction path already uses),
    predicts it with the given, already-loaded `pipeline` (never refit, never retrained),
    and folds the prediction back in as a source="FORECAST" row (price and fuel filled
    via finalize_live_row(), weather left NaN like every other target row) before moving
    to the next day. A later day's price_lag/rolling/fuel-lag features therefore see
    this run's own earlier forecast days exactly as they would see real observations --
    without ever being labeled as real ones.

    Args:
        live_fuel_snapshot: CEYPETCO's current fuel prices, if available (see
            app.services.fetch_live_fuel_price()) -- used for every forecast day whose
            date its reading covers (LiveFuelPriceSnapshot.covers()); earlier days (or
            no snapshot at all) fall back to forward-filling history_df's own last
            known value, same as finalize_live_row() everywhere else. This function
            never invents a future fuel-price change.
        use_live_weather: if True (default), missing weather is fetched live from
            Open-Meteo when historical coverage doesn't reach far enough back; if
            False, forecasting fails with ForecastDataUnavailableError the first day
            that would have needed a live fetch, rather than silently proceeding
            without weather.

    Raises:
        ForecastInputError: target_date is unparseable, (vegetable, market) has no
            historical rows, or target_date is not strictly after the latest known
            actual price date (use the existing single-step prediction flow instead).
        ForecastDataUnavailableError: a required day's weather (or any other model
            feature) genuinely can't be obtained -- never fabricated. Names the day
            forecasting reached and the day that failed.
    """
    try:
        target_date = pd.Timestamp(target_date)
    except (ValueError, TypeError) as error:
        raise ForecastInputError(f"forecast_future_price: could not parse target_date {target_date!r}") from error

    group_history = history_df.loc[(history_df["vegetable"] == vegetable) & (history_df["market"] == market)]
    if group_history.empty:
        raise ForecastInputError(
            f"forecast_future_price: no historical data for vegetable={vegetable!r}, market={market!r}"
        )

    # "Actual" means a real observation: known price, and not itself a FORECAST row or
    # a runtime-only IMPUTED_HISTORICAL estimate from a prior imputation pass -- the
    # forecast must always start from real HARTI ground truth, never from an estimate.
    non_real_sources = {FORECAST_SOURCE_LABEL, IMPUTED_HISTORICAL_SOURCE_LABEL}
    is_real_observation = group_history["price"].notna()
    if "source" in group_history.columns:
        is_real_observation &= ~group_history["source"].isin(non_real_sources)
    actual_group_history = group_history.loc[is_real_observation]
    imputed_dates_in_input = sorted(
        group_history.loc[group_history["source"] == IMPUTED_HISTORICAL_SOURCE_LABEL, "date"]
        if "source" in group_history.columns
        else []
    )
    if actual_group_history.empty:
        raise ForecastInputError(
            f"forecast_future_price: no row with a known (non-NaN) price for "
            f"vegetable={vegetable!r}, market={market!r} -- nothing to forecast from"
        )
    latest_actual_date = pd.Timestamp(actual_group_history["date"].max())
    if target_date <= latest_actual_date:
        raise ForecastInputError(
            f"forecast_future_price: target_date {target_date.date()} is not after the latest "
            f"known actual price date {latest_actual_date.date()} for {vegetable}/{market} -- "
            "use the existing single-step prediction flow for a same-day or past date instead."
        )

    config = ConfigLoader(config_path)
    weather_lag_days = config.get("feature_engineering.weather_lag_days", 14)

    forecast_history = history_df.copy()
    daily_forecast: list[DailyForecastPoint] = []
    warnings: list[str] = []
    any_live_weather_used = False
    any_seasonal_weather_used = False
    any_archive_weather_used = False
    weather_cache: dict = {}  # shared across this one vegetable's days, avoiding one request per day

    current_date = latest_actual_date + pd.Timedelta(days=1)
    while current_date <= target_date:
        try:
            if use_live_weather:
                input_df = build_live_prediction_input(
                    forecast_history, current_date, vegetable, market, config_path, weather_cache=weather_cache
                )
            else:
                input_df = build_prediction_input(forecast_history, current_date, vegetable, market)
        except LiveWeatherError as error:
            required_weather_date = current_date - pd.Timedelta(days=weather_lag_days)
            raise ForecastDataUnavailableError(
                f"Cannot forecast {target_date.date()} -- forecasting day {current_date.date()} "
                f"needs the missing feature(s) rainfall_mm_lag{weather_lag_days}d/"
                f"temperature_c_lag{weather_lag_days}d/humidity_pct_lag{weather_lag_days}d, which "
                f"require a weather reading for {required_weather_date.date()} "
                f"({weather_lag_days} day(s) before {current_date.date()}). That reading is not in "
                f"the loaded historical weather and could not be fetched live: {error} "
                f"Forecasting succeeded up to {(current_date - pd.Timedelta(days=1)).date()}."
            ) from error

        if len(input_df) > len(forecast_history) + 1:
            # build_live_prediction_input() always appends [context_row, target_row]
            # in that order when a live fetch was needed -- the second-to-last row is
            # the context row, tagged by which Open-Meteo endpoint actually supplied
            # it (see that function's docstring).
            context_source = str(input_df.iloc[-2]["source"])
            if context_source == "ARCHIVE_WEATHER":
                any_archive_weather_used = True
            elif context_source != LOCAL_WEATHER_SOURCE_LABEL:
                any_live_weather_used = True
            if context_source == "LIVE_WEATHER_SEASONAL":
                any_seasonal_weather_used = True

        try:
            predicted = pipeline.predict(
                input_df, target_date=current_date, vegetable=vegetable, market=market
            )
        except FeatureAlignmentError as error:
            raise ForecastDataUnavailableError(
                f"Cannot forecast {target_date.date()}. {error} Forecasting succeeded up to "
                f"{(current_date - pd.Timedelta(days=1)).date()}."
            ) from error

        predicted_price = float(predicted.loc[0, "predicted_price"])
        daily_forecast.append(DailyForecastPoint(date=current_date, predicted_price=predicted_price))

        forecast_row = pd.DataFrame(
            {
                "date": [current_date],
                "vegetable": [vegetable],
                "market": [market],
                "source": [FORECAST_SOURCE_LABEL],
                "price": [predicted_price],
                "rainfall_mm": [float("nan")],
                "temperature_c": [float("nan")],
                "humidity_pct": [float("nan")],
            }
        )
        forecast_row = finalize_live_row(forecast_history, forecast_row, live_fuel_snapshot)
        forecast_history = pd.concat([forecast_history, forecast_row], ignore_index=True)

        current_date += pd.Timedelta(days=1)

    history_sources = set(group_history["source"].dropna()) if "source" in group_history.columns else set()
    if "SAMPLE" in history_sources:
        data_sources = ["Bundled synthetic sample prices (not real market data)", "Sample weather data"]
    elif history_sources & {"HARTI", "HARTI_LIVE"}:
        data_sources = ["Historical HARTI price data", "Historical weather data"]
    else:
        data_sources = ["Uploaded price data", "Weather from the uploaded data"]
    if "source" in group_history.columns and str(
        group_history.loc[group_history["date"] == latest_actual_date, "source"].iloc[0]
    ) in ("HARTI_LIVE",):
        data_sources.insert(1, "Latest live HARTI bulletin")
    if any_archive_weather_used:
        data_sources.append("Open-Meteo historical weather archive")
    if any_live_weather_used:
        data_sources.append("Open-Meteo weather forecast")
    if any_seasonal_weather_used:
        data_sources.append("Open-Meteo seasonal weather forecast (EC46, beyond 16-day horizon)")
    if live_fuel_snapshot is not None:
        data_sources.append("Current CEYPETCO fuel price")
    else:
        data_sources.append("Historical fuel price (forward-filled)")
        warnings.append(
            "Live CEYPETCO fuel price was not available for this forecast -- used the last "
            "known historical fuel price instead."
        )
    if imputed_dates_in_input:
        data_sources.append("Runtime-imputed historical estimate (non-HARTI)")
        warnings.append(
            "Some historical HARTI dates were unavailable. Temporary estimated historical "
            "values were used only to calculate required model features. Original HARTI "
            f"data was not modified. Imputed date(s): "
            f"{', '.join(d.date().isoformat() for d in imputed_dates_in_input)}."
        )

    return ForecastResult(
        vegetable=vegetable,
        market=market,
        latest_actual_date=latest_actual_date,
        target_date=target_date,
        forecast_horizon_days=(target_date - latest_actual_date).days,
        predicted_price=daily_forecast[-1].predicted_price,
        daily_forecast=daily_forecast,
        data_sources=data_sources,
        warnings=warnings,
        imputed_historical_dates=imputed_dates_in_input,
    )
