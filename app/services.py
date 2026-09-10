"""Streamlit-free logic backing the app: loading sample data, assembling prediction input,
and training a demo model. Kept separate from the page files so it's testable with plain pytest.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.config_loader import ConfigLoader
from src.data_collection.harti_live_client import (
    HartiLiveClient,
    HartiLiveDataError,
    HartiLiveSnapshot,
)
from src.data_collection.live_fuel_price_client import (
    CeypetcoLiveFuelPriceClient,
    LiveFuelPriceError,
    LiveFuelPriceSnapshot,
)
from src.data_collection.live_weather_client import (
    LiveWeatherError,
    OpenMeteoWeatherClient,
)
from src.data_collection.production_region_resolver import (
    ProductionRegionResolver,
    ResolvedWeatherLocation,
)
from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
from src.feature_engineering.production_region_weather import (
    apply_production_region_weather,
)
from src.models.artifact_metadata import (
    ARTIFACT_VERSION,
    ArtifactMetadata,
    new_training_run_id,
    save_artifact_metadata,
)
from src.models.pipeline import ModelTrainingPipeline, ModelTrainingResult
from src.prediction.predictor import InferenceArtifact, save_inference_artifact
from src.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "sample_data" / "sample_history.csv"


def resolve_demo_models_dir(config_path: str = "config/config.yaml") -> str:
    """The directory train_demo_model() persists to -- NEVER config's paths.models_dir.

    Defaults to <paths.models_dir>/demo so it stays isolated even under a minimal
    test config that only sets paths.models_dir, without requiring every config
    (including test fixtures) to separately define paths.demo_models_dir. A real
    deployment can still set paths.demo_models_dir explicitly to move it elsewhere.
    Public (not a train_demo_model() implementation detail) so callers -- notably
    the Predict page -- can load a demo-trained Predictor from the same place
    train_demo_model() just wrote it to, without duplicating this resolution rule.
    """
    config = ConfigLoader(config_path)
    explicit = config.get("paths.demo_models_dir")
    if explicit:
        return explicit
    return str(Path(config.get("paths.models_dir", "models")) / "demo")


def load_sample_history(path: Path | str = SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Load the bundled synthetic demo dataset (source="SAMPLE", not real HARTI/Dambulla data)."""
    return pd.read_csv(path, parse_dates=["date"])


def load_real_market_history(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Load and preprocess the project's real historical HARTI/weather/fuel dataset --
    the same data src/models are trained on -- as the historical context "Latest HARTI
    Data" predictions merge the live bulletin's newest row into.

    Distinct from load_sample_history(): that loads the small bundled SYNTHETIC demo
    dataset. This runs the exact load -> clean -> transform -> merge sequence
    scripts/train.py uses, plus the same production-region weather correction, so the
    result is what a "Historical Dataset" prediction is already working from --
    "Latest HARTI Data" mode only adds one more real row on top of it (see
    merge_harti_live_price()).
    """
    preprocessing = PreprocessingPipeline(config_path=config_path)
    preprocessing.load_data()
    preprocessing.clean_data()
    preprocessing.transform_data()
    merged = preprocessing.merge_data()
    return apply_production_region_weather(merged, config_path=config_path)


def resolve_weather_region(vegetable: str, config_path: str = "config/config.yaml") -> ResolvedWeatherLocation:
    """Resolve which location's weather a vegetable's prediction should use.

    Thin wrapper around ProductionRegionResolver for the UI to call directly (to
    display "Weather region: X") without needing to know the mapping file's path
    or the resolver's constructor.
    """
    return ProductionRegionResolver(config_path=config_path).resolve(vegetable)


def build_prediction_input(
    history_df: pd.DataFrame,
    target_date: pd.Timestamp,
    vegetable: str,
    market: str,
) -> pd.DataFrame:
    """Append one row for target_date with price left NaN -- the row PredictionPipeline will predict.

    Weather columns are also left NaN for this row: PredictionPipeline never
    reads a row's own weather values to compute that row's features (weather
    features are looked up from earlier dates only), so a forecast isn't
    needed to request a prediction.

    If history_df carries diesel_price/petrol_price (a model trained with
    fuel_prices configured needs diesel_price_lag_1d/petrol_price_lag_1d --
    see config.feature_engineering.fuel_price_lag_days), pd.concat's default
    outer join already fills this row's fuel-price columns with NaN, same as
    price/weather above -- no special-casing needed here for that.
    """
    target_row = pd.DataFrame(
        {
            "date": [pd.Timestamp(target_date)],
            "vegetable": [vegetable],
            "market": [market],
            "source": ["USER_REQUEST"],
            "price": [float("nan")],
            "rainfall_mm": [float("nan")],
            "temperature_c": [float("nan")],
            "humidity_pct": [float("nan")],
        }
    )
    return pd.concat([history_df, target_row], ignore_index=True)


def build_live_prediction_input(
    history_df: pd.DataFrame,
    target_date: pd.Timestamp,
    vegetable: str,
    market: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Like build_prediction_input, but backfills a missing weather-lag reading with live data.

    FeatureEngineer's WeatherFeatureGenerator computes rainfall_mm_lag14d/
    temperature_c_lag14d/humidity_pct_lag14d for a row dated D from the
    weather reading at exactly D - weather_lag_days (see weather_features.py's
    docstring and config's feature_engineering.weather_lag_days) -- an
    exact-date lookup, not "the latest known weather" and not "today's
    weather". So the ONE date that matters here is `required_date =
    target_date - weather_lag_days`, regardless of how far target_date itself
    is from history_df's newest row.

    If history_df already has a valid (non-NaN) weather reading for
    required_date, this is identical to build_prediction_input -- no API
    call is made. Otherwise (most commonly: the weather series hasn't caught
    up to required_date yet, e.g. a daily price feed paired with a weather
    source that updates less often) it fetches required_date's reading from
    Open-Meteo and inserts exactly one weather-only context row for it
    (price left NaN, source="LIVE_WEATHER") -- never a row for target_date's
    own (same-day) weather, which build_prediction_input already correctly
    leaves NaN (see its docstring: same-day weather is never read for a
    row's own features, only earlier rows' weather is looked up).

    required_date beyond Open-Meteo's own 16-day forecast reach (e.g. a distant
    recursive forecast day whose weather_lag_days lookback still lands in the
    future): falls back to Open-Meteo's separate Seasonal Forecast API
    (OpenMeteoWeatherClient.fetch_seasonal_daily_weather(), a real, distinct,
    same-provider endpoint reaching up to 46 days ahead), never attempted for a
    required_date in the past. The inserted row is then tagged
    source="LIVE_WEATHER_SEASONAL" instead of "LIVE_WEATHER" -- same schema,
    same treatment everywhere downstream, just distinguishable for reporting
    which endpoint actually supplied it. Still never fabricated: if required_date
    is beyond even this fallback's reach, or Open-Meteo has nothing for it either
    way, LiveWeatherError names every endpoint that was tried.

    The fetch itself is resolved to the vegetable's evidence-based production
    region (config/crop_production_regions.yaml) via resolve_weather_region(),
    not always the market's coordinates -- e.g. a Carrot request fetches
    Nuwara Eliya's weather here, not Dambulla's. A vegetable with no
    sufficiently evidenced region falls back to the market's own configured
    coordinates (weather_api section), identical to this function's original
    behavior.

    The context row uses the same (vegetable, market) as the request because it
    matters, not just for a readable label: WeatherFeatureGenerator groups its
    weather-lag lookup by vegetable (see feature_engineer.py), so this row's
    weather is only ever picked up by this same vegetable's future lag features,
    never blended into another vegetable's. A row at a date no other row shares
    also cannot change any other row's own price-lag/rolling features (those are
    grouped by (vegetable, market) with a strictly backward-looking, exact-date
    join).

    Callers must pass this same target_date/vegetable/market into
    PredictionPipeline.predict() so only the real target row is scored -- the
    inserted context row has an unknown price too and must not be treated as
    a row to predict.

    Raises:
        LiveWeatherError: if weather_api.enabled is false in config while
            required_date's weather is genuinely missing, or if Open-Meteo
            cannot be reached, errors, or doesn't cover required_date. Never
            silently proceeds with a fabricated or mislabeled substitute.
    """
    target_date = pd.Timestamp(target_date)
    config = ConfigLoader(config_path)
    weather_lag_days = config.get("feature_engineering.weather_lag_days", 14)
    required_date = target_date - pd.Timedelta(days=weather_lag_days)

    weather_columns = ["rainfall_mm", "temperature_c", "humidity_pct"]
    existing = history_df.loc[history_df["date"] == required_date, weather_columns]
    already_covered = not existing.empty and bool(existing.notna().all(axis=1).any())

    if already_covered:
        return build_prediction_input(history_df, target_date, vegetable, market)

    if not config.get("weather_api.enabled", True):
        raise LiveWeatherError(
            f"Live weather is disabled (weather_api.enabled: false) but no weather reading "
            f"exists for {required_date.date()}, needed for {target_date.date()}'s "
            f"{weather_lag_days}-day weather lag feature. Enable weather_api.enabled or supply "
            f"weather data for that date some other way."
        )

    today = pd.Timestamp.now().normalize()
    past_days = max(0, (today - required_date).days)
    forecast_days = max(1, (required_date - today).days + 1)

    resolved_location = resolve_weather_region(vegetable, config_path=config_path)
    client = OpenMeteoWeatherClient.for_coordinates(
        resolved_location.latitude, resolved_location.longitude, config_path=config_path
    )
    live_weather = client.fetch_recent_daily_weather(past_days=past_days, forecast_days=forecast_days)
    match = live_weather.loc[live_weather["date"] == required_date]
    context_source = "LIVE_WEATHER"

    attempted_sources = [f"Open-Meteo forecast (past_days={past_days}, forecast_days={forecast_days})"]
    if match.empty and required_date > today:
        # required_date is beyond fetch_recent_daily_weather()'s own MAX_FORECAST_DAYS
        # (16 days) -- fall back to Open-Meteo's separate Seasonal Forecast API
        # (ECMWF EC46 sub-seasonal model), a real, distinct, same-provider endpoint
        # reaching up to MAX_SEASONAL_FORECAST_DAYS (46) days ahead -- see
        # fetch_seasonal_daily_weather()'s docstring. Never attempted for a
        # required_date in the past: a seasonal forecast is only meaningful going
        # forward from today, unlike a genuinely missing historical reading.
        seasonal_forecast_days = (required_date - today).days + 1
        seasonal_weather = client.fetch_seasonal_daily_weather(forecast_days=seasonal_forecast_days)
        match = seasonal_weather.loc[seasonal_weather["date"] == required_date]
        context_source = "LIVE_WEATHER_SEASONAL"
        attempted_sources.append(f"Open-Meteo seasonal forecast (forecast_days={seasonal_forecast_days})")

    if match.empty:
        raise LiveWeatherError(
            f"None of Open-Meteo's endpoints ({'; '.join(attempted_sources)}) returned a weather "
            f"reading for {required_date.date()}, needed for {target_date.date()}'s "
            f"{weather_lag_days}-day weather lag feature."
        )
    reading = match.iloc[0]

    context_row = pd.DataFrame(
        {
            "date": [required_date],
            "vegetable": [vegetable],
            "market": [market],
            "source": [context_source],
            "price": [float("nan")],
            "rainfall_mm": [reading["rainfall_mm"]],
            "temperature_c": [reading["temperature_c"]],
            "humidity_pct": [reading["humidity_pct"]],
        }
    )

    extended_history = pd.concat([history_df, context_row], ignore_index=True)
    return build_prediction_input(extended_history, target_date, vegetable, market)


def fetch_harti_latest_snapshot(config_path: str = "config/config.yaml") -> HartiLiveSnapshot:
    """Fetch the most recent HARTI daily bulletin's prices for config's harti.target_market.

    Thin wrapper (Streamlit-free, independently testable) around HartiLiveClient,
    mirroring resolve_weather_region()'s role for OpenMeteoWeatherClient.

    Raises:
        HartiLiveDataError: if harti.latest_data_enabled is false, or the live
            fetch/parse fails for any reason (network, timeout, changed bulletin
            structure, no data for any tracked vegetable). This is HARTI's own latest
            published bulletin, not a real-time feed -- see HartiLiveSnapshot.bulletin_date.
    """
    config = ConfigLoader(config_path)
    if not config.get("harti.latest_data_enabled", True):
        raise HartiLiveDataError(
            "Latest HARTI Data is disabled (harti.latest_data_enabled: false in config.yaml)"
        )
    return HartiLiveClient.from_config(config_path).fetch_latest_prices()


def finalize_live_row(
    history_df: pd.DataFrame,
    row: pd.DataFrame,
    live_fuel_snapshot: LiveFuelPriceSnapshot | None = None,
) -> pd.DataFrame:
    """Fill a single real HARTI-sourced (or forecast) row's weather/fuel columns before
    it's appended.

    Shared by merge_harti_live_price(), backfill_harti_price_gap(), and
    app/forecasting.py's recursive forecast loop -- all append a single
    (vegetable, market, date, price) row and need the exact same weather-NaN and fuel
    treatment, whether that price is a real HARTI observation or a model prediction
    being folded into the forecast history for the next recursive step.

    Fuel price: prefers live_fuel_snapshot's CEYPETCO reading for a column when it
    actually covers this row's date (see LiveFuelPriceSnapshot.covers() -- a reading is
    only valid on or after its own effective date), since that is a freshly verified,
    currently-in-effect price rather than an assumption. Otherwise (no snapshot, the
    column isn't in it, or this row's date predates the reading's effective date --
    e.g. a backfilled gap day before the latest revision) falls back to forward-filling
    history_df's own last known value -- the same domain rule DataCleaner already
    applies elsewhere for fuel-price gaps (a step-valued administrative price, genuinely
    constant between CEYPETCO revisions -- carrying the last known price forward is the
    correct representation, not an approximation) rather than inventing a new one.
    Never applied to price or weather: those are looked up by lag features at an exact
    earlier date, never required at this row's own date.
    """
    row = row.copy()
    for column in ("rainfall_mm", "temperature_c", "humidity_pct"):
        if column not in row.columns:
            row[column] = float("nan")

    row_date = pd.Timestamp(row["date"].iloc[0])
    for column in ("diesel_price", "petrol_price"):
        if column not in history_df.columns:
            continue
        if live_fuel_snapshot is not None and live_fuel_snapshot.covers(column, row_date):
            row[column] = live_fuel_snapshot.prices[column]
            continue
        last_known = history_df[column].dropna()
        row[column] = last_known.iloc[-1] if not last_known.empty else float("nan")

    return row


def fetch_live_fuel_price(config_path: str = "config/config.yaml") -> LiveFuelPriceSnapshot:
    """Fetch CEYPETCO's current, currently-in-effect retail fuel prices.

    Thin wrapper (Streamlit-free, independently testable) around
    CeypetcoLiveFuelPriceClient, mirroring fetch_harti_latest_snapshot()'s /
    resolve_weather_region()'s role for their own live clients.

    Raises:
        LiveFuelPriceError: if fuel_api.enabled is false, or the live fetch/parse
            fails for any reason (network, timeout, changed page structure, a
            configured fuel-price column with no matching product on the page).
    """
    config = ConfigLoader(config_path)
    if not config.get("fuel_api.enabled", True):
        raise LiveFuelPriceError(
            "Live fuel price is disabled (fuel_api.enabled: false in config.yaml)"
        )
    return CeypetcoLiveFuelPriceClient.from_config(config_path).fetch_latest_prices()


def merge_harti_live_price(
    history_df: pd.DataFrame,
    live_snapshot: HartiLiveSnapshot,
    vegetable: str,
    market: str,
    live_fuel_snapshot: LiveFuelPriceSnapshot | None = None,
) -> pd.DataFrame:
    """Append the live bulletin's newest observed (vegetable, market) price onto history_df.

    Returns history_df with exactly one additional real row (source="HARTI_LIVE") --
    NOT yet a to-predict target row. Callers compose this with build_prediction_input()/
    build_live_prediction_input() exactly as they already do with any other history_df:
    e.g. group_history["date"].max() naturally becomes this new row's date afterward,
    so the existing "predict the day after the latest known date" UI logic needs no
    changes to pick it up, and the existing use_live_weather branch needs no changes
    either -- it will simply look for this row's weather instead of an older one.

    Forward-fills a fuel-price gap for the live row using history_df's own last known
    diesel/petrol price: fuel_price_lag_days (config, default 1) means a model trained
    with fuel-price features needs a fuel reading for the live row's own date, which
    the historical fuel-price CSV won't have reached if it wasn't refreshed as recently
    as HARTI's own bulletin. This reuses DataCleaner's own documented rule for fuel
    price elsewhere (a step-valued administrative price, genuinely constant between
    CEYPETCO revisions -- carrying the last known price forward is the correct
    representation, not an approximation) rather than inventing a new one. Never
    applied to price or weather: those are looked up by lag features at an exact
    earlier date, never required at the live row's own date.

    Callers predicting the day after this row should call
    backfill_harti_price_gap() first if the historical CSV has gone stale relative to
    this bulletin -- otherwise a multi-day gap between history_df's last known date and
    live_snapshot.bulletin_date starves price_lag_7d/rolling_std_7d of real observations
    (see that function's docstring).

    live_fuel_snapshot: CEYPETCO's current fuel prices (fetch_live_fuel_price()), if
    available -- used in place of the forward-fill above for a fuel column whose live
    reading actually covers this row's date (see finalize_live_row()). Omit (the
    default) to always forward-fill, e.g. when live fuel price couldn't be fetched.

    Raises:
        HartiLiveDataError: if live_snapshot has no row for (vegetable, market) --
            e.g. that vegetable had no reading ("-") in this particular bulletin.
    """
    live_rows = live_snapshot.data[
        (live_snapshot.data["vegetable"] == vegetable) & (live_snapshot.data["market"] == market)
    ]
    if live_rows.empty:
        available = sorted(
            live_snapshot.data.loc[live_snapshot.data["market"] == market, "vegetable"].unique()
        )
        raise HartiLiveDataError(
            f"HARTI's latest bulletin ({live_snapshot.bulletin_date.date()}) has no '{market}' "
            f"price for '{vegetable}'. Vegetables it did report a '{market}' price for: {available}"
        )

    live_row = finalize_live_row(history_df, live_rows.iloc[[0]], live_fuel_snapshot)
    return pd.concat([history_df, live_row], ignore_index=True)


def backfill_harti_price_gap(
    history_df: pd.DataFrame,
    live_snapshot: HartiLiveSnapshot,
    vegetable: str,
    market: str,
    config_path: str = "config/config.yaml",
    live_fuel_snapshot: LiveFuelPriceSnapshot | None = None,
) -> pd.DataFrame:
    """Fill (vegetable, market)'s real HARTI bulletins for any calendar day between
    history_df's last known date for that group and live_snapshot.bulletin_date.

    Why this exists: "Latest HARTI Data" merges in exactly one new row (the live
    bulletin's own date) via merge_harti_live_price(). As long as the historical CSV
    (data/raw/harti_prices.csv) is refreshed at least as often as HARTI publishes (1-2
    days), that's a one-day gap at most and every lag/rolling feature has the dense
    daily history it needs. If the CSV has instead gone stale for longer than that, the
    gap between its last real observation and the live bulletin's date can exceed a
    lag/rolling window (e.g. 7 days): price_lag_7d then has no exact match 7 days back,
    and price_rolling_std_7d's trailing window contains at most the single new live
    point -- too few observations for a defined standard deviation. Predictor correctly
    refuses to fabricate either (see predictor.py's FeatureAlignmentError).

    The fix is to close the gap with more real data, not fabricate it: HARTI's own
    archive (src/data_collection/harti_live_client.py) publishes one bulletin per
    calendar day going back to 2015, and HartiLiveClient.fetch_prices_for_date() can
    already fetch any specific past day's bulletin -- this was implemented and tested
    at the client level but never wired into the app's "Latest HARTI Data" flow (see
    README.md section 10's former "known limitation"). This function is that wiring: it
    fetches each gap day's real bulletin and keeps this (vegetable, market)'s reading
    from it, if HARTI published one and it had a reading for this vegetable that day.

    A day HARTI didn't publish a bulletin for, or that bulletin had no reading for this
    exact vegetable ("-" in the PDF), is silently skipped (logged) -- never fabricated.
    In that case the corresponding lag/rolling feature legitimately stays NaN and
    Predictor still raises FeatureAlignmentError exactly as before; this function can
    only add real data it actually finds, never guarantee the gap closes completely.

    Only backfills the most recent `max(feature_engineering.lag_periods_days +
    rolling_windows_days)` days before the live bulletin's date (default: 30) --
    fetching further back would reach bulletins no configured lag/rolling window can
    use anyway. Returns history_df unchanged if there is no gap (the group's last known
    date is already the day before the live bulletin, or later).

    live_fuel_snapshot: forwarded to finalize_live_row() for each backfilled row --
    see merge_harti_live_price()'s docstring.
    """
    group_history = history_df.loc[
        (history_df["vegetable"] == vegetable) & (history_df["market"] == market)
    ]
    if group_history.empty:
        return history_df

    config = ConfigLoader(config_path)
    lag_periods_days = config.get("feature_engineering.lag_periods_days", [1, 7, 14, 30])
    rolling_windows_days = config.get("feature_engineering.rolling_windows_days", [7, 14, 30])
    max_window_days = max([*lag_periods_days, *rolling_windows_days], default=30)

    last_known_date = group_history["date"].max()
    gap_start = max(
        last_known_date + pd.Timedelta(days=1),
        live_snapshot.bulletin_date - pd.Timedelta(days=max_window_days),
    )
    gap_end = live_snapshot.bulletin_date - pd.Timedelta(days=1)
    if gap_start > gap_end:
        return history_df

    client = HartiLiveClient.from_config(config_path)
    filled_rows = [
        row
        for row in (
            _fetch_and_finalize_harti_day(client, day, vegetable, market, history_df, live_fuel_snapshot)
            for day in pd.date_range(gap_start, gap_end, freq="D")
        )
        if row is not None
    ]

    if not filled_rows:
        return history_df

    logger.info(
        "backfill_harti_price_gap: filled %d real HARTI bulletin(s) for '%s'/'%s' between "
        "%s and %s",
        len(filled_rows),
        vegetable,
        market,
        gap_start.date(),
        gap_end.date(),
    )
    return pd.concat([history_df, *filled_rows], ignore_index=True)


def _fetch_and_finalize_harti_day(
    client: HartiLiveClient,
    day: pd.Timestamp,
    vegetable: str,
    market: str,
    history_df: pd.DataFrame,
    live_fuel_snapshot: LiveFuelPriceSnapshot | None,
) -> pd.DataFrame | None:
    """Fetch one real HARTI bulletin for `day` and return its finalized (vegetable,
    market) row, or None if HARTI never published one that day or it had no reading
    for this vegetable -- never fabricated, in either case. Shared by
    backfill_harti_price_gap() (a contiguous forward gap) and
    backfill_internal_harti_gaps() (any isolated hole in the trailing lookback window).
    """
    try:
        day_snapshot = client.fetch_prices_for_date(day)
    except HartiLiveDataError as error:
        logger.info(
            "no usable HARTI bulletin for %s (%s) -- skipped, not fabricated", day.date(), error
        )
        return None

    day_rows = day_snapshot.data[
        (day_snapshot.data["vegetable"] == vegetable) & (day_snapshot.data["market"] == market)
    ]
    if day_rows.empty:
        logger.info(
            "%s's bulletin had no '%s' price for '%s' -- skipped, not fabricated",
            day.date(),
            market,
            vegetable,
        )
        return None

    return finalize_live_row(history_df, day_rows.iloc[[0]], live_fuel_snapshot)


def backfill_internal_harti_gaps(
    history_df: pd.DataFrame,
    vegetable: str,
    market: str,
    config_path: str = "config/config.yaml",
    live_fuel_snapshot: LiveFuelPriceSnapshot | None = None,
) -> pd.DataFrame:
    """Fill any ISOLATED missing calendar day within (vegetable, market)'s own recent
    price history -- not the contiguous "recent" gap backfill_harti_price_gap() closes
    (between the historical CSV's last date and a live bulletin's date), but any single
    day *within* the already-loaded history that HARTI simply has no row for, anywhere
    in the trailing max(lag_periods_days + rolling_windows_days) days up to and
    including the group's own latest known date.

    Why this exists: LagFeatureGenerator's exact-date join (see lag_features.py) means
    price_lag_30d for some future forecast day D is only populated when a row for
    EXACTLY D-30 exists. Recursive forecasting (app/forecasting.py) starts from the
    group's latest known date and only ever adds NEW rows going forward -- it never
    revisits the trailing history behind that starting point. A single isolated hole
    back there (e.g. a day HARTI's own bulletin never covered this market/vegetable,
    even though the CSV otherwise reaches back plenty far) silently starves whichever
    future day's lag feature needs exactly that date, with nothing in the recursive
    loop itself able to fix it -- that date is in the past relative to every forecast
    day, never a date recursion produces a row for.

    This fills any such isolated hole with a REAL bulletin fetched from HARTI's own
    archive (HartiLiveClient.fetch_prices_for_date(), which already serves any
    specific past date, not just the most recent one) -- never fabricated. A day HARTI
    genuinely never published for this (vegetable, market) stays missing (verified by
    fetching, not assumed): the corresponding lag/rolling feature legitimately remains
    NaN, and Predictor/forecast_future_price still raise their normal, undisguised
    errors for whichever specific forecast day needed it -- this function can only add
    real data it actually finds, never guarantee every lookback day is covered.

    Returns history_df unchanged if the group has no historical rows, or if every day
    in the trailing window already has one.
    """
    group_history = history_df.loc[
        (history_df["vegetable"] == vegetable) & (history_df["market"] == market) & history_df["price"].notna()
    ]
    if group_history.empty:
        return history_df

    config = ConfigLoader(config_path)
    lag_periods_days = config.get("feature_engineering.lag_periods_days", [1, 7, 14, 30])
    rolling_windows_days = config.get("feature_engineering.rolling_windows_days", [7, 14, 30])
    max_window_days = max([*lag_periods_days, *rolling_windows_days], default=30)

    last_known_date = group_history["date"].max()
    window_start = last_known_date - pd.Timedelta(days=max_window_days - 1)
    existing_dates = set(group_history["date"])  # dates with a REAL price -- a NaN-price
    # row (e.g. a weather-only context row) does not count as "having" this day's price
    missing_days = [
        day
        for day in pd.date_range(window_start, last_known_date, freq="D")
        if day not in existing_dates
    ]
    if not missing_days:
        return history_df

    client = HartiLiveClient.from_config(config_path)
    filled_rows = [
        row
        for row in (
            _fetch_and_finalize_harti_day(client, day, vegetable, market, history_df, live_fuel_snapshot)
            for day in missing_days
        )
        if row is not None
    ]

    if not filled_rows:
        return history_df

    logger.info(
        "backfill_internal_harti_gaps: filled %d isolated real HARTI bulletin(s) for '%s'/'%s' "
        "within the trailing %d-day window ending %s",
        len(filled_rows),
        vegetable,
        market,
        max_window_days,
        last_known_date.date(),
    )
    return pd.concat([history_df, *filled_rows], ignore_index=True)


IMPUTED_HISTORICAL_SOURCE_LABEL = "IMPUTED_HISTORICAL"


def impute_missing_historical_prices(
    history_df: pd.DataFrame,
    vegetable: str,
    market: str,
    config_path: str = "config/config.yaml",
    live_fuel_snapshot: LiveFuelPriceSnapshot | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Last-resort fill for a calendar day within (vegetable, market)'s trailing
    lag/rolling lookback window that HARTI genuinely has no bulletin for anywhere --
    verified by calling this only AFTER backfill_internal_harti_gaps() has already
    tried a real archive fetch and failed to find one. This is priority 2 of 2: real
    HARTI data (backfill_internal_harti_gaps()) always wins when it exists; this
    function only ever fills a date that is STILL missing from history_df.

    Every row this produces is tagged source="IMPUTED_HISTORICAL"
    (IMPUTED_HISTORICAL_SOURCE_LABEL) -- never "HARTI_LIVE" or any other real-source
    label, so it can never be mistaken for an actual observation by anything that
    reads this DataFrame later (a chart, a data_sources list, another audit). It exists
    only in the DataFrame this function returns; nothing here writes to
    data/raw/harti_prices.csv or any other file, and no existing real row is ever
    modified -- this only ever ADDS a row for a date with no row at all yet.

    Deterministic estimation strategy, using only REAL surrounding price observations
    for this (vegetable, market) -- never a later date's forecast, and never another
    imputed date, since this runs once over history_df before any forecast row exists:
      - a real price both before AND after the gap -> linear interpolation between them.
      - only a real price before the gap (within this group's own history) -> that
        price carried forward (flat) -- there is no second real anchor to slope toward.
      - only a real price after the gap -> that price carried backward (flat), same
        reasoning in reverse.
      - no real price on either side (the group has no real data at all near this
        window) -> left unfilled; FeatureAlignmentError is still the correct outcome
        for a request with no real anchor whatsoever to estimate from.

    Returns:
        (updated_history_df, imputed_dates) -- imputed_dates lists every date this
        call actually filled (empty if none were needed), so a caller can warn about
        it explicitly rather than the estimate passing silently as if it were real.
    """
    group_history = history_df.loc[
        (history_df["vegetable"] == vegetable) & (history_df["market"] == market) & history_df["price"].notna()
    ]
    if group_history.empty:
        return history_df, []

    config = ConfigLoader(config_path)
    lag_periods_days = config.get("feature_engineering.lag_periods_days", [1, 7, 14, 30])
    rolling_windows_days = config.get("feature_engineering.rolling_windows_days", [7, 14, 30])
    max_window_days = max([*lag_periods_days, *rolling_windows_days], default=30)

    last_known_date = group_history["date"].max()
    window_start = last_known_date - pd.Timedelta(days=max_window_days - 1)
    existing_dates = set(group_history["date"])  # dates with a REAL price already
    window_missing_dates = [
        day
        for day in pd.date_range(window_start, last_known_date, freq="D")
        if day not in existing_dates
    ]
    if not window_missing_dates:
        return history_df, []

    # Interpolate/edge-fill over the group's ENTIRE real price history (not just the
    # trailing window) so a gap near the window's own edge still has every real
    # anchor available to it -- only the results actually inside the window are used.
    real_prices = (
        group_history.dropna(subset=["price"])
        .drop_duplicates(subset="date")
        .set_index("date")["price"]
        .sort_index()
    )
    if real_prices.empty:
        return history_df, []

    full_range = pd.date_range(min(window_start, real_prices.index.min()), last_known_date, freq="D")
    dense = real_prices.reindex(full_range)
    estimated = dense.interpolate(method="linear", limit_area="inside").ffill().bfill()

    imputed_dates = [day for day in window_missing_dates if pd.notna(estimated.get(day))]
    if not imputed_dates:
        return history_df, []

    imputed_rows = [
        finalize_live_row(
            history_df,
            pd.DataFrame(
                {
                    "date": [day],
                    "vegetable": [vegetable],
                    "market": [market],
                    "source": [IMPUTED_HISTORICAL_SOURCE_LABEL],
                    "price": [float(estimated[day])],
                    "rainfall_mm": [float("nan")],
                    "temperature_c": [float("nan")],
                    "humidity_pct": [float("nan")],
                }
            ),
            live_fuel_snapshot,
        )
        for day in imputed_dates
    ]

    logger.warning(
        "impute_missing_historical_prices: %s/%s has no real HARTI observation for %d "
        "date(s) anywhere in the archive -- filled with a deterministic runtime-only "
        "estimate (source='%s', never written to disk, never treated as real) for "
        "feature computation only: %s",
        vegetable,
        market,
        len(imputed_dates),
        IMPUTED_HISTORICAL_SOURCE_LABEL,
        [day.date().isoformat() for day in imputed_dates],
    )
    return pd.concat([history_df, *imputed_rows], ignore_index=True), imputed_dates


def train_demo_model(
    history_df: pd.DataFrame,
    config_path: str,
    output_dir: str | None = None,
) -> ModelTrainingResult:
    """Train every configured model on history_df and persist the best one + its inference artifact.

    This is the one place in the whole project that actually wires
    TrainTestDatasetBuilder -> ModelTrainingPipeline -> save_inference_artifact
    together end to end outside of a test, reachable from the Streamlit Predict
    page for both the bundled synthetic sample data and any user-uploaded CSV.

    Persists to output_dir (or, when omitted, config's paths.demo_models_dir /
    "<paths.models_dir>/demo") -- deliberately NEVER config's paths.models_dir
    itself, regardless of what history_df actually contains. This function has
    no way to tell "the real production dataset, reloaded" apart from "the
    synthetic sample, or a user's arbitrary upload" -- history_df is just a
    DataFrame -- so it must not be trusted to decide when overwriting the real
    best_model.joblib is safe. It once did (writing straight to
    paths.models_dir), and a demo/sandbox retrain from this page silently
    clobbered the real trained model as a result (see models/backups/
    *_SYNTHETIC_DEMO_OVERWRITE_backup_*.joblib). The production model is only
    ever produced by scripts/train.py, run explicitly and separately.
    """
    resolved_output_dir = output_dir if output_dir is not None else resolve_demo_models_dir(config_path)

    builder = TrainTestDatasetBuilder(config_path=config_path)
    X_train, X_test, y_train, y_test = builder.build(history_df)

    result = ModelTrainingPipeline(config_path=config_path).run(
        X_train, X_test, y_train, y_test, output_dir=resolved_output_dir
    )

    training_run_id = new_training_run_id()
    save_inference_artifact(
        InferenceArtifact(builder.fitted_encoder, list(X_train.columns), training_run_id=training_run_id),
        output_dir=resolved_output_dir,
        config_path=config_path,
    )
    save_artifact_metadata(
        ArtifactMetadata(
            artifact_version=ARTIFACT_VERSION,
            training_run_id=training_run_id,
            trained_at=datetime.now().isoformat(timespec="seconds"),
            model_type=type(result.best_model).__name__,
            model_name=result.best_model_name,
            training_data={
                "source": "app-provided demo/sandbox data (bundled synthetic sample or a "
                "user upload) -- NOT the real production HARTI dataset",
                "row_count": len(history_df),
            },
            feature_count=X_train.shape[1],
            feature_columns=list(X_train.columns),
            metrics=result.metrics[result.best_model_name],
        ),
        output_dir=resolved_output_dir,
        config_path=config_path,
    )
    return result
