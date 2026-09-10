"""Streamlit page: train a demo model if needed, then request a price prediction."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.forecasting import (
    ForecastDataUnavailableError,
    ForecastInputError,
    forecast_future_price,
    max_supported_forecast_date,
)
from app.services import (
    IMPUTED_HISTORICAL_SOURCE_LABEL,
    backfill_harti_price_gap,
    backfill_internal_harti_gaps,
    build_live_prediction_input,
    build_prediction_input,
    fetch_harti_latest_snapshot,
    fetch_live_fuel_price,
    impute_missing_historical_prices,
    load_real_market_history,
    load_sample_history,
    merge_harti_live_price,
    resolve_demo_models_dir,
    resolve_weather_region,
    train_demo_model,
)
from config.config_loader import ConfigLoader
from src.data_collection.harti_live_client import HartiLiveDataError, HartiLiveSnapshot
from src.data_collection.live_fuel_price_client import LiveFuelPriceError, LiveFuelPriceSnapshot
from src.data_collection.live_weather_client import LiveWeatherError
from src.feature_engineering.feature_engineer import FeatureEngineeringError
from src.feature_engineering.production_region_weather import (
    apply_production_region_weather,
)
from src.models.artifact_metadata import ArtifactConsistencyError
from src.prediction.pipeline import PredictionInputError, PredictionPipeline
from src.prediction.predictor import (
    FeatureAlignmentError,
    ModelArtifactNotFoundError,
    Predictor,
    TransformationArtifactNotFoundError,
)

CONFIG_PATH = "config/config.yaml"
REQUIRED_COLUMNS = {"date", "vegetable", "market", "price"}
_HARTI_CACHE_TTL_SECONDS = ConfigLoader(CONFIG_PATH).get("harti.cache_minutes", 60) * 60
_HARTI_LATEST_DATA_ENABLED = ConfigLoader(CONFIG_PATH).get("harti.latest_data_enabled", True)
_FUEL_CACHE_TTL_SECONDS = ConfigLoader(CONFIG_PATH).get("fuel_api.cache_minutes", 60) * 60
_WEATHER_LAG_DAYS = ConfigLoader(CONFIG_PATH).get("feature_engineering.weather_lag_days", 14)

st.set_page_config(page_title="Predict | Vegetable Price Prediction", page_icon="📈")
st.title("📈 Predict a vegetable price")


def _render_weather_data_card(
    vegetable: str,
    market: str,
    resolved_location,
    weather_date_label: str,
    weather_values: dict[str, float] | None,
) -> None:
    """Renders the 'Weather Data Used' transparency card: which location's weather fed
    this prediction, its coordinates, and (when available) the actual values used --
    never fabricated, never silently swapped in for the market's location."""
    with st.container(border=True):
        st.markdown("**🌦️ Weather Data Used**")
        if resolved_location.is_production_region:
            location_label = resolved_location.region_name
            location_note = "evidence-based cultivation/production region for this vegetable"
        else:
            location_label = f"{market} (market location)"
            location_note = (
                f"⚠ fallback -- no sufficiently evidenced cultivation-region mapping exists for "
                f"'{vegetable}' in config/crop_production_regions.yaml; this is the project's "
                f"documented fallback, not a guess at {vegetable}'s actual cultivation location"
            )
        st.markdown(
            f"- **Vegetable:** {vegetable}\n"
            f"- **Weather Location:** {location_label}\n"
            f"- **Coordinates:** {resolved_location.latitude:.4f}, {resolved_location.longitude:.4f}\n"
            f"- **Weather Source:** Open-Meteo\n"
            f"- **Weather Date:** {weather_date_label}\n"
            f"- **Data Used For:** Price Prediction (as {_WEATHER_LAG_DAYS}-day-lagged features)"
        )
        st.caption(location_note)
        if weather_values:
            cols = st.columns(len(weather_values))
            for col, (label, value) in zip(cols, weather_values.items()):
                col.metric(label, value)
        st.caption(
            "Weather features used by this model: rainfall, temperature, humidity (14-day lag). "
            "Wind speed is not part of this project's weather schema/model -- not shown, not fabricated."
        )


@st.cache_resource
def _load_predictor(config_path: str) -> Predictor:
    return Predictor.load(config_path=config_path)


@st.cache_resource
def _load_demo_predictor(config_path: str, models_dir: str) -> Predictor:
    """Loads a Predictor from the sandbox/demo directory train_demo_model() writes to --
    kept as a separate cache entry (and a separate on-disk directory) from
    _load_predictor()'s production model so training a sandbox model can never
    shadow or be confused with the real one."""
    return Predictor.load(config_path=config_path, models_dir=models_dir)


@st.cache_data
def _load_sample_history_cached() -> pd.DataFrame:
    return load_sample_history()


@st.cache_data
def _load_real_market_history_cached() -> pd.DataFrame:
    return load_real_market_history(CONFIG_PATH)


@st.cache_data(ttl=_HARTI_CACHE_TTL_SECONDS)
def _fetch_harti_snapshot_cached() -> HartiLiveSnapshot:
    return fetch_harti_latest_snapshot(CONFIG_PATH)


@st.cache_data(ttl=_FUEL_CACHE_TTL_SECONDS)
def _fetch_live_fuel_price_cached() -> LiveFuelPriceSnapshot:
    return fetch_live_fuel_price(CONFIG_PATH)


# ---------------------------------------------------------------------------
# 1. Historical data source
# ---------------------------------------------------------------------------

st.header("1. Historical data")
_data_source_options = ["Bundled sample data (synthetic)", "Upload my own CSV"]
if _HARTI_LATEST_DATA_ENABLED:
    _data_source_options.append("Latest HARTI Data (real)")
data_source = st.radio("Data source", _data_source_options, horizontal=True)

history_df = None
live_snapshot: HartiLiveSnapshot | None = None
if data_source == "Bundled sample data (synthetic)":
    history_df = _load_sample_history_cached()
    st.caption(
        f"Loaded `app/sample_data/sample_history.csv` -- {len(history_df)} synthetic row(s), "
        f"{history_df['date'].min().date()} to {history_df['date'].max().date()}. Not real market data."
    )
elif data_source == "Upload my own CSV":
    uploaded_file = st.file_uploader(
        "CSV with columns: date, vegetable, market, price, rainfall_mm, temperature_c, humidity_pct "
        "(also include diesel_price, petrol_price if the loaded model was trained with fuel-price "
        "data -- otherwise a prediction needing those lag features will fail with a clear error)",
        type="csv",
    )
    if uploaded_file is not None:
        try:
            history_df = pd.read_csv(uploaded_file, parse_dates=["date"])
        except Exception as error:  # noqa: BLE001 -- surfaced directly to the user, not swallowed
            st.error(f"Could not read that CSV: {error}")
            st.stop()
else:  # "Latest HARTI Data (real)"
    try:
        history_df = _load_real_market_history_cached()
        live_snapshot = _fetch_harti_snapshot_cached()
    except HartiLiveDataError as error:
        st.error(f"Couldn't fetch the latest HARTI bulletin: {error}")
        st.stop()

    refresh_col, info_col = st.columns([1, 5])
    with refresh_col:
        if st.button("Refresh"):
            _fetch_harti_snapshot_cached.clear()
            st.rerun()
    with info_col:
        st.caption(
            f"Historical context: real HARTI/Dambulla dataset (`data/raw/harti_prices.csv`, "
            f"{len(history_df)} rows, {history_df['date'].min().date()} to "
            f"{history_df['date'].max().date()}). **Latest HARTI bulletin: "
            f"{live_snapshot.bulletin_date.date()}** ({len(live_snapshot.data)} vegetable prices, "
            f"fetched {live_snapshot.retrieved_at.strftime('%Y-%m-%d %H:%M')}). This is HARTI's own "
            f"most recently *published* daily bulletin, not a real-time feed -- HARTI typically "
            f"publishes 1-2 days behind today, and not every market reports every day."
        )

if history_df is None:
    st.info("Upload a CSV to continue.")
    st.stop()

missing_columns = REQUIRED_COLUMNS - set(history_df.columns)
if missing_columns:
    st.error(f"Uploaded data is missing required column(s): {sorted(missing_columns)}")
    st.stop()

# Corrects each row's same-day weather to its vegetable's evidence-based production
# region (config/crop_production_regions.yaml) where one is mapped -- e.g. Carrot
# rows get Nuwara Eliya's weather instead of the market's. A vegetable with no
# sufficiently evidenced region is left unchanged (market-location weather).
history_df = apply_production_region_weather(history_df, config_path=CONFIG_PATH)

with st.expander("Preview loaded data"):
    st.dataframe(history_df.head(20))

# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

st.header("2. Model")

demo_models_dir = resolve_demo_models_dir(CONFIG_PATH)

try:
    production_predictor = _load_predictor(CONFIG_PATH)
except (ModelArtifactNotFoundError, TransformationArtifactNotFoundError):
    production_predictor = None
except ArtifactConsistencyError as error:
    st.error(f"Production model artifacts under `models/` are inconsistent and cannot be used: {error}")
    production_predictor = None

try:
    demo_predictor = _load_demo_predictor(CONFIG_PATH, demo_models_dir)
except (ModelArtifactNotFoundError, TransformationArtifactNotFoundError):
    demo_predictor = None
except ArtifactConsistencyError as error:
    st.error(f"Sandbox model artifacts under `{demo_models_dir}` are inconsistent and cannot be used: {error}")
    demo_predictor = None


def _train_sandbox_model(button_label: str) -> None:
    """Trains on the data loaded above and saves ONLY to demo_models_dir -- never to `models/`,
    regardless of which data source is loaded. See app/services.py::train_demo_model()'s
    docstring for why this function must never be trusted to decide when overwriting the
    real production model (models/best_model.joblib) is safe."""
    if st.button(button_label):
        with st.spinner("Training..."):
            try:
                result = train_demo_model(history_df, CONFIG_PATH)
            except Exception as error:  # noqa: BLE001 -- training failures are shown, not swallowed
                st.error(f"Training failed: {error}")
                st.stop()
        st.success(
            f"Sandbox model trained and saved under `{demo_models_dir}` (production model, if any, "
            f"is untouched). Best model: **{result.best_model_name}** "
            f"({result.metrics[result.best_model_name]})"
        )
        _load_demo_predictor.clear()
        st.rerun()


if production_predictor is not None:
    st.success("Production model loaded from `models/` -- trained on the real HARTI dataset by `scripts/train.py`.")
    predictor = production_predictor
    with st.expander("Train a sandbox model on the data loaded above"):
        st.caption(f"Saved separately under `{demo_models_dir}` -- never overwrites the production model above.")
        _train_sandbox_model("Train sandbox model")
else:
    st.warning(
        "No production model found under `models/`. Run `scripts/train.py` to train one on the real "
        "HARTI dataset. In the meantime, you can train a temporary sandbox model on the data loaded "
        f"above (saved under `{demo_models_dir}`, never as the production model) to try the app."
    )
    _train_sandbox_model("Train sandbox model on loaded data")
    if demo_predictor is None:
        st.stop()
    st.info("No production model is available -- using a **sandbox** model for this session instead.")
    predictor = demo_predictor

# ---------------------------------------------------------------------------
# 3. Prediction request
# ---------------------------------------------------------------------------

st.header("3. Request a prediction")

vegetable = st.selectbox("Vegetable", sorted(history_df["vegetable"].unique()))
markets_for_vegetable = sorted(history_df.loc[history_df["vegetable"] == vegetable, "market"].unique())
market = st.selectbox("Market", markets_for_vegetable)

if live_snapshot is not None:
    live_fuel_snapshot: LiveFuelPriceSnapshot | None = None
    try:
        live_fuel_snapshot = _fetch_live_fuel_price_cached()
    except LiveFuelPriceError as error:
        st.caption(
            f"Live fuel price unavailable ({error}) -- falling back to the last known "
            "fuel price from historical data."
        )
    else:
        st.caption(
            "**Live fuel price (CEYPETCO):** "
            + ", ".join(
                f"{column} Rs.{price:.2f} (effective "
                f"{live_fuel_snapshot.effective_dates[column].date()})"
                for column, price in live_fuel_snapshot.prices.items()
            )
        )

    try:
        history_df = backfill_harti_price_gap(
            history_df, live_snapshot, vegetable, market, CONFIG_PATH,
            live_fuel_snapshot=live_fuel_snapshot,
        )
        history_df = merge_harti_live_price(
            history_df, live_snapshot, vegetable, market, live_fuel_snapshot=live_fuel_snapshot
        )
    except HartiLiveDataError as error:
        st.error(f"Can't use the latest HARTI bulletin for this selection: {error}")
        st.stop()

resolved_location = resolve_weather_region(vegetable, config_path=CONFIG_PATH)
_coords_caption = f"({resolved_location.latitude:.4f}, {resolved_location.longitude:.4f})"
if resolved_location.is_production_region:
    st.caption(
        f"**Vegetable:** {vegetable} &nbsp;|&nbsp; **Market:** {market} &nbsp;|&nbsp; "
        f"**Weather region:** {resolved_location.region_name} {_coords_caption} (evidence-based "
        f"production region, not the market) &nbsp;|&nbsp; **Weather source:** Open-Meteo"
    )
else:
    st.caption(
        f"**Vegetable:** {vegetable} &nbsp;|&nbsp; **Market:** {market} &nbsp;|&nbsp; "
        f"**Weather region:** {market} {_coords_caption} (market location -- no sufficiently "
        f"evidenced production region for {vegetable}, falling back to market weather) "
        f"&nbsp;|&nbsp; **Weather source:** Open-Meteo"
    )

group_history = history_df[(history_df["vegetable"] == vegetable) & (history_df["market"] == market)]
actual_group_history = group_history.loc[group_history["price"].notna()]
latest_actual_date = actual_group_history["date"].max()
default_target = (latest_actual_date + pd.Timedelta(days=1)).date()

prediction_type = st.radio(
    "Prediction type",
    ["Latest / Next Available Prediction", "Future Date Forecast (multi-step)"],
    horizontal=True,
    help=(
        "'Latest / Next Available Prediction' scores one date directly, same as before. "
        "'Future Date Forecast' reaches a distant date by recursively predicting every day in "
        "between first and feeding each day's own prediction into the next -- see README.md "
        "section 12."
    ),
)
use_live_weather = st.checkbox(
    "Fetch latest weather from Open-Meteo automatically (recommended)",
    value=True,
    help=(
        "If the loaded history's weather doesn't reach far enough back for a required date's "
        "14-day weather lag feature, fetch the missing day(s) from Open-Meteo instead of "
        "failing. Uncheck to predict/forecast using only the weather already in the loaded "
        "history."
    ),
)

if prediction_type == "Latest / Next Available Prediction":
    target_date = st.date_input(
        "Date to predict",
        value=default_target,
        min_value=default_target,
    )
    st.caption(
        "Predicting further beyond the last historical date needs correspondingly more lag/rolling "
        "history (up to 30 days per config.feature_engineering). If it's missing, you'll see a clear "
        "error naming the exact feature(s) that couldn't be computed -- never a fabricated result. "
        "For a date far enough ahead to need more than that, use **Future Date Forecast** instead."
    )

    if st.button("Predict", type="primary"):
        target_ts = pd.Timestamp(target_date)

        try:
            if use_live_weather:
                input_df = build_live_prediction_input(history_df, target_ts, vegetable, market, CONFIG_PATH)
            else:
                input_df = build_prediction_input(history_df, target_ts, vegetable, market)
        except LiveWeatherError as error:
            st.error(f"Couldn't get live weather data: {error}")
            st.stop()

        pipeline = PredictionPipeline(config_path=CONFIG_PATH, predictor=predictor)

        try:
            result = pipeline.predict(input_df, target_date=target_ts, vegetable=vegetable, market=market)
        except FeatureEngineeringError as error:
            st.error(f"Missing required data: {error}")
        except PredictionInputError as error:
            st.error(f"Invalid input: {error}")
        except FeatureAlignmentError as error:
            st.error(f"Not enough historical context to predict: {error}")
        except (ModelArtifactNotFoundError, TransformationArtifactNotFoundError) as error:
            st.error(f"Model not available: {error}")
        else:
            predicted_price = float(result.loc[0, "predicted_price"])
            st.metric(label=f"Predicted price -- {vegetable} at {market} on {target_date}", value=f"{predicted_price:.2f}")

            _weather_lookup_date = target_ts - pd.Timedelta(days=_WEATHER_LAG_DAYS)
            _weather_values = None
            try:
                _engineered_row = pipeline.get_engineered_features(
                    input_df, target_date=target_ts, vegetable=vegetable, market=market
                )
                _weather_values = {
                    "Rainfall": f"{_engineered_row['rainfall_mm_lag14d']:.1f} mm",
                    "Temperature": f"{_engineered_row['temperature_c_lag14d']:.1f} °C",
                    "Humidity": f"{_engineered_row['humidity_pct_lag14d']:.0f} %",
                }
            except (PredictionInputError, KeyError):
                pass  # display the card without values rather than fail the page
            _render_weather_data_card(
                vegetable, market, resolved_location, str(_weather_lookup_date.date()), _weather_values
            )

            if live_snapshot is not None:
                st.caption(
                    f"Data source: **Latest HARTI Data** -- bulletin dated {live_snapshot.bulletin_date.date()}, "
                    "not real-time."
                )
            else:
                st.caption(f"Data source: {data_source}")

            fig, ax = plt.subplots()
            ax.plot(group_history["date"], group_history["price"], marker="o", markersize=3, label="Historical price")
            ax.scatter([pd.Timestamp(target_date)], [predicted_price], color="red", zorder=5, label="Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            fig.autofmt_xdate()
            st.pyplot(fig)

else:  # "Future Date Forecast (multi-step)"
    estimated_max_date = max_supported_forecast_date(latest_actual_date, CONFIG_PATH)
    st.caption(
        f"Estimated furthest forecastable date given the live weather provider's own forward "
        f"limit: **~{estimated_max_date.date()}**. The loaded history's own weather coverage can "
        "extend this further; the real limit is only known once you generate the forecast."
    )
    forecast_date = st.date_input(
        "Forecast date",
        value=default_target,
        min_value=default_target,
        key="forecast_date_input",
    )
    forecast_horizon_preview = (pd.Timestamp(forecast_date) - latest_actual_date).days
    st.caption(
        f"Latest actual data: **{latest_actual_date.date()}** &nbsp;|&nbsp; Forecast horizon: "
        f"**{forecast_horizon_preview} day(s) ahead**"
    )

    if st.button("Generate Forecast", type="primary"):
        live_fuel_snapshot_for_forecast: LiveFuelPriceSnapshot | None = None
        try:
            live_fuel_snapshot_for_forecast = _fetch_live_fuel_price_cached()
        except LiveFuelPriceError as error:
            st.caption(f"Live fuel price unavailable ({error}) -- using historical fuel price fallback.")

        forecast_input_history = history_df
        if live_snapshot is not None:
            # Recursive forecasting depends on the ENTIRE trailing lag/rolling window
            # before latest_actual_date staying dense -- a single isolated day HARTI's
            # bulletin never covered (anywhere in that window, not just the recent
            # stale-CSV gap backfill_harti_price_gap() already closed above) would
            # otherwise silently starve a later forecast day's price_lag_{7,14,30}d.
            # Only real bulletins are ever added here; see that function's docstring.
            forecast_input_history = backfill_internal_harti_gaps(
                forecast_input_history, vegetable, market, CONFIG_PATH,
                live_fuel_snapshot=live_fuel_snapshot_for_forecast,
            )

        # Priority 2, last resort: a date backfill_internal_harti_gaps() just confirmed
        # HARTI genuinely has no bulletin for (a real archive fetch was attempted and
        # found nothing) still needs SOME value for its exact-date lag/rolling lookup.
        # This estimates it deterministically from real surrounding observations only
        # -- never a forecast row -- and tags it source="IMPUTED_HISTORICAL" so it can
        # never be mistaken for a real observation; see that function's docstring.
        forecast_input_history, imputed_dates_for_forecast = impute_missing_historical_prices(
            forecast_input_history, vegetable, market, CONFIG_PATH,
            live_fuel_snapshot=live_fuel_snapshot_for_forecast,
        )

        pipeline = PredictionPipeline(config_path=CONFIG_PATH, predictor=predictor)

        with st.spinner(f"Forecasting {forecast_horizon_preview} day(s) ahead, recursively..."):
            try:
                forecast_result = forecast_future_price(
                    forecast_input_history,
                    vegetable,
                    market,
                    forecast_date,
                    pipeline,
                    CONFIG_PATH,
                    use_live_weather=use_live_weather,
                    live_fuel_snapshot=live_fuel_snapshot_for_forecast,
                )
            except ForecastInputError as error:
                st.error(f"Invalid forecast request: {error}")
                st.stop()
            except ForecastDataUnavailableError as error:
                st.error(f"Cannot forecast that far ahead: {error}")
                st.stop()
            except (ModelArtifactNotFoundError, TransformationArtifactNotFoundError) as error:
                st.error(f"Model not available: {error}")
                st.stop()

        st.subheader("📈 Future Vegetable Price Forecast")
        st.warning(
            "⚠ **FORECAST** -- generated recursively from the latest available actual market "
            "data. This is a model estimate, not a guaranteed future price."
        )
        info_cols = st.columns(3)
        info_cols[0].metric("Vegetable", forecast_result.vegetable)
        info_cols[1].metric("Market", forecast_result.market)
        info_cols[2].metric("Forecast Horizon", f"{forecast_result.forecast_horizon_days} day(s) ahead")
        st.caption(
            f"Latest actual data: **{forecast_result.latest_actual_date.date()}** &nbsp;|&nbsp; "
            f"Forecast date: **{forecast_result.target_date.date()}**"
        )
        _forecast_weather_lookup_date = forecast_result.target_date - pd.Timedelta(days=_WEATHER_LAG_DAYS)
        _forecast_weather_values = None
        try:
            # Re-derives the SAME final-day input build_future_price() already computed
            # internally (weather doesn't depend on the recursive price chain) purely to
            # read back its engineered values for display -- side-effect-free, does not
            # affect forecast_result, which was already computed above.
            if use_live_weather:
                _final_day_input = build_live_prediction_input(
                    forecast_input_history, forecast_result.target_date, vegetable, market, CONFIG_PATH
                )
            else:
                _final_day_input = build_prediction_input(
                    forecast_input_history, forecast_result.target_date, vegetable, market
                )
            _engineered_row = pipeline.get_engineered_features(
                _final_day_input, target_date=forecast_result.target_date, vegetable=vegetable, market=market
            )
            _forecast_weather_values = {
                "Rainfall": f"{_engineered_row['rainfall_mm_lag14d']:.1f} mm",
                "Temperature": f"{_engineered_row['temperature_c_lag14d']:.1f} °C",
                "Humidity": f"{_engineered_row['humidity_pct_lag14d']:.0f} %",
            }
        except (LiveWeatherError, PredictionInputError, KeyError):
            pass  # display the card without values rather than fail the page
        _render_weather_data_card(
            vegetable, market, resolved_location, str(_forecast_weather_lookup_date.date()), _forecast_weather_values
        )
        st.metric(
            label=(
                f"Forecasted Price -- {forecast_result.vegetable} at {forecast_result.market} on "
                f"{forecast_result.target_date.date()}"
            ),
            value=f"Rs. {forecast_result.predicted_price:.2f} / kg",
        )
        st.caption("**Data source:** " + " + ".join(forecast_result.data_sources))
        for warning_message in forecast_result.warnings:
            st.caption(f"⚠ {warning_message}")

        daily_forecast_df = pd.DataFrame(
            {
                "date": [point.date for point in forecast_result.daily_forecast],
                "forecasted_price": [point.predicted_price for point in forecast_result.daily_forecast],
            }
        )
        with st.expander("Daily forecast path"):
            st.dataframe(daily_forecast_df)
            if forecast_result.imputed_historical_dates:
                imputed_mask = forecast_input_history["source"] == IMPUTED_HISTORICAL_SOURCE_LABEL
                imputed_points = forecast_input_history.loc[imputed_mask, ["date", "price"]].sort_values("date")
                st.dataframe(imputed_points.rename(columns={"price": "imputed_estimated_price"}))

            fig, ax = plt.subplots()
            ax.plot(
                actual_group_history["date"], actual_group_history["price"],
                marker="o", markersize=3, label="Historical price (actual)",
            )
            if forecast_result.imputed_historical_dates:
                imputed_mask = forecast_input_history["source"] == IMPUTED_HISTORICAL_SOURCE_LABEL
                imputed_points = forecast_input_history.loc[imputed_mask].sort_values("date")
                ax.scatter(
                    imputed_points["date"], imputed_points["price"],
                    marker="x", color="gray", zorder=4,
                    label="Imputed historical (estimated -- not real HARTI data)",
                )
            ax.plot(
                daily_forecast_df["date"], daily_forecast_df["forecasted_price"],
                marker="o", markersize=3, linestyle="--", color="orange", label="Forecast (recursive)",
            )
            ax.scatter(
                [forecast_result.target_date], [forecast_result.predicted_price],
                color="red", zorder=5, label="Requested forecast date",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            fig.autofmt_xdate()
            st.pyplot(fig)
