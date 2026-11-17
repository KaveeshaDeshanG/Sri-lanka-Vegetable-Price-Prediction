"""Streamlit page: train a demo model if needed, then request a price prediction."""

import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)  # config and data paths are relative to the project root

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app import ui
from app.i18n import t
from app.catalogue import (
    data_freshness,
    data_version,
    display_name,
    last_run_problem,
    localized,
    past_date_notice,
    series_status,
    update_notice,
    update_status,
)
from app.data_refresh import run_data_update
from app.source_records import (
    cbsl_records,
    dcs_records,
    harti_dambulla_records,
    harti_other_market_records,
    latest_dambulla_overview,
    mapping_note,
    records_for_series,
)
from app.forecasting import (
    ForecastDataUnavailableError,
    ForecastInputError,
    ForecastResult,
    direct_forecast,
    forecast_future_price,
    max_supported_forecast_date,
)
from app.services import (
    FUEL_COLUMNS,
    IMPUTED_HISTORICAL_SOURCE_LABEL,
    HistoryInputError,
    backfill_harti_price_gap,
    backfill_internal_harti_gaps,
    build_live_prediction_input,
    build_prediction_input,
    fetch_harti_latest_snapshot,
    fetch_live_fuel_price,
    forecaster_problems,
    impute_missing_historical_prices,
    load_real_market_history,
    load_sample_history,
    merge_harti_live_price,
    missing_historical_datasets,
    prepare_uploaded_history,
    resolve_demo_models_dir,
    resolve_weather_region,
    train_demo_model,
)
from config.config_loader import ConfigLoader
from src.data_collection.data_validator import DataValidationError
from src.data_collection.harti_live_client import HartiLiveClient, HartiLiveDataError, HartiLiveSnapshot
from src.data_collection.live_fuel_price_client import LiveFuelPriceError, LiveFuelPriceSnapshot
from src.data_collection.live_weather_client import LiveWeatherError
from src.feature_engineering.feature_engineer import FeatureEngineeringError
from src.feature_engineering.forecast_features import ForecastFeatureBuilder
from src.feature_engineering.production_region_weather import (
    apply_production_region_weather,
)
from src.models.direct_forecaster import MAX_HORIZON_DAYS, DirectForecaster
from src.models.artifact_metadata import (
    ArtifactConsistencyError,
    ArtifactMetadataNotFoundError,
    load_artifact_metadata,
)
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
_STALE_AFTER_DAYS = 30
_MANY_IMPUTED_DAYS = 7
# The page offers two data sources. Real data is the project's stored official records, kept current
# by scripts/daily_update.py (scheduled, or "Check official sources now" below). The former
# "Bundled sample data" and live "Latest HARTI Data" options are no longer offered; their loaders
# (app/services.py: load_sample_history, fetch_harti_latest_snapshot, ...) are kept and tested.
SOURCE_UPLOAD = "Upload CSV"
SOURCE_REAL = "Latest Real Data Prediction"
SOURCE_HISTORICAL = SOURCE_REAL  # the name the code below has always used for the real-data path
SOURCE_LATEST = None  # live-bulletin mode: not offered, so live_snapshot below is always None
# Horizons offered for real data, with the Model Performance band each was scored in.
_HORIZON_CHOICES = {"1 day": 1, "7 days": 7, "14 days": 14, "30 days": 30}
_HORIZON_BANDS = {1: "1", 7: "4-7", 14: "8-14", 30: "22-30"}
# What a damaged CSV can raise while loading: pandas' parser and decoding errors are ValueErrors.
_DATA_READ_ERRORS = (DataValidationError, ValueError, KeyError, TypeError, OSError)
# A sandbox model stays on disk, so it may have been trained before this data was loaded.
_SANDBOX_IN_USE = (
    "Using a **sandbox** model from the most recent sandbox training, not the project's saved models. "
    "It may have been trained on different data; train again above to use this data."
)  # shown through t()

st.set_page_config(page_title="Predict | Vegetable Price Prediction", page_icon="📈", layout="wide", initial_sidebar_state="auto")
ui.apply_theme()
ui.page_header(
    "Predict a vegetable price",
    "Forecast from the latest official prices, which you can inspect source by source, or from your own CSV.",
    "Price forecasting",
)


def _render_weather_data_card(
    vegetable: str,
    market: str,
    resolved_location,
    weather_date_label: str,
    weather_values: dict[str, float] | None,
    used_for: str | None = None,
) -> None:
    """Renders the 'Weather Data Used' transparency card: which location's weather fed
    this prediction, its coordinates, and (when available) the actual values used --
    never fabricated, never silently swapped in for the market's location."""
    if used_for is None:
        used_for = t("Price Prediction (as {days}-day-lagged features)", days=_WEATHER_LAG_DAYS)
    with st.container(border=True):
        st.markdown(t("**🌦️ Weather Data Used**"))
        if resolved_location.is_production_region:
            location_label = resolved_location.region_name
            location_note = t("evidence-based cultivation/production region for this vegetable")
        else:
            location_label = t("{market} (market location)", market=market)
            location_note = t(
                "⚠ fallback -- no sufficiently evidenced cultivation-region mapping exists for "
                "'{vegetable}' in config/crop_production_regions.yaml; this is the project's "
                "documented fallback, not a guess at {vegetable}'s actual cultivation location",
                vegetable=vegetable,
            )
        st.markdown(
            f"- **{t('Vegetable')}:** {localized(vegetable)}\n"
            f"- **{t('Weather Location')}:** {location_label}\n"
            f"- **{t('Coordinates')}:** {resolved_location.latitude:.4f}, {resolved_location.longitude:.4f}\n"
            f"- **{t('Weather Source')}:** Open-Meteo\n"
            f"- **{t('Weather Date')}:** {weather_date_label}\n"
            f"- **{t('Data Used For')}:** {used_for}"
        )
        st.caption(location_note)
        if weather_values:
            ui.stat_cards([{"label": label, "value": value} for label, value in weather_values.items()], compact=True)
        st.caption(
            t(
                "Weather used by this model: rainfall, temperature and humidity. "
                "Wind speed is not part of this project's weather schema/model -- not shown, not fabricated."
            )
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
def _load_real_market_history_cached(version: str) -> pd.DataFrame:
    """Cached per data_version(), so a daily update is picked up without restarting the app."""
    return load_real_market_history(CONFIG_PATH)


_DATA_VERSION = data_version()
# Read on every run (a small file), so a scheduled update's result shows on the next click.
_UPDATE_STATUS = update_status(Path(ConfigLoader(CONFIG_PATH).get("paths.raw_data_dir", "data/raw")) / "update_log.jsonl")


@st.cache_data(ttl=_HARTI_CACHE_TTL_SECONDS)
def _fetch_harti_snapshot_cached() -> HartiLiveSnapshot:
    return fetch_harti_latest_snapshot(CONFIG_PATH)


@st.cache_data(ttl=_FUEL_CACHE_TTL_SECONDS)
def _fetch_live_fuel_price_cached() -> LiveFuelPriceSnapshot:
    return fetch_live_fuel_price(CONFIG_PATH)


@st.cache_resource(ttl=_HARTI_CACHE_TTL_SECONDS)
def _harti_client() -> HartiLiveClient:
    """One shared client, which remembers the archive index and bulletins it has read, so a
    widget interaction doesn't re-download them."""
    return HartiLiveClient.from_config(CONFIG_PATH)


@st.cache_data(ttl=_HARTI_CACHE_TTL_SECONDS, show_spinner="Reading HARTI bulletins...")
def _history_with_latest_bulletin(
    vegetable: str,
    market: str,
    bulletin_date: str,
    _history_df: pd.DataFrame,
    _live_snapshot: HartiLiveSnapshot,
    _live_fuel_snapshot: LiveFuelPriceSnapshot | None,
) -> pd.DataFrame:
    """The history plus real bulletins closing any gap up to the latest one. Cached by
    (vegetable, market, bulletin_date); underscore arguments aren't part of the cache key."""
    backfilled = backfill_harti_price_gap(
        _history_df, _live_snapshot, vegetable, market, CONFIG_PATH,
        live_fuel_snapshot=_live_fuel_snapshot, client=_harti_client(),
    )
    return merge_harti_live_price(backfilled, _live_snapshot, vegetable, market, live_fuel_snapshot=_live_fuel_snapshot)


def _history_for_features(
    history: pd.DataFrame, vegetable: str, market: str, live_fuel: LiveFuelPriceSnapshot | None, use_archive: bool
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Close holes in the trailing lag/rolling window so exact-date features can be computed:
    real archived bulletins first (when using live HARTI data), then labelled estimates."""
    if use_archive:
        history = backfill_internal_harti_gaps(
            history, vegetable, market, CONFIG_PATH, live_fuel_snapshot=live_fuel, client=_harti_client()
        )
    return impute_missing_historical_prices(history, vegetable, market, CONFIG_PATH, live_fuel_snapshot=live_fuel)


def _load_model_files(load, description: str, training_script: str = "scripts/train.py"):
    """Load a model, reporting unusable model files on the page rather than crashing it."""
    try:
        return load()
    except (ModelArtifactNotFoundError, TransformationArtifactNotFoundError):
        return None
    except ArtifactConsistencyError as error:
        st.error(t("{description} the model files are inconsistent and cannot be used: {error}", description=t(description), error=error))
    except Exception as error:  # noqa: BLE001 -- a corrupt file can raise almost anything while unpickling
        st.error(
            t(
                "{description} the model files could not be read, so they may be damaged "
                "({error}). Restore them from a backup, or run `{script}` to recreate them.",
                description=t(description),
                error=f"{type(error).__name__}: {error}",
                script=training_script,
            )
        )
    return None


def _missing_required_columns(model: Predictor, history: pd.DataFrame) -> list[str]:
    """Fuel columns this model's features need that the loaded data doesn't have, or has only blank."""
    return [
        column
        for column in FUEL_COLUMNS
        if any(feature.startswith(f"{column}_lag") for feature in model.feature_columns)
        and (column not in history.columns or history[column].notna().sum() == 0)
    ]


@st.cache_resource
def _load_forecaster_version(version: str) -> DirectForecaster:
    try:
        return DirectForecaster.load(CONFIG_PATH)
    except FileNotFoundError as error:
        raise ModelArtifactNotFoundError(str(error)) from error


@st.cache_data(ttl=_HARTI_CACHE_TTL_SECONDS, show_spinner="Preparing forecast features...")
def _forecast_origin_rows(cache_key: str, _history: pd.DataFrame, _latest_markets: pd.DataFrame | None) -> pd.DataFrame:
    """Features for every vegetable and date. Cached by cache_key alone."""
    return ForecastFeatureBuilder(CONFIG_PATH, extra_market_rows=_latest_markets).build_origin_features(_history)


def _render_direct_result(
    result: ForecastResult, origin_rows: pd.DataFrame, actual_history: pd.DataFrame, location, source_label: str, multi_day: bool
) -> None:
    """Show a direct forecast: the price, its path (for several days), weather and sources."""
    if multi_day:
        st.subheader(t("📈 Future Vegetable Price Forecast"))
        st.warning(
            t(
                "⚠ **FORECAST** -- a model estimate from the latest actual market data, not a guaranteed "
                "future price. Error grows with distance; see **Model Performance** for measured accuracy."
            )
        )
        columns = st.columns(3)
        columns[0].metric(t("Vegetable"), localized(result.vegetable))
        columns[1].metric(t("Market"), result.market)
        columns[2].metric(t("Forecast Horizon"), t("{days} day(s) ahead", days=result.forecast_horizon_days))
    label_template = "Forecasted Price -- {vegetable} at {market} on {date}" if multi_day else "Predicted Price -- {vegetable} at {market} on {date}"
    with st.container(key="vp_result_direct"):
        st.metric(
            label=t(label_template, vegetable=localized(result.vegetable), market=result.market, date=result.target_date.date()),
            value=f"Rs. {result.predicted_price:.2f} / {t('kg')}",
        )
        _latest_actual = actual_history.loc[actual_history["date"] == result.latest_actual_date, "price"]
        _latest_price = f" (Rs. {_latest_actual.iloc[-1]:,.2f} / {t('kg')})" if not _latest_actual.empty else ""
        st.caption(
            t(
                "Latest actual data: **{date}**{price}. Each day's forecast uses only what was "
                "known then, including other markets' published prices. **Data source:** {source}.",
                date=result.latest_actual_date.date(),
                price=_latest_price,
                source=t(source_label),
            )
        )
        st.caption(f"**{t('Model used')}:** {t(_model_used)}")

    chart_column, detail_column = st.columns([3, 2], gap="medium")
    origin_row = origin_rows[
        (origin_rows["vegetable"] == result.vegetable) & (origin_rows["origin"] == result.latest_actual_date)
    ].iloc[0]
    values = {
        t("Rainfall (last 7 days)"): origin_row["rain_sum_7"],
        t("Temperature (7-day mean)"): origin_row["temp_mean_7"],
        t("Humidity (7-day mean)"): origin_row["humidity_mean_7"],
    }
    units = dict(zip(values, ("mm", "°C", "%"), strict=True))
    path = pd.DataFrame(
        {"date": [p.date for p in result.daily_forecast], "forecasted_price": [p.predicted_price for p in result.daily_forecast]}
    )
    with detail_column:
        _render_weather_data_card(
            result.vegetable,
            result.market,
            location,
            t("The 7 days up to {date}", date=result.latest_actual_date.date()),
            {name: f"{value:.1f} {units[name]}" for name, value in values.items() if pd.notna(value)} or None,
            used_for=t("Forecasting, from observed weather only (no future weather is needed)"),
        )
        if multi_day:
            with st.expander(t("Daily forecast path")):
                st.dataframe(
                    ui.translated_columns(pd.DataFrame({"Date": path["date"].dt.date, "Forecast (Rs / kg)": path["forecasted_price"].round(2)})),
                    hide_index=True,
                )
    recent = actual_history[actual_history["date"] >= result.latest_actual_date - pd.Timedelta(days=180)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(recent["date"], recent["price"], marker="o", markersize=2.5, color=ui.SERIES["actual"], label=t("Actual price"))
    if multi_day:
        ax.plot(path["date"], path["forecasted_price"], marker="o", markersize=3, linestyle="--", color=ui.SERIES["forecast"], label=t("Forecast"))
    ax.scatter([result.target_date], [result.predicted_price], s=70, color=ui.SERIES["target"], edgecolors=ui.COLORS["background"], zorder=5, label=t("Requested date"))
    ax.set_xlabel(t("Date"))
    ax.set_ylabel(t("Price (Rs / kg)"))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    with chart_column:
        st.pyplot(fig)
    plt.close(fig)


def _model_summary() -> str:
    try:
        metadata = load_artifact_metadata(config_path=CONFIG_PATH)
    except (ArtifactMetadataNotFoundError, ValueError, TypeError):
        return t("Earlier comparison model loaded from `models/`.")
    data = metadata.training_data
    return t(
        "Earlier comparison model loaded: **{model}**, trained on {rows} rows of data from {start} to {end}.",
        model=metadata.model_type,
        rows=data.get("train_rows", "?"),
        start=data.get("start_date", "?"),
        end=data.get("end_date", "?"),
    )


# ---------------------------------------------------------------------------
# Latest Real Data Prediction: source status, collected records, prediction, freshness
# ---------------------------------------------------------------------------

# Source key in update_log.jsonl -> (name, what it publishes, days after which its newest publication is
# behind its usual schedule). HARTI publishes daily (longest 2026 gap: 4 days); CBSL on working days;
# DCS weekly, as "week n of a month", whose age is counted from the week's first day (day 1, 8, 15 or 22).
_OFFICIAL_SOURCES = {
    "harti": ("HARTI", "Daily wholesale bulletin, every market", 3),
    "cbsl": ("CBSL", "Daily Price Report, wholesale and retail", 4),
    "dcs": ("DCS", "Weekly retail prices, Colombo District", 21),
}
# DCS's 21 days (three weekly cycles), from its bulletins' own PDF creation dates (16 consecutive bulletins,
# 2024 and 2026; reports/DATA_SOURCES.md, "DCS freshness"): a bulletin appears 3-20 days after its week
# starts, and a "week 4" label starts on the 22nd but runs to the month's end, so on schedule the newest
# week is up to 20 days old just before the next one appears (median 15). 14 days flagged 10 of those 16
# cycles as stale (9 on schedule); 21 flags only October 2026, when week 2 came 15 days late.


def _publication_date(value) -> pd.Timestamp | None:
    """A source's newest publication as a date: '2026-11-12', or DCS's '2026-11 week 1' (its week's first day)."""
    if not value:
        return None
    week = re.fullmatch(r"(\d{4})-(\d{2}) week (\d)", str(value))
    if week:
        year, month, number = map(int, week.groups())
        return pd.Timestamp(year=year, month=month, day=1) + pd.Timedelta(days=7 * (number - 1))
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _render_source_status() -> None:
    st.header(t("1. Official source status"))
    refresh = st.session_state.pop("real_data_refresh_result", None)
    if refresh is not None:
        getattr(st, refresh.level)(refresh.message)
        if refresh.output_tail and refresh.level != "success":
            with st.expander(t("Update log (last lines)")):
                st.code(refresh.output_tail, language="text")
    status = _UPDATE_STATUS
    if status.last_attempt_failed:
        st.warning(t("The latest update attempt failed ({error}). The status below is from the last completed check.", error=status.last_error))
    today = pd.Timestamp.now().normalize()
    for column, (key, (name, publishes, behind_days)) in zip(st.columns(3, gap="small"), _OFFICIAL_SOURCES.items()):
        info = status.sources.get(key)
        with column, st.container(border=True):
            st.markdown(f"**{name}**  \n{t(publishes)}")
            if info is None:
                st.info(t("Not recorded yet: the last completed check predates source-by-source status. Use **Check official sources now**."))
                continue
            if info["status"] == "failed":
                st.error(t("Couldn't be checked at the last update. Its stored records are shown unchanged."))
                st.caption(t("Error: {error}", error=info.get("error", t("no details"))))
            else:
                st.success(t("Checked successfully"))
            newest = _publication_date(info.get("newest_published"))
            age = (today - newest).days if newest is not None else None
            lines = [
                f"**{t('Newest publication found')}:** {info.get('newest_published') or t('unknown')}"
                + (" " + t("({days} day(s) ago)", days=age) if age is not None else ""),
                f"**{t('Last checked')}:** {pd.Timestamp(info['checked_at']):%Y-%m-%d %H:%M}",
            ]
            if info["status"] != "failed":
                new = info.get("new_records", 0)
                collected = (t("{n} record(s)", n=new) if key == "harti" else t("{n} report(s)", n=new)) if new else t("none, nothing newer was published")
                lines.append(f"**{t('New data collected')}:** {collected}")
            for problem in info.get("problems", []):
                lines.append(f"**{t('Import problem')}:** {problem}")
            if age is not None and age > behind_days:
                lines.append(f"**{t('Stale')}:** " + t("the newest {name} publication is {days} days old", name=name, days=age))
            st.markdown("  \n".join(lines))
    others = [
        f"{t(name.capitalize())}: {t('couldn' + chr(39) + 't be updated') if status.sources[name]['status'] == 'failed' else t('checked')}"
        for name in ("weather", "fuel")
        if name in status.sources
    ]
    st.caption(
        (
            t("Last completed check of all sources: **{when}**.", when=f"{status.last_success:%Y-%m-%d %H:%M}") + " "
            if status.last_success is not None
            else t("No completed check recorded.") + " "
        )
        + t("Checks run automatically every 3 hours (Windows task `VegetablePriceUpdate`), and on request below.")
        + " "
        + ("; ".join(others) + "." if others else "")
    )
    problem = last_run_problem()
    if problem:
        st.warning(problem + " " + t("The next scheduled update retries automatically."))
    freshness_rows = data_freshness()
    with st.expander(ui.freshness_summary(freshness_rows) + " " + t("(all sources, including weather and fuel)")):
        ui.data_freshness_panel(freshness_rows)
    if st.button(t("Check official sources now"), help=t("Downloads any newly published HARTI, CBSL and DCS reports (about a minute). Nothing already stored is changed.")):
        with st.spinner(t("Checking HARTI, CBSL and DCS for newly published reports...")):
            st.session_state["real_data_refresh_result"] = run_data_update()
        st.rerun()


@st.cache_data(show_spinner="Reading the collected official records...")
def _load_source_records(version: str) -> dict[str, pd.DataFrame]:
    """Each source's stored records with their traceability fields. Cached per data_version()."""
    return {
        "harti_dambulla": harti_dambulla_records(CONFIG_PATH),
        "harti_other": harti_other_market_records(CONFIG_PATH),
        "cbsl": cbsl_records(CONFIG_PATH),
        "dcs": dcs_records(CONFIG_PATH),
    }


# Record tables: the columns a reader needs first (when, where, the prices, the document and when it was
# collected) fit side by side at 1440 px; the descriptive ones that follow are reached by scrolling the
# table sideways, while the pinned date and market stay in view. No column is dropped.
_RECORD_COLUMN_ORDER = [
    "Observation date",
    "Market / location",
    "Minimum",
    "Maximum",
    "Average / midpoint",
    "Unit",
    "Source document",
    "Collected",
    "Price type",
    "Published",
    "Vegetable (as published)",
    "Source",
]


def _record_column_config() -> dict:
    text = st.column_config.TextColumn  # prices arrive pre-formatted, so a missing one can say why
    return {
        "Observation date": text(t("Observation date"), width=126, pinned=True),
        "Market / location": text(t("Market / location"), width=130, pinned=True, help=t("Market, or the area the price covers")),
        "Minimum": text(t("Min"), width=72, help=t("Lowest price in the source's published range")),
        "Maximum": text(t("Max"), width=72, help=t("Highest price in the source's published range")),
        "Average / midpoint": text(t("Average / midpoint"), width=128, help=t("HARTI: midpoint of its range (the model's price). CBSL: its single price. DCS: weekly average.")),
        "Unit": text(t("Unit"), width=92),
        "Source document": st.column_config.LinkColumn(t("Document"), display_text=t("Open"), width=84, help=t("The official bulletin or report this record was read from")),
        "Collected": text(t("Collected"), width=128, help=t("When this project downloaded the document")),
        "Price type": text(t("Price type"), width=136),
        "Published": text(t("Published"), width=112),
        "Vegetable (as published)": text(t("Vegetable (as published)"), width=170),
        "Source": text(t("Source"), width=250),
    }


def _render_records(frame: pd.DataFrame, key: str, missing: str = "not published") -> None:  # missing: English, translated below
    """A record table: dates as dates, missing values named, never shown as zero."""
    if frame.empty:
        st.info(t("No records."))
        return
    shown = frame.copy()
    for column in ("Observation date", "Published"):
        shown[column] = shown[column].map(lambda v: v.date().isoformat() if isinstance(v, pd.Timestamp) else str(v))
    not_recorded, missing_text = t("not recorded"), t(missing)
    shown["Collected"] = shown["Collected"].map(lambda v: f"{v:%Y-%m-%d %H:%M}" if pd.notna(v) else not_recorded)
    for column in ("Minimum", "Maximum", "Average / midpoint"):
        shown[column] = shown[column].map(lambda v: f"{v:,.2f}" if pd.notna(v) else missing_text)
    if "Price type" in shown.columns:
        shown["Price type"] = shown["Price type"].map(lambda v: t(v) if isinstance(v, str) else v)
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        key=f"records_{key}",
        column_order=[column for column in _RECORD_COLUMN_ORDER if column in shown.columns],
        column_config=_record_column_config(),
    )
    st.caption(
        t(
            "{n} record(s), {columns} columns. ↔ Scroll the table sideways (scrollbar under it, or "
            "Shift + mouse wheel) to reach the last columns, Vegetable (as published) and Source; the date and market stay pinned.",
            n=len(shown),
            columns=len(_RECORD_COLUMN_ORDER),
        )
    )


def _render_vegetable_records(series: str, tables: dict[str, pd.DataFrame]) -> None:
    """The selected vegetable's newest records, source by source, and the one the forecast starts from."""
    records = records_for_series(series, tables)
    dambulla = records["harti_dambulla"]
    if dambulla.empty:
        st.error(t("No HARTI Dambulla price is stored for {series}, so no prediction is possible. Nothing is estimated in its place.", series=display_name(series)))
        st.stop()
    start = dambulla.iloc[0]
    with st.container(border=True):
        st.markdown(
            t(
                "**Forecast starting point (used by the model):** {source}, {series}, observed "
                "**{date}**: Rs **{price:,.2f}** per kg (midpoint of "
                "{low:,.2f} to {high:,.2f}), wholesale. ",
                source=start["Source"],
                series=display_name(series),
                date=f"{start['Observation date']:%Y-%m-%d}",
                price=start["Average / midpoint"],
                low=start["Minimum"],
                high=start["Maximum"],
            )
            + f"[{t('Bulletin')}]({str(start['Source document']).replace(' ', '%20')}) · {t('collected')} "
            + (f"{start['Collected']:%Y-%m-%d %H:%M}" if pd.notna(start["Collected"]) else t("time not recorded"))
        )

    def newest(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
        keep = frame["Observation date"].drop_duplicates().head(periods)
        return frame[frame["Observation date"].isin(keep)]

    note = mapping_note(series)
    tabs = st.tabs([
        t("HARTI Dambulla · used by the model"),
        t("HARTI other markets · model inputs"),
        t("CBSL · reference only"),
        t("DCS · reference only"),
    ])
    with tabs[0]:
        st.caption(t("The forecaster's own series: the midpoint of HARTI's published min-max range, Rs per kg. Newest 10 bulletins."))
        _render_records(newest(dambulla, 10), "dambulla")
    with tabs[1]:
        st.caption(
            t(
                "The same bulletins' other markets. The model uses their prices relative to Dambulla as inputs; it doesn't "
                "forecast them. Newest 3 bulletins. Min and max are shown where the bulletin's range has been extracted."
            )
        )
        _render_records(newest(records["harti_other"], 3), "other", missing="not extracted")
    with tabs[2]:
        st.caption(
            t(
                "Collected by CBSL separately from HARTI, so its prices differ and are never combined with HARTI's. CBSL prints "
                "one price per market (no range). Not used by the model. Newest 3 reports."
            )
            + (" " + t("Matching: {note}.", note=note) if note else "")
        )
        cbsl = newest(records["cbsl"], 3)
        if cbsl.empty:
            st.info(t("CBSL's report has no item that clearly matches {series}.", series=display_name(series)))
        for price_type in ("Wholesale", "Retail"):
            part = cbsl[cbsl["Price type"] == price_type]
            if not part.empty:
                st.markdown(f"**{t(price_type)}**")
                _render_records(part, f"cbsl_{price_type}")
    with tabs[3]:
        st.caption(
            t(
                "Retail weekly averages for Colombo District. DCS publishes a week ('week n of a month'), not a date. "
                "Retail, so never comparable with HARTI's wholesale prices. Not used by the model. Newest 4 weeks."
            )
            + (" " + t("Matching: {note}.", note=note) if note else "")
        )
        dcs = newest(records["dcs"], 4)
        if dcs.empty:
            st.info(t("DCS's bulletin has no item that clearly matches {series}.", series=display_name(series)))
        else:
            _render_records(dcs, "dcs")


@st.cache_data
def _band_metrics(version: str) -> dict[str, dict]:
    path = Path(ConfigLoader(CONFIG_PATH).get("paths.metrics_dir", "reports/metrics")) / "direct_forecaster_metrics.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {"period": f"{report.get('test_start')} to {report.get('test_end')}", **{band["days_ahead"]: band for band in report.get("bands", [])}}


def _render_real_prediction() -> None:
    st.header(t("4. Prediction"))
    st.caption(
        t(
            "**Model:** Ridge direct forecaster, one model per number of days ahead (1 to {days}), trained on "
            "data through {date}. **Its inputs:** HARTI Dambulla prices, other HARTI markets' "
            "prices, Open-Meteo weather for the vegetable's growing region, and CEYPETCO fuel prices (next-day model only). "
            "CBSL and DCS records are not inputs.",
            days=MAX_HORIZON_DAYS,
            date=forecaster.trained_through.date(),
        )
    )
    another = "Another date (up to {days} days)"
    labels = [*_HORIZON_CHOICES, another]
    choice = st.radio(
        t("Forecast period"), labels, horizontal=True, key="real_forecast_period", format_func=lambda option: t(option, days=MAX_HORIZON_DAYS)
    )
    if choice in _HORIZON_CHOICES:
        horizon = _HORIZON_CHOICES[choice]
        target = latest_actual_date + pd.Timedelta(days=horizon)
    else:
        picked = st.date_input(
            t("Forecast date"),
            value=(latest_actual_date + pd.Timedelta(days=1)).date(),
            min_value=(latest_actual_date + pd.Timedelta(days=1)).date(),
            max_value=(latest_actual_date + pd.Timedelta(days=MAX_HORIZON_DAYS)).date(),
            key="real_forecast_date",
        )
        target = pd.Timestamp(picked)
        horizon = (target - latest_actual_date).days
    st.caption(
        t(
            "Forecast date **{target}**: {days} day(s) after the latest actual price ({latest}). "
            "Forecasts count from the latest published price, not from today.",
            target=target.date(),
            days=horizon,
            latest=latest_actual_date.date(),
        )
    )
    if not st.button(t("Predict"), type="primary", key="real_predict"):
        return
    _show_direct_forecast(target, multi_day=horizon > 1)

    try:
        path = direct_forecast(
            forecaster, _direct_rows, vegetable, market, latest_actual_date + pd.Timedelta(days=MAX_HORIZON_DAYS), latest_actual_date, [_direct_sources]
        ).daily_forecast
    except (ForecastInputError, ForecastDataUnavailableError):
        return
    metrics = _band_metrics(_DATA_VERSION)
    rows = []
    for days in _HORIZON_CHOICES.values():
        point, band = path[days - 1], metrics.get(_HORIZON_BANDS[days], {})
        rows.append(
            {
                "Days ahead": days,
                "Forecast date": point.date.date().isoformat(),
                "Forecast (Rs per kg)": round(point.predicted_price, 2),
                "Accuracy measured for": t("{band} day ahead", band=_HORIZON_BANDS[days]) if _HORIZON_BANDS[days] == "1" else t("{band} days ahead", band=_HORIZON_BANDS[days]),
                "Measured MAE (Rs per kg)": round(band["model_mae"], 2) if band else None,
                "Measured MAPE (%)": round(band["model_mape"], 1) if band else None,
                "MAE of holding the last price (Rs per kg)": round(band["hold_mae"], 2) if band else None,
            }
        )
    st.markdown(t("**Forecasts at 1, 7, 14 and 30 days**"))
    # Short headers and fixed widths so all seven columns fit at 1440 px; the full names are in each header's tooltip.
    number = st.column_config.NumberColumn
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        key="real_horizon_table",
        column_config={
            "Days ahead": number(t("Days ahead"), width=88, format="%d"),
            "Forecast date": st.column_config.TextColumn(t("Forecast date"), width=112),
            "Forecast (Rs per kg)": number(t("Forecast (Rs/kg)"), width=124, format="%.2f", help=t("Forecast (Rs per kg)")),
            "Accuracy measured for": st.column_config.TextColumn(t("Accuracy measured for"), width=164, help=t("The range of days ahead whose test errors are shown")),
            "Measured MAE (Rs per kg)": number(t("Model MAE (Rs/kg)"), width=138, format="%.2f", help=t("Measured MAE (Rs per kg) of the model over that range")),
            "Measured MAPE (%)": number(t("Model MAPE (%)"), width=122, format="%.1f", help=t("Measured MAPE (%) of the model over that range")),
            "MAE of holding the last price (Rs per kg)": number(
                t("Hold-last-price MAE (Rs/kg)"), width=196, format="%.2f", help=t("MAE of holding the last price (Rs per kg): the naive baseline over that range")
            ),
        },
    )
    st.caption(
        t(
            "**Model MAE** and **Hold-last-price MAE** are the average error in Rs per kg of this model and of simply "
            "repeating the last price, over the range in **Accuracy measured for**. "
            "Each forecast comes from that day's own model, starting from Rs {price:,.2f} on "
            "{date}. Accuracy was measured on all {n} vegetables over the held-out "
            "test period {period} (`reports/metrics/direct_forecaster_metrics.json`), not on "
            "{vegetable} alone; see **Model Performance**.",
            price=actual_group_history["price"].iloc[-1],
            date=latest_actual_date.date(),
            n=len(forecaster.vegetables),
            period=metrics.get("period", t("unknown")),
            vegetable=display_name(vegetable),
        )
    )


def _render_freshness(series: str, market_name: str, latest: pd.Timestamp, level, message, stale) -> None:
    st.header(t("5. Data freshness and warnings"))
    if message:
        (st.warning if level == "warning" else st.info)(message)
    if stale:
        st.warning(stale)
    failed = [_OFFICIAL_SOURCES[key][0] for key, info in _UPDATE_STATUS.sources.items() if key in _OFFICIAL_SOURCES and info.get("status") == "failed"]
    if failed:
        st.warning(t("{sources} couldn't be checked at the last update, so newer records from it may exist.", sources=", ".join(failed)))
    if latest + pd.Timedelta(days=1) < pd.Timestamp.now().normalize():
        st.caption(
            t(
                "The forecast path starts on {date}, the day after the latest actual price. "
                "Its days before today have already passed: they are forecasts, not recorded prices.",
                date=(latest + pd.Timedelta(days=1)).date(),
            )
        )
    st.caption(
        t(
            "Prices are HARTI's published Dambulla wholesale prices for {series} at {market}, not real-time "
            "prices. No price is invented: a day HARTI didn't publish stays missing in the records. For its inputs, the model "
            "carries the last real price forward over at most 7 missing days (`MAX_CARRY_DAYS`), and treats older gaps as unknown.",
            series=display_name(series),
            market=market_name,
        )
    )


# ---------------------------------------------------------------------------
# 1. Historical data source
# ---------------------------------------------------------------------------

_missing_datasets = missing_historical_datasets(CONFIG_PATH)
_data_source_options = [SOURCE_UPLOAD] + ([] if _missing_datasets else [SOURCE_REAL])
if _missing_datasets:
    st.warning(
        t(
            "Latest Real Data Prediction is unavailable because of missing file(s): "
            "{files}. Only an uploaded CSV can be used. "
            "`reports/DATA_SOURCES.md` explains how to rebuild them.",
            files=", ".join(f"`{path}`" for path in _missing_datasets),
        )
    )
data_source = st.radio(
    t("Data source"),
    _data_source_options,
    index=len(_data_source_options) - 1,
    horizontal=True,
    format_func=t,
    help=t(
        "Latest Real Data Prediction uses the official HARTI, CBSL and DCS records this project has collected. "
        "Upload CSV uses only your file, with a different model."
    ),
)
REAL_MODE = data_source == SOURCE_REAL

history_df = None
live_snapshot: HartiLiveSnapshot | None = None
if REAL_MODE:
    _render_source_status()
else:
    st.header(t("1. Your data"))
if data_source == SOURCE_UPLOAD:
    uploaded_file = st.file_uploader(
        t(
            "CSV with columns: date (YYYY-MM-DD), vegetable, market, price. Optional: rainfall_mm, temperature_c, "
            "humidity_pct, and diesel_price, petrol_price (needed if the loaded model uses fuel prices)."
        ),
        type="csv",
    )
    if uploaded_file is not None:
        try:
            history_df, _upload_notes = prepare_uploaded_history(pd.read_csv(uploaded_file))
        except HistoryInputError as error:
            st.error(str(error))
            st.stop()
        except pd.errors.EmptyDataError:
            st.error(t("That file is empty. It needs a header row and at least one row of data."))
            st.stop()
        except Exception as error:  # noqa: BLE001 -- surfaced directly to the user, not swallowed
            st.error(t("Could not read that file as a comma-separated CSV saved in UTF-8. Details: {error}", error=error))
            st.stop()
        for _note in _upload_notes:
            st.warning(_note)
else:  # SOURCE_REAL: the stored official records, as last imported
    try:
        history_df = _load_real_market_history_cached(_DATA_VERSION)
    except _DATA_READ_ERRORS as error:
        st.error(t("Couldn't load the real datasets: {error}", error=error))
        st.stop()

if history_df is None:
    st.info(t("Upload a CSV to continue."))
    st.stop()

missing_columns = REQUIRED_COLUMNS - set(history_df.columns)
if missing_columns:
    st.error(t("Uploaded data is missing required column(s): {columns}", columns=sorted(missing_columns)))
    st.stop()

# Corrects each row's same-day weather to its vegetable's evidence-based production
# region (config/crop_production_regions.yaml) where one is mapped -- e.g. Carrot
# rows get Nuwara Eliya's weather instead of the market's. A vegetable with no
# sufficiently evidenced region is left unchanged (market-location weather).
_region_weather_path = ConfigLoader(CONFIG_PATH).get("paths.raw_files.weather_data_by_region")
if not _region_weather_path or not Path(_region_weather_path).exists():
    st.warning(
        t(
            "The regional weather file is missing, so this data keeps its own weather rather than each "
            "vegetable's production-region weather, even where the weather details below name a region."
        )
    )
try:
    history_df = apply_production_region_weather(history_df, config_path=CONFIG_PATH)
except (ValueError, KeyError, TypeError, OSError) as error:
    st.warning(
        t(
            "The regional weather file could not be read ({error}), so this data keeps its own weather "
            "rather than each vegetable's production-region weather, even where the weather details below name a region.",
            error=type(error).__name__,
        )
    )

if not REAL_MODE:
    with st.expander(t("Preview loaded data")):
        st.dataframe(history_df.head(20))

# ---------------------------------------------------------------------------
# 2. Model (shown as its own section for uploaded data; inside the prediction for real data)
# ---------------------------------------------------------------------------

if not REAL_MODE:
    st.header(t("2. Model"))

demo_models_dir = resolve_demo_models_dir(CONFIG_PATH)

# Real data always uses the direct forecaster. If it can't run as it was evaluated, this stops and
# says why, and the earlier model is used only if the user explicitly asks for it.
forecaster = None
if data_source in (SOURCE_HISTORICAL, SOURCE_LATEST):
    _problems = forecaster_problems(CONFIG_PATH)
    if _problems:
        st.error(
            t("**The Ridge direct forecaster can't run:**")
            + "\n"
            + "\n".join(f"- {problem}" for problem in _problems)
            + "\n\n"
            + t(
                "Restore the file(s) from a backup. `reports/DATA_SOURCES.md` explains how to rebuild datasets, "
                "and `scripts/train_forecaster.py` recreates the model."
            )
        )
    else:
        forecaster = _load_model_files(lambda: _load_forecaster_version(_DATA_VERSION), "Ridge direct forecaster:", "scripts/train_forecaster.py")
    if forecaster is None and not st.checkbox(
        t("Use the earlier comparison model instead (a different model, with different accuracy)"),
        key="use_earlier_model_for_real_data",
    ):
        st.info(t("No prediction is available from this data source until the forecaster can run."))
        st.stop()
use_direct = forecaster is not None
production_predictor = None if use_direct else _load_model_files(lambda: _load_predictor(CONFIG_PATH), "Earlier comparison model:")
demo_predictor = _load_model_files(lambda: _load_demo_predictor(CONFIG_PATH, demo_models_dir), "Sandbox model:")


def _train_sandbox_model(button_label: str) -> None:
    """Trains on the data loaded above and saves ONLY to demo_models_dir -- never to `models/`,
    regardless of which data source is loaded. See app/services.py::train_demo_model()'s
    docstring for why this function must never be trusted to decide when overwriting the
    real production model (models/best_model.joblib) is safe."""
    if st.button(button_label):
        with st.spinner(t("Training...")):
            try:
                result = train_demo_model(history_df, CONFIG_PATH)
            except Exception as error:  # noqa: BLE001 -- training failures are shown, not swallowed
                st.error(t("Training failed: {error}", error=error))
                st.stop()
        st.success(
            t(
                "Sandbox model trained and saved under `{folder}` (the saved project models are "
                "untouched). Best model: **{model}** ({metrics})",
                folder=demo_models_dir,
                model=result.best_model_name,
                metrics=result.metrics[result.best_model_name],
            )
        )
        _load_demo_predictor.clear()
        st.rerun()


_missing_for_production = (
    _missing_required_columns(production_predictor, history_df) if production_predictor is not None else []
)
if use_direct:
    predictor = None
    _model_used = "Ridge direct forecaster, with a separate model for each day up to 30 days ahead."
    if not REAL_MODE:
        st.success(
            t(
                "Ridge direct forecaster loaded: one model for each day up to {days} days ahead, trained on data "
                "through {date}. See **Model Performance** for measured accuracy.",
                days=MAX_HORIZON_DAYS,
                date=forecaster.trained_through.date(),
            )
        )
elif production_predictor is not None and not _missing_for_production:
    st.success(_model_summary())
    predictor = production_predictor
    _model_used = "Earlier comparison model (51 features), not the Ridge direct forecaster."
    if data_source in (SOURCE_HISTORICAL, SOURCE_LATEST):
        st.warning(
            t(
                "**You chose the earlier comparison model.** The direct forecaster's accuracy on **Model Performance** "
                "doesn't apply to these results."
            )
        )
    else:
        st.info(
            t(
                "**This data uses the earlier comparison model, not the Ridge direct forecaster.** The forecaster needs "
                "other markets' published prices and regional weather on the same dates, which only the project's real "
                "datasets have. The accuracy on **Model Performance** was measured on real Dambulla prices and doesn't "
                "carry over to other data."
            )
        )
    with st.expander(t("Train a sandbox model on the data loaded above")):
        st.caption(t("Saved separately under `{folder}` -- never overwrites the project's saved models.", folder=demo_models_dir))
        _train_sandbox_model(t("Train sandbox model"))
elif production_predictor is not None:
    st.warning(
        t(
            "The earlier comparison model needs {columns}, which this data doesn't have. To "
            "try the app on this data, train a temporary sandbox model (saved under `{folder}`, never over the "
            "project's saved models). Its results describe only the data it was trained on.",
            columns=", ".join(_missing_for_production),
            folder=demo_models_dir,
        )
    )
    _train_sandbox_model(t("Train sandbox model on loaded data"))
    if demo_predictor is None or _missing_required_columns(demo_predictor, history_df):
        st.stop()
    st.info(t(_SANDBOX_IN_USE))
    predictor = demo_predictor
    _model_used = "Temporary sandbox model, from the most recent sandbox training."
else:
    st.warning(
        t(
            "The earlier comparison model isn't available from `models/` (`scripts/train.py` creates it). You can train a "
            "temporary sandbox model on the data loaded above (saved under `{folder}`, never over the project's "
            "saved models) to try the app.",
            folder=demo_models_dir,
        )
    )
    _train_sandbox_model(t("Train sandbox model on loaded data"))
    if demo_predictor is None or _missing_required_columns(demo_predictor, history_df):
        st.stop()
    st.info(t(_SANDBOX_IN_USE))
    predictor = demo_predictor
    _model_used = "Temporary sandbox model, from the most recent sandbox training."

# ---------------------------------------------------------------------------
# 3. Prediction request
# ---------------------------------------------------------------------------

if REAL_MODE:
    st.header(t("2. Latest collected data"))
    _source_tables = _load_source_records(_DATA_VERSION)
    with st.expander(t("Newest HARTI Dambulla price of every vegetable (what a forecast would start from)")):
        _render_records(latest_dambulla_overview(_source_tables["harti_dambulla"]), key="overview")
    # The chosen vegetable's records are drawn here, in section 2, once it is chosen in section 3 below:
    # one selector drives both, so the tables and the prediction always describe the same vegetable.
    _vegetable_records_area = st.container()
    st.header(t("3. Vegetable and market"))
else:
    st.header(t("3. Request a prediction"))

_priced_rows = history_df[history_df["price"].notna()]
if _priced_rows.empty:
    st.error(t("The loaded data has no usable prices."))
    st.stop()
if use_direct:
    # Only series the forecaster was trained on and whose latest price is within its reach: a
    # forecast from an older price would describe dates that have already passed.
    _status = series_status(_priced_rows, forecaster.vegetables)
    _supported = _status[_status["prediction_supported"]]
    _unsupported = _status[~_status["prediction_supported"]]
    if _supported.empty:
        st.error(t("No price series is recent enough to forecast. Run `scripts/update_data.py` to import the latest bulletins."))
        st.stop()
    vegetable = st.selectbox(t("Vegetable"), _supported["series"].tolist(), format_func=display_name, key="predict_vegetable")
    if not _unsupported.empty:
        with st.expander(t("{n} recorded price series can't be forecast, and why", n=len(_unsupported))):
            st.dataframe(
                ui.translated_columns(pd.DataFrame(
                    {
                        "Series": _unsupported["display_name"],
                        "Recorded from": _unsupported["first"].dt.date,
                        "Latest price": _unsupported["latest"].dt.date,
                        "Reason": _unsupported["reason"],
                    }
                )),
                hide_index=True,
                width="stretch",
            )
            st.caption(t("Their recorded prices are still on the **Price History** page."))
else:
    vegetable = st.selectbox(t("Vegetable"), sorted(_priced_rows["vegetable"].unique()), key="predict_vegetable")
if REAL_MODE:
    with _vegetable_records_area:
        st.markdown(t("**Records for {vegetable}** (the vegetable chosen in section 3)", vegetable=display_name(vegetable)))
        _render_vegetable_records(vegetable, _source_tables)
market =st.selectbox(t("Market"), sorted(_priced_rows.loc[_priced_rows["vegetable"] == vegetable, "market"].unique()))
_update_level, _update_message = None, None
if use_direct:
    ui.data_source_panel(
        "HARTI daily bulletins", market, t("Wholesale (Rs/kg)"),
        _priced_rows.loc[_priced_rows["vegetable"] == vegetable, "date"].max(),
        extra=t("Forecasts run from the latest recorded price, up to {days} days ahead.", days=MAX_HORIZON_DAYS),
    )
    # Stored data: when the official sources were last checked, and whether the data may be behind
    # (shown under "Data freshness and warnings").
    _series_rows = _priced_rows[(_priced_rows["vegetable"] == vegetable) & (_priced_rows["market"] == market)].sort_values("date")
    _update_level, _update_message = update_notice(_UPDATE_STATUS, _series_rows["date"].iloc[-1])
    st.caption(
        t(
            "**Latest actual price:** Rs. {price:,.2f} / kg on {date} &nbsp;|&nbsp; **Last successful data update:** ",
            price=_series_rows["price"].iloc[-1],
            date=_series_rows["date"].iloc[-1].date(),
        )
        + (f"{_UPDATE_STATUS.last_success:%Y-%m-%d %H:%M}" if _UPDATE_STATUS.last_success is not None else t("none recorded"))
    )

if use_direct and vegetable not in forecaster.vegetables:
    st.error(t("**{vegetable}** has too little price history for the forecaster to have been trained on it.", vegetable=localized(vegetable)))
    st.stop()
_unseen = [] if use_direct else predictor.unseen_categories(vegetable=vegetable, market=market)
if _unseen:
    st.warning(
        t(
            "This model never saw **{items}** during training. It can still return a number from the "
            "recent prices, but nothing specific to this selection is behind it, so treat it with caution.",
            items=", ".join(_unseen),
        )
    )

live_fuel_snapshot: LiveFuelPriceSnapshot | None = None
if live_snapshot is not None:
    try:
        live_fuel_snapshot = _fetch_live_fuel_price_cached()
    except LiveFuelPriceError as error:
        st.caption(t("Live fuel price unavailable ({error}) -- falling back to the last known fuel price from historical data.", error=error))
    else:
        st.caption(
            t("**Live fuel price (CEYPETCO):** ")
            + ", ".join(
                t("{fuel} Rs.{price:.2f} (effective {date})", fuel=column, price=price, date=live_fuel_snapshot.effective_dates[column].date())
                for column, price in live_fuel_snapshot.prices.items()
            )
        )

    try:
        history_df = _history_with_latest_bulletin(
            vegetable, market, f"{live_snapshot.bulletin_date.date()}|{_DATA_VERSION}", history_df, live_snapshot, live_fuel_snapshot
        )
    except HartiLiveDataError as error:
        st.warning(t("Using historical data only for this selection: {error}", error=error))
        live_snapshot = None

resolved_location = resolve_weather_region(vegetable, config_path=CONFIG_PATH)
_coords_caption = f"({resolved_location.latitude:.4f}, {resolved_location.longitude:.4f})"
if resolved_location.is_production_region:
    st.caption(
        t(
            "**Vegetable:** {vegetable} &nbsp;|&nbsp; **Market:** {market} &nbsp;|&nbsp; "
            "**Weather region:** {region} {coords} (evidence-based "
            "production region, not the market) &nbsp;|&nbsp; **Weather source:** Open-Meteo",
            vegetable=localized(vegetable),
            market=market,
            region=resolved_location.region_name,
            coords=_coords_caption,
        )
    )
else:
    st.caption(
        t(
            "**Vegetable:** {vegetable} &nbsp;|&nbsp; **Market:** {market} &nbsp;|&nbsp; "
            "**Weather region:** {market} {coords} (market location -- no sufficiently "
            "evidenced production region for {vegetable}, falling back to market weather) "
            "&nbsp;|&nbsp; **Weather source:** Open-Meteo",
            vegetable=localized(vegetable),
            market=market,
            coords=_coords_caption,
        )
    )

group_history = history_df[(history_df["vegetable"] == vegetable) & (history_df["market"] == market)]
actual_group_history = group_history.loc[group_history["price"].notna()]
latest_actual_date = actual_group_history["date"].max()
default_target = (latest_actual_date + pd.Timedelta(days=1)).date()

_days_since_reading = (pd.Timestamp.now().normalize() - latest_actual_date).days
_stale_message = (
    t(
        "The latest price for {vegetable} at {market} is from **{date}**, "
        "{days} days ago. A prediction here continues from that date, not from today.",
        vegetable=vegetable,
        market=market,
        date=latest_actual_date.date(),
        days=_days_since_reading,
    )
    if _days_since_reading > _STALE_AFTER_DAYS
    else None
)
if _stale_message and not REAL_MODE:
    st.warning(_stale_message)

_direct_rows = None
if use_direct:
    _latest_markets = live_snapshot.all_markets if live_snapshot is not None else None
    _rows_key = (f"{vegetable}|{live_snapshot.bulletin_date.date()}" if live_snapshot is not None else "historical") + f"|{_DATA_VERSION}"
    _direct_rows = _forecast_origin_rows(_rows_key, history_df, _latest_markets)
    _direct_sources = "HARTI bulletins (Dambulla and other markets), Open-Meteo archive weather, CEYPETCO fuel prices"
    if live_snapshot is not None:
        _direct_sources += f", including the bulletin dated {live_snapshot.bulletin_date.date()} (not real-time)"


def _show_direct_forecast(requested_date, multi_day: bool) -> None:
    """Run and display a direct forecast, reporting a refusal on the page."""
    try:
        with st.spinner(t("Calculating the forecast...")):
            result = direct_forecast(
                forecaster, _direct_rows, vegetable, market, requested_date, latest_actual_date, [_direct_sources]
            )
    except (ForecastInputError, ForecastDataUnavailableError) as error:
        st.error(t("Cannot forecast that date: {error}", error=error))
        return
    _render_direct_result(result, _direct_rows, actual_group_history, resolved_location, _direct_sources, multi_day)


if REAL_MODE and use_direct:
    _render_real_prediction()
    _render_freshness(vegetable, market, latest_actual_date, _update_level, _update_message, _stale_message)
    st.stop()

prediction_type = st.radio(
    t("Prediction type"),
    ["Latest / Next Available Prediction", "Future Date Forecast (multi-step)"],
    horizontal=True,
    format_func=t,
    help=(
        t(
            "Both options use a separate model for each number of days ahead, up to {days}. "
            "'Future Date Forecast' also shows the daily path to the chosen date.",
            days=MAX_HORIZON_DAYS,
        )
        if use_direct
        else t(
            "'Latest / Next Available Prediction' scores one date directly. 'Future Date Forecast' reaches a "
            "distant date by predicting every day in between and feeding each prediction into the next, so errors "
            "can build up -- see README.md section 12."
        )
    ),
)
use_live_weather = use_direct or st.checkbox(
    t("Fetch latest weather from Open-Meteo automatically (recommended)"),
    value=True,
    help=t(
        "If the loaded history's weather doesn't reach far enough back for a required date's "
        "14-day weather lag feature, fetch the missing day(s) from Open-Meteo instead of "
        "failing. Uncheck to predict/forecast using only the weather already in the loaded "
        "history."
    ),
)


if prediction_type == "Latest / Next Available Prediction":
    target_date = st.date_input(
        t("Date to predict"),
        value=default_target,
        min_value=default_target,
    )
    _past = past_date_notice(target_date, latest_actual_date)
    if _past:
        st.warning(_past)
    if not use_direct:
        st.caption(
            t(
                "Predicting further beyond the last historical date needs correspondingly more lag/rolling "
                "history (up to 30 days per config.feature_engineering). If it's missing, you'll see a clear "
                "error naming the exact feature(s) that couldn't be computed -- never a fabricated result. "
                "For a date far enough ahead to need more than that, use **Future Date Forecast** instead."
            )
        )

    if st.button(t("Predict"), type="primary"):
        if use_direct:
            _show_direct_forecast(target_date, multi_day=False)
            st.stop()

        target_ts = pd.Timestamp(target_date)

        # A day with no reading anywhere in the lag/rolling window would otherwise refuse
        # the prediction outright. Same labelled handling as Future Date Forecast below.
        prediction_history, imputed_dates = _history_for_features(
            history_df, vegetable, market, live_fuel_snapshot, use_archive=live_snapshot is not None
        )
        if imputed_dates:
            _severity = st.error if len(imputed_dates) > _MANY_IMPUTED_DAYS else st.warning
            _severity(
                t(
                    "{n} of the last 30 days have no real price for {vegetable} at {market}, so "
                    "estimates from surrounding real prices fill them for the model's features: "
                    "{dates}. Original data is unchanged.",
                    n=len(imputed_dates),
                    vegetable=vegetable,
                    market=market,
                    dates=", ".join(day.date().isoformat() for day in imputed_dates),
                )
            )

        try:
            if use_live_weather:
                input_df = build_live_prediction_input(prediction_history, target_ts, vegetable, market, CONFIG_PATH)
            else:
                input_df = build_prediction_input(prediction_history, target_ts, vegetable, market)
        except LiveWeatherError as error:
            st.error(t("Couldn't get live weather data: {error}", error=error))
            st.stop()

        pipeline = PredictionPipeline(config_path=CONFIG_PATH, predictor=predictor)

        try:
            result = pipeline.predict(input_df, target_date=target_ts, vegetable=vegetable, market=market)
        except FeatureEngineeringError as error:
            st.error(t("Missing required data: {error}", error=error))
        except PredictionInputError as error:
            st.error(t("Invalid input: {error}", error=error))
        except FeatureAlignmentError as error:
            st.error(t("Not enough historical context to predict: {error}", error=error))
        except (ModelArtifactNotFoundError, TransformationArtifactNotFoundError) as error:
            st.error(t("Model not available: {error}", error=error))
        else:
            predicted_price = float(result.loc[0, "predicted_price"])
            with st.container(key="vp_result_next"):
                st.metric(
                    label=t("Predicted price -- {vegetable} at {market} on {date}", vegetable=vegetable, market=market, date=target_date),
                    value=f"Rs. {predicted_price:.2f} / {t('kg')}",
                )

            _weather_lookup_date = target_ts - pd.Timedelta(days=_WEATHER_LAG_DAYS)
            _weather_values = None
            try:
                _engineered_row = pipeline.get_engineered_features(
                    input_df, target_date=target_ts, vegetable=vegetable, market=market
                )
                _weather_values = {
                    t("Rainfall"): f"{_engineered_row[f'rainfall_mm_lag{_WEATHER_LAG_DAYS}d']:.1f} mm",
                    t("Temperature"): f"{_engineered_row[f'temperature_c_lag{_WEATHER_LAG_DAYS}d']:.1f} °C",
                    t("Humidity"): f"{_engineered_row[f'humidity_pct_lag{_WEATHER_LAG_DAYS}d']:.0f} %",
                }
            except (PredictionInputError, KeyError):
                pass  # display the card without values rather than fail the page
            _render_weather_data_card(
                vegetable, market, resolved_location, str(_weather_lookup_date.date()), _weather_values
            )

            if live_snapshot is not None:
                st.caption(t("Data source: **Latest HARTI Data** -- bulletin dated {date}, not real-time.", date=live_snapshot.bulletin_date.date()))
            else:
                st.caption(t("Data source: {source}", source=t(data_source)))
            st.caption(f"**{t('Model used')}:** {t(_model_used)}")

            fig, ax = plt.subplots(figsize=(8, 4.4))
            ax.plot(group_history["date"], group_history["price"], marker="o", markersize=2.5, color=ui.SERIES["actual"], label=t("Historical price"))
            ax.scatter([pd.Timestamp(target_date)], [predicted_price], s=70, color=ui.SERIES["target"], edgecolors=ui.COLORS["background"], zorder=5, label=t("Prediction"))
            ax.set_xlabel(t("Date"))
            ax.set_ylabel(t("Price"))
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

else:  # "Future Date Forecast (multi-step)"
    if use_direct:
        st.caption(
            t(
                "The furthest date available is **{date}**, {days} days after the latest actual price.",
                date=(latest_actual_date + pd.Timedelta(days=MAX_HORIZON_DAYS)).date(),
                days=MAX_HORIZON_DAYS,
            )
        )
    else:
        estimated_max_date = max_supported_forecast_date(latest_actual_date, CONFIG_PATH)
        st.caption(
            t(
                "Estimated furthest forecastable date given the live weather provider's own forward "
                "limit: **~{date}**. The loaded history's own weather coverage can "
                "extend this further; the real limit is only known once you generate the forecast.",
                date=estimated_max_date.date(),
            )
        )
    forecast_date = st.date_input(
        t("Forecast date"),
        value=default_target,
        min_value=default_target,
        key="forecast_date_input",
    )
    forecast_horizon_preview = (pd.Timestamp(forecast_date) - latest_actual_date).days
    st.caption(
        t(
            "Latest actual data: **{date}** &nbsp;|&nbsp; Forecast horizon: **{days} day(s) ahead**",
            date=latest_actual_date.date(),
            days=forecast_horizon_preview,
        )
    )
    _past = past_date_notice(forecast_date, latest_actual_date)
    if _past:
        st.warning(_past)
    elif pd.Timestamp(default_target) < pd.Timestamp.now().normalize():
        st.caption(
            t(
                "The forecast path starts on {date}, the day after the latest actual data; its days before "
                "today have already passed and are forecasts, not recorded prices.",
                date=default_target,
            )
        )

    if forecast_horizon_preview > 7:
        st.warning(
            t(
                "Forecast error grows quickly with distance. See the **Model Performance** page for measured "
                "accuracy at each range before relying on a date this far ahead."
            )
        )

    if st.button(t("Generate Forecast"), type="primary"):
        if use_direct:
            _show_direct_forecast(forecast_date, multi_day=True)
            st.stop()

        live_fuel_snapshot_for_forecast: LiveFuelPriceSnapshot | None = None
        try:
            live_fuel_snapshot_for_forecast = _fetch_live_fuel_price_cached()
        except LiveFuelPriceError as error:
            st.caption(t("Live fuel price unavailable ({error}) -- using historical fuel price fallback.", error=error))

        # Recursive forecasting depends on the ENTIRE trailing lag/rolling window staying
        # dense. Real archived bulletins fill holes first; a date HARTI genuinely has none
        # for is then estimated from real surrounding prices and tagged IMPUTED_HISTORICAL.
        forecast_input_history, imputed_dates_for_forecast = _history_for_features(
            history_df, vegetable, market, live_fuel_snapshot_for_forecast, use_archive=live_snapshot is not None
        )

        pipeline = PredictionPipeline(config_path=CONFIG_PATH, predictor=predictor)

        with st.spinner(t("Forecasting {days} day(s) ahead, recursively...", days=forecast_horizon_preview)):
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
                st.error(t("Invalid forecast request: {error}", error=error))
                st.stop()
            except ForecastDataUnavailableError as error:
                st.error(t("Cannot forecast that far ahead: {error}", error=error))
                st.stop()
            except (ModelArtifactNotFoundError, TransformationArtifactNotFoundError) as error:
                st.error(t("Model not available: {error}", error=error))
                st.stop()

        st.subheader(t("📈 Future Vegetable Price Forecast"))
        st.warning(
            t(
                "⚠ **FORECAST** -- generated recursively from the latest available actual market "
                "data. This is a model estimate, not a guaranteed future price."
            )
        )
        info_cols = st.columns(3)
        info_cols[0].metric(t("Vegetable"), forecast_result.vegetable)
        info_cols[1].metric(t("Market"), forecast_result.market)
        info_cols[2].metric(t("Forecast Horizon"), t("{days} day(s) ahead", days=forecast_result.forecast_horizon_days))
        st.caption(
            t(
                "Latest actual data: **{latest}** &nbsp;|&nbsp; Forecast date: **{date}**",
                latest=forecast_result.latest_actual_date.date(),
                date=forecast_result.target_date.date(),
            )
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
                t("Rainfall"): f"{_engineered_row[f'rainfall_mm_lag{_WEATHER_LAG_DAYS}d']:.1f} mm",
                t("Temperature"): f"{_engineered_row[f'temperature_c_lag{_WEATHER_LAG_DAYS}d']:.1f} °C",
                t("Humidity"): f"{_engineered_row[f'humidity_pct_lag{_WEATHER_LAG_DAYS}d']:.0f} %",
            }
        except (LiveWeatherError, PredictionInputError, KeyError):
            pass  # display the card without values rather than fail the page
        _render_weather_data_card(
            vegetable, market, resolved_location, str(_forecast_weather_lookup_date.date()), _forecast_weather_values
        )
        with st.container(key="vp_result_recursive"):
            st.metric(
                label=t(
                    "Forecasted Price -- {vegetable} at {market} on {date}",
                    vegetable=forecast_result.vegetable,
                    market=forecast_result.market,
                    date=forecast_result.target_date.date(),
                ),
                value=f"Rs. {forecast_result.predicted_price:.2f} / {t('kg')}",
            )
            st.caption(f"**{t('Data source')}:** " + " + ".join(t(source) for source in forecast_result.data_sources))
            st.caption(f"**{t('Model used')}:** {t(_model_used)} {t('Reached by predicting one day at a time.')}")
        for warning_message in forecast_result.warnings:
            st.caption(f"⚠ {warning_message}")

        daily_forecast_df = pd.DataFrame(
            {
                "date": [point.date for point in forecast_result.daily_forecast],
                "forecasted_price": [point.predicted_price for point in forecast_result.daily_forecast],
            }
        )
        with st.expander(t("Daily forecast path")):
            st.dataframe(daily_forecast_df)
            if forecast_result.imputed_historical_dates:
                imputed_mask = forecast_input_history["source"] == IMPUTED_HISTORICAL_SOURCE_LABEL
                imputed_points = forecast_input_history.loc[imputed_mask, ["date", "price"]].sort_values("date")
                st.dataframe(imputed_points.rename(columns={"price": "imputed_estimated_price"}))

            fig, ax = plt.subplots(figsize=(8, 4.4))
            ax.plot(
                actual_group_history["date"], actual_group_history["price"],
                marker="o", markersize=2.5, color=ui.SERIES["actual"], label=t("Historical price (actual)"),
            )
            if forecast_result.imputed_historical_dates:
                imputed_mask = forecast_input_history["source"] == IMPUTED_HISTORICAL_SOURCE_LABEL
                imputed_points = forecast_input_history.loc[imputed_mask].sort_values("date")
                ax.scatter(
                    imputed_points["date"], imputed_points["price"],
                    marker="x", color=ui.SERIES["estimate"], zorder=4,
                    label=t("Imputed historical (estimated -- not real HARTI data)"),
                )
            ax.plot(
                daily_forecast_df["date"], daily_forecast_df["forecasted_price"],
                marker="o", markersize=3, linestyle="--", color=ui.SERIES["forecast"], label=t("Forecast (recursive)"),
            )
            ax.scatter(
                [forecast_result.target_date], [forecast_result.predicted_price],
                s=70, color=ui.SERIES["target"], edgecolors=ui.COLORS["background"], zorder=5, label=t("Requested forecast date"),
            )
            ax.set_xlabel(t("Date"))
            ax.set_ylabel(t("Price"))
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
