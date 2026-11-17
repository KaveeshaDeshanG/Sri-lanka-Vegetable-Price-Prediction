"""Streamlit page: from a cultivation start date to the expected harvest window and the price around it."""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)  # config and data paths are relative to the project root

import altair as alt
import pandas as pd
import streamlit as st

from app import ui
from app.catalogue import NO_DURATION_REASONS, data_freshness, data_version, display_name
from app.cultivation import (
    MIN_SEASONAL_OBSERVATIONS,
    MIN_SEASONAL_YEARS,
    SEASONAL_PAD_DAYS,
    SEASONAL_REFERENCE_YEARS,
    NEUTRAL_CHANGE_PCT,
    CultivationReferenceError,
    CurrentPrice,
    PriceComparison,
    PriceResult,
    build_plan,
    compare_prices,
    current_price,
    latest_start_in_reach,
    load_crop_timings,
    load_error_bands,
    select_price,
)
from app.services import forecaster_problems, historical_dataset_problems, load_real_market_history
from app.source_records import harti_dambulla_records
from config.config_loader import ConfigLoader
from src.data_collection.data_validator import DataValidationError
from src.data_collection.production_region_resolver import ProductionRegionResolver
from src.feature_engineering.forecast_features import ForecastFeatureBuilder
from src.models.direct_forecaster import MAX_HORIZON_DAYS, DirectForecaster

CONFIG_PATH = "config/config.yaml"
REFERENCE_PATH = "data/reference/doa_harvest_durations_v2.csv"
REFERENCE_RETRIEVED = "2026-11-09"
_config = ConfigLoader(CONFIG_PATH)
ERROR_BANDS_PATH = Path(_config.get("paths.metrics_dir", "reports/metrics")) / "forecast_error_bands.json"
OTHER_LOCATION = "Other / not listed"
NO_SERIES = "No HARTI price series for this crop"

st.set_page_config(page_title="Cultivation Planner | Vegetable Price Prediction", page_icon="🌱", layout="wide", initial_sidebar_state="auto")
# The planner uses HARTI Dambulla prices, HORDI timings, and the weather and fuel the forecaster
# reads. CBSL and DCS are not used here, so the sidebar doesn't list them.
ui.apply_theme(page_sources=(ui.SOURCE_HARTI, ui.SOURCE_HORDI, ui.SOURCE_WEATHER, ui.SOURCE_FUEL))
ui.page_header(
    "Cultivation planner",
    "From your cultivation start date to the expected harvest window, the price around it and a suggested selling period.",
    "Plan a crop",
)

# How current each source's data is (app/catalogue.py data_freshness): read from the files and the update log.
_freshness = data_freshness()
with st.expander(ui.freshness_summary(_freshness)):
    ui.data_freshness_panel(_freshness)
    st.caption(
        "Each source publishes on its own schedule, so no figure here is real-time. The automatic update checks every "
        "source every 3 hours and adds anything newly published."
    )


# Every loader is cached per data_version(), so a daily update or a retrained model is picked up
# by a running app on the next interaction.
_DATA_VERSION = data_version()


@st.cache_data
def _history(version: str = _DATA_VERSION) -> pd.DataFrame:
    return load_real_market_history(CONFIG_PATH)


@st.cache_resource
def _forecaster(version: str = _DATA_VERSION) -> DirectForecaster:
    return DirectForecaster.load(CONFIG_PATH)


@st.cache_data(show_spinner="Preparing forecast features...")
def _origin_rows(_history_frame: pd.DataFrame, version: str = _DATA_VERSION) -> pd.DataFrame:
    return ForecastFeatureBuilder(CONFIG_PATH).build_origin_features(_history_frame)


@st.cache_data
def _regional_weather(version: str = _DATA_VERSION) -> pd.DataFrame | None:
    path = _config.get("paths.raw_files.weather_data_by_region")
    try:
        return pd.read_csv(path, parse_dates=["date"]) if path and Path(path).exists() else None
    except (ValueError, OSError):
        return None


@st.cache_data
def _dambulla_records(version: str = _DATA_VERSION) -> pd.DataFrame | None:
    try:
        return harti_dambulla_records(CONFIG_PATH)
    except (ValueError, KeyError, OSError):
        return None


def _model_weather_region(series: str) -> tuple[str, str]:
    """The weather region the forecaster uses for this series: its display text, and its name in the
    regional weather file (the market's own location, Dambulla, when no production region is evidenced)."""
    location = ProductionRegionResolver(config_path=CONFIG_PATH).resolve(series)
    if location.is_production_region:
        return f"{location.region_name} (main production region)", location.region_name
    return "Dambulla (the market's location; no evidenced production region)", "Dambulla"


def _fmt_date(day: pd.Timestamp) -> str:
    return f"{day.day} {day.strftime('%b %Y')}"


# Each kind of price has its own colour, so a historical reference is never mistaken for a forecast.
_BADGE = {
    "model_forecast": "blue",
    "recorded_actual": "green",
    "seasonal_reference": "orange",
    "latest_actual": "gray",
    "unavailable": "red",
}


def _rs(value: float) -> str:
    return f"Rs. {value:,.2f}"


def _price_card(result: PriceResult) -> dict:
    """The plan's price card: the price and, underneath, what kind of price it is."""
    if result.kind == "unavailable":
        return {"label": "Price at harvest", "value": "No verified price", "sub": "no price series for this crop"}
    sub = {
        "model_forecast": f"model forecast for {_fmt_date(result.date)}" if result.date is not None else "model forecast",
        "recorded_actual": "recorded in the harvest window (actual)",
        "seasonal_reference": "same weeks of earlier years: history, not a forecast",
        "latest_actual": f"latest actual price, {result.age_days} day(s) old: not a prediction",
    }[result.kind]
    return {"label": "Price at harvest", "value": f"Rs. {result.price:,.0f} / kg", "sub": sub, "accent": result.kind == "model_forecast"}


def _render_price(result: PriceResult, plan, timing, series: str | None, window_text: str, today: pd.Timestamp) -> None:
    """The price section: one clearly labelled price, where it comes from, and what it is not."""
    with st.container(border=True):
        st.markdown(f"**Price type:** :{_BADGE[result.kind]}-badge[{result.label}]")
        days_ahead = (plan.harvest_start - today).days
        when = (
            f"{days_ahead} day(s) from today" if days_ahead > 0 else ("today" if days_ahead == 0 else f"{-days_ahead} day(s) ago")
        )
        lines = [
            f"- **Vegetable:** {timing.crop}, {timing.variety}"
            + (f" (HARTI price series: {display_name(series)})" if series else ""),
            f"- **{timing.date_label.split(' (')[0]}:** {_fmt_date(plan.start_date)}",
            f"- **Expected harvest:** {window_text} ({when})",
        ]
        if result.kind == "unavailable":
            lines.append("- **Price:** none. No verified price exists, so none is shown (not Rs 0).")
            lines.append(f"- **Missing source:** {PRICE_SOURCE_NOTE}")
        else:
            statistic = {
                "model_forecast": "forecast",
                "recorded_actual": "median of the prices recorded in the window",
                "seasonal_reference": "median of the same weeks in earlier years",
                "latest_actual": "newest recorded price (midpoint of HARTI's minimum and maximum)",
            }[result.kind]
            lines.append(f"- **Price:** **{_rs(result.price)} per kg** ({statistic})")
            if result.low is not None and result.high is not None:
                range_text = {
                    "model_forecast": "the likely range, the middle 80% of past forecast errors for all vegetables together",
                    "recorded_actual": "the lowest to highest recorded in the window",
                    "seasonal_reference": "the lowest to highest recorded in those weeks",
                    "latest_actual": "HARTI's minimum to maximum on that day",
                }[result.kind]
                lines.append(f"- **Range:** {_rs(result.low)} to {_rs(result.high)} per kg: {range_text}")
            if result.kind in ("seasonal_reference", "recorded_actual"):
                lines.append(f"- **Median / average:** {_rs(result.median)} / {_rs(result.mean)} per kg")
            if result.kind == "model_forecast":
                lines.append(
                    f"- **Forecast date and horizon:** {_fmt_date(result.date)}, {result.horizon} day(s) after the latest "
                    f"recorded price ({_fmt_date(plan.latest_price_date)})"
                )
            if result.kind == "latest_actual":
                lines.append(f"- **Observation date:** {_fmt_date(result.date)} ({result.age_days} day(s) before today)")
            if result.kind in ("seasonal_reference", "recorded_actual"):
                lines.append(
                    f"- **Historical period:** {result.period}"
                    if result.kind == "seasonal_reference"
                    else f"- **Period:** {result.period}"
                )
                lines.append(
                    f"- **Records used:** {result.observations} daily price(s), years {', '.join(str(y) for y in result.years)}"
                )
            lines.append(
                f"- **Source:** {result.source} · **Market:** {result.market} · **Price type:** {result.price_type} · "
                f"**Unit:** {result.unit}"
            )
        st.markdown("\n".join(lines))
        st.markdown(f"**What this is:** {result.explanation}")
        if result.limitation:
            st.caption(f"**Limitation:** {result.limitation}")


def _render_comparison(
    current: CurrentPrice, harvest: PriceResult, comparison: PriceComparison, plan, crop: str, today: pd.Timestamp
) -> None:
    """Today's actual price and the harvest-time price side by side, and the change between them."""
    now_meta = [("Vegetable", crop)]
    if current.price is not None:
        now_meta.append(("Price date", f"{_fmt_date(current.date)} ({current.age_days} day(s) ago), latest recorded"))
        if current.low is not None and current.high is not None:
            now_meta.append(("Day's range", f"Rs. {current.low:,.2f} to {current.high:,.2f} (this price is the midpoint)"))
        if current.planting_price is not None:
            now_meta.append(
                ("On your planting date", f"Rs. {current.planting_price:,.2f} (recorded {_fmt_date(current.planting_price_date)})")
            )
    else:
        now_meta.append(("Why", current.reason))
    now_meta.append(("Market", f"{current.market} wholesale, {current.source} (the same for every growing location)"))
    now_card = {
        "tag": "Actual price · now" if current.is_current else ("Last recorded price" if current.price is not None else "No recorded price"),
        "title": "Current market price (planning time)",
        "value": current.price,
        "value_text": "No current price",
        "accent": ui.COLORS["primary"] if current.is_current else ui.COLORS["muted"],
        "meta": now_meta,
    }

    harvest_meta = [("Vegetable", crop)]
    harvest_date = harvest.date if harvest.kind == "model_forecast" and harvest.date is not None else plan.harvest_start
    harvest_meta.append(("Expected harvest", _fmt_date(harvest_date)))
    if plan.latest_price_date is not None and harvest.kind in ("model_forecast", "seasonal_reference"):
        after_latest = (harvest_date - plan.latest_price_date).days
        from_today = (harvest_date - today).days
        horizon = f"{after_latest} days after the latest recorded price ({from_today} days from today)"
        if plan.stale:
            horizon = f"{from_today} days from today; no forecast, as the latest recorded price is too old"
        elif harvest.kind == "seasonal_reference":
            horizon += f": beyond the model's {MAX_HORIZON_DAYS}-day forecast"
        harvest_meta.append(("Horizon", horizon))
    if harvest.kind == "model_forecast":
        if harvest.low is not None and harvest.high is not None:
            harvest_meta.append(("Likely range", f"Rs. {harvest.low:,.2f} to {harvest.high:,.2f}"))
        harvest_meta.append(("Note", "A model forecast, not a guaranteed price"))
        tag, title, accent = "Model forecast", "Expected price at harvest", ui.COLORS["info"]
    elif harvest.kind == "seasonal_reference":
        harvest_meta.append(
            ("Based on", f"{harvest.observations} prices in the same weeks of {harvest.years[0]}–{harvest.years[-1]}")
        )
        harvest_meta.append(("Range", f"Rs. {harvest.low:,.2f} to {harvest.high:,.2f}"))
        tag, title, accent = "Seasonal reference · not a forecast", "Typical price at harvest time", ui.COLORS["forecast"]
    elif harvest.kind == "recorded_actual":
        harvest_meta.append(("Based on", f"{harvest.observations} price(s) recorded in the harvest window"))
        tag, title, accent = "Actual · harvest passed", "Price recorded at harvest", ui.COLORS["secondary"]
    else:
        harvest_meta.append(
            ("Why", "No forecast or seasonal reference covers the harvest dates."
             if harvest.kind == "latest_actual" else "No verified price exists for this crop.")
        )
        tag, title, accent = "No harvest-time estimate", "Expected price at harvest", ui.COLORS["muted"]
    harvest_card = {
        "tag": tag,
        "title": title,
        "value": harvest.price if harvest.kind in ("model_forecast", "seasonal_reference", "recorded_actual") else None,
        "value_text": "No estimate",
        "accent": accent,
        "meta": harvest_meta,
    }

    if comparison.comparable:
        sign = "+" if comparison.difference >= 0 else "−"
        headline = f"{sign}Rs. {abs(comparison.difference):,.2f} / kg ({sign}{abs(comparison.percent):.1f}%)"
        subject = "The forecast price at harvest" if harvest.kind == "model_forecast" else "The typical price at harvest time"
        text = {
            "higher": f"{subject} is HIGHER than today's price.",
            "lower": f"{subject} is LOWER than today's price.",
            "similar": f"{subject} is about the same as today's price (within ±{NEUTRAL_CHANGE_PCT:.0f}%).",
        }[comparison.direction]
        if harvest.kind == "model_forecast":
            note = "A forecast, not a guaranteed price."
            if comparison.within_forecast_range:
                note += " Today's price lies inside the forecast's likely range, so the change is within its normal error."
        else:
            note = (
                "Compared with what the price has been in these weeks in earlier years (history, not a forecast); "
                "earlier years' prices are not adjusted for inflation."
            )
        delta = {"direction": comparison.direction, "headline": headline, "text": text, "note": note}
    else:
        delta = {"direction": None, "headline": "No comparison", "text": comparison.reason}
    ui.price_comparison([now_card, harvest_card], delta)


PRICE_SOURCE_NOTE = (
    "HARTI's daily bulletin doesn't publish this crop, so `data/raw/harti_prices.csv` and the other-market file "
    "`data/raw/harti_all_markets.csv` have no series for it, and the project has no validated mapping to another series."
)


try:
    timings = load_crop_timings(REFERENCE_PATH)
except CultivationReferenceError as error:
    st.error(f"The harvest reference data can't be used: {error} `data/reference/README.md` explains where it comes from.")
    st.stop()

_data_problems = historical_dataset_problems(CONFIG_PATH)
if _data_problems:
    st.error("**The price datasets can't be used:**\n" + "\n".join(f"- {problem}" for problem in _data_problems))
    st.stop()
try:
    history = _history(_DATA_VERSION)
except (DataValidationError, ValueError, KeyError, TypeError, OSError) as error:
    st.error(f"Couldn't load the real datasets: {error}")
    st.stop()
_priced_series = set(history.loc[history["price"].notna(), "vegetable"])

forecaster = origin_rows = None
_model_problems = forecaster_problems(CONFIG_PATH)
if _model_problems:
    st.warning(
        "**Price forecasts are unavailable, so only harvest dates and earlier years' prices are shown:**\n"
        + "\n".join(f"- {problem}" for problem in _model_problems)
    )
else:
    try:
        forecaster = _forecaster(_DATA_VERSION)
        origin_rows = _origin_rows(history, _DATA_VERSION)
    except Exception as error:  # noqa: BLE001 -- a corrupt model file can raise almost anything while unpickling
        forecaster = origin_rows = None
        st.warning(f"Price forecasts are unavailable, because the forecaster couldn't be loaded ({type(error).__name__}: {error}).")
bands = load_error_bands(ERROR_BANDS_PATH)

st.header("1. Your crop")
_crops = list(dict.fromkeys(t.crop for t in timings))
_left, _middle, _right = st.columns(3, gap="medium")
with _left:
    crop = st.selectbox("Vegetable", _crops, key="planner_crop")
_crop_rows = [t for t in timings if t.crop == crop]
with _middle:
    _variety = st.selectbox(
        "Variety or guidance",
        [t.variety for t in _crop_rows],
        key=f"planner_variety_{crop}",
        format_func=lambda name: next(f"{t.variety} — {t.duration_text}" for t in _crop_rows if t.variety == name),
        help="Released varieties have their own published durations; crop guidance applies to the crop in general.",
    )
timing = next(t for t in _crop_rows if t.variety == _variety)
_series_options = [label for label in timing.price_labels if label in _priced_series]
with _right:
    price_label = st.selectbox(
        "Market price series (HARTI, Dambulla wholesale)",
        _series_options or [NO_SERIES],
        key=f"planner_series_{crop}",
        format_func=lambda name: name if name == NO_SERIES else display_name(name),
        help="The HARTI daily wholesale series used for this crop's prices. Some crops have more than one.",
    )
if price_label == NO_SERIES:
    price_label = None

with st.container(border=True):
    st.markdown(f"**Published timing:** {timing.duration_text} to **{timing.event}**, counted from **{timing.date_label.split(' (')[0].lower()}**.")
    st.markdown(f"> {timing.source_text}")
    st.caption(f"Department of Agriculture, HORDI ({timing.scope}), retrieved {REFERENCE_RETRIEVED}: {timing.source_url}")
    st.caption(f"**How the dates are calculated:** {timing.calculation_basis}")
if timing.start_stage == "not stated":
    st.warning(
        "This source doesn't say which date its duration counts from (sowing, planting or transplanting). "
        "Enter the date you consider the start, and treat the harvest dates as approximate."
    )
elif timing.start_stage in ("nursery sowing", "flowering", "seeding", "transplanting", "field planting"):
    st.info(f"Enter the **{timing.date_label.lower()}**. {timing.stage_note}")

_today = pd.Timestamp.now().normalize()
# Say before Generate how far the forecast reaches for this crop, so a farmer planting today isn't
# left to discover after submitting that no forecast is possible.
if price_label:
    _series_latest = history.loc[(history["vegetable"] == price_label) & history["price"].notna(), "date"].max()
    _reach_end = _series_latest + pd.Timedelta(days=MAX_HORIZON_DAYS)
    _stage = timing.date_label.split(" (")[0].lower()
    if (_today - _series_latest).days > MAX_HORIZON_DAYS:
        st.warning(
            f"**No price forecast is possible for {display_name(price_label)}:** its latest HARTI Dambulla price is from "
            f"{_series_latest.date()}, more than {MAX_HORIZON_DAYS} days ago. The plan shows harvest dates and earlier "
            "years' prices only."
        )
    else:
        _last_start = latest_start_in_reach(timing, _reach_end)
        _note = (
            " That date has passed, so a crop started today or later gets harvest dates and earlier years' prices "
            "(history), not a forecast."
            if _last_start < _today
            else " A later date gets harvest dates and earlier years' prices (history), not a forecast."
        )
        st.info(
            f"**Forecast reach:** the price model forecasts up to **{_fmt_date(_reach_end)}**, {MAX_HORIZON_DAYS} days after "
            f"the latest HARTI Dambulla price ({_fmt_date(_series_latest)}). With {timing.duration_text} to {timing.event}, "
            f"a forecast for the harvest needs a {_stage} on or before **{_fmt_date(_last_start)}**.{_note}"
        )
_regions = sorted(_regional_weather(_DATA_VERSION)["region"].unique()) if _regional_weather(_DATA_VERSION) is not None else []
_model_region_text, _model_region = _model_weather_region(price_label) if price_label else (None, None)
with st.form("cultivation_plan"):
    _date_column, _location_column = st.columns(2, gap="medium")
    with _date_column:
        start_date = st.date_input(
            timing.date_label,
            value=_today.date(),
            min_value=(_today - pd.Timedelta(days=730)).date(),
            max_value=(_today + pd.Timedelta(days=365)).date(),
            help=timing.stage_note + " A past date shows a crop already in the ground.",
        )
    with _location_column:
        _location_options = [*_regions, OTHER_LOCATION]
        location = st.selectbox(
            "Weather to display",
            _location_options,
            index=_location_options.index(_model_region) if _model_region in _location_options else 0,
            key=f"planner_weather_{price_label}",
            help=(
                "Only changes the Recent weather panel. It doesn't change the harvest dates (national HORDI guidance) or the "
                "price forecast, which always uses the weather of the vegetable's main production region"
                + (f": {_model_region_text}." if _model_region_text else ".")
            ),
        )
    submitted = st.form_submit_button("Generate Cultivation Plan", type="primary")

with st.expander("All published durations"):
    st.dataframe(
        pd.DataFrame(
            {
                "Crop": [t.crop for t in timings],
                "Variety or guidance": [t.variety for t in timings],
                "Duration": [t.duration_text for t in timings],
                "Counted from": [t.start_stage for t in timings],
                "To": [t.event for t in timings],
                "Source wording": [t.source_text for t in timings],
                "Source": [t.source_url for t in timings],
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Source": st.column_config.LinkColumn("Source")},
    )
    st.caption(
        "Price series with no published harvest duration, so not offered here: "
        + "; ".join(f"**{display_name(name)}**: {reason}" for name, reason in NO_DURATION_REASONS.items())
    )

if not submitted:
    st.info("Choose a vegetable, variety and date, then select **Generate Cultivation Plan**.")
    st.stop()

actual = None
if price_label:
    actual = history.loc[(history["vegetable"] == price_label) & history["price"].notna(), ["date", "price"]].sort_values("date")
with st.spinner("Building the cultivation plan..."):
    try:
        plan = build_plan(timing, price_label, start_date, actual, forecaster, origin_rows, bands, today=_today)
    except ValueError as error:
        st.error(f"Couldn't build the plan: {error}")
        st.stop()

st.header("2. Your cultivation plan")
_is_window = plan.harvest_start != plan.harvest_end
_window = _fmt_date(plan.harvest_start) + (f" – {_fmt_date(plan.harvest_end)}" if _is_window else "")
_window_label = {
    "first harvest": "Expected first harvest",
    "harvesting period": "Expected harvesting period",
    "maturity": "Expected maturity",
    "maturity for consumption": "Expected maturity",
}.get(timing.event, "Expected harvest")
_latest_record = None
if price_label:
    _records = _dambulla_records(_DATA_VERSION)
    if _records is not None:
        _newest = _records[_records["series"] == price_label].sort_values("Observation date").tail(1)
        if not _newest.empty and _newest["Observation date"].iloc[0] == plan.latest_price_date:
            _row = _newest.iloc[0]
            _latest_record = {
                "date": _row["Observation date"],
                "price": _row["Average / midpoint"],
                "low": _row["Minimum"],
                "high": _row["Maximum"],
                "document": _row["Source document"],
            }
price_result = select_price(plan, actual, _today, _latest_record)
_single_day = plan.fully_forecast and plan.window_days == 1
if plan.forecast_days and not plan.fully_forecast:
    _sell = {
        "label": "Suggested selling period",
        "value": "No suggestion",
        "sub": f"the forecast covers {plan.forecast_days} of {plan.window_days} harvest days",
    }
elif plan.best_day and plan.clear_advantage:
    _sell = {"label": "Suggested selling day", "value": _fmt_date(plan.best_day["date"]), "sub": f"forecast Rs. {plan.best_day['price']:.0f} / kg"}
elif _single_day:
    _sell = {"label": "Suggested selling period", "value": "Sell as harvested", "sub": "the published timing gives one harvest day"}
elif plan.best_day:
    _sell = {"label": "Suggested selling period", "value": "Sell as harvested", "sub": "no day in the window is clearly better"}
else:
    _sell = {"label": "Suggested selling period", "value": "Not forecast", "sub": "no forecast covers the harvest dates"}
ui.stat_cards(
    [
        {"label": "Growing duration", "value": timing.duration_text, "sub": f"to {timing.event}, from {timing.start_stage} (published)"},
        {"label": _window_label, "value": _window, "sub": f"counted from {_fmt_date(plan.start_date)}"},
        _price_card(price_result),
        _sell,
    ]
)

st.header("3. Price at harvest")
_current = current_price(actual, _today, _latest_record, plan.start_date, crop=timing.crop)
_render_comparison(
    _current, price_result, compare_prices(_current, price_result), plan, display_name(price_label) if price_label else timing.crop, _today
)
_render_price(price_result, plan, timing, price_label, _window, _today)

if plan.latest_price_date is not None:
    ui.data_source_panel(
        "HARTI daily bulletin", "Dambulla", "Wholesale (Rs/kg)", plan.latest_price_date, _today,
        # A stale series' freshness warning already says no forecast is offered.
        extra="" if plan.stale else f"Forecasts reach {plan.forecast_end.date()}, {MAX_HORIZON_DAYS} days after the latest recorded price.",
    )
    _start = None if _latest_record is None else pd.Series(
        {
            "Average / midpoint": _latest_record["price"],
            "Minimum": _latest_record["low"],
            "Maximum": _latest_record["high"],
            "Source document": _latest_record["document"],
        }
    )
    _heading = (
        "Forecast starting point (used by the model)"
        if plan.forecast is not None
        else "Latest recorded price (actual, not a forecast)"
    )
    if _start is not None and pd.notna(_start["Minimum"]) and pd.notna(_start["Maximum"]):
        _price_text = (
            f"Rs **{_start['Average / midpoint']:,.2f}** per kg (midpoint of {_start['Minimum']:,.2f} to {_start['Maximum']:,.2f})"
        )
    else:
        _price_text = f"Rs **{actual['price'].iloc[-1]:,.2f}** per kg"
    _link = ""
    if _start is not None and pd.notna(_start["Source document"]):
        _link = f" [Bulletin]({str(_start['Source document']).replace(' ', '%20')})"
    with st.container(border=True):
        st.markdown(
            f"**{_heading}:** HARTI daily bulletin, Dambulla, {display_name(price_label)}, observed "
            f"**{plan.latest_price_date:%Y-%m-%d}**: {_price_text}, wholesale.{_link}"
        )
        if plan.forecast is not None:
            st.caption(
                "**What the forecast uses:** recorded HARTI prices (Dambulla and other markets) up to that date, and observed "
                f"Open-Meteo weather for {_model_region_text} up to then. No weather forecast is used. Fuel prices enter "
                "only the 1-day-ahead forecast. CBSL and DCS prices are not used."
            )
if plan.status == "passed":
    st.warning(
        f"This harvest window ended before the latest recorded price ({plan.latest_price_date.date()}), so the crop should "
        "already be harvested. The prices actually recorded in that window are shown below."
    )
elif plan.status == "beyond" and not plan.stale:  # a stale series has its own warning instead
    st.info(
        f"The harvest window starts after {plan.forecast_end.date()}, the furthest date the price model can forecast "
        f"({MAX_HORIZON_DAYS} days after the latest recorded price). No price is predicted for it. Prices from the same weeks "
        "of earlier years are shown instead (section 3 and the table below); they are history, not a prediction."
    )
if plan.best_day and plan.clear_advantage:
    st.success(
        f"The forecast is highest on **{_fmt_date(plan.best_day['date'])}** (Rs. {plan.best_day['price']:.0f} / kg), and even "
        "its low estimate is above the forecast for the first harvest day."
    )
elif _single_day:
    st.info(
        "The published timing gives a single harvest day, so there is no window to compare. Forecasts for the days around "
        "it are in the chart, with their likely range."
    )
elif plan.best_day and plan.fully_forecast:
    st.info(
        "Forecast prices in the harvest window differ by less than the forecast's normal error, so no single day is clearly "
        "better. Selling as the crop is ready is reasonable."
    )
for warning in plan.warnings:
    st.warning(warning)

if actual is not None and not actual.empty:
    st.subheader("Price trend around the harvest")
    recent = actual[actual["date"] >= plan.latest_price_date - pd.Timedelta(days=150)]
    layers = []
    shade = pd.DataFrame({"start": [plan.harvest_start], "end": [plan.harvest_end + pd.Timedelta(days=1)]})
    layers.append(
        alt.Chart(shade).mark_rect(opacity=0.12).encode(x="start:T", x2="end:T", color=alt.value(ui.COLORS["primary"]), tooltip=[alt.Tooltip("start:T", title="Harvest from"), alt.Tooltip("end:T", title="Until")])
    )
    layers.append(
        alt.Chart(recent).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title="Date"), y=alt.Y("price:Q", title="Rs / kg"), color=alt.value(ui.SERIES["actual"]),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("price:Q", title="Recorded Rs/kg", format=".0f")],
        )
    )
    if plan.forecast is not None:
        forecast = plan.forecast
        if forecast["low"].notna().any():
            layers.append(
                alt.Chart(forecast).mark_area(opacity=0.18).encode(x="date:T", y="low:Q", y2="high:Q", color=alt.value(ui.SERIES["forecast"]))
            )
        layers.append(
            alt.Chart(forecast).mark_line(strokeDash=[6, 4], strokeWidth=2.5, point=alt.OverlayMarkDef(color=ui.SERIES["forecast"], size=28)).encode(
                x="date:T", y="price:Q", color=alt.value(ui.SERIES["forecast"]),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("price:Q", title="Forecast Rs/kg", format=".0f"),
                    alt.Tooltip("low:Q", title="Likely low", format=".0f"),
                    alt.Tooltip("high:Q", title="Likely high", format=".0f"),
                ],
            )
        )
    st.altair_chart(alt.layer(*layers).properties(height=340).interactive(bind_y=False), width="stretch")
    st.caption(
        f"Green line: recorded {display_name(price_label)} prices at Dambulla (HARTI, wholesale). "
        + (
            "Dashed amber line: Ridge direct forecast. Shaded: its likely range, the middle 80% of the forecast's errors "
            "on the test period for all vegetables together at that many days ahead "
            f"(not measured for {display_name(price_label)} alone). "
            if plan.forecast is not None
            else "No forecast is drawn: it can't reach these dates. "
        )
        + "Green band: the expected harvest window. Drag or scroll to zoom."
    )

if plan.window_forecast is not None and not plan.window_forecast.empty:
    with st.expander("Forecast for each harvest day"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Date": plan.window_forecast["date"].dt.date,
                    "Days ahead": plan.window_forecast["horizon"],
                    "Forecast (Rs/kg)": plan.window_forecast["price"].round(2),
                    "Likely low": plan.window_forecast["low"].round(0),
                    "Likely high": plan.window_forecast["high"].round(0),
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if (plan.window_forecast["date"] <= _today).any():
            st.caption(
                f"Dates up to today ({_fmt_date(_today)}) are forecasts too: no price has been recorded for them yet. "
                f"All forecasts start from the {plan.latest_price_date:%Y-%m-%d} price."
            )
        if not plan.fully_forecast:
            st.caption(f"The forecast covers {plan.forecast_days} of the {plan.window_days} days in the harvest window.")

if plan.recorded_in_window is not None:
    st.subheader("Prices recorded in the harvest window")
    st.dataframe(
        pd.DataFrame({"Date": plan.recorded_in_window["date"].dt.date, "Recorded (Rs/kg)": plan.recorded_in_window["price"]}),
        hide_index=True,
        width="stretch",
    )
    st.caption("Recorded HARTI Dambulla wholesale prices: actual history, not a forecast.")

if plan.same_season is not None and not plan.same_season.empty:
    st.subheader("Prices in the same weeks of earlier years")
    st.dataframe(plan.same_season.round(0), hide_index=True, width="stretch")
    st.caption(
        f"Recorded {display_name(price_label)} prices at Dambulla (HARTI, wholesale, Rs/kg) in {len(plan.same_season)} earlier "
        f"year(s), for the harvest dates ± {SEASONAL_PAD_DAYS} days. The seasonal reference in section 3 uses the "
        f"{SEASONAL_REFERENCE_YEARS} most recent of these years, and needs at least {MIN_SEASONAL_OBSERVATIONS} prices in "
        f"{MIN_SEASONAL_YEARS} years. This shows past variation. It is history, not a forecast."
    )
elif plan.seasonal is not None:
    st.info(
        f"No {display_name(price_label)} price was recorded in the same weeks (± {SEASONAL_PAD_DAYS} days) of any earlier "
        "year, so there is no seasonal reference."
    )

_source_column, _weather_column = st.columns(2, gap="medium")
with _source_column:
    with st.container(border=True):
        st.markdown("**Where the harvest timing comes from**")
        st.markdown(f"> {timing.source_text}")
        st.caption(
            f"Department of Agriculture, HORDI ({timing.scope}, {timing.variety}), retrieved {REFERENCE_RETRIEVED}: {timing.source_url}. "
            "Published guidance, not a model prediction. Actual timing varies with variety, season and management."
        )
with _weather_column:
    with st.container(border=True):
        st.markdown(f"**Recent weather: {location}**")
        weather = _regional_weather(_DATA_VERSION)
        if location == OTHER_LOCATION or weather is None:
            st.caption("No saved weather for this location. Choose one of the listed locations to see recent conditions.")
        else:
            local = weather[weather["region"] == location].sort_values("date")
            last = local["date"].max()
            month = local[local["date"] > last - pd.Timedelta(days=30)]
            ui.stat_cards(
                [
                    {"label": "Rainfall, last 30 days", "value": f"{month['rainfall_mm'].sum():.0f} mm"},
                    {"label": "Mean temperature", "value": f"{month['temperature_c'].mean():.1f} °C"},
                    {"label": "Mean humidity", "value": f"{month['humidity_pct'].mean():.0f} %"},
                ],
                compact=True,
            )
            st.caption(
                f"Open-Meteo archive, 30 days to {last.date()}. Shown for context only: it doesn't change the harvest dates "
                "or the price forecast."
                + (f" The forecast uses the weather of {_model_region_text}." if _model_region_text and location != _model_region else "")
            )
