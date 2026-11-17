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
from app.catalogue import NO_DURATION_REASONS, crop_label, data_freshness, data_version, display_name
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
from app.i18n import current_language, format_date, t
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
        t(
            "Each source publishes on its own schedule, so no figure here is real-time. The automatic update checks every "
            "source every 3 hours and adds anything newly published."
        )
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


@st.cache_data(show_spinner=False)
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
        return t("{region} (main production region)", region=location.region_name), location.region_name
    return t("Dambulla (the market's location; no evidenced production region)"), "Dambulla"


def _fmt_date(day: pd.Timestamp) -> str:
    return format_date(day)


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


def _per_kg(value: str) -> str:
    return t("{price} per kg", price=value)


def _price_card(result: PriceResult) -> dict:
    """The plan's price card: the price and, underneath, what kind of price it is."""
    if result.kind == "unavailable":
        return {"label": t("Price at harvest"), "value": t("No verified price"), "sub": t("no price series for this crop")}
    sub = {
        "model_forecast": t("model forecast for {date}", date=_fmt_date(result.date)) if result.date is not None else t("model forecast"),
        "recorded_actual": t("recorded in the harvest window (actual)"),
        "seasonal_reference": t("same weeks of earlier years: history, not a forecast"),
        "latest_actual": t("latest actual price, {days} day(s) old: not a prediction", days=result.age_days),
    }[result.kind]
    return {"label": t("Price at harvest"), "value": f"Rs. {result.price:,.0f} / {t('kg')}", "sub": sub, "accent": result.kind == "model_forecast"}


def _render_price(result: PriceResult, plan, timing, series: str | None, window_text: str, today: pd.Timestamp) -> None:
    """The price section: one clearly labelled price, where it comes from, and what it is not."""
    with st.container(border=True):
        st.markdown(f"**{t('Price type')}:** :{_BADGE[result.kind]}-badge[{result.label}]")
        days_ahead = (plan.harvest_start - today).days
        when = (
            t("{days} day(s) from today", days=days_ahead)
            if days_ahead > 0
            else (t("today") if days_ahead == 0 else t("{days} day(s) ago", days=-days_ahead))
        )
        lines = [
            f"- **{t('Vegetable')}:** {crop_label(timing.crop)}, {t(timing.variety)}"
            + (" " + t("(HARTI price series: {series})", series=display_name(series)) if series else ""),
            f"- **{timing.date_label.split(' (')[0]}:** {_fmt_date(plan.start_date)}",
            f"- **{t('Expected harvest')}:** {window_text} ({when})",
        ]
        if result.kind == "unavailable":
            lines.append(f"- **{t('Price')}:** " + t("none. No verified price exists, so none is shown (not Rs 0)."))
            lines.append(f"- **{t('Missing source')}:** {t(PRICE_SOURCE_NOTE)}")
        else:
            statistic = {
                "model_forecast": t("forecast"),
                "recorded_actual": t("median of the prices recorded in the window"),
                "seasonal_reference": t("median of the same weeks in earlier years"),
                "latest_actual": t("newest recorded price (midpoint of HARTI's minimum and maximum)"),
            }[result.kind]
            lines.append(f"- **{t('Price')}:** **{_per_kg(_rs(result.price))}** ({statistic})")
            if result.low is not None and result.high is not None:
                range_text = {
                    "model_forecast": t("the likely range, the middle 80% of past forecast errors for all vegetables together"),
                    "recorded_actual": t("the lowest to highest recorded in the window"),
                    "seasonal_reference": t("the lowest to highest recorded in those weeks"),
                    "latest_actual": t("HARTI's minimum to maximum on that day"),
                }[result.kind]
                lines.append(f"- **{t('Range')}:** " + t("{low} to {high} per kg", low=_rs(result.low), high=_rs(result.high)) + f": {range_text}")
            if result.kind in ("seasonal_reference", "recorded_actual"):
                lines.append(f"- **{t('Median / average')}:** {_per_kg(f'{_rs(result.median)} / {_rs(result.mean)}')}")
            if result.kind == "model_forecast":
                lines.append(
                    f"- **{t('Forecast date and horizon')}:** "
                    + t(
                        "{date}, {days} day(s) after the latest recorded price ({latest})",
                        date=_fmt_date(result.date),
                        days=result.horizon,
                        latest=_fmt_date(plan.latest_price_date),
                    )
                )
            if result.kind == "latest_actual":
                lines.append(
                    f"- **{t('Observation date')}:** {_fmt_date(result.date)} (" + t("{days} day(s) before today", days=result.age_days) + ")"
                )
            if result.kind in ("seasonal_reference", "recorded_actual"):
                lines.append(
                    f"- **{t('Historical period')}:** {result.period}"
                    if result.kind == "seasonal_reference"
                    else f"- **{t('Period')}:** {result.period}"
                )
                lines.append(
                    f"- **{t('Records used')}:** "
                    + t("{n} daily price(s), years {years}", n=result.observations, years=", ".join(str(y) for y in result.years))
                )
            lines.append(
                f"- **{t('Source')}:** {result.source} · **{t('Market')}:** {result.market} · **{t('Price type')}:** {t(result.price_type)} · "
                f"**{t('Unit')}:** {result.unit}"
            )
        st.markdown("\n".join(lines))
        st.markdown(f"**{t('What this is')}:** {result.explanation}")
        if result.limitation:
            st.caption(f"**{t('Limitation')}:** {result.limitation}")


def _render_comparison(
    current: CurrentPrice, harvest: PriceResult, comparison: PriceComparison, plan, crop: str, today: pd.Timestamp
) -> None:
    """Today's actual price and the harvest-time price side by side, and the change between them."""
    now_meta = [(t("Vegetable"), crop)]
    if current.price is not None:
        now_meta.append((t("Price date"), t("{date} ({days} day(s) ago), latest recorded", date=_fmt_date(current.date), days=current.age_days)))
        if current.low is not None and current.high is not None:
            now_meta.append(
                (t("Day's range"), t("Rs. {low:,.2f} to {high:,.2f} (this price is the midpoint)", low=current.low, high=current.high))
            )
        if current.planting_price is not None:
            now_meta.append(
                (
                    t("On your planting date"),
                    t("Rs. {price:,.2f} (recorded {date})", price=current.planting_price, date=_fmt_date(current.planting_price_date)),
                )
            )
    else:
        now_meta.append((t("Why"), current.reason))
    now_meta.append(
        (t("Market"), t("{market} wholesale, {source} (the same for every growing location)", market=current.market, source=current.source))
    )
    now_card = {
        "tag": t("Actual price · now") if current.is_current else (t("Last recorded price") if current.price is not None else t("No recorded price")),
        "title": t("Current market price (planning time)"),
        "value": current.price,
        "value_text": t("No current price"),
        "accent": ui.COLORS["primary"] if current.is_current else ui.COLORS["muted"],
        "meta": now_meta,
    }

    harvest_meta = [(t("Vegetable"), crop)]
    harvest_date = harvest.date if harvest.kind == "model_forecast" and harvest.date is not None else plan.harvest_start
    harvest_meta.append((t("Expected harvest"), _fmt_date(harvest_date)))
    if plan.latest_price_date is not None and harvest.kind in ("model_forecast", "seasonal_reference"):
        after_latest = (harvest_date - plan.latest_price_date).days
        from_today = (harvest_date - today).days
        horizon = t("{after} days after the latest recorded price ({today} days from today)", after=after_latest, today=from_today)
        if plan.stale:
            horizon = t("{today} days from today; no forecast, as the latest recorded price is too old", today=from_today)
        elif harvest.kind == "seasonal_reference":
            horizon += t(": beyond the model's {days}-day forecast", days=MAX_HORIZON_DAYS)
        harvest_meta.append((t("Horizon"), horizon))
    if harvest.kind == "model_forecast":
        if harvest.low is not None and harvest.high is not None:
            harvest_meta.append((t("Likely range"), t("Rs. {low:,.2f} to {high:,.2f}", low=harvest.low, high=harvest.high)))
        harvest_meta.append((t("Note"), t("A model forecast, not a guaranteed price")))
        tag, title, accent = t("Model forecast"), t("Expected price at harvest"), ui.COLORS["info"]
    elif harvest.kind == "seasonal_reference":
        harvest_meta.append(
            (t("Based on"), t("{n} prices in the same weeks of {first}–{last}", n=harvest.observations, first=harvest.years[0], last=harvest.years[-1]))
        )
        harvest_meta.append((t("Range"), t("Rs. {low:,.2f} to {high:,.2f}", low=harvest.low, high=harvest.high)))
        tag, title, accent = t("Seasonal reference · not a forecast"), t("Typical price at harvest time"), ui.COLORS["forecast"]
    elif harvest.kind == "recorded_actual":
        harvest_meta.append((t("Based on"), t("{n} price(s) recorded in the harvest window", n=harvest.observations)))
        tag, title, accent = t("Actual · harvest passed"), t("Price recorded at harvest"), ui.COLORS["secondary"]
    else:
        harvest_meta.append(
            (
                t("Why"),
                t("No forecast or seasonal reference covers the harvest dates.")
                if harvest.kind == "latest_actual"
                else t("No verified price exists for this crop."),
            )
        )
        tag, title, accent = t("No harvest-time estimate"), t("Expected price at harvest"), ui.COLORS["muted"]
    harvest_card = {
        "tag": tag,
        "title": title,
        "value": harvest.price if harvest.kind in ("model_forecast", "seasonal_reference", "recorded_actual") else None,
        "value_text": t("No estimate"),
        "accent": accent,
        "meta": harvest_meta,
    }

    if comparison.comparable:
        sign = "+" if comparison.difference >= 0 else "−"
        headline = f"{sign}Rs. {abs(comparison.difference):,.2f} / {t('kg')} ({sign}{abs(comparison.percent):.1f}%)"
        subject = t("The forecast price at harvest") if harvest.kind == "model_forecast" else t("The typical price at harvest time")
        text = {
            "higher": t("{subject} is HIGHER than today's price.", subject=subject),
            "lower": t("{subject} is LOWER than today's price.", subject=subject),
            "similar": t("{subject} is about the same as today's price (within ±{pct:.0f}%).", subject=subject, pct=NEUTRAL_CHANGE_PCT),
        }[comparison.direction]
        if harvest.kind == "model_forecast":
            note = t("A forecast, not a guaranteed price.")
            if comparison.within_forecast_range:
                note += " " + t("Today's price lies inside the forecast's likely range, so the change is within its normal error.")
        else:
            note = t(
                "Compared with what the price has been in these weeks in earlier years (history, not a forecast); "
                "earlier years' prices are not adjusted for inflation."
            )
        delta = {"direction": comparison.direction, "headline": headline, "text": text, "note": note}
    else:
        delta = {"direction": None, "headline": t("No comparison"), "text": comparison.reason}
    ui.price_comparison([now_card, harvest_card], delta)


PRICE_SOURCE_NOTE = (
    "HARTI's daily bulletin doesn't publish this crop, so `data/raw/harti_prices.csv` and the other-market file "
    "`data/raw/harti_all_markets.csv` have no series for it, and the project has no validated mapping to another series."
)


try:
    timings = load_crop_timings(REFERENCE_PATH)
except CultivationReferenceError as error:
    st.error(t("The harvest reference data can't be used: {error} `data/reference/README.md` explains where it comes from.", error=error))
    st.stop()

_data_problems = historical_dataset_problems(CONFIG_PATH)
if _data_problems:
    st.error(t("**The price datasets can't be used:**") + "\n" + "\n".join(f"- {problem}" for problem in _data_problems))
    st.stop()
try:
    history = _history(_DATA_VERSION)
except (DataValidationError, ValueError, KeyError, TypeError, OSError) as error:
    st.error(t("Couldn't load the real datasets: {error}", error=error))
    st.stop()
_priced_series = set(history.loc[history["price"].notna(), "vegetable"])

forecaster = origin_rows = None
_model_problems = forecaster_problems(CONFIG_PATH)
if _model_problems:
    st.warning(
        t("**Price forecasts are unavailable, so only harvest dates and earlier years' prices are shown:**")
        + "\n"
        + "\n".join(f"- {problem}" for problem in _model_problems)
    )
else:
    try:
        forecaster = _forecaster(_DATA_VERSION)
        with st.spinner(t("Preparing forecast features...")):
            origin_rows = _origin_rows(history, _DATA_VERSION)
    except Exception as error:  # noqa: BLE001 -- a corrupt model file can raise almost anything while unpickling
        forecaster = origin_rows = None
        st.warning(
            t("Price forecasts are unavailable, because the forecaster couldn't be loaded ({error}).", error=f"{type(error).__name__}: {error}")
        )
bands = load_error_bands(ERROR_BANDS_PATH)

st.header(t("1. Your crop"))
_crops = list(dict.fromkeys(row.crop for row in timings))
_left, _middle, _right = st.columns(3, gap="medium")
with _left:
    crop = st.selectbox(t("Vegetable"), _crops, key="planner_crop", format_func=crop_label)
_crop_rows = [row for row in timings if row.crop == crop]
with _middle:
    _variety = st.selectbox(
        t("Variety or guidance"),
        [row.variety for row in _crop_rows],
        key=f"planner_variety_{crop}",
        format_func=lambda name: next(f"{t(row.variety)} — {row.duration_text}" for row in _crop_rows if row.variety == name),
        help=t("Released varieties have their own published durations; crop guidance applies to the crop in general."),
    )
timing = next(row for row in _crop_rows if row.variety == _variety)
_series_options = [label for label in timing.price_labels if label in _priced_series]
with _right:
    price_label = st.selectbox(
        t("Market price series (HARTI, Dambulla wholesale)"),
        _series_options or [NO_SERIES],
        key=f"planner_series_{crop}",
        format_func=lambda name: t(name) if name == NO_SERIES else display_name(name),
        help=t("The HARTI daily wholesale series used for this crop's prices. Some crops have more than one."),
    )
if price_label == NO_SERIES:
    price_label = None

with st.container(border=True):
    st.markdown(
        t(
            "**Published timing:** {duration} to **{event}**, counted from **{stage}**.",
            duration=timing.duration_text,
            event=timing.event_text,
            stage=timing.date_label.split(" (")[0].lower(),
        )
    )
    st.markdown(f"> {timing.source_text}")
    if current_language() != "en":
        st.caption(t("Quoted as published by the Department of Agriculture (in English)."))
    st.caption(
        t(
            "Department of Agriculture, HORDI ({scope}), retrieved {retrieved}: {url}",
            scope=t(timing.scope),
            retrieved=REFERENCE_RETRIEVED,
            url=timing.source_url,
        )
    )
    st.caption(f"**{t('How the dates are calculated')}:** {timing.calculation_basis}")
if timing.start_stage == "not stated":
    st.warning(
        t(
            "This source doesn't say which date its duration counts from (sowing, planting or transplanting). "
            "Enter the date you consider the start, and treat the harvest dates as approximate."
        )
    )
elif timing.start_stage in ("nursery sowing", "flowering", "seeding", "transplanting", "field planting"):
    st.info(t("Enter the **{label}**.", label=timing.date_label.lower()) + f" {timing.stage_note}")

_today = pd.Timestamp.now().normalize()
# Say before Generate how far the forecast reaches for this crop, so a farmer planting today isn't
# left to discover after submitting that no forecast is possible.
if price_label:
    _series_latest = history.loc[(history["vegetable"] == price_label) & history["price"].notna(), "date"].max()
    _reach_end = _series_latest + pd.Timedelta(days=MAX_HORIZON_DAYS)
    _stage = timing.date_label.split(" (")[0].lower()
    if (_today - _series_latest).days > MAX_HORIZON_DAYS:
        st.warning(
            t(
                "**No price forecast is possible for {series}:** its latest HARTI Dambulla price is from "
                "{date}, more than {days} days ago. The plan shows harvest dates and earlier "
                "years' prices only.",
                series=display_name(price_label),
                date=_series_latest.date(),
                days=MAX_HORIZON_DAYS,
            )
        )
    else:
        _last_start = latest_start_in_reach(timing, _reach_end)
        _note = (
            t(
                "That date has passed, so a crop started today or later gets harvest dates and earlier years' prices "
                "(history), not a forecast."
            )
            if _last_start < _today
            else t("A later date gets harvest dates and earlier years' prices (history), not a forecast.")
        )
        st.info(
            t(
                "**Forecast reach:** the price model forecasts up to **{reach}**, {days} days after "
                "the latest HARTI Dambulla price ({latest}). With {duration} to {event}, "
                "a forecast for the harvest needs a {stage} on or before **{last_start}**.",
                reach=_fmt_date(_reach_end),
                days=MAX_HORIZON_DAYS,
                latest=_fmt_date(_series_latest),
                duration=timing.duration_text,
                event=timing.event_text,
                stage=_stage,
                last_start=_fmt_date(_last_start),
            )
            + " "
            + _note
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
            help=timing.stage_note + " " + t("A past date shows a crop already in the ground."),
        )
    with _location_column:
        _location_options = [*_regions, OTHER_LOCATION]
        location = st.selectbox(
            t("Weather to display"),
            _location_options,
            index=_location_options.index(_model_region) if _model_region in _location_options else 0,
            key=f"planner_weather_{price_label}",
            format_func=lambda option: t(option) if option == OTHER_LOCATION else option,  # region names stay as stored
            help=(
                t(
                    "Only changes the Recent weather panel. It doesn't change the harvest dates (national HORDI guidance) or the "
                    "price forecast, which always uses the weather of the vegetable's main production region"
                )
                + (f": {_model_region_text}." if _model_region_text else ".")
            ),
        )
    submitted = st.form_submit_button(t("Generate Cultivation Plan"), type="primary")

with st.expander(t("All published durations")):
    st.dataframe(
        ui.translated_columns(
            pd.DataFrame(
                {
                    "Crop": [crop_label(row.crop) for row in timings],
                    "Variety or guidance": [t(row.variety) for row in timings],
                    "Duration": [row.duration_text for row in timings],
                    "Counted from": [row.stage_text for row in timings],
                    "To": [row.event_text for row in timings],
                    "Source wording": [row.source_text for row in timings],
                    "Source": [row.source_url for row in timings],
                }
            )
        ),
        hide_index=True,
        width="stretch",
        column_config={t("Source"): st.column_config.LinkColumn(t("Source"))},
    )
    st.caption(
        t("Price series with no published harvest duration, so not offered here: ")
        + "; ".join(f"**{display_name(name)}**: {t(reason)}" for name, reason in NO_DURATION_REASONS.items())
    )

if not submitted:
    st.info(t("Choose a vegetable, variety and date, then select **Generate Cultivation Plan**."))
    st.stop()

actual = None
if price_label:
    actual = history.loc[(history["vegetable"] == price_label) & history["price"].notna(), ["date", "price"]].sort_values("date")
with st.spinner(t("Building the cultivation plan...")):
    try:
        plan = build_plan(timing, price_label, start_date, actual, forecaster, origin_rows, bands, today=_today)
    except ValueError as error:
        st.error(t("Couldn't build the plan: {error}", error=error))
        st.stop()

st.header(t("2. Your cultivation plan"))
_is_window = plan.harvest_start != plan.harvest_end
_window = _fmt_date(plan.harvest_start) + (f" – {_fmt_date(plan.harvest_end)}" if _is_window else "")
_window_label = {
    "first harvest": t("Expected first harvest"),
    "harvesting period": t("Expected harvesting period"),
    "maturity": t("Expected maturity"),
    "maturity for consumption": t("Expected maturity"),
}.get(timing.event, t("Expected harvest"))
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
        "label": t("Suggested selling period"),
        "value": t("No suggestion"),
        "sub": t("the forecast covers {covered} of {total} harvest days", covered=plan.forecast_days, total=plan.window_days),
    }
elif plan.best_day and plan.clear_advantage:
    _sell = {
        "label": t("Suggested selling day"),
        "value": _fmt_date(plan.best_day["date"]),
        "sub": t("forecast Rs. {price:.0f} / kg", price=plan.best_day["price"]),
    }
elif _single_day:
    _sell = {"label": t("Suggested selling period"), "value": t("Sell as harvested"), "sub": t("the published timing gives one harvest day")}
elif plan.best_day:
    _sell = {"label": t("Suggested selling period"), "value": t("Sell as harvested"), "sub": t("no day in the window is clearly better")}
else:
    _sell = {"label": t("Suggested selling period"), "value": t("Not forecast"), "sub": t("no forecast covers the harvest dates")}
ui.stat_cards(
    [
        {
            "label": t("Growing duration"),
            "value": timing.duration_text,
            "sub": t("to {event}, from {stage} (published)", event=timing.event_text, stage=timing.stage_text),
        },
        {"label": _window_label, "value": _window, "sub": t("counted from {date}", date=_fmt_date(plan.start_date))},
        _price_card(price_result),
        _sell,
    ]
)

st.header(t("3. Price at harvest"))
_current = current_price(actual, _today, _latest_record, plan.start_date, crop=timing.crop)
_render_comparison(
    _current, price_result, compare_prices(_current, price_result), plan, display_name(price_label) if price_label else crop_label(timing.crop), _today
)
_render_price(price_result, plan, timing, price_label, _window, _today)

if plan.latest_price_date is not None:
    ui.data_source_panel(
        "HARTI daily bulletin", "Dambulla", t("Wholesale (Rs/kg)"), plan.latest_price_date, _today,
        # A stale series' freshness warning already says no forecast is offered.
        extra=""
        if plan.stale
        else t("Forecasts reach {date}, {days} days after the latest recorded price.", date=plan.forecast_end.date(), days=MAX_HORIZON_DAYS),
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
        t("Forecast starting point (used by the model)")
        if plan.forecast is not None
        else t("Latest recorded price (actual, not a forecast)")
    )
    if _start is not None and pd.notna(_start["Minimum"]) and pd.notna(_start["Maximum"]):
        _price_text = t(
            "Rs **{price:,.2f}** per kg (midpoint of {low:,.2f} to {high:,.2f})",
            price=_start["Average / midpoint"],
            low=_start["Minimum"],
            high=_start["Maximum"],
        )
    else:
        _price_text = t("Rs **{price:,.2f}** per kg", price=actual["price"].iloc[-1])
    _link = ""
    if _start is not None and pd.notna(_start["Source document"]):
        _link = f" [{t('Bulletin')}]({str(_start['Source document']).replace(' ', '%20')})"
    with st.container(border=True):
        st.markdown(
            t(
                "**{heading}:** HARTI daily bulletin, Dambulla, {series}, observed **{date}**: {price}, wholesale.",
                heading=_heading,
                series=display_name(price_label),
                date=f"{plan.latest_price_date:%Y-%m-%d}",
                price=_price_text,
            )
            + _link
        )
        if plan.forecast is not None:
            st.caption(
                t(
                    "**What the forecast uses:** recorded HARTI prices (Dambulla and other markets) up to that date, and observed "
                    "Open-Meteo weather for {region} up to then. No weather forecast is used. Fuel prices enter "
                    "only the 1-day-ahead forecast. CBSL and DCS prices are not used.",
                    region=_model_region_text,
                )
            )
if plan.status == "passed":
    st.warning(
        t(
            "This harvest window ended before the latest recorded price ({date}), so the crop should "
            "already be harvested. The prices actually recorded in that window are shown below.",
            date=plan.latest_price_date.date(),
        )
    )
elif plan.status == "beyond" and not plan.stale:  # a stale series has its own warning instead
    st.info(
        t(
            "The harvest window starts after {date}, the furthest date the price model can forecast "
            "({days} days after the latest recorded price). No price is predicted for it. Prices from the same weeks "
            "of earlier years are shown instead (section 3 and the table below); they are history, not a prediction.",
            date=plan.forecast_end.date(),
            days=MAX_HORIZON_DAYS,
        )
    )
if plan.best_day and plan.clear_advantage:
    st.success(
        t(
            "The forecast is highest on **{date}** (Rs. {price:.0f} / kg), and even "
            "its low estimate is above the forecast for the first harvest day.",
            date=_fmt_date(plan.best_day["date"]),
            price=plan.best_day["price"],
        )
    )
elif _single_day:
    st.info(
        t(
            "The published timing gives a single harvest day, so there is no window to compare. Forecasts for the days around "
            "it are in the chart, with their likely range."
        )
    )
elif plan.best_day and plan.fully_forecast:
    st.info(
        t(
            "Forecast prices in the harvest window differ by less than the forecast's normal error, so no single day is clearly "
            "better. Selling as the crop is ready is reasonable."
        )
    )
for warning in plan.warnings:
    st.warning(warning)

if actual is not None and not actual.empty:
    st.subheader(t("Price trend around the harvest"))
    recent = actual[actual["date"] >= plan.latest_price_date - pd.Timedelta(days=150)]
    layers = []
    shade = pd.DataFrame({"start": [plan.harvest_start], "end": [plan.harvest_end + pd.Timedelta(days=1)]})
    layers.append(
        alt.Chart(shade).mark_rect(opacity=0.12).encode(
            x="start:T", x2="end:T", color=alt.value(ui.COLORS["primary"]),
            tooltip=[ui.date_tooltip("start:T", t("Harvest from")), ui.date_tooltip("end:T", t("Until"))],
        )
    )
    layers.append(
        alt.Chart(recent).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=t("Date"), axis=ui.date_axis()), y=alt.Y("price:Q", title=t("Rs / kg")), color=alt.value(ui.SERIES["actual"]),
            tooltip=[ui.date_tooltip("date:T", t("Date")), alt.Tooltip("price:Q", title=t("Recorded Rs/kg"), format=".0f")],
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
                    ui.date_tooltip("date:T", t("Date")),
                    alt.Tooltip("price:Q", title=t("Forecast Rs/kg"), format=".0f"),
                    alt.Tooltip("low:Q", title=t("Likely low"), format=".0f"),
                    alt.Tooltip("high:Q", title=t("Likely high"), format=".0f"),
                ],
            )
        )
    st.altair_chart(alt.layer(*layers).properties(height=340).interactive(bind_y=False), width="stretch")
    st.caption(
        t("Green line: recorded {series} prices at Dambulla (HARTI, wholesale).", series=display_name(price_label))
        + " "
        + (
            t(
                "Dashed amber line: Ridge direct forecast. Shaded: its likely range, the middle 80% of the forecast's errors "
                "on the test period for all vegetables together at that many days ahead "
                "(not measured for {series} alone).",
                series=display_name(price_label),
            )
            if plan.forecast is not None
            else t("No forecast is drawn: it can't reach these dates.")
        )
        + " "
        + t("Green band: the expected harvest window. Drag or scroll to zoom.")
    )

if plan.window_forecast is not None and not plan.window_forecast.empty:
    with st.expander(t("Forecast for each harvest day")):
        st.dataframe(
            ui.translated_columns(
                pd.DataFrame(
                    {
                        "Date": plan.window_forecast["date"].dt.date,
                        "Days ahead": plan.window_forecast["horizon"],
                        "Forecast (Rs/kg)": plan.window_forecast["price"].round(2),
                        "Likely low": plan.window_forecast["low"].round(0),
                        "Likely high": plan.window_forecast["high"].round(0),
                    }
                )
            ),
            hide_index=True,
            width="stretch",
        )
        if (plan.window_forecast["date"] <= _today).any():
            st.caption(
                t(
                    "Dates up to today ({today}) are forecasts too: no price has been recorded for them yet. "
                    "All forecasts start from the {latest} price.",
                    today=_fmt_date(_today),
                    latest=f"{plan.latest_price_date:%Y-%m-%d}",
                )
            )
        if not plan.fully_forecast:
            st.caption(t("The forecast covers {covered} of the {total} days in the harvest window.", covered=plan.forecast_days, total=plan.window_days))

if plan.recorded_in_window is not None:
    st.subheader(t("Prices recorded in the harvest window"))
    st.dataframe(
        ui.translated_columns(pd.DataFrame({"Date": plan.recorded_in_window["date"].dt.date, "Recorded (Rs/kg)": plan.recorded_in_window["price"]})),
        hide_index=True,
        width="stretch",
    )
    st.caption(t("Recorded HARTI Dambulla wholesale prices: actual history, not a forecast."))

if plan.same_season is not None and not plan.same_season.empty:
    st.subheader(t("Prices in the same weeks of earlier years"))
    st.dataframe(ui.translated_columns(plan.same_season.round(0)), hide_index=True, width="stretch")
    st.caption(
        t(
            "Recorded {series} prices at Dambulla (HARTI, wholesale, Rs/kg) in {years} earlier "
            "year(s), for the harvest dates ± {pad} days. The seasonal reference in section 3 uses the "
            "{recent} most recent of these years, and needs at least {min_n} prices in "
            "{min_years} years. This shows past variation. It is history, not a forecast.",
            series=display_name(price_label),
            years=len(plan.same_season),
            pad=SEASONAL_PAD_DAYS,
            recent=SEASONAL_REFERENCE_YEARS,
            min_n=MIN_SEASONAL_OBSERVATIONS,
            min_years=MIN_SEASONAL_YEARS,
        )
    )
elif plan.seasonal is not None:
    st.info(
        t(
            "No {series} price was recorded in the same weeks (± {pad} days) of any earlier "
            "year, so there is no seasonal reference.",
            series=display_name(price_label),
            pad=SEASONAL_PAD_DAYS,
        )
    )

_source_column, _weather_column = st.columns(2, gap="medium")
with _source_column, st.container(border=True):
    st.markdown(f"**{t('Where the harvest timing comes from')}**")
    st.markdown(f"> {timing.source_text}")
    st.caption(
        t(
            "Department of Agriculture, HORDI ({scope}, {variety}), retrieved {retrieved}: {url}. "
            "Published guidance, not a model prediction. Actual timing varies with variety, season and management.",
            scope=t(timing.scope),
            variety=t(timing.variety),
            retrieved=REFERENCE_RETRIEVED,
            url=timing.source_url,
        )
    )
with _weather_column, st.container(border=True):
    st.markdown(f"**{t('Recent weather: {location}', location=t(location) if location == OTHER_LOCATION else location)}**")
    weather = _regional_weather(_DATA_VERSION)
    if location == OTHER_LOCATION or weather is None:
        st.caption(t("No saved weather for this location. Choose one of the listed locations to see recent conditions."))
    else:
        local = weather[weather["region"] == location].sort_values("date")
        last = local["date"].max()
        month = local[local["date"] > last - pd.Timedelta(days=30)]
        ui.stat_cards(
            [
                {"label": t("Rainfall, last 30 days"), "value": f"{month['rainfall_mm'].sum():.0f} mm"},
                {"label": t("Mean temperature"), "value": f"{month['temperature_c'].mean():.1f} °C"},
                {"label": t("Mean humidity"), "value": f"{month['humidity_pct'].mean():.0f} %"},
            ],
            compact=True,
        )
        st.caption(
            t(
                "Open-Meteo archive, 30 days to {date}. Shown for context only: it doesn't change the harvest dates "
                "or the price forecast.",
                date=last.date(),
            )
            + (" " + t("The forecast uses the weather of {region}.", region=_model_region_text) if _model_region_text and location != _model_region else "")
        )
