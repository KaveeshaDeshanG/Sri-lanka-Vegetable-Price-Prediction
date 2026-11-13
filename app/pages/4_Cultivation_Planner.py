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
from app.catalogue import NO_DURATION_REASONS, data_version, display_name
from app.cultivation import CultivationReferenceError, build_plan, load_crop_timings, load_error_bands
from app.services import forecaster_problems, historical_dataset_problems, load_real_market_history
from config.config_loader import ConfigLoader
from src.data_collection.data_validator import DataValidationError
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
ui.apply_theme()
ui.page_header(
    "Cultivation planner",
    "From your cultivation start date to the expected harvest window, the price around it and a suggested selling period.",
    "Plan a crop",
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


def _fmt_date(day: pd.Timestamp) -> str:
    return f"{day.day} {day.strftime('%b %Y')}"


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
_regions = sorted(_regional_weather(_DATA_VERSION)["region"].unique()) if _regional_weather(_DATA_VERSION) is not None else []
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
        location = st.selectbox(
            "Growing location", [*_regions, OTHER_LOCATION],
            help="Used to show recent weather. The published durations are national guidance and don't vary by location.",
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
if plan.window_forecast is not None and not plan.window_forecast.empty:
    _first = plan.window_forecast.iloc[0]
    _price_card = {
        "label": "Forecast price at harvest",
        "value": f"Rs. {_first['price']:.0f} / kg",
        "sub": (f"likely Rs. {_first['low']:.0f}–{_first['high']:.0f} on {_fmt_date(_first['date'])}" if pd.notna(_first["low"]) else f"on {_fmt_date(_first['date'])}"),
        "accent": True,
    }
else:
    if plan.status == "no_prices":
        _reason = "no price series for this crop"
    elif plan.stale:
        _reason = "latest recorded price is too old"
    elif plan.status == "passed":
        _reason = "harvest window has passed"
    elif forecaster is None:
        _reason = "the forecaster isn't available"
    else:
        _reason = f"outside the forecast's {MAX_HORIZON_DAYS}-day reach"
    _price_card = {"label": "Forecast price at harvest", "value": "Not available", "sub": _reason}
_single_day = plan.window_forecast is not None and len(plan.window_forecast) == 1
if plan.best_day and plan.clear_advantage:
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
        _price_card,
        _sell,
    ]
)

if plan.latest_price_date is not None:
    ui.data_source_panel(
        "HARTI daily bulletin", "Dambulla", "Wholesale (Rs/kg)", plan.latest_price_date, _today,
        extra=f"Forecasts reach {plan.forecast_end.date()}, {MAX_HORIZON_DAYS} days after the latest recorded price.",
    )
if plan.status == "passed":
    st.warning(
        f"This harvest window ended before the latest recorded price ({plan.latest_price_date.date()}), so the crop should "
        "already be harvested. The prices actually recorded in that window are shown below."
    )
elif plan.status == "beyond":
    st.info(
        f"The harvest window starts after {plan.forecast_end.date()}, the furthest date the price model can forecast "
        f"({MAX_HORIZON_DAYS} days after the latest recorded price). No price is predicted for it. Prices from the same weeks "
        "of earlier years are shown instead; they are history, not a prediction."
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
elif plan.best_day:
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
        + ("Dashed amber line: Ridge direct forecast, with its likely range (the middle 80% of test-period errors) shaded. " if plan.forecast is not None else "No forecast is drawn: it can't reach these dates. ")
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
        f"Recorded {display_name(price_label)} prices at Dambulla in {len(plan.same_season)} earlier year(s). "
        "This shows past variation. It is history, not a forecast."
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
            st.caption(f"Open-Meteo archive, 30 days to {last.date()}. Shown for context; it doesn't change the harvest dates.")
