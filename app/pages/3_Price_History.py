"""Streamlit page: explore recorded prices: HARTI daily wholesale by market, or DCS weekly retail."""

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
from app.catalogue import data_version, display_name
from app.services import break_long_gaps, load_real_market_history, missing_historical_datasets, resolve_weather_region
from config.config_loader import ConfigLoader
from src.data_collection.data_validator import DataValidationError
from src.feature_engineering.forecast_features import MAX_CARRY_DAYS
from src.preprocessing.transformer import DataTransformer

CONFIG_PATH = "config/config.yaml"
SOURCE_HARTI = "HARTI daily bulletins: wholesale, by market"
SOURCE_DCS = "DCS weekly bulletins: retail, Colombo District"
SOURCE_CBSL = "CBSL daily report: wholesale and retail"
_config = ConfigLoader(CONFIG_PATH)

st.set_page_config(page_title="Price History | Vegetable Price Prediction", page_icon="🗓️", layout="wide", initial_sidebar_state="auto")
ui.apply_theme()
ui.page_header(
    "Price history",
    "Explore recorded prices from the official bulletins. Wholesale and retail, and each market, are kept separate.",
    "Market data",
)


@st.cache_data
def _load_history(version: str) -> pd.DataFrame:
    return load_real_market_history(CONFIG_PATH)


@st.cache_data
def _load_markets(version: str) -> pd.DataFrame | None:
    path = _config.get("paths.raw_files.harti_all_markets")
    if not path or not Path(path).exists():
        return None
    panel = DataTransformer().standardize_vegetable_names(pd.read_csv(path, parse_dates=["published", "reported"]))
    # A market that hasn't reported again can repeat its last reading in the next bulletin: one
    # observation per market, vegetable and reporting date, as first published.
    return panel.sort_values("published").drop_duplicates(["market", "vegetable", "reported"], keep="first")


@st.cache_data
def _load_dcs(version: str) -> pd.DataFrame | None:
    path = _config.get("paths.raw_files.dcs_weekly_retail")
    if not path or not Path(path).exists():
        return None
    table = pd.read_csv(path)
    # A plotting position only: DCS publishes "week n of a month", not dates.
    table["plot_position"] = pd.to_datetime(dict(year=table["year"], month=table["month"], day=1)) + pd.to_timedelta((table["week_of_month"] - 1) * 7, unit="D")
    return table


@st.cache_data
def _load_cbsl(version: str) -> pd.DataFrame | None:
    path = _config.get("paths.raw_files.cbsl_daily_prices")
    if not path or not Path(path).exists():
        return None
    table = pd.read_csv(path, parse_dates=["date"])
    return table[table["vegetable"].notna() & table["price"].notna()]


_missing_datasets = missing_historical_datasets(CONFIG_PATH)
if _missing_datasets:
    st.warning(
        f"Price history is unavailable because of missing file(s): {', '.join(f'`{path}`' for path in _missing_datasets)}. "
        "`reports/DATA_SOURCES.md` explains how to rebuild them."
    )
    st.stop()
try:
    _version = data_version()
    history = _load_history(_version)
    markets = _load_markets(_version)
    dcs = _load_dcs(_version)
    cbsl = _load_cbsl(_version)
except (DataValidationError, ValueError, KeyError, TypeError, OSError) as error:  # a damaged CSV can raise any of these
    st.error(f"Couldn't load the real datasets: {error}")
    st.stop()

_sources = (
    [SOURCE_HARTI]
    + ([SOURCE_CBSL] if cbsl is not None and not cbsl.empty else [])
    + ([SOURCE_DCS] if dcs is not None and not dcs.empty else [])
)
source = st.radio("Data source", _sources, horizontal=True, key="history_source")

if source == SOURCE_CBSL:
    _filters = st.columns([1, 1, 1], gap="medium")
    with _filters[0]:
        item = st.selectbox("Item", sorted(cbsl["item"].unique()), key="history_cbsl_item")
    group = cbsl[cbsl["item"] == item]
    with _filters[1]:
        price_type = st.radio("Price type", sorted(group["price_type"].unique(), reverse=True), horizontal=True, key="history_cbsl_type")
    group = group[group["price_type"] == price_type]
    with _filters[2]:
        chosen_markets = st.multiselect("Markets", sorted(group["market"].unique()), default=sorted(group["market"].unique()), key=f"history_cbsl_markets_{price_type}")
    selected = group[group["market"].isin(chosen_markets)].sort_values("date")
    if selected.empty:
        st.info("Choose at least one market.")
        st.stop()
    ui.data_source_panel(
        "CBSL Daily Price Report", ", ".join(chosen_markets), f"{price_type.capitalize()} ({selected['unit'].iloc[0]})", selected["date"].max(),
        extra="Published on working days. Collected by CBSL, separately from HARTI.",
    )
    st.subheader(f"{item}: {price_type} price, CBSL")
    # Only the markets shown appear in the legend; each keeps its own colour whichever are shown.
    _shown = [market for market in ui.MARKET_COLORS if market in set(selected["market"])]
    chart = alt.Chart(selected).mark_line(point=alt.OverlayMarkDef(size=16), strokeWidth=2).encode(
        x=alt.X("date:T", title="Report date"),
        y=alt.Y("price:Q", title=f"Rs per {selected['unit'].iloc[0].split('/')[-1]}"),
        color=alt.Color("market:N", title="Market", scale=alt.Scale(domain=_shown, range=[ui.MARKET_COLORS[m] for m in _shown]), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("date:T", title="Report date"), alt.Tooltip("market:N"), alt.Tooltip("price:Q", title="Rs", format=",.2f")],
    )
    st.altair_chart(chart.properties(height=320).interactive(bind_y=False), width="stretch")
    st.caption(
        "Central Bank of Sri Lanka, Daily Price Report: each report's 'Today' column, dated with the report. Wholesale "
        "and retail are separate, and CBSL's prices are collected separately from HARTI's, so the two sources' Dambulla "
        "prices differ and are never combined. Weekends and holidays have no report."
    )
    with st.expander("View as a table"):
        st.dataframe(selected[["date", "market", "price_type", "item", "unit", "price", "source_url"]], hide_index=True, width="stretch",
                     column_config={"source_url": st.column_config.LinkColumn("Source")})
    st.stop()

if source == SOURCE_DCS:
    _filters = st.columns([1, 1], gap="medium")
    with _filters[0]:
        vegetable = st.selectbox("Vegetable", sorted(dcs["vegetable"].unique()), key="history_dcs_vegetable")
    group = dcs[dcs["vegetable"] == vegetable].sort_values("plot_position")
    years = sorted(group["year"].unique())
    with _filters[1]:
        year_range = st.select_slider("Years", options=years, value=(years[0], years[-1]), key="history_dcs_years") if len(years) > 1 else (years[0], years[0])
    selected = group[group["year"].between(year_range[0], year_range[1])]
    latest = selected.iloc[-1]
    ui.stat_cards(
        [
            {"label": "Data source", "value": "DCS weekly retail prices"},
            {"label": "Market", "value": "Colombo District (main markets)"},
            {"label": "Price type", "value": f"Retail, weekly average ({latest['unit']})"},
            {"label": "Latest week", "value": latest["period"], "accent": True},
        ],
        compact=True,
    )
    columns = st.columns(4)
    columns[0].metric("Latest weekly average", f"Rs. {latest['price']:,.2f}", help=latest["period"])
    columns[1].metric("Average", f"Rs. {selected['price'].mean():,.0f}")
    columns[2].metric("Lowest", f"Rs. {selected['price'].min():,.0f}")
    columns[3].metric("Highest", f"Rs. {selected['price'].max():,.0f}")
    st.subheader(f"{vegetable}: retail price, Colombo District")
    chart = alt.Chart(selected).mark_line(point=alt.OverlayMarkDef(size=14), strokeWidth=2).encode(
        x=alt.X("plot_position:T", title="Month (weekly points placed by week number)"),
        y=alt.Y("price:Q", title=f"Rs per {latest['unit']}"),
        color=alt.value(ui.SERIES["actual"]),
        tooltip=[alt.Tooltip("period:N", title="Published week"), alt.Tooltip("price:Q", title="Average Rs", format=",.2f"), alt.Tooltip("item:N", title="DCS item")],
    )
    st.altair_chart(chart.properties(height=320).interactive(bind_y=False), width="stretch")
    st.caption(
        "Department of Census and Statistics weekly retail bulletins: each bulletin's own 'this week' average. "
        "DCS publishes the week only as 'week n of a month', so points are placed by week number, not by a calendar date. "
        "Retail prices: not comparable with HARTI's wholesale prices and never combined with them."
    )
    with st.expander("View as a table"):
        st.dataframe(selected[["period", "item", "unit", "price", "range_low", "range_high", "source_file"]], hide_index=True, width="stretch")
    st.stop()

_market_list = ["Dambulla"] + (sorted(set(markets["market"]) - {"Dambulla"}) if markets is not None else [])
_filters = st.columns([1, 1, 1], gap="medium")
with _filters[0]:
    market = st.selectbox("Market", _market_list, key="history_market")
if market == "Dambulla":
    series_frame = history[history["price"].notna()][["date", "vegetable", "price", "rainfall_mm", "temperature_c", "humidity_pct", "diesel_price", "petrol_price"]]
else:
    series_frame = markets[markets["market"] == market].rename(columns={"reported": "date"})[["date", "vegetable", "price", "published"]]
with _filters[1]:
    vegetable = st.selectbox("Vegetable", sorted(series_frame["vegetable"].unique(), key=display_name), format_func=display_name, key=f"history_vegetable_{market}")
group = series_frame[series_frame["vegetable"] == vegetable].sort_values("date")

first_date, last_date = group["date"].min().date(), group["date"].max().date()
with _filters[2]:
    selected_range = st.date_input("Date range", value=(first_date, last_date), min_value=first_date, max_value=last_date, key=f"history_range_{market}_{vegetable}")
if len(selected_range) != 2:
    st.info("Choose an end date to continue.")
    st.stop()
start, end = pd.Timestamp(selected_range[0]), pd.Timestamp(selected_range[1])
selected = group[group["date"].between(start, end)]
if selected.empty:
    st.warning(f"No recorded prices for {display_name(vegetable)} at {market} in that range. Try a wider one.")
    st.stop()

ui.data_source_panel("HARTI daily bulletins", market, "Wholesale (Rs/kg)", group["date"].max(), extra=f"Series runs {first_date} to {last_date}.")
span_days = (end - start).days + 1
columns = st.columns(4)
columns[0].metric("Latest", f"Rs. {selected['price'].iloc[-1]:,.0f}", help=str(selected["date"].iloc[-1].date()))
columns[1].metric("Average", f"Rs. {selected['price'].mean():,.0f}")
columns[2].metric("Lowest", f"Rs. {selected['price'].min():,.0f}")
columns[3].metric("Highest", f"Rs. {selected['price'].max():,.0f}")

st.subheader(f"{display_name(vegetable)} wholesale price at {market}")
st.line_chart(break_long_gaps(selected.set_index("date")["price"]), x_label="Date", y_label="Rs / kg", color=ui.SERIES["actual"])
st.caption(
    f"Midpoint of HARTI's published daily range at {market}. {len(selected)} recorded days of {span_days} in this range. "
    f"The line breaks where there is no reading for more than {MAX_CARRY_DAYS} days; shorter gaps are joined."
    + ("" if market == "Dambulla" else " Dates are the market's own reporting dates.")
)

if market == "Dambulla":
    location = resolve_weather_region(vegetable, config_path=CONFIG_PATH)
    st.subheader("Weather")
    if location.is_production_region:
        st.caption(f"From **{location.region_name}**, this vegetable's main production region.")
    else:
        st.caption("From Dambulla, because no single production region is evidenced for this vegetable.")
    weather = selected.set_index("date")
    _weather_series = (
        ("rainfall_mm", "Rainfall", "mm"),
        ("temperature_c", "Mean temperature", "°C"),
        ("humidity_pct", "Mean humidity", "%"),
    )
    for tab, (column, _title, unit) in zip(st.tabs([title for _, title, _ in _weather_series]), _weather_series, strict=True):
        with tab:
            st.line_chart(break_long_gaps(weather[column]), x_label="Date", y_label=unit, height=220, color=ui.WEATHER_COLORS[column])

with st.expander("View as a table"):
    st.dataframe(selected, hide_index=True, width="stretch")
