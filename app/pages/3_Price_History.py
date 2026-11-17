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
from app.i18n import t
from app.catalogue import data_freshness, data_version, display_name, localized
from app.source_records import load_dcs_weekly
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
    # DCS's printed 0.00 ("not available that week") is read as missing, never as Rs 0; the CSV is unchanged.
    table = load_dcs_weekly(CONFIG_PATH)
    if table is None:
        return None
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
        t(
            "Price history is unavailable because of missing file(s): {files}. "
            "`reports/DATA_SOURCES.md` explains how to rebuild them.",
            files=", ".join(f"`{path}`" for path in _missing_datasets),
        )
    )
    st.stop()
try:
    _version = data_version()
    history = _load_history(_version)
    markets = _load_markets(_version)
    dcs = _load_dcs(_version)
    cbsl = _load_cbsl(_version)
except (DataValidationError, ValueError, KeyError, TypeError, OSError) as error:  # a damaged CSV can raise any of these
    st.error(t("Couldn't load the real datasets: {error}", error=error))
    st.stop()

_sources = (
    [SOURCE_HARTI]
    + ([SOURCE_CBSL] if cbsl is not None and not cbsl.empty else [])
    + ([SOURCE_DCS] if dcs is not None and not dcs.empty else [])
)
source = st.radio(t("Data source"), _sources, horizontal=True, key="history_source", format_func=t)

if source == SOURCE_CBSL:
    _filters = st.columns([1, 1, 1], gap="medium")
    with _filters[0]:
        item = st.selectbox(t("Item"), sorted(cbsl["item"].unique()), key="history_cbsl_item")
    group = cbsl[cbsl["item"] == item]
    with _filters[1]:
        price_type = st.radio(
            t("Price type"), sorted(group["price_type"].unique(), reverse=True), horizontal=True, key="history_cbsl_type", format_func=t
        )
    group = group[group["price_type"] == price_type]
    with _filters[2]:
        chosen_markets = st.multiselect(t("Markets"), sorted(group["market"].unique()), default=sorted(group["market"].unique()), key=f"history_cbsl_markets_{price_type}")
    selected = group[group["market"].isin(chosen_markets)].sort_values("date")
    if selected.empty:
        st.info(t("Choose at least one market."))
        st.stop()
    ui.data_source_panel(
        "CBSL Daily Price Report", ", ".join(chosen_markets), f"{t(price_type.capitalize())} ({selected['unit'].iloc[0]})", selected["date"].max(),
        extra=t("Published on working days. Collected by CBSL, separately from HARTI."),
    )
    st.subheader(t("{item}: {price_type} price, CBSL", item=item, price_type=t(price_type)))
    # Only the markets shown appear in the legend; each keeps its own colour whichever are shown.
    _shown = [market for market in ui.MARKET_COLORS if market in set(selected["market"])]
    chart = alt.Chart(selected).mark_line(point=alt.OverlayMarkDef(size=16), strokeWidth=2).encode(
        x=alt.X("date:T", title=t("Report date"), axis=ui.date_axis()),
        y=alt.Y("price:Q", title=t("Rs per {unit}", unit=selected["unit"].iloc[0].split("/")[-1])),
        color=alt.Color("market:N", title=t("Market"), scale=alt.Scale(domain=_shown, range=[ui.MARKET_COLORS[m] for m in _shown]), legend=alt.Legend(orient="top")),
        tooltip=[ui.date_tooltip("date:T", t("Report date")), alt.Tooltip("market:N", title=t("Market")), alt.Tooltip("price:Q", title="Rs", format=",.2f")],
    )
    st.altair_chart(chart.properties(height=320).interactive(bind_y=False), width="stretch")
    st.caption(
        t(
            "Central Bank of Sri Lanka, Daily Price Report: each report's 'Today' column, dated with the report. Wholesale "
            "and retail are separate, and CBSL's prices are collected separately from HARTI's, so the two sources' Dambulla "
            "prices differ and are never combined. Weekends and holidays have no report."
        )
    )
    with st.expander(t("View as a table")):
        st.dataframe(selected[["date", "market", "price_type", "item", "unit", "price", "source_url"]], hide_index=True, width="stretch",
                     column_config={"source_url": st.column_config.LinkColumn(t("Source"))})
    st.stop()

if source == SOURCE_DCS:
    _filters = st.columns([1, 1], gap="medium")
    with _filters[0]:
        vegetable = st.selectbox(t("Vegetable"), sorted(dcs["vegetable"].unique()), key="history_dcs_vegetable", format_func=localized)
    group = dcs[dcs["vegetable"] == vegetable].sort_values("plot_position")
    years = sorted(group["year"].unique())
    with _filters[1]:
        year_range = st.select_slider(t("Years"), options=years, value=(years[0], years[-1]), key="history_dcs_years") if len(years) > 1 else (years[0], years[0])
    selected = group[group["year"].between(year_range[0], year_range[1])].copy()
    # Weeks DCS printed 0.00 ("not available") are missing: not plotted, not in the statistics, and the
    # line breaks there instead of joining across them.
    selected["segment"] = selected["price_unavailable"].cumsum()
    unavailable = selected[selected["price_unavailable"]]
    selected_priced = selected[~selected["price_unavailable"]]
    if selected_priced.empty:
        st.warning(
            t(
                "DCS published no price for {vegetable} in the selected years (every week was printed as unavailable).",
                vegetable=localized(vegetable),
            )
        )
        st.stop()
    latest = selected_priced.iloc[-1]
    ui.stat_cards(
        [
            {"label": t("Data source"), "value": t("DCS weekly retail prices")},
            {"label": t("Market"), "value": t("Colombo District (main markets)")},
            {"label": t("Price type"), "value": t("Retail, weekly average ({unit})", unit=latest["unit"])},
            {"label": t("Latest week"), "value": latest["period"], "accent": True},
        ],
        compact=True,
    )
    columns = st.columns(4)
    columns[0].metric(t("Latest weekly average"), f"Rs. {latest['price']:,.2f}", help=latest["period"])
    columns[1].metric(t("Average"), f"Rs. {selected_priced['price'].mean():,.0f}")
    columns[2].metric(t("Lowest"), f"Rs. {selected_priced['price'].min():,.0f}")
    columns[3].metric(t("Highest"), f"Rs. {selected_priced['price'].max():,.0f}")
    st.subheader(t("{vegetable}: retail price, Colombo District", vegetable=localized(vegetable)))
    chart = alt.Chart(selected_priced).mark_line(point=alt.OverlayMarkDef(size=14), strokeWidth=2).encode(
        x=alt.X("plot_position:T", title=t("Month (weekly points placed by week number)"), axis=ui.date_axis()),
        y=alt.Y("price:Q", title=t("Rs per {unit}", unit=latest["unit"])),
        detail="segment:N",
        color=alt.value(ui.SERIES["actual"]),
        tooltip=[
            alt.Tooltip("period:N", title=t("Published week")),
            alt.Tooltip("price:Q", title=t("Average Rs"), format=",.2f"),
            alt.Tooltip("item:N", title=t("DCS item")),
        ],
    )
    st.altair_chart(chart.properties(height=320).interactive(bind_y=False), width="stretch")
    st.caption(
        t(
            "Department of Census and Statistics weekly retail bulletins: each bulletin's own 'this week' average. "
            "DCS publishes the week only as 'week n of a month', so points are placed by week number, not by a calendar date. "
            "Retail prices: not comparable with HARTI's wholesale prices and never combined with them."
        )
    )
    if not unavailable.empty:
        st.info(
            t(
                "DCS printed no price (0.00) for {vegetable} in {n} week(s): {weeks}. "
                "Those weeks are shown as unavailable, not as Rs 0: they aren't plotted or counted, and the line breaks there.",
                vegetable=localized(vegetable),
                n=len(unavailable),
                weeks=", ".join(unavailable["period"]),
            )
        )
    with st.expander(t("View as a table")):
        table = selected[["period", "item", "unit", "price", "range_low", "range_high", "source_file"]].copy()
        _unavailable_text = t("unavailable (DCS printed 0.00)")
        for column in ("price", "range_low", "range_high"):
            table[column] = table[column].map(lambda value: f"{value:,.2f}" if pd.notna(value) else _unavailable_text)
        st.dataframe(table, hide_index=True, width="stretch")
    st.stop()

_market_list = ["Dambulla"] + (sorted(set(markets["market"]) - {"Dambulla"}) if markets is not None else [])
_filters = st.columns([1, 1, 1], gap="medium")
with _filters[0]:
    market = st.selectbox(t("Market"), _market_list, key="history_market")
if market == "Dambulla":
    series_frame = history[history["price"].notna()][["date", "vegetable", "price", "rainfall_mm", "temperature_c", "humidity_pct", "diesel_price", "petrol_price"]]
else:
    series_frame = markets[markets["market"] == market].rename(columns={"reported": "date"})[["date", "vegetable", "price", "published"]]
with _filters[1]:
    vegetable = st.selectbox(
        t("Vegetable"), sorted(series_frame["vegetable"].unique(), key=display_name), format_func=display_name, key=f"history_vegetable_{market}"
    )
group = series_frame[series_frame["vegetable"] == vegetable].sort_values("date")

first_date, last_date = group["date"].min().date(), group["date"].max().date()
with _filters[2]:
    selected_range = st.date_input(t("Date range"), value=(first_date, last_date), min_value=first_date, max_value=last_date, key=f"history_range_{market}_{vegetable}")
if len(selected_range) != 2:
    st.info(t("Choose an end date to continue."))
    st.stop()
start, end = pd.Timestamp(selected_range[0]), pd.Timestamp(selected_range[1])
selected = group[group["date"].between(start, end)]
if selected.empty:
    st.warning(t("No recorded prices for {vegetable} at {market} in that range. Try a wider one.", vegetable=display_name(vegetable), market=market))
    st.stop()

ui.data_source_panel(
    "HARTI daily bulletins", market, t("Wholesale (Rs/kg)"), group["date"].max(), extra=t("Series runs {first} to {last}.", first=first_date, last=last_date)
)
span_days = (end - start).days + 1
columns = st.columns(4)
columns[0].metric(t("Latest"), f"Rs. {selected['price'].iloc[-1]:,.0f}", help=str(selected["date"].iloc[-1].date()))
columns[1].metric(t("Average"), f"Rs. {selected['price'].mean():,.0f}")
columns[2].metric(t("Lowest"), f"Rs. {selected['price'].min():,.0f}")
columns[3].metric(t("Highest"), f"Rs. {selected['price'].max():,.0f}")

st.subheader(t("{vegetable} wholesale price at {market}", vegetable=display_name(vegetable), market=market))
ui.line_chart(break_long_gaps(selected.set_index("date")["price"]), t("Rs / kg"), ui.SERIES["actual"])
st.caption(
    t(
        "Midpoint of HARTI's published daily range at {market}. {n} recorded days of {span} in this range. "
        "The line breaks where there is no reading for more than {gap} days; shorter gaps are joined.",
        market=market,
        n=len(selected),
        span=span_days,
        gap=MAX_CARRY_DAYS,
    )
    + ("" if market == "Dambulla" else " " + t("Dates are the market's own reporting dates."))
)

if market == "Dambulla":
    location = resolve_weather_region(vegetable, config_path=CONFIG_PATH)
    st.subheader(t("Weather"))
    if location.is_production_region:
        st.caption(t("From **{region}**, this vegetable's main production region.", region=location.region_name))
    else:
        st.caption(t("From Dambulla, because no single production region is evidenced for this vegetable."))
    weather = selected.set_index("date")
    _weather_series = (
        ("rainfall_mm", "Rainfall", "mm"),
        ("temperature_c", "Mean temperature", "°C"),
        ("humidity_pct", "Mean humidity", "%"),
    )
    for tab, (column, _title, unit) in zip(st.tabs([t(title) for _, title, _ in _weather_series]), _weather_series, strict=True):
        with tab:
            ui.line_chart(break_long_gaps(weather[column]), unit, ui.WEATHER_COLORS[column], height=220)

with st.expander(t("View as a table")):
    st.dataframe(selected, hide_index=True, width="stretch")
