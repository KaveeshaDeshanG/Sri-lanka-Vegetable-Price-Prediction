"""Streamlit entry point: project overview, model status and navigation."""

import json
import os
import sys
from pathlib import Path

# Streamlit's runner puts this script's own directory on sys.path, not the
# project root -- add the root explicitly so `import app.services` / `import
# src...` / `import config...` resolve the same way they do under pytest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)  # config and data paths are relative to the project root

import streamlit as st

from app import ui
from app.i18n import t
from app.catalogue import data_freshness, data_version, freshness, last_run_problem, latest_update, load_manifest
from app.data_refresh import run_data_update
from app.services import forecaster_problems, historical_dataset_problems
from config.config_loader import ConfigLoader
from src.models.direct_forecaster import DirectForecaster

CONFIG_PATH = "config/config.yaml"

st.set_page_config(page_title="Sri Lankan Vegetable Price Prediction", page_icon="🥕", layout="wide", initial_sidebar_state="auto")
ui.apply_theme()


@st.cache_resource
def _load_forecaster(version: str) -> DirectForecaster:
    """Cached per data_version(), so a retrained model is shown without restarting the app."""
    return DirectForecaster.load(CONFIG_PATH)


def _saved_accuracy() -> dict | None:
    """The forecaster's saved evaluation report, if it's there and readable."""
    path = Path(ConfigLoader(CONFIG_PATH).get("paths.metrics_dir", "reports/metrics")) / "direct_forecaster_metrics.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return report if isinstance(report, dict) and report.get("bands") else None
    except (OSError, ValueError):
        return None


ui.page_header(
    "Sri Lankan Vegetable Price Prediction",
    "AI-powered Sri Lankan vegetable price prediction system using historical market and weather data.",
    "Dambulla wholesale market · agricultural price analytics",
)

_problems = historical_dataset_problems(CONFIG_PATH) + forecaster_problems(CONFIG_PATH)
_forecaster = None
if not _problems:
    try:
        _forecaster = _load_forecaster(data_version())
    except Exception as error:  # noqa: BLE001 -- a corrupt file can raise almost anything while unpickling
        _problems.append(
            t("The forecaster's model file could not be read, so it may be damaged ({error}).", error=f"{type(error).__name__}: {error}")
        )

if _forecaster is not None:
    st.info(
        t(
            "**Real data:** the Ridge direct forecaster is trained on real HARTI market prices through "
            "{date}, with real weather and fuel-price data "
            "(`reports/DATA_SOURCES.md`). The **Predict** page also offers a small **synthetic** sample dataset for "
            "trying the app; results from it describe synthetic history, not a real market.",
            date=_forecaster.trained_through.date(),
        )
    )
    _report = _saved_accuracy()
    if _report is not None:
        _next_day = _report["bands"][0]
        ui.stat_cards(
            [
                {"label": t("Next-day error"), "value": f"{_next_day['model_mape']:.1f}%", "sub": t("MAE Rs. {mae:.2f} per kg", mae=_next_day["model_mae"])},
                {"label": t("Vs holding the price"), "value": t("{gain:.1f}% lower", gain=_next_day["gain_pct"]), "sub": t("average next-day error"), "accent": True},
                {"label": t("Forecast range"), "value": t("1–{days} days", days=_report["max_horizon_days"]), "sub": t("one Ridge model per day")},
                {"label": t("Vegetables"), "value": str(len(_forecaster.vegetables)), "sub": t("data through {date}", date=_forecaster.trained_through.date())},
            ]
        )
else:
    st.error(
        t("**The Ridge direct forecaster can't run on real data:**")
        + "\n"
        + "\n".join(f"- {problem}" for problem in _problems)
        + "\n\n"
        + t(
            "Restore the file(s) from a backup. `reports/DATA_SOURCES.md` explains how to rebuild datasets, and "
            "`scripts/train_forecaster.py` recreates the model."
        )
    )

st.header(t("Data in use"))
st.subheader(t("Data freshness"))
_problem = last_run_problem(ConfigLoader(CONFIG_PATH).get("paths.raw_data_dir", "data/raw"))
if _problem:
    st.warning(_problem + " " + t("The data shown is what was stored before it; the next scheduled update retries automatically."))
ui.data_freshness_panel(data_freshness(ConfigLoader(CONFIG_PATH).get("paths.raw_data_dir", "data/raw")))
st.caption(
    t(
        "**Latest Available Data**: nothing newer has been published. **Recently Updated**: new data added in the last 24 hours. "
        "**Update Pending**: newer data exists or the source hasn't been checked for over 24 hours. **Source Unavailable**: the last "
        "check failed, so the stored data is shown unchanged. No source is real-time: each publishes on its own schedule."
    )
)
st.subheader(t("Datasets"))
_manifest = load_manifest(ConfigLoader(CONFIG_PATH).get("paths.raw_files.data_manifest", "data/raw/data_manifest.json"))
if _manifest is None:
    st.caption(t("`data/raw/data_manifest.json` is missing; run `scripts/update_data.py` to record where each dataset comes from."))
else:
    _labels = {
        "data/raw/harti_prices.csv": "HARTI daily bulletins: Dambulla wholesale",
        "data/raw/harti_all_markets.csv": "HARTI daily bulletins: every market, wholesale",
        "data/raw/cbsl_daily_prices.csv": "CBSL Daily Price Report: wholesale and retail, kept apart",
        "data/raw/dcs_weekly_retail_prices.csv": "DCS weekly retail prices: Colombo District",
        "data/raw/fuel_prices.csv": "CEYPETCO fuel prices",
        "data/raw/weather_data_by_region.csv": "Open-Meteo weather, 8 locations",
    }
    st.dataframe(
        [
            {
                t("Dataset"): t(_labels.get(entry["file"], entry["file"])),
                t("Price type"): t(entry.get("price_type", "")) if entry.get("price_type") else "",
                t("First"): entry.get("first_observation", ""),
                t("Latest"): entry.get("latest_observation", ""),
                t("Rows"): entry["rows"],
                t("Source"): entry["source_url"],
            }
            for entry in _manifest["datasets"]
        ],
        hide_index=True,
        width="stretch",
        column_config={t("Source"): st.column_config.LinkColumn(t("Source"))},
    )
    _harti = next((e for e in _manifest["datasets"] if e["file"] == "data/raw/harti_prices.csv"), None)
    if _harti:
        _fresh = freshness(_harti["latest_observation"])
        if _fresh.level != "current":
            st.warning(_fresh.message + " " + t("Use **Check for new official data** below (or run `scripts/daily_update.py`) to import newer bulletins."))
    st.caption(
        t(
            "Read from the files themselves when they were last built ({date}). Wholesale and retail prices are separate datasets.",
            date=_manifest["generated_at"][:10],
        )
    )

_update = latest_update(ConfigLoader(CONFIG_PATH).get("paths.raw_data_dir", "data/raw") + "/update_log.jsonl")
if _update is not None:
    _harti = _update.get("harti", {})
    _parts = [
        t("Last daily update **{when}**", when=_update.get("finished_at", "")[:16].replace("T", " ")),
        t("newest HARTI bulletin listed {date}", date=_harti.get("newest_listed", t("unknown"))),
        t(
            "{dambulla} Dambulla and {all_markets} all-market prices added",
            dambulla=_harti.get("dambulla_rows_added", 0),
            all_markets=_harti.get("all_market_rows_added", 0),
        ),
    ]
    if _update.get("cbsl", {}).get("days_added"):
        _parts.append(t("CBSL reports added: {days}", days=", ".join(_update["cbsl"]["days_added"][-3:])))
    st.caption(
        "; ".join(_parts)
        + ". "
        + t(
            "Updates import newly published official bulletins (`scripts/daily_update.py`, run by the scheduled task or the button below); prices are not real-time."
        )
    )
    if _harti.get("unreadable"):
        st.warning(
            t(
                "HARTI bulletin(s) {dates} couldn't be read, so their prices are missing. The bulletin layout may have changed.",
                dates=", ".join(_harti["unreadable"]),
            )
        )

# Runs only when clicked: downloads newly published HARTI, CBSL and DCS bulletins plus weather and
# fuel, appending to the datasets (with a backup first). Pages pick the new files up through data_version().
_refresh = st.session_state.pop("data_refresh_result", None)
if _refresh is not None:
    getattr(st, _refresh.level)(_refresh.message)
    if _refresh.output_tail and _refresh.level != "success":
        with st.expander(t("Update log (last lines)")):
            st.code(_refresh.output_tail, language="text")
if st.button(
    t("Check for new official data"),
    help=t(
        "Downloads any HARTI, CBSL or DCS bulletins published since the last update, plus weather and fuel prices. "
        "Takes about a minute. Existing data is backed up first and never overwritten."
    ),
):
    with st.spinner(t("Checking HARTI, CBSL, DCS, Open-Meteo and CEYPETCO for newly published data...")):
        st.session_state["data_refresh_result"] = run_data_update()
    st.rerun()
_decision_path = Path(ConfigLoader(CONFIG_PATH).get("paths.metrics_dir", "reports/metrics")) / "retrain_decision.json"
try:
    _decision = json.loads(_decision_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    _decision = None
if _decision:
    _verdict = (
        t("accepted and saved")
        if _decision.get("saved")
        else (t("accepted (dry run, not saved)") if _decision.get("accepted") else t("rejected; the saved model was kept"))
    )
    _gain = _decision["overall"]["gain_pct"]
    st.caption(
        t(
            "Last retraining check {date}: {verdict}. Held-out prices {start} to {end}: the candidate's MAE was "
            "{gain:.1f}% {direction} than holding the last price.",
            date=_decision["decided_at"][:10],
            verdict=_verdict,
            start=_decision["holdout_start"],
            end=_decision["holdout_end"],
            gain=abs(_gain),
            direction=t("lower") if _gain >= 0 else t("higher"),
        )
        + (" " + t("Reasons: {reasons}.", reasons="; ".join(_decision["reasons_rejected"])) if _decision.get("reasons_rejected") else "")
    )

_sources = ui.available_sources(CONFIG_PATH)
_has_cbsl = any(name.startswith("CBSL") for name in _sources)
_has_dcs = any(name.startswith("DCS") for name in _sources)
_extra_sources = (t(", CBSL's daily wholesale and retail report") if _has_cbsl else "") + (t(", DCS weekly retail prices") if _has_dcs else "")
_history_views = (
    t("HARTI daily wholesale prices for each market")
    + (t(", CBSL's daily wholesale and retail prices for Pettah, Dambulla and Narahenpita") if _has_cbsl else "")
    + (t(", or DCS weekly retail prices for Colombo District") if _has_dcs else "")
)

st.header(t("How it works"))
ui.step_cards(
    [
        (t("Collect real data"), t("HARTI daily wholesale bulletins (each market separate){extra}, regional weather and fuel prices, updated daily.", extra=_extra_sources)),
        (t("Build features"), t("Only what was published by the forecast date, so nothing leaks from the future.")),
        (t("Forecast directly"), t("One Ridge model per day ahead, from 1 to 30, predicting the change from the last price.")),
        (t("Measure honestly"), t("Tested on later data, against simply holding the last price.")),
    ]
)
st.markdown(
    t(
        """
    Each number of days ahead has its own model, so a forecast uses only real data, not earlier
    predictions, and doesn't need future weather: one Ridge model per day ahead, from 1 to 30.
    Forecasts are for the Dambulla wholesale market.

    - **Predict**: request a next-day price or a forecast up to 30 days ahead. Real data always uses
      the Ridge direct forecaster, and each result names the model behind it. Uploaded and sample
      data use a different model, because the forecaster needs the project's other datasets.
    - **Model Performance**: the forecaster's measured accuracy, compared with holding the last price
      and with the earlier Linear Regression, Random Forest and XGBoost models.
    - **Price History**: explore recorded prices: {history_views}. Each source and price type is kept
      separate.
    - **Cultivation Planner**: choose a crop and variety and enter the start date its published timing
      counts from (sowing, nursery sowing, planting or transplanting). It shows the expected harvest
      window (the Department of Agriculture's published timing, not a prediction), the forecast price
      where the harvest falls within 30 days of the latest price, and a suggested selling period.
    """,
        history_views=_history_views,
    )
)

st.subheader(t("Get started"))
_links = st.columns(4)
with _links[0]:
    st.page_link("pages/1_Predict.py", label=t("Go to Prediction →"), icon="📈")
with _links[1]:
    st.page_link("pages/2_Model_Performance.py", label=t("Go to Model Performance →"), icon="📊")
with _links[2]:
    st.page_link("pages/3_Price_History.py", label=t("Go to Price History →"), icon="🗓️")
with _links[3]:
    st.page_link("pages/4_Cultivation_Planner.py", label=t("Go to Cultivation Planner →"), icon="🌱")
