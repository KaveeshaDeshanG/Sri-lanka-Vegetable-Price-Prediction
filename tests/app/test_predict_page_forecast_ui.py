"""The Predict page through Streamlit's own AppTest harness: two data sources, and the five
sections of Latest Real Data Prediction (source status, collected records, vegetable and market,
prediction, freshness). Runs against the project's real datasets and saved forecaster.

Replaces the earlier tests of the four-option page ("Historical dataset (real)" with its
"Prediction type" radio), which no longer exists for real data.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import catalogue

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "app" / "pages" / "1_Predict.py")
pytestmark = pytest.mark.skipif(not Path("models/direct_forecaster.joblib").exists(), reason="forecaster not built")


def _page() -> AppTest:
    at = AppTest.from_file(PAGE_PATH, default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def test_only_two_data_sources_and_real_data_is_the_default():
    at = _page()
    source = next(r for r in at.radio if r.label == "Data source")
    assert list(source.options) == ["Upload CSV", "Latest Real Data Prediction"]
    assert source.value == "Latest Real Data Prediction"
    headers = [h.value for h in at.header]
    assert headers == [
        "1. Official source status",
        "2. Latest collected data",
        "3. Vegetable and market",
        "4. Prediction",
        "5. Data freshness and warnings",
    ]


def test_forecast_periods_are_1_7_14_and_30_days():
    at = _page()
    period = next(r for r in at.radio if r.label == "Forecast period")
    assert list(period.options)[:4] == ["1 day", "7 days", "14 days", "30 days"]
    assert period.value == "1 day"
    assert any(b.label == "Predict" for b in at.button)
    assert "CBSL and DCS records are not inputs" in _text(at)


def test_beans_forecast_equals_the_saved_forecaster_and_names_its_starting_point():
    from app.forecasting import direct_forecast
    from app.services import load_real_market_history
    from src.feature_engineering.forecast_features import ForecastFeatureBuilder
    from src.models.direct_forecaster import DirectForecaster

    history = load_real_market_history("config/config.yaml")
    beans = history[(history["vegetable"] == "Beans") & history["price"].notna()].sort_values("date")
    latest = beans["date"].iloc[-1]
    rows = ForecastFeatureBuilder("config/config.yaml").build_origin_features(history)
    expected = direct_forecast(DirectForecaster.load(), rows, "Beans", "Dambulla", latest + pd.Timedelta(days=7), latest, ["HARTI"])

    at = _page()
    next(s for s in at.selectbox if s.label == "Vegetable").set_value("Beans")
    at.run()
    next(r for r in at.radio if r.label == "Forecast period").set_value("7 days")
    at.run()
    next(b for b in at.button if b.label == "Predict").click()
    at.run()
    assert not at.exception, at.exception

    shown = [m.value for m in at.metric]
    assert f"Rs. {expected.predicted_price:.2f} / kg" in shown
    text = _text(at)
    assert f"observed **{latest:%Y-%m-%d}**" in text and f"Rs **{beans['price'].iloc[-1]:,.2f}** per kg" in text
    assert "Forecasts at 1, 7, 14 and 30 days" in text


def test_an_unreachable_harti_is_shown_as_failed_not_as_current(monkeypatch):
    real = catalogue.update_status

    def harti_down(*args, **kwargs):
        status = real(*args, **kwargs)
        sources = {**status.sources, "harti": {"status": "failed", "checked_at": "2026-11-15T13:00:00", "error": "ConnectionError: www.harti.gov.lk unreachable"}}
        return catalogue.UpdateStatus(status.last_success, status.newest_listed, False, None, False, sources)

    monkeypatch.setattr(catalogue, "update_status", harti_down)
    at = _page()
    assert any("Couldn't be checked" in e.value for e in at.error)
    assert any("HARTI couldn't be checked" in w.value for w in at.warning)


def _in_order(at) -> list[tuple[str, str]]:
    """The page's headers, Vegetable selector and record headings, top to bottom."""
    found = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            if child.type == "header":
                found.append(("header", child.value))
            elif child.type == "selectbox":
                found.append(("selectbox", child.label))
            elif child.type == "markdown" and str(child.value).startswith("**Records for"):
                found.append(("records", child.value))
            walk(child)

    walk(at.main)
    return found


def test_the_vegetable_is_chosen_in_section_3_and_drives_the_section_2_records():
    at = _page()
    order = _in_order(at)
    position = {item: index for index, item in enumerate(order)}
    vegetable = position[("selectbox", "Vegetable")]
    assert position[("header", "3. Vegetable and market")] < vegetable < position[("header", "4. Prediction")]
    records = next(index for index, item in enumerate(order) if item[0] == "records")
    assert position[("header", "2. Latest collected data")] < records < position[("header", "3. Vegetable and market")]

    next(s for s in at.selectbox if s.label == "Vegetable").set_value("Carrot")
    at.run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "**Records for Carrot**" in text and "Dambulla, Carrot, observed" in text
    assert "**Vegetable:** Carrot" in text  # the prediction's own data panel names the same vegetable


def _dcs_week_aged(low: int, high: int) -> tuple[str, int]:
    """A real DCS-style week label ('YYYY-MM week n', week starting day 1, 8, 15 or 22) aged low..high days."""
    today = pd.Timestamp.now().normalize()
    for age in range(low, high + 1):
        start = today - pd.Timedelta(days=age)
        if start.day in (1, 8, 15, 22):
            return f"{start:%Y-%m} week {(start.day - 1) // 7 + 1}", age
    raise AssertionError("no week start in range")


@pytest.mark.parametrize("ages, stale", [((12, 21), False), ((22, 31), True)])
def test_dcs_is_stale_only_after_three_weekly_cycles(monkeypatch, ages, stale):
    label, age = _dcs_week_aged(*ages)
    real = catalogue.update_status

    def with_dcs_week(*args, **kwargs):
        status = real(*args, **kwargs)
        dcs = {"status": "ok", "checked_at": "2026-11-15T16:00:00", "newest_published": label, "new_records": 0, "problems": []}
        return catalogue.UpdateStatus(status.last_success, status.newest_listed, False, None, False, {**status.sources, "dcs": dcs})

    monkeypatch.setattr(catalogue, "update_status", with_dcs_week)
    at = _page()
    text = _text(at)
    assert f"{label} ({age} day(s) ago)" in text
    assert (f"the newest DCS publication is {age} days old" in text) is stale


def test_upload_csv_waits_for_a_file_without_using_official_data():
    at = _page()
    next(r for r in at.radio if r.label == "Data source").set_value("Upload CSV")
    at.run()
    assert not at.exception
    assert any("Upload a CSV to continue" in i.value for i in at.info)
    assert [h.value for h in at.header] == ["1. Your data"]
