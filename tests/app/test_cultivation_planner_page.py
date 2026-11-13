"""The Cultivation Planner page, run through Streamlit's own test harness."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[2] / "app" / "pages" / "4_Cultivation_Planner.py")


def _select(crop: str, variety: str | None = None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    assert not at.exception
    at.selectbox(key="planner_crop").set_value(crop)
    at.run()
    if variety:
        at.selectbox(key=f"planner_variety_{crop}").set_value(variety)
        at.run()
    assert not at.exception
    return at


def _run(crop: str, start: str, variety: str | None = None) -> AppTest:
    at = _select(crop, variety)
    at.date_input[0].set_value(pd.Timestamp(start).date())
    next(b for b in at.button if b.label == "Generate Cultivation Plan").click()
    at.run()
    assert not at.exception
    return at


def _markdown(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_page_waits_for_the_generate_button():
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    assert not at.exception
    assert any("Generate Cultivation Plan" in i.value for i in at.info)
    assert any(b.label == "Generate Cultivation Plan" for b in at.button)


def test_a_harvest_within_reach_shows_the_forecast_price():
    at = _run("Carrot", "2026-07-26")
    text = _markdown(at)
    assert "13 Nov 2026" in text  # 110 days after 26 Jul 2026 (New Kuroda)
    assert "Forecast price at harvest" in text and "Rs." in text
    assert "the published timing gives one harvest day" in text
    assert any(df.value.shape[0] > 0 for df in at.dataframe)


def test_a_harvest_beyond_reach_shows_history_not_a_forecast():
    at = _run("Cabbage", "2026-11-09")
    assert any("history, not a prediction" in i.value for i in at.info)
    assert "Not available" in _markdown(at)


def test_a_source_that_states_no_start_point_is_flagged():
    at = _run("Potato", "2026-08-01")
    assert any("doesn't say" in w.value for w in at.warning)
    assert at.date_input[0].label.startswith("Start date (the source doesn't say")


def test_the_date_asked_for_is_the_sources_own_start_stage():
    assert _select("Capsicum").date_input[0].label == "Nursery sowing date"
    assert _select("Brinjal").date_input[0].label == "Transplanting date"
    assert _select("Beans", "Wade").date_input[0].label == "Seeding (sowing) date"
    assert _select("Pumpkin", "A.N.K.").date_input[0].label == "Flowering date"


def test_choosing_a_variety_changes_the_dates():
    vine = _run("Beans", "2026-09-01", "Vine types (crop guidance)")
    bush = _run("Beans", "2026-09-01", "Bush types (crop guidance)")
    assert "31 Oct 2026" in _markdown(vine)  # 60 days
    assert "16 Oct 2026" in _markdown(bush)  # 45 days


def test_months_are_calendar_months():
    at = _run("Leeks", "2026-01-10")
    assert "25 May 2026" in _markdown(at)  # 4½ months after transplanting


def test_a_crop_with_no_price_series_shows_dates_only():
    at = _run("Cauliflower", "2026-11-09")
    text = _markdown(at)
    assert "8 Jan 2027" in text and "23 Jan 2027" in text  # 60-75 days after transplanting
    assert "no price series for this crop" in text
    assert any("No market price series" in w.value for w in at.warning)
