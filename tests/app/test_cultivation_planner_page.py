"""The Cultivation Planner page, run through Streamlit's own test harness."""

from pathlib import Path

import pandas as pd
import pytest
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


def _run(crop: str, start: str, variety: str | None = None, weather: str | None = None, series: str | None = None) -> AppTest:
    at = _select(crop, variety)
    if series:
        at.selectbox(key=f"planner_series_{crop}").set_value(series)
        at.run()
    if weather:
        next(s for s in at.selectbox if s.label == "Weather to display").set_value(weather)
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


def _carrot_start(days_after_latest_price: int = 7) -> str:
    """A New Kuroda carrot planting date (110 days to harvest) whose harvest falls this many days after
    the latest recorded price, so the test stays within the forecast's reach as new data arrives."""
    return str((_latest("Carrot") + pd.Timedelta(days=days_after_latest_price - 110)).date())


def test_a_harvest_within_reach_shows_the_forecast_price():
    start = _carrot_start()
    at = _run("Carrot", start)
    text = _markdown(at)
    harvest = pd.Timestamp(start) + pd.Timedelta(days=110)
    assert _day(harvest) in text  # 110 days after planting (New Kuroda)
    assert "Price at harvest" in text and "Rs." in text and ":blue-badge[Model Forecast]" in text
    assert "the published timing gives one harvest day" in text
    assert any(df.value.shape[0] > 0 for df in at.dataframe)


def test_a_harvest_beyond_reach_shows_history_not_a_forecast():
    at = _run("Cabbage", "2026-11-09")
    assert any("history, not a prediction" in i.value for i in at.info)
    text = _markdown(at)
    assert ":orange-badge[Historical Seasonal Price Reference – Not a Forecast]" in text
    assert "Model Forecast" not in text and "Records used:" in text


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


# --- fixes from the Cultivation Planner audit, 2026-11-15 -----------------------------------------


def _latest(series: str) -> pd.Timestamp:
    from app.services import load_real_market_history

    history = load_real_market_history("config/config.yaml")
    return history.loc[(history["vegetable"] == series) & history["price"].notna(), "date"].max()


def _day(day: pd.Timestamp) -> str:
    return f"{day.day} {day.strftime('%b %Y')}"


def test_the_forecast_reach_is_stated_before_generate():
    at = _select("Carrot")
    reach_end = _latest("Carrot") + pd.Timedelta(days=30)
    last_start = reach_end - pd.Timedelta(days=110)  # New Kuroda: 110 days after planting
    notice = next(i.value for i in at.info if "Forecast reach" in i.value)
    assert f"forecasts up to **{_day(reach_end)}**" in notice
    assert f"planting date on or before **{_day(last_start)}**" in notice


def test_the_weather_choice_never_changes_the_forecast():
    """Found in the audit: the 'Growing location' box looked like it tailored the prediction."""
    default = _run("Carrot", _carrot_start())
    weather = next(s for s in default.selectbox if s.label == "Weather to display")
    assert weather.value == "Nuwara Eliya"  # defaults to the region the model uses for carrot
    jaffna = _run("Carrot", _carrot_start(), weather="Jaffna")

    def price(at):
        text = _markdown(at)
        return text[text.index("Price at harvest"):][:120]

    assert price(default) == price(jaffna)
    assert any("doesn't change the harvest dates or the price forecast" in c.value and "Nuwara Eliya (main production region)" in c.value for c in jaffna.caption)


def test_the_starting_record_and_model_inputs_are_named():
    at = _run("Carrot", _carrot_start())
    latest = _latest("Carrot")
    text = _markdown(at)
    assert f"**Forecast starting point (used by the model):** HARTI daily bulletin, Dambulla, Carrot, observed **{latest:%Y-%m-%d}**" in text
    assert "(midpoint of " in text
    captions = " ".join(c.value for c in at.caption)
    assert "Nuwara Eliya (main production region)" in captions and "No weather forecast is used" in captions
    assert "all vegetables together" in captions and "middle 80% of past forecast errors for all vegetables together" in text


def test_a_partly_forecast_window_gets_no_selling_suggestion():
    at = _run("Manioc", "2026-02-20")
    text = _markdown(at)
    assert "No suggestion" in text and "Sell as harvested" not in text
    assert any("covers only" in w.value for w in at.warning)


def test_a_stale_series_gets_one_clear_warning_not_a_beyond_reach_notice():
    at = _run("Cabbage", "2026-08-20", series="Cabbage (Nuwara Eliya)")
    assert not any("furthest date the price model can forecast" in i.value for i in at.info)
    assert any("No price forecast is possible" in w.value for w in at.warning)
    assert any("no forecast is shown" in w.value for w in at.warning)


def test_the_sidebar_lists_only_the_sources_this_page_uses():
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert "Used on this page" in sidebar and "HARTI daily bulletins" in sidebar and "Open-Meteo weather" in sidebar
    assert "CBSL" not in sidebar and "DCS" not in sidebar



# --- a price, or the reason there is none, for every plan (fix of 2026-11-15) ---------------------


@pytest.mark.parametrize(
    ("crop", "start", "badge"),
    [
        ("Beans", "2026-11-15", ":orange-badge[Historical Seasonal Price Reference – Not a Forecast]"),  # 45 days
        ("Carrot", "2026-11-15", ":orange-badge[Historical Seasonal Price Reference – Not a Forecast]"),  # 110 days
        ("Manioc", "2026-11-15", ":orange-badge[Historical Seasonal Price Reference – Not a Forecast]"),  # 9-12 months
        ("Cauliflower", "2026-11-15", ":red-badge[No Verified Price Available]"),  # no HARTI series
    ],
)
def test_the_price_section_is_always_shown_and_labelled(crop, start, badge):
    at = _run(crop, start)
    assert "3. Price at harvest" in [h.value for h in at.header]
    text = _markdown(at)
    assert badge in text and "**What this is:**" in text
    assert "Rs. 0.00" not in text and "Not available" not in text


def test_a_crop_without_prices_names_the_missing_source():
    text = _markdown(_run("Cauliflower", "2026-11-15"))
    assert "**Missing source:**" in text and "harti_prices.csv" in text and "not Rs 0" in text


def test_a_long_harvest_shows_its_records_and_years():
    text = _markdown(_run("Manioc", "2026-11-15"))
    assert "**Historical period:**" in text and "**Records used:**" in text and "**Median / average:**" in text
    assert "Source:** HARTI daily bulletin" in text and "**Unit:** Rs/kg" in text
