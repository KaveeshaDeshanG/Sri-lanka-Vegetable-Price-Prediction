"""The Cultivation Planner must use published durations as-is, count from the source's own stage,
calculate dates correctly, and show only the real forecaster's prices within its reach."""

import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.cultivation import (
    EVENTS,
    START_STAGES,
    UNITS,
    CropTiming,
    CultivationReferenceError,
    add_duration,
    build_plan,
    harvest_window,
    load_crop_timings,
    load_error_bands,
    same_season_history,
    suggest_selling_day,
)
from scripts.forecast_error_bands import error_bands
from src.models.direct_forecaster import MAX_HORIZON_DAYS

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data" / "reference" / "doa_harvest_durations_v2.csv"
HEADER = "crop,variety,price_labels,min_value,max_value,unit,start_stage,event,repeat_harvest,scope,source_url,source_text\n"
TIMINGS = load_crop_timings(REFERENCE)


def _independent(start: date, value: float, unit: str) -> date:
    """A second implementation, with the standard library only, to check add_duration against."""
    if unit == "days":
        return start + timedelta(days=int(value))
    if unit == "weeks":
        return start + timedelta(days=7 * int(value))

    def plus_months(day: date, months: int) -> date:
        index = day.month - 1 + months
        year, month = day.year + index // 12, index % 12 + 1
        return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))

    whole = int(value)
    base = plus_months(start, whole)
    if value == whole:
        return base
    following = plus_months(start, whole + 1)
    return base + timedelta(days=int((value - whole) * (following - base).days))


def _timing(min_value=60, max_value=70, unit="days", start_stage="planting", event="harvest", repeat="") -> CropTiming:
    return CropTiming("Test crop", "Crop guidance", ["Carrot"], min_value, max_value, unit, start_stage, event, repeat, "crop guidance", "https://doa.gov.lk/x/", "text")


def _actual(start="2024-01-01", end="2026-11-06", price=100.0) -> pd.DataFrame:
    dates = pd.date_range(start, end)
    return pd.DataFrame({"date": dates, "price": price + np.arange(len(dates)) % 30})


# --- the reference file -------------------------------------------------------------------------


def test_reference_file_is_complete_and_cited():
    assert len(TIMINGS) == 77
    assert len({(t.crop, t.variety) for t in TIMINGS}) == 77
    prices = pd.read_csv(ROOT / "data" / "raw" / "harti_prices.csv").replace({"vegetable": {"Eggplant": "Brinjal"}})
    known = set(prices["vegetable"])
    for timing in TIMINGS:
        assert 0 < timing.min_value <= timing.max_value
        assert timing.unit in UNITS and timing.start_stage in START_STAGES and timing.event in EVENTS
        assert timing.source_url.startswith("https://doa.gov.lk/")
        assert timing.source_text and "Ã" not in timing.source_text and "â€" not in timing.source_text  # no mojibake
        assert set(timing.price_labels) <= known, (timing.crop, timing.variety)


def test_crops_harti_never_priced_have_no_price_link():
    """HARTI prints "Dambala (Wing Beans)" and "Thithbatu" rows but never a price in them, so the
    planner must not offer a price series for those crops."""
    prices = pd.read_csv(ROOT / "data" / "raw" / "harti_all_markets.csv")
    assert not prices["vegetable"].isin(["Wing Beans", "Thibbatu"]).any()
    for timing in TIMINGS:
        if timing.crop in ("Winged Bean", "Thibbatu"):
            assert timing.price_labels == [], (timing.crop, timing.variety)


def test_source_wording_matches_the_saved_official_pages():
    """Every quote appears word for word on the DOA page it cites (saved on retrieval, 2026-11-09)."""
    pages = json.loads((ROOT / "data/reference/sources/doa_hordi_pages_2026-11-09.json").read_text(encoding="utf-8"))
    texts = {}
    for page in pages:
        texts[page["url"]] = page.get("text", "")
        for variety in page.get("varieties", []):
            texts[variety["url"]] = texts[variety["final"]] = variety["text"]
    normalise = lambda text: " ".join(str(text).split())  # noqa: E731
    for timing in TIMINGS:
        assert normalise(timing.source_text) in normalise(texts[timing.source_url]), (timing.crop, timing.variety)


@pytest.mark.parametrize(
    ("crop", "variety", "stage", "duration"),
    [
        ("Capsicum", "Crop guidance", "nursery sowing", "75 days"),
        ("Brinjal", "Crop guidance", "transplanting", "10–12 weeks"),
        ("Leeks", "Crop guidance", "transplanting", "4½ months"),
        ("Beans", "Wade", "seeding", "45 days"),
        ("Beans", "Sanjaya", "planting", "45 days"),
        ("Ela Batu", "GK Elabatu 1 (Manike)", "field planting", "75–80 days"),
        ("Pumpkin", "A.N.K.", "flowering", "40 days"),
        ("Radish", "Japanese Ball", "not stated", "45–55 days"),
        ("Sweet Potato", "CARI -9", "not stated", "3½–4 months"),
        ("Manioc", "Kirikawadi", "not stated", "6–12 months"),
    ],
)
def test_start_stage_and_unit_are_kept_as_published(crop, variety, stage, duration):
    timing = next(t for t in TIMINGS if t.crop == crop and t.variety == variety)
    assert timing.start_stage == stage
    assert timing.duration_text == duration


def test_every_start_stage_asks_for_its_own_date():
    labels = [label for label, _ in START_STAGES.values()]
    assert len(set(labels)) == len(labels)
    assert START_STAGES["nursery sowing"][0] == "Nursery sowing date"
    assert START_STAGES["transplanting"][0] == "Transplanting date"
    assert START_STAGES["field planting"][0] == "Field planting date"
    assert "doesn't say" in START_STAGES["not stated"][0]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("crop,min_value\nX,1\n", "missing column"),
        ("", "could not be read"),
        (HEADER + "X,V,Carrot,90,60,days,planting,harvest,,crop guidance,https://doa.gov.lk/x,t\n", "invalid duration range"),
        (HEADER + "X,V,Carrot,0,10,days,planting,harvest,,crop guidance,https://doa.gov.lk/x,t\n", "invalid duration range"),
        (HEADER + "X,V,Carrot,4.5,5,days,planting,harvest,,crop guidance,https://doa.gov.lk/x,t\n", "fractional"),
        (HEADER + "X,V,Carrot,4.3,5,months,planting,harvest,,crop guidance,https://doa.gov.lk/x,t\n", "half months"),
        (HEADER + "X,V,Carrot,1,2,fortnights,planting,harvest,,crop guidance,https://doa.gov.lk/x,t\n", "unknown unit"),
        (HEADER + "X,V,Carrot,1,2,days,germination,harvest,,crop guidance,https://doa.gov.lk/x,t\n", "unknown start stage"),
        (HEADER + "X,V,Carrot,1,2,days,planting,ripening,,crop guidance,https://doa.gov.lk/x,t\n", "unknown event"),
        (HEADER + "X,V,Carrot,1,2,days,planting,harvest,,crop guidance,https://example.com/x,t\n", "no Department of Agriculture source"),
        (HEADER + "X,V,Carrot,abc,2,days,planting,harvest,,crop guidance,https://doa.gov.lk/x,t\n", "non-numeric"),
        (HEADER + "X,V,Carrot,1,2,days,planting,harvest,,crop guidance,https://doa.gov.lk/x,t\n" * 2, "listed twice"),
    ],
)
def test_a_damaged_reference_file_is_reported(tmp_path, content, message):
    path = tmp_path / "ref.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CultivationReferenceError, match=message):
        load_crop_timings(path)


def test_a_missing_reference_file_is_reported(tmp_path):
    with pytest.raises(CultivationReferenceError, match="missing"):
        load_crop_timings(tmp_path / "nope.csv")


# --- date calculations --------------------------------------------------------------------------

BOUNDARY_STARTS = [date(2024, 2, 29), date(2025, 1, 31), date(2025, 12, 20), date(2026, 11, 9), date(2023, 8, 31)]


@pytest.mark.parametrize("timing", TIMINGS, ids=[f"{t.crop}-{t.variety}" for t in TIMINGS])
@pytest.mark.parametrize("start", BOUNDARY_STARTS, ids=str)
def test_every_crop_and_variety_window_matches_an_independent_calculation(timing, start):
    first, last = harvest_window(timing, start)
    assert first.date() == _independent(start, timing.min_value, timing.unit)
    assert last.date() == _independent(start, timing.max_value, timing.unit)
    assert first <= last and first > pd.Timestamp(start)
    if timing.min_value == timing.max_value:
        assert first == last  # a single published value is a single day, not a window


@pytest.mark.parametrize(
    ("start", "value", "unit", "expected"),
    [
        ("2024-02-28", 1, "days", "2024-02-29"),  # leap year
        ("2025-02-28", 1, "days", "2025-03-01"),  # not a leap year
        ("2025-12-20", 45, "days", "2026-02-03"),  # across the year boundary
        ("2026-01-01", 10, "weeks", "2026-03-12"),  # 70 days exactly
        ("2025-01-31", 1, "months", "2025-02-28"),  # month end, clamped to February
        ("2024-01-31", 1, "months", "2024-02-29"),  # leap February
        ("2026-01-10", 4.5, "months", "2026-05-25"),  # 10 May + 15 of the 31 days to 10 June
        ("2025-11-30", 3.5, "months", "2026-03-15"),  # 28 Feb + 15 of the 30 days to 30 Mar
        ("2025-10-15", 3, "months", "2026-01-15"),  # across the year boundary
        ("2026-03-01", 12, "months", "2027-03-01"),
    ],
)
def test_add_duration_boundaries(start, value, unit, expected):
    assert add_duration(start, value, unit) == pd.Timestamp(expected)


def test_no_month_is_treated_as_30_days():
    """9 months from 1 January is 1 October (273 days), not 270 days later."""
    assert add_duration("2026-01-01", 9, "months") == pd.Timestamp("2026-10-01")
    assert add_duration("2026-01-01", 9, "months") != pd.Timestamp("2026-01-01") + pd.Timedelta(days=270)


@pytest.mark.parametrize(("value", "unit"), [(0, "days"), (-3, "days"), (1.5, "days"), (5, "years")])
def test_invalid_durations_are_refused(value, unit):
    with pytest.raises(ValueError):
        add_duration("2026-01-01", value, unit)


def test_a_missing_start_date_is_refused():
    with pytest.raises(ValueError, match="start date"):
        build_plan(_timing(), "Carrot", None, _actual(), None, None, None, today=pd.Timestamp("2026-11-09"))


# --- plans ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "status"),
    [("2026-07-01", "passed"), ("2026-09-15", "in_reach"), ("2026-10-01", "partly_in_reach"), ("2026-11-09", "beyond")],
)
def test_status_depends_on_the_forecasts_reach(start, status):
    """60-70 day crop; the latest price is 2026-11-06, so forecasts reach 2026-12-06."""
    plan = build_plan(_timing(), "Carrot", start, _actual(), None, None, None, today=pd.Timestamp("2026-11-09"))
    assert plan.status == status


def test_beyond_reach_shows_earlier_years_not_a_forecast():
    plan = build_plan(_timing(), "Carrot", "2026-11-09", _actual(), None, None, None, today=pd.Timestamp("2026-11-09"))
    assert plan.forecast is None and plan.window_forecast is None and plan.best_day is None
    assert list(plan.same_season["Period"]) == ["2026-01-08 to 2026-01-18", "2025-01-08 to 2025-01-18", "2024-01-08 to 2024-01-18"]


def test_a_passed_window_shows_the_prices_actually_recorded_in_it():
    plan = build_plan(_timing(), "Carrot", "2026-07-01", _actual(), None, None, None, today=pd.Timestamp("2026-11-09"))
    assert plan.recorded_in_window["date"].min() == pd.Timestamp("2026-08-30")
    assert plan.recorded_in_window["date"].max() == pd.Timestamp("2026-09-09")


def test_a_crop_without_a_price_series_gets_dates_only():
    plan = build_plan(_timing(), None, "2026-11-09", None, object(), object(), None, today=pd.Timestamp("2026-11-09"))
    assert plan.status == "no_prices" and plan.forecast is None
    assert plan.harvest_start == pd.Timestamp("2027-01-08")
    assert any("No market price series" in warning for warning in plan.warnings)


def test_a_stale_price_series_gets_history_not_an_old_forecast():
    """Found in the final audit: a series last priced 485 days ago showed a 2025 'forecast'."""

    class Forecaster:
        def forecast(self, *args, **kwargs):
            raise AssertionError("no forecast should be requested for a stale series")

    actual = _actual(end="2025-07-12")
    plan = build_plan(_timing(90, 110), "Carrot", "2025-04-20", actual, Forecaster(), object(), None, today=pd.Timestamp("2026-11-09"))
    assert plan.stale and plan.forecast is None and plan.window_forecast is None and plan.best_day is None
    assert not plan.same_season.empty
    assert any("no forecast is shown" in warning for warning in plan.warnings)


def test_a_current_series_is_not_stale():
    plan = build_plan(_timing(), "Carrot", "2026-11-09", _actual(), None, None, None, today=pd.Timestamp("2026-11-09"))
    assert not plan.stale


@pytest.mark.parametrize(
    ("stage", "event", "repeat", "expected"),
    [
        ("not stated", "harvest", "", "doesn't say"),
        ("flowering", "harvest", "", "counts from flowering"),
        ("planting", "first harvest", "every 4 days", "first harvest"),
    ],
)
def test_timing_caveats_are_shown(stage, event, repeat, expected):
    plan = build_plan(_timing(start_stage=stage, event=event, repeat=repeat), "Carrot", "2026-11-09", _actual(), None, None, None, today=pd.Timestamp("2026-11-09"))
    assert any(expected in warning for warning in plan.warnings)


def test_same_season_history_uses_only_recorded_prices():
    actual = pd.DataFrame({"date": pd.to_datetime(["2025-03-01", "2025-03-05", "2024-07-01"]), "price": [100.0, 140.0, 999.0]})
    table = same_season_history(actual, pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-10"))
    assert len(table) == 1
    assert table.iloc[0]["Days recorded"] == 2 and table.iloc[0]["Median (Rs/kg)"] == 120.0


def test_same_season_history_with_no_matching_years_is_empty():
    actual = pd.DataFrame({"date": pd.to_datetime(["2026-10-01"]), "price": [100.0]})
    assert same_season_history(actual, pd.Timestamp("2027-03-01"), pd.Timestamp("2027-03-10")).empty


def _window(prices, lows):
    dates = pd.date_range("2026-11-10", periods=len(prices))
    return pd.DataFrame({"date": dates, "price": prices, "low": lows, "high": [p * 1.2 for p in prices]})


def test_a_clearly_better_selling_day_is_identified():
    best, clear = suggest_selling_day(_window([100.0, 105.0, 140.0], [90.0, 95.0, 120.0]))
    assert best["date"] == pd.Timestamp("2026-11-12") and clear


def test_a_small_difference_is_not_called_clear():
    best, clear = suggest_selling_day(_window([100.0, 104.0], [90.0, 95.0]))
    assert best["price"] == 104.0 and not clear


def test_unmeasured_ranges_never_claim_a_clear_advantage():
    _, clear = suggest_selling_day(_window([100.0, 200.0], [np.nan, np.nan]))
    assert not clear


def test_no_window_means_no_suggestion():
    assert suggest_selling_day(None) == (None, False)


def test_error_bands_are_quantiles_of_log_error():
    scored = pd.DataFrame({"horizon": [1] * 5, "target_price": [100.0, 110, 90, 100, 120], "prediction": [100.0] * 5})
    band = error_bands(scored)[0]
    assert band["horizon"] == 1 and band["rows"] == 5
    assert band["q50"] == pytest.approx(0.0)
    assert band["q10"] < 0 < band["q90"]


def test_saved_error_bands_cover_every_horizon():
    bands = load_error_bands(ROOT / "reports" / "metrics" / "forecast_error_bands.json")
    assert sorted(bands) == list(range(1, 31))
    assert all(band["q10"] < band["q90"] for band in bands.values())


def test_unreadable_error_bands_give_none(tmp_path):
    bad = tmp_path / "bands.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_error_bands(bad) is None
    assert load_error_bands(tmp_path / "missing.json") is None


def _real_inputs():
    import os

    from app.services import load_real_market_history
    from src.feature_engineering.forecast_features import ForecastFeatureBuilder
    from src.models.direct_forecaster import DirectForecaster

    os.chdir(ROOT)
    history = load_real_market_history("config/config.yaml")
    origin_rows = ForecastFeatureBuilder("config/config.yaml").build_origin_features(history)
    return history, origin_rows, DirectForecaster.load("config/config.yaml")


def test_planned_harvest_price_is_the_trained_forecasters_output():
    """End to end on the real data and model: the price for the harvest date is the forecaster's
    own output for exactly that date, and every forecast date is within 30 days of the latest price."""
    history, origin_rows, forecaster = _real_inputs()
    actual = history[(history["vegetable"] == "Carrot") & history["price"].notna()][["date", "price"]]
    latest = actual["date"].max()
    carrot = next(t for t in TIMINGS if t.crop == "Carrot" and t.variety == "New Kuroda")
    start = latest + pd.Timedelta(days=7) - pd.Timedelta(days=110)  # first harvest 7 days after the latest price
    bands = load_error_bands(ROOT / "reports" / "metrics" / "forecast_error_bands.json")

    plan = build_plan(carrot, "Carrot", start, actual, forecaster, origin_rows, bands, today=latest)

    assert plan.status == "in_reach" and plan.harvest_start == latest + pd.Timedelta(days=7)
    assert plan.forecast["date"].max() == latest + pd.Timedelta(days=MAX_HORIZON_DAYS)
    assert list(plan.forecast["horizon"]) == list(range(1, MAX_HORIZON_DAYS + 1))
    direct = forecaster.forecast(origin_rows, "Carrot", latest, [7])["predicted_price"].iloc[0]
    assert plan.window_forecast.iloc[0]["date"] == plan.harvest_start
    assert plan.window_forecast.iloc[0]["price"] == pytest.approx(direct)
    expected_low = direct * np.exp(json.loads((ROOT / "reports/metrics/forecast_error_bands.json").read_text())["bands"][6]["q10"])
    assert plan.window_forecast.iloc[0]["low"] == pytest.approx(expected_low)
