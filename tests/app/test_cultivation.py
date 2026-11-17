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
    latest_start_in_reach,
    load_crop_timings,
    load_error_bands,
    same_season_history,
    select_price,
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
    # the harvest window (8-18 Jan) widened by SEASONAL_PAD_DAYS on each side, in each earlier year
    assert list(plan.same_season["Period"]) == ["2026-01-01 to 2026-01-25", "2025-01-01 to 2025-01-25", "2024-01-01 to 2024-01-25"]


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


# --- reach and coverage (fixes from the Cultivation Planner audit, 2026-11-15) ---------------------


class _PathForecaster:
    """Stands in for the forecaster: a flat 100 Rs/kg path, so only the plan's own logic is tested."""

    def forecast(self, origin_rows, vegetable, origin, horizons):
        return pd.DataFrame(
            {
                "horizon": horizons,
                "target_date": [origin + pd.Timedelta(days=h) for h in horizons],
                "predicted_price": [100.0 + h for h in horizons],
                "last_price": 100.0,
            }
        )


@pytest.mark.parametrize("timing", TIMINGS, ids=lambda t: f"{t.crop}-{t.variety}")
def test_latest_start_in_reach_is_the_last_date_whose_harvest_is_forecast(timing):
    end = pd.Timestamp("2026-12-12")
    start = latest_start_in_reach(timing, end)
    assert add_duration(start, timing.min_value, timing.unit) <= end
    assert add_duration(start + pd.Timedelta(days=1), timing.min_value, timing.unit) > end


def test_latest_start_in_reach_examples():
    sanjaya = next(t for t in TIMINGS if t.variety == "Sanjaya")
    new_kuroda = next(t for t in TIMINGS if t.variety == "New Kuroda")
    assert latest_start_in_reach(sanjaya, "2026-12-12") == pd.Timestamp("2026-10-28")  # 45 days
    assert latest_start_in_reach(new_kuroda, "2026-12-12") == pd.Timestamp("2026-08-24")  # 110 days


def test_a_partly_forecast_window_gets_no_selling_suggestion():
    """Found in the audit: Manioc's advice rested on 23 of its window's 93 days."""
    plan = build_plan(_timing(), "Carrot", "2026-10-01", _actual(), _PathForecaster(), object(), None, today=pd.Timestamp("2026-11-09"))
    assert plan.status == "partly_in_reach"
    assert 0 < plan.forecast_days < plan.window_days and not plan.fully_forecast
    assert plan.best_day is None and not plan.clear_advantage
    assert any(f"covers only {plan.forecast_days} of the {plan.window_days} days" in w for w in plan.warnings)


def test_a_fully_forecast_window_still_gets_a_suggestion():
    plan = build_plan(_timing(), "Carrot", "2026-09-15", _actual(), _PathForecaster(), object(), None, today=pd.Timestamp("2026-11-09"))
    assert plan.status == "in_reach" and plan.fully_forecast and plan.forecast_days == plan.window_days == 11
    assert plan.best_day is not None and plan.best_day["date"] == plan.harvest_end  # the path rises


# --- the price a plan leads with, for every harvest distance (fix of 2026-11-15) ------------------
# Root cause of the missing prices: beyond the forecaster's 30-day reach, and for stale series, the
# plan's price was a hard-coded "Not available", and the earlier-years table used only the exact
# harvest days (2-10 prices in all for a one-day harvest). select_price() now always leads with a
# labelled price, or says why none exists.

HORIZONS = [1, 7, 14, 30, 45, 60, 90, 120, 180, 365]  # days after the latest recorded price


@pytest.fixture(scope="module")
def real():
    return _real_inputs()


def _days_timing(days: int, labels=("x",)) -> CropTiming:
    return CropTiming("Test crop", "Crop guidance", list(labels), days, days, "days", "planting", "harvest", "", "crop guidance", "https://doa.gov.lk/x/", "text")


def _independent_seasonal(prices: pd.DataFrame, harvest: pd.Timestamp) -> tuple[float, int, list[int]]:
    """Median, count and years of the same weeks (harvest ± 7 days) in the 5 most recent earlier years with prices."""
    windows = []
    for years_back in range(1, 13):
        day = date(harvest.year - years_back, harvest.month, min(harvest.day, 28) if harvest.month == 2 else harvest.day)
        low, high = pd.Timestamp(day - timedelta(days=7)), pd.Timestamp(day + timedelta(days=7))
        found = prices[(prices["date"] >= low) & (prices["date"] <= high) & (prices["price"] > 0)]
        if len(found):
            windows.append(found)
    used = pd.concat(windows[:5])
    return float(used["price"].median()), len(used), sorted(used["date"].dt.year.unique().tolist())


@pytest.mark.parametrize("vegetable", ["Beans", "Carrot", "Manioc"])
@pytest.mark.parametrize("days", HORIZONS)
def test_every_harvest_distance_gets_a_correctly_labelled_price(real, vegetable, days):
    history, origin_rows, forecaster = real
    actual = history[(history["vegetable"] == vegetable) & history["price"].notna()][["date", "price"]].sort_values("date")
    latest = actual["date"].max()
    harvest = latest + pd.Timedelta(days=days)
    timing = _days_timing(60, [vegetable])
    plan = build_plan(timing, vegetable, harvest - pd.Timedelta(days=60), actual, forecaster, origin_rows, None, today=latest)
    assert plan.harvest_start == harvest  # the planting-to-harvest arithmetic is unchanged
    result = select_price(plan, actual, latest)
    assert result.price is not None and result.price > 0
    assert (result.source, result.market, result.price_type, result.unit) == ("HARTI daily bulletin", "Dambulla", "Wholesale", "Rs/kg")
    if days <= MAX_HORIZON_DAYS:
        direct = forecaster.forecast(origin_rows, vegetable, latest, [days])["predicted_price"].iloc[0]
        assert result.kind == "model_forecast" and result.label == "Model Forecast"
        assert result.horizon == days and result.date == harvest
        assert result.price == pytest.approx(direct)
    else:
        # never a model output beyond its reach: nothing was even requested from the forecaster
        assert plan.forecast is None and plan.window_forecast is None
        assert result.kind == "seasonal_reference"
        assert result.label == "Historical Seasonal Price Reference – Not a Forecast"
        median, count, years = _independent_seasonal(actual, harvest)
        assert result.price == pytest.approx(median) and result.observations == count and result.years == years
        assert result.observations >= 10 and len({p.split(" to ")[0][:4] for p in result.period.split("; ")}) >= 3
        assert result.low > 0 and result.low <= result.median <= result.high
        assert "not a prediction" in result.explanation


@pytest.mark.parametrize("days", HORIZONS)
def test_a_stale_series_never_gets_a_forecast(real, days):
    history, origin_rows, forecaster = real
    series = "Cabbage (Nuwara Eliya)"
    actual = history[(history["vegetable"] == series) & history["price"].notna()][["date", "price"]].sort_values("date")
    today = pd.Timestamp("2026-11-15")
    assert (today - actual["date"].max()).days > MAX_HORIZON_DAYS  # the precondition: its last price is from 2025-07-12
    harvest = today + pd.Timedelta(days=days)
    plan = build_plan(_days_timing(90, [series]), series, harvest - pd.Timedelta(days=90), actual, forecaster, origin_rows, None, today=today)
    result = select_price(plan, actual, today)
    assert plan.stale and plan.forecast is None
    assert result.kind in ("seasonal_reference", "latest_actual") and result.price > 0
    assert "more than 30 days old" in result.explanation
    if result.kind == "latest_actual":
        assert result.date == pd.Timestamp("2025-07-12") and result.age_days == (today - pd.Timestamp("2025-07-12")).days


def test_a_crop_with_no_price_series_gets_an_explanation_not_a_price():
    cauliflower = next(t for t in TIMINGS if t.crop == "Cauliflower")
    plan = build_plan(cauliflower, None, "2026-11-15", None, None, None, None, today=pd.Timestamp("2026-11-15"))
    result = select_price(plan, None, "2026-11-15")
    assert result.kind == "unavailable" and result.price is None and result.label == "No Verified Price Available"
    assert "no price series" in result.explanation and "harti_prices.csv" in result.explanation


def test_too_few_seasonal_prices_fall_back_to_the_latest_actual_price():
    # prices only from mid-2025 on: the same weeks of earlier years hold one year's prices at most
    actual = _actual(start="2025-06-01", end="2026-11-06")
    plan = build_plan(_timing(), "Carrot", "2026-11-09", actual, None, None, None, today=pd.Timestamp("2026-11-09"))
    record = {"date": pd.Timestamp("2026-11-06"), "price": 120.0, "low": 110.0, "high": 130.0}
    result = select_price(plan, actual, "2026-11-09", record)
    assert not plan.seasonal["sufficient"]
    assert result.kind == "latest_actual" and result.label == "Latest Available Actual Price – Not a Future Prediction"
    assert (result.price, result.low, result.high, result.age_days) == (120.0, 110.0, 130.0, 3)
    assert f"Only {plan.seasonal['observations']} price(s)" in result.explanation


def test_no_same_season_prices_at_all_still_shows_the_latest_price():
    actual = _actual(start="2026-09-01", end="2026-11-06")
    plan = build_plan(_timing(), "Carrot", "2026-11-09", actual, None, None, None, today=pd.Timestamp("2026-11-09"))
    result = select_price(plan, actual, "2026-11-09")
    assert result.kind == "latest_actual" and result.price == actual["price"].iloc[-1]
    assert "No price was recorded in the same weeks" in result.explanation


def test_zero_coded_prices_are_never_counted_as_rs_0():
    actual = _actual()
    january = actual["date"].dt.month == 1
    actual.loc[january & actual["date"].dt.day.between(10, 20), "price"] = 0.0  # coded 0: no price published
    actual.loc[january & (actual["date"].dt.day == 5), "price"] = np.nan
    plan = build_plan(_timing(), "Carrot", "2026-11-09", actual, None, None, None, today=pd.Timestamp("2026-11-09"))
    result = select_price(plan, actual, "2026-11-09")
    # harvest 8-18 Jan 2027, so the same weeks are 1-25 January: 1-4, 6-9 and 21-25 January keep real prices
    assert result.kind == "seasonal_reference" and result.low > 0
    assert result.observations == 3 * 13  # 2024, 2025 and 2026
    assert (plan.same_season["Lowest (Rs/kg)"] > 0).all()


def test_a_passed_window_leads_with_the_prices_recorded_in_it():
    actual = _actual()
    plan = build_plan(_timing(), "Carrot", "2026-07-01", actual, None, None, None, today=pd.Timestamp("2026-11-09"))
    result = select_price(plan, actual, "2026-11-09")
    recorded = actual[actual["date"].between("2026-08-30", "2026-09-09")]["price"]
    assert result.kind == "recorded_actual" and result.price == recorded.median() and result.observations == len(recorded)
