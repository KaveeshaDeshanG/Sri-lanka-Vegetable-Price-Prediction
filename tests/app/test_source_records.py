"""app/source_records.py: each official source's records, kept apart and fully traceable."""

from pathlib import Path

import pandas as pd
import pytest

from app import source_records as sr
from app.catalogue import HARTI_SERIES

RAW = Path("data/raw")
needs_data = pytest.mark.skipif(not (RAW / "harti_prices_detail.csv").exists(), reason="real datasets not built")


@pytest.mark.parametrize(
    ("raw", "label"),
    [("1 kg.", "Rs per 1 kg"), ("1Kg.", "Rs per 1 kg"), ("1 Kg", "Rs per 1 kg"), ("1kg.", "Rs per 1 kg"), ("500g", "Rs per 500 g"), ("Bunch", "Rs per bunch")],
)
def test_dcs_unit_spellings_are_unified_but_different_units_stay_different(raw, label):
    assert sr.dcs_unit(raw) == label


def test_the_name_map_only_names_real_series_and_items():
    mapping = sr.source_item_map()
    assert set(mapping.index) <= set(HARTI_SERIES)
    if (RAW / "cbsl_daily_prices.csv").exists():
        cbsl_items = set(pd.read_csv(RAW / "cbsl_daily_prices.csv")["item"])
        assert {item for items in mapping["cbsl_items"] for item in items} <= cbsl_items
    if (RAW / "dcs_weekly_retail_prices.csv").exists():
        dcs = set(pd.read_csv(RAW / "dcs_weekly_retail_prices.csv")["vegetable"])
        assert {veg for vegetables in mapping["dcs_vegetables"] for veg in vegetables} <= dcs


def test_ambiguous_names_are_left_unmapped():
    mapping = sr.source_item_map()
    for series in ("Brinjal", "Brinjals"):
        assert mapping.loc[series, "cbsl_items"] == [] and mapping.loc[series, "dcs_vegetables"] == []
        assert "not mapped" in mapping.loc[series, "note"]


def _frame(**columns):
    base = {column: [None] for column in sr.COLUMNS}
    base.update({key: [value] if not isinstance(value, list) else value for key, value in columns.items()})
    length = max(len(v) for v in base.values())
    return pd.DataFrame({k: v * length if len(v) == 1 else v for k, v in base.items()})


def test_records_stay_per_source_and_missing_values_stay_missing():
    sources = {
        "harti_dambulla": _frame(series="Beans", **{"Observation date": pd.Timestamp("2026-11-12"), "Market / location": "Dambulla", "Minimum": 380.0, "Maximum": 480.0, "Average / midpoint": 430.0}),
        "harti_other": _frame(series="Carrot", **{"Observation date": pd.Timestamp("2026-11-12"), "Market / location": "Kandy"}),
        "cbsl": _frame(item=["Beans", "Beans"], **{"Observation date": [pd.Timestamp("2026-11-13")] * 2, "Market / location": ["Dambulla", "Dambulla"],
                                                    "Price type": ["Wholesale", "Retail"], "Average / midpoint": [375.0, 405.0], "Minimum": [float("nan")] * 2}),
        "dcs": _frame(vegetable="Beans (Butter)", sort_key=202611.1, **{"Market / location": "Colombo"}),
    }
    result = sr.records_for_series("Beans", sources)
    assert len(result["harti_dambulla"]) == 1 and result["harti_dambulla"]["Average / midpoint"].iloc[0] == 430.0
    assert result["harti_other"].empty  # another vegetable's records never leak in
    assert sorted(result["cbsl"]["Price type"]) == ["Retail", "Wholesale"]  # two rows, never averaged together
    assert result["cbsl"]["Minimum"].isna().all()  # CBSL prints no range: missing, not zero
    assert result["dcs"].empty  # Butter beans aren't mapped to HARTI's (green) Beans


@needs_data
def test_the_forecast_starting_point_is_the_newest_stored_dambulla_price():
    from app.services import load_real_market_history

    dambulla = sr.harti_dambulla_records()
    history = load_real_market_history("config/config.yaml")
    beans = history[(history["vegetable"] == "Beans") & history["price"].notna()].sort_values("date").iloc[-1]
    newest = sr.latest_dambulla_overview(dambulla).set_index("Vegetable (as published)").loc["Beans"]
    assert newest["Observation date"] == beans["date"] and newest["Average / midpoint"] == beans["price"]
    assert newest["Minimum"] <= newest["Average / midpoint"] <= newest["Maximum"]
    assert str(newest["Source document"]).startswith("https://www.harti.gov.lk/")


@needs_data
def test_every_record_is_traceable_to_its_document():
    for frame in (sr.harti_dambulla_records(), sr.cbsl_records(), sr.dcs_records()):
        assert frame["Source document"].notna().all()
    assert sr.harti_other_market_records()["Source document"].notna().mean() > 0.99
