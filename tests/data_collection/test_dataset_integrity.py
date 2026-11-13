"""The imported datasets: no duplicates, markets and price types kept apart, and a manifest that matches them."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.update_data import harti_entries, rebuild_dcs
from src.data_collection.harti_live_client import MARKET_NAMES, MAX_REPORTING_LAG_DAYS

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def dambulla():
    return pd.read_csv(RAW / "harti_prices_detail.csv", parse_dates=["date", "bulletin_date"])


@pytest.fixture(scope="module")
def markets():
    return pd.read_csv(RAW / "harti_all_markets.csv", parse_dates=["published", "reported"])


@pytest.fixture(scope="module")
def dcs():
    return pd.read_csv(RAW / "dcs_weekly_retail_prices.csv")


def test_dambulla_prices_are_unique_real_readings(dambulla):
    assert not dambulla.duplicated(["date", "vegetable"]).any()
    assert set(dambulla["market"]) == {"Dambulla"}
    assert (dambulla["low"] <= dambulla["high"]).all() and (dambulla["price"] > 0).all()
    assert (dambulla["price"] == (dambulla["low"] + dambulla["high"]) / 2).all()
    lag = (dambulla["bulletin_date"] - dambulla["date"]).dt.days
    assert lag.between(0, MAX_REPORTING_LAG_DAYS).all()  # a reading is never dated after the bulletin that published it
    assert dambulla["source_pdf"].str.startswith("https://www.harti.gov.lk/").all()
    assert dambulla["date"].min() < pd.Timestamp("2016-01-01")  # the archive back to 2015 is imported, not only 2024 on


def test_every_market_is_its_own_series(markets):
    assert not markets.duplicated(["published", "market", "vegetable"]).any()
    assert set(markets["market"]) <= set(MARKET_NAMES)
    lag = (markets["published"] - markets["reported"]).dt.days
    assert lag.between(0, MAX_REPORTING_LAG_DAYS).all()


def test_the_dambulla_column_agrees_between_the_two_harti_files(dambulla, markets):
    """Both files are read from the same bulletins, so Dambulla's readings must agree where both exist."""
    same_day = dambulla[dambulla["bulletin_date"] == dambulla["date"]].assign(vegetable=lambda f: f["vegetable"].replace({"Eggplant": "Brinjal"}))
    panel = markets[markets["market"] == "Dambulla"]
    joined = same_day.merge(panel, left_on=["bulletin_date", "vegetable"], right_on=["published", "vegetable"], suffixes=("", "_panel"))
    assert len(joined) > 0.95 * len(same_day)
    assert (joined["price"] == joined["price_panel"]).all()


def test_retail_prices_are_a_separate_dataset(dcs, dambulla, markets):
    assert not dcs.duplicated(["year", "month", "week_of_month", "vegetable"]).any()
    assert set(dcs["price_type"]) == {"retail (weekly average)"}
    assert set(dcs["market"]) == {"Colombo District (DCS main markets)"}
    assert dcs["week_of_month"].between(1, 5).all() and dcs["month"].between(1, 12).all()
    assert "price_type" not in dambulla.columns and "price_type" not in markets.columns
    assert not set(dcs["source_file"]) & set(dambulla["source_pdf"])


def test_the_manifest_matches_the_files():
    manifest = json.loads((RAW / "data_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["file"]: entry for entry in manifest["datasets"]}
    prices = pd.read_csv(RAW / "harti_prices.csv", parse_dates=["date"])
    assert entries["data/raw/harti_prices.csv"]["latest_observation"] == str(prices["date"].max().date())
    assert entries["data/raw/harti_prices.csv"]["first_observation"] == str(prices["date"].min().date())
    assert entries["data/raw/harti_prices.csv"]["rows"] == len(prices)
    panel = pd.read_csv(RAW / "harti_all_markets.csv", parse_dates=["published"])
    assert entries["data/raw/harti_all_markets.csv"]["latest_observation"] == str(panel["published"].max().date())
    dcs = pd.read_csv(RAW / "dcs_weekly_retail_prices.csv").sort_values(["year", "month", "week_of_month"])
    assert entries["data/raw/dcs_weekly_retail_prices.csv"]["latest_observation"] == dcs["period"].iloc[-1]
    for entry in entries.values():
        assert entry["source_url"].startswith("https://") and entry["retrieved_at"]


def test_url_corrections_replace_only_the_folder_links(tmp_path):
    listing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-29", "2025-06-30"]),
            "url": ["https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/", "https://www.harti.gov.lk/x.pdf"],
        }
    )
    listing.to_csv(tmp_path / "archive_entries_listing.csv", index=False)
    entries = harti_entries(None, offline=True, cache_dir=tmp_path).set_index("date")["url"]
    assert entries[pd.Timestamp("2025-06-29")].endswith("2025/December/daily_29-06-2025.pdf")
    assert entries[pd.Timestamp("2025-06-30")] == "https://www.harti.gov.lk/x.pdf"


def test_two_bulletins_for_the_same_week_are_both_kept_out(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy(FIXTURES / "DCSB-WRP-2026-11-W1.pdf", cache / "DCSB-WRP-2026-11-W1.pdf")
    shutil.copy(FIXTURES / "DCSB-WRP-2026-11-W1.pdf", cache / "reissue.pdf")  # same week again
    shutil.copy(FIXTURES / "DCSB-WRP-2020-04-W1.pdf", cache / "DCSB-WRP-2020-04-W1.pdf")
    summary = rebuild_dcs(cache, tmp_path, None)
    table = pd.read_csv(tmp_path / "dcs_weekly_retail_prices.csv")
    assert set(table["period"]) == {"2020-04 week 1"}
    assert summary["problems"] == 2
