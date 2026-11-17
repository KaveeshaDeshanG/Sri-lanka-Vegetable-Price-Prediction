"""Checks on the project's real datasets: the newest records reach the app intact.

These read data/raw itself, so they are skipped when the datasets haven't been built. Nothing here
hard-codes a year: "current" means the year of the newest published price.
"""

from pathlib import Path

import pandas as pd
import pytest

from app.catalogue import load_manifest
from app.services import load_real_market_history

RAW = Path("data/raw")
pytestmark = pytest.mark.skipif(not (RAW / "harti_prices.csv").exists(), reason="real datasets not built")


@pytest.fixture(scope="module")
def harti():
    return pd.read_csv(RAW / "harti_prices.csv", parse_dates=["date"])


def test_every_date_parses_and_none_is_in_the_future(harti):
    assert harti["date"].notna().all()
    assert harti["date"].max() <= pd.Timestamp.now().normalize()


def test_the_newest_year_has_a_full_run_of_bulletins(harti):
    newest = harti["date"].max()
    this_year = harti[harti["date"].dt.year == newest.year]
    # From January to the newest bulletin, at least one price per month.
    assert set(this_year["date"].dt.month) == set(range(1, newest.month + 1))


def test_no_price_is_stored_twice(harti):
    assert not harti.duplicated(["date", "vegetable", "market"]).any()
    markets = pd.read_csv(RAW / "harti_all_markets.csv")
    assert not markets.duplicated(["published", "reported", "market", "vegetable"]).any()


def test_the_app_loader_keeps_every_price_including_the_newest(harti):
    loaded = load_real_market_history("config/config.yaml")
    priced = loaded[loaded["price"].notna()]
    assert len(priced) == len(harti)
    assert priced["date"].max() == harti["date"].max()


def test_the_manifest_reports_the_files_real_latest_dates(harti):
    manifest = load_manifest(RAW / "data_manifest.json")
    entry = next(e for e in manifest["datasets"] if e["file"] == "data/raw/harti_prices.csv")
    assert entry["latest_observation"] == str(harti["date"].max().date())
    assert entry["rows"] == len(harti)


def test_wholesale_and_retail_stay_in_separate_records():
    cbsl = pd.read_csv(RAW / "cbsl_daily_prices.csv")
    assert set(cbsl["price_type"]) <= {"wholesale", "retail"}
    assert not cbsl.duplicated(["date", "item", "market", "price_type"]).any()
    dcs = pd.read_csv(RAW / "dcs_weekly_retail_prices.csv")
    assert dcs["price_type"].str.startswith("retail").all()
