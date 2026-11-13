"""Older HARTI bulletin layouts and DCS weekly retail bulletins, read from real official PDFs."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_harti_prices import extract_market_rows
from src.data_collection.dcs_retail_parser import parse_dcs_weekly_pdf
from src.data_collection.harti_live_client import (
    HartiLiveClient,
    canonical_market_name,
    parse_all_markets,
    resolve_vegetable,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HARTI_2015 = FIXTURES / "harti_daily_2015-06-22.pdf"


@pytest.mark.parametrize(
    ("header", "market"),
    [
        ("Pettah Market", "Pettah"),
        ("Petha Market", "Pettah"),
        ("Kappetipola Market", "Keppetipola"),
        ("a Kappetipola Market", "Keppetipola"),
        ("T'thegama Market", "Thambuththegama"),
        ("hambuththegam Market", "Thambuththegama"),
        ("Peliyagoda Market", "Peliyagoda"),
        ("Dambulla Market", "Dambulla"),
        ("Market", None),
    ],
)
def test_older_market_spellings_resolve_to_one_market_each(header, market):
    assert canonical_market_name(header) == market


@pytest.mark.parametrize(
    ("label", "vegetable"),
    [
        ("Brinjals (Other)", "Brinjals (Other)"),
        ("Brinjals (Village)", "Brinjals (Village)"),
        ("Brinjals", "Brinjals"),
        ("Eggplant", "Eggplant"),
        ("Bitter Gourd (Village)", "Bitter Gourd (Village)"),
        ("Dambala (Wing Beans)", "Wing Beans"),
        ("Thithbatu", "Thibbatu"),
        ("Gingerly", None),
        ("Black Gram", None),
    ],
)
def test_older_row_labels_stay_separate_series(label, vegetable):
    """An older label is its own series: never folded into a later label without evidence."""
    assert resolve_vegetable(label) == vegetable


def test_a_2015_bulletin_reads_the_dambulla_column_including_the_merged_label_rows():
    """2015-2020 bulletins print six labels in one tall cell; each must land on its own price row."""
    extract = extract_market_rows(HARTI_2015.read_bytes(), pd.Timestamp("2015-06-22"), "Dambulla")
    assert extract.error is None and extract.market_date == pd.Timestamp("2015-06-22")
    prices = {row["vegetable"]: (row["low"], row["high"]) for row in extract.rows}
    # Values as printed in the PDF's Dambulla column (checked against the page text).
    assert prices["Beans"] == (190.0, 210.0)
    assert prices["Brinjals (Other)"] == (25.0, 35.0)
    assert prices["Potato (Nuwara Eliya)"] == (110.0, 112.0)
    assert prices["Big Onion (Imported)"] == (85.0, 90.0)
    assert "Potato (Welimada)" not in prices  # "-" in the Dambulla column: no reading, no row
    assert "Brinjals (Village)" not in prices


def test_a_2015_bulletin_keeps_each_market_separate():
    table = HartiLiveClient()._extract_vegetable_table(HARTI_2015.read_bytes())
    panel = parse_all_markets(table, pd.Timestamp("2015-06-22"))
    assert set(panel["market"]) == {"Pettah", "Kandy", "Dambulla", "Meegoda", "Norochchole", "Thambuththegama"}
    beans = panel[panel["vegetable"] == "Beans"].set_index("market")["price"]
    assert beans["Pettah"] == 230.0 and beans["Dambulla"] == 200.0 and beans["Kandy"] == 195.0
    assert not panel.duplicated(["market", "vegetable"]).any()
    meegoda = panel[panel["market"] == "Meegoda"]["reported"].unique()
    assert list(meegoda) == [pd.Timestamp("2015-06-21")]  # Meegoda's own date, a day behind the bulletin


def test_dcs_weekly_bulletin_reads_this_weeks_averages_only():
    bulletin = parse_dcs_weekly_pdf((FIXTURES / "DCSB-WRP-2026-11-W1.pdf").read_bytes(), "DCSB-WRP-2026-11-W1.pdf")
    assert bulletin.error is None
    assert (bulletin.year, bulletin.month, bulletin.week_of_month) == (2026, 11, 1)
    rows = {row["item"]: row for row in bulletin.rows}
    assert rows["Ash Plantain"]["this_week_avg"] == 368.57
    assert rows["Ash Plantain"]["previous_week_avg"] == 362.96
    assert (rows["Ash Plantain"]["range_low"], rows["Ash Plantain"]["range_high"]) == (320.0, 400.0)
    assert rows["Tomatoe - No 1."]["vegetable"] == "Tomato" and rows["Tomatoe - No 1."]["this_week_avg"] == 883.20
    assert rows["Potatoes - Imported"]["vegetable"] == "Potato (Imported)"  # an indented sub-item joins its group
    assert rows["B.Onions - Local"]["this_week_avg"] == 356.67
    assert rows["Bandakka"]["vegetable"] == "Ladies Fingers"
    assert rows["Coconut ( Large )"]["vegetable"] is None  # not a vegetable: kept out of the dataset


def test_dcs_bulletin_without_a_price_range_column_is_still_read():
    bulletin = parse_dcs_weekly_pdf((FIXTURES / "DCSB-WRP-2020-04-W1.pdf").read_bytes(), "DCSB-WRP-2020-04-W1.pdf")
    assert bulletin.error is None and (bulletin.year, bulletin.month, bulletin.week_of_month) == (2020, 4, 1)
    ash = next(row for row in bulletin.rows if row["item"] == "Ash Plantain")
    assert ash["this_week_avg"] == 151.83 and ash["range_low"] is None


def test_dcs_bulletin_whose_title_contradicts_its_file_name_is_refused():
    bulletin = parse_dcs_weekly_pdf((FIXTURES / "DCSB-WRP-2026-11-W1.pdf").read_bytes(), "DCSB-WRP-2026-10-W1.pdf")
    assert bulletin.error and "doesn't match" in bulletin.error and not bulletin.rows


def test_an_unreadable_dcs_file_is_reported_not_raised():
    assert parse_dcs_weekly_pdf(b"not a pdf", "x.pdf").error
