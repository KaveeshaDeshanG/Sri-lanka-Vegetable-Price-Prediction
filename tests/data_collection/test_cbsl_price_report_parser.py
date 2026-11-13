"""The CBSL Daily Price Report parser, checked against real reports (tests/fixtures).

Expected values were read from the rendered 09 November 2026 and 04 August 2021 tables.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data_collection.cbsl_price_report_parser import normalise_item, parse_cbsl_price_report, report_date_from_name

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _table(name: str) -> pd.DataFrame:
    report = parse_cbsl_price_report((FIXTURES / name).read_bytes(), name)
    assert report.error is None
    return pd.DataFrame(report.rows)


def _price(table, item, price_type, market):
    hit = table[(table["item"] == item) & (table["price_type"] == price_type) & (table["market"] == market)]
    assert len(hit) == 1, (item, price_type, market)
    return hit["price"].iloc[0]


@pytest.fixture(scope="module")
def nov_2026():
    return _table("price_report_20261109_e.pdf")


@pytest.mark.parametrize(
    "item, price_type, market, expected",
    [
        ("Beans", "wholesale", "Pettah", 600.0),
        ("Beans", "wholesale", "Dambulla", 440.0),
        ("Beans", "retail", "Pettah", 700.0),
        ("Beans", "retail", "Dambulla", 470.0),
        ("Beans", "retail", "Narahenpita", 800.0),
        ("Carrot", "wholesale", "Dambulla", 575.0),
        ("Carrot", "retail", "Dambulla", 605.0),
        ("Tomato", "wholesale", "Pettah", 650.0),
        ("Green Chilli", "retail", "Pettah", 400.0),
        ("Lime", "wholesale", "Pettah", 1200.0),  # printed "1,200.00"
        ("Lime", "retail", "Narahenpita", 2000.0),
    ],
)
def test_todays_prices_match_the_printed_table(nov_2026, item, price_type, market, expected):
    assert _price(nov_2026, item, price_type, market) == expected


def test_only_todays_column_is_kept_and_dated_with_the_report(nov_2026):
    assert set(nov_2026["date"]) == {pd.Timestamp("2026-11-09")}
    # "Last Friday" Beans wholesale at Pettah was 500; only today's 600 is stored.
    assert 500.0 not in set(nov_2026[(nov_2026["item"] == "Beans") & (nov_2026["market"] == "Pettah")]["price"])
    assert not nov_2026.duplicated(["date", "item", "market", "price_type"]).any()


def test_wholesale_and_retail_stay_apart(nov_2026):
    pairs = set(zip(nov_2026["price_type"], nov_2026["market"]))
    assert {("wholesale", "Pettah"), ("wholesale", "Dambulla"), ("retail", "Pettah"), ("retail", "Dambulla"), ("retail", "Narahenpita")} <= pairs
    assert ("wholesale", "Narahenpita") not in pairs  # CBSL publishes no wholesale price there


def test_not_available_stays_missing(nov_2026):
    # Red Onion (Imp) is "n.a." in today's Dambulla columns.
    assert pd.isna(_price(nov_2026, "Red Onion (Imp)", "wholesale", "Dambulla"))
    assert _price(nov_2026, "Red Onion (Imp)", "wholesale", "Pettah") == 535.0


def test_vegetable_items_are_mapped_and_others_are_not(nov_2026):
    mapped = nov_2026.drop_duplicates("item").set_index("item")["vegetable"]
    assert mapped["Snake gourd"] == "Snake Gourd"
    assert mapped["Potato (Imp)"] == "Potato"
    assert pd.isna(mapped["Coconut (Avg.)"])
    assert set(nov_2026.loc[nov_2026["item"] == "Coconut (Avg.)", "unit"]) == {"Rs./Nut"}


def test_older_2021_layout_and_misprint():
    table = _table("price_report_20210804e.pdf")
    assert set(table["date"]) == {pd.Timestamp("2021-08-04")}
    assert _price(table, "Beans", "wholesale", "Pettah") == 255.0  # "Yesterday" was 260
    assert _price(table, "Beans", "retail", "Narahenpita") == 280.0
    assert "Red Onion (Imp)" in set(table["item"])  # printed "Red Onion (lmp)"
    assert "Red Onion (lmp)" in set(table["item_as_printed"])


def test_a_report_whose_table_date_disagrees_with_its_name_is_refused():
    report = parse_cbsl_price_report((FIXTURES / "price_report_20261109_e.pdf").read_bytes(), "price_report_20261106_e.pdf")
    assert report.rows == [] and "file name" in report.error


def test_unreadable_input_is_reported_not_raised():
    assert parse_cbsl_price_report(b"not a pdf", "price_report_20261109_e.pdf").error.startswith("could not open PDF")


def test_helpers():
    assert report_date_from_name("price_report_20210804e.pdf") == pd.Timestamp("2021-08-04")
    assert report_date_from_name("other.pdf") is None
    assert normalise_item("Red  Onion (lmp)") == "Red Onion (Imp)"
