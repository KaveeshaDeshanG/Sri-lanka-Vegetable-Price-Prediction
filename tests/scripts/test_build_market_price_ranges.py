"""scripts/build_market_price_ranges.py: display-only min-max ranges that can never disagree with the
stored other-market prices. Uses the real 2026-11-09 HARTI bulletin in tests/fixtures."""

import shutil
from pathlib import Path

import pandas as pd

from scripts.build_market_price_ranges import RANGES_FILE, update_ranges
from src.data_collection.harti_live_client import HartiLiveClient, parse_all_markets
from src.preprocessing.transformer import DataTransformer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "harti_daily_2026-11-09.pdf"
DAY = pd.Timestamp("2026-11-09")


def _table():
    return HartiLiveClient()._extract_vegetable_table(FIXTURE.read_bytes())


def test_the_default_parse_is_unchanged_and_the_range_is_optional():
    table = _table()
    plain = parse_all_markets(table, DAY)
    ranged = parse_all_markets(table, DAY, keep_range=True)
    assert list(plain.columns) == ["published", "reported", "market", "vegetable", "price"]
    assert list(ranged.columns) == [*plain.columns, "low", "high"]
    pd.testing.assert_frame_equal(plain, ranged[plain.columns])
    assert ((ranged["low"] + ranged["high"]) / 2 == ranged["price"]).all() and (ranged["low"] <= ranged["high"]).all()


def test_ranges_are_added_only_where_they_match_the_stored_price(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir(), cache.mkdir()
    shutil.copy(FIXTURE, cache / "2026-11-09_0000000000000000.pdf")
    panel = DataTransformer().standardize_vegetable_names(parse_all_markets(_table(), DAY))
    tampered = panel.index[(panel["market"] == "Kandy") & (panel["vegetable"] == "Beans")][0]
    panel.loc[tampered, "price"] += 1  # a stored price the bulletin doesn't support
    panel.to_csv(raw / "harti_all_markets.csv", index=False)

    report = update_ranges(raw, cache)
    ranges = pd.read_csv(raw / RANGES_FILE)
    others = panel[panel["market"] != "Dambulla"]
    assert "Dambulla" not in set(ranges["market"])  # its ranges are already in harti_prices_detail.csv
    assert report["rows_not_matching_stored_price"] == 1 and report["rows_added"] == len(others) - 1
    assert not ((ranges["market"] == "Kandy") & (ranges["vegetable"] == "Beans")).any()
    merged = others.astype({"published": str, "reported": str}).merge(ranges.astype({"published": str, "reported": str}), on=["published", "reported", "market", "vegetable"])
    assert ((merged["low"] + merged["high"]) / 2 == merged["price"]).all()

    again = update_ranges(raw, cache)  # incremental: the bulletin is already done
    assert again["bulletins_read"] == 0 and again["rows_added"] == 0
    assert len(pd.read_csv(raw / RANGES_FILE)) == len(ranges)
