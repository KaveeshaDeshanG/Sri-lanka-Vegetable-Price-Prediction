"""Data freshness labels (app/catalogue.py data_freshness): every source, read from its file and the update log.

Small datasets in the project's real formats are written to a temporary folder; one test also checks
the project's real data.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.catalogue import DATA_SOURCES, FRESHNESS_LABELS, data_freshness, last_run_problem

NOW = pd.Timestamp("2026-11-17 21:00")
PAGES = Path(__file__).resolve().parents[2] / "app"


def _source(key, status="ok", checked="2026-11-17T16:00:12", **extra):
    return {key: {"status": status, "checked_at": checked, **extra}}


def _run(sources, status="ok", finished="2026-11-17T16:00:55", **extra):
    return {"started_at": finished, "status": status, "finished_at": finished, "sources": sources, **extra}


@pytest.fixture
def raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"date": ["2026-11-16", "2026-11-17"], "vegetable": "Beans", "market": "Dambulla", "price": [430.0, 450.0]}).to_csv(raw / "harti_prices.csv", index=False)
    pd.DataFrame({"date": ["2026-11-16", "2026-11-17"], "item": "Beans", "price": [400.0, 420.0]}).to_csv(raw / "cbsl_daily_prices.csv", index=False)
    pd.DataFrame({"year": [2026, 2026], "month": [10, 11], "week_of_month": [4, 1], "price": [500.0, 510.0]}).to_csv(raw / "dcs_weekly_retail_prices.csv", index=False)
    days = pd.date_range("2026-11-10", "2026-11-14")
    pd.DataFrame({"date": list(days) * 2, "region": ["Badulla"] * 5 + ["Dambulla"] * 5, "rainfall_mm": 1.0}).to_csv(raw / "weather_data_by_region.csv", index=False)
    fuel_days = pd.date_range("2026-10-25", "2026-11-17")
    pd.DataFrame({"date": fuel_days, "diesel_price": [392.0 if d < pd.Timestamp("2026-11-02") else 397.0 for d in fuel_days],
                  "petrol_price": [414.0 if d < pd.Timestamp("2026-11-02") else 424.0 for d in fuel_days]}).to_csv(raw / "fuel_prices.csv", index=False)
    return raw


def _log(raw, *runs):
    (raw / "update_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")


def _labels(raw, now=NOW):
    return {row.source.key: row for row in data_freshness(raw, now)}


def _all_ok(harti_new=0):
    return {
        **_source("harti", newest_published="2026-11-17", new_records=harti_new),
        **_source("cbsl", newest_published="2026-11-17", new_records=0),
        **_source("dcs", newest_published="2026-11 week 1", new_records=0),
        **_source("weather"),
        **_source("fuel"),
    }


def test_every_source_is_reported_with_its_latest_stored_data(raw):
    _log(raw, _run(_all_ok()))
    rows = _labels(raw)
    assert list(rows) == [s.key for s in DATA_SOURCES] == ["harti", "cbsl", "dcs", "weather", "fuel"]
    assert {k: r.latest_stored for k, r in rows.items()} == {
        "harti": "2026-11-17", "cbsl": "2026-11-17", "dcs": "2026-11 week 1", "weather": "2026-11-14", "fuel": "2026-11-02",
    }
    assert all(r.label == "Latest Available Data" for r in rows.values())
    assert rows["harti"].last_success == pd.Timestamp("2026-11-17 16:00:12")
    assert "diesel Rs 397, petrol 92 Rs 424" in rows["fuel"].note  # the revision in force, and its date
    assert rows["weather"].source.category == "Observed weather, not a forecast"


def test_new_records_in_the_last_day_are_recently_updated(raw):
    _log(raw, _run(_all_ok(harti_new=192)))
    assert _labels(raw)["harti"].label == "Recently Updated"
    assert _labels(raw, NOW + pd.Timedelta(hours=30))["harti"].label == "Update Pending"  # and a day later, unchecked


def test_a_failed_check_is_source_unavailable_and_keeps_the_stored_data(raw):
    sources = {**_all_ok(), **_source("cbsl", status="failed", checked="2026-11-17T19:00:30", error="ConnectionError: cbsl.gov.lk unreachable")}
    _log(raw, _run(_all_ok()), _run(sources, status="partial", finished="2026-11-17T19:01:00"))
    cbsl = _labels(raw)["cbsl"]
    assert cbsl.label == "Source Unavailable" and "unreachable" in cbsl.note and "unchanged" in cbsl.note
    assert cbsl.latest_stored == "2026-11-17" and cbsl.last_success == pd.Timestamp("2026-11-17 16:00:12")


def test_a_publication_not_stored_yet_is_update_pending(raw):
    sources = {**_all_ok(), **_source("dcs", newest_published="2026-11 week 2", new_records=0)}
    _log(raw, _run(sources))
    dcs = _labels(raw)["dcs"]
    assert dcs.label == "Update Pending" and "2026-11 week 2" in dcs.note


def test_no_check_for_a_day_is_update_pending(raw):
    _log(raw, _run(_all_ok()))
    rows = _labels(raw, pd.Timestamp("2026-11-18 18:00"))
    assert all(r.label == "Update Pending" for r in rows.values())
    assert "Not checked for 25 hours" in rows["harti"].note


def test_a_source_never_checked_is_update_pending(raw):
    _log(raw)
    assert {r.label for r in _labels(raw).values()} == {"Update Pending"}


def test_weather_more_than_its_archive_delay_behind_is_update_pending(raw):
    _log(raw, _run(_all_ok()))
    assert _labels(raw)["weather"].label == "Latest Available Data"  # 14 Nov on 17 Nov: the normal 3-day delay
    assert _labels(raw, pd.Timestamp("2026-11-19 10:00"))["weather"].label == "Update Pending"


def test_an_interrupted_last_run_is_reported(raw):
    interrupted = {"started_at": "2026-11-17T19:00:12", "status": "interrupted", "error": "the update (process 17820) stopped before finishing",
                   "finished_at": "2026-11-17T21:30:00"}
    _log(raw, _run(_all_ok()), interrupted)
    assert "was interrupted" in last_run_problem(raw) and "17820" in last_run_problem(raw)
    assert _labels(raw)["harti"].label == "Latest Available Data"  # the last completed check still stands
    _log(raw, _run(_all_ok()))
    assert last_run_problem(raw) is None


def test_a_missing_data_file_is_reported_not_invented(raw):
    (raw / "cbsl_daily_prices.csv").unlink()
    _log(raw, _run(_all_ok()))
    cbsl = _labels(raw)["cbsl"]
    assert cbsl.latest_stored is None and cbsl.latest_stored_date is None and "couldn't be read" in cbsl.note


def test_the_projects_real_data_gets_a_valid_label_for_every_source():
    rows = data_freshness("data/raw")
    assert len(rows) == 5 and all(row.label in FRESHNESS_LABELS for row in rows)
    assert all(row.latest_stored for row in rows)


@pytest.mark.parametrize("page", ["Home.py", "pages/2_Model_Performance.py", "pages/3_Price_History.py", "pages/4_Cultivation_Planner.py"])
def test_every_page_shows_data_freshness(page):
    at = AppTest.from_file(str(PAGES / page), default_timeout=240).run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert 'class="vp-fresh"' in html and "HARTI daily bulletin" in html and "CEYPETCO fuel prices" in html
    if page != "Home.py":
        assert any(e.label.startswith("Data freshness:") for e in at.expander)
