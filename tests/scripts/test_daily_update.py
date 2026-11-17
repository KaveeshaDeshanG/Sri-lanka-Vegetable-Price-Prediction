"""scripts/daily_update.py: incremental, validated, idempotent imports that never change stored rows.

Network access is replaced with the real bulletins in tests/fixtures; every file is written to a
temporary folder.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import daily_update
from scripts.daily_update import Backup, import_cbsl, import_harti, single_run, validate_dambulla_rows, UpdateError
from config.config_loader import ConfigLoader

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HARTI_URL = "https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2026/November/Vegetable%20Pricenew%20ex1(2026.11.09).pdf"
TODAY = pd.Timestamp("2026-11-09")


@pytest.fixture
def raw_dir(tmp_path):
    """A small stored dataset ending 2026-11-06, in the project's real formats."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        [
            {"date": "2026-11-06", "vegetable": "Beans", "market": "Dambulla", "price": 450.0, "low": 400.0, "high": 500.0,
             "raw_cell": "400.00 - 500.00", "bulletin_date": "2026-11-06", "source_pdf": "https://example/old.pdf"},
            {"date": "2026-11-06", "vegetable": "Carrot", "market": "Dambulla", "price": 500.0, "low": 480.0, "high": 520.0,
             "raw_cell": "480.00 - 520.00", "bulletin_date": "2026-11-06", "source_pdf": "https://example/old.pdf"},
        ]
    ).to_csv(raw / "harti_prices_detail.csv", index=False)
    pd.read_csv(raw / "harti_prices_detail.csv")[["date", "vegetable", "market", "price"]].to_csv(raw / "harti_prices.csv", index=False)
    pd.DataFrame(
        [{"published": "2026-11-06", "reported": "2026-11-06", "market": "Kandy", "vegetable": "Carrot", "price": 510.0}]
    ).to_csv(raw / "harti_all_markets.csv", index=False)
    return raw


@pytest.fixture
def offline_harti(monkeypatch, tmp_path):
    """The archive lists 2026-11-09 (and an older, already stored day); downloads read the fixture."""
    entries = pd.DataFrame({"date": pd.to_datetime(["2026-11-06", "2026-11-09"]), "url": ["https://example/old.pdf", HARTI_URL]})
    monkeypatch.setattr(daily_update, "harti_entries", lambda config, offline, cache_dir: entries)
    calls = []

    def fake_download(url, path, timeout, attempts=3):
        calls.append(url)
        data = (FIXTURES / "harti_daily_2026-11-09.pdf").read_bytes()
        path.write_bytes(data)
        return data

    monkeypatch.setattr(daily_update, "download_pdf", fake_download)
    return calls


def _run_harti(raw_dir, tmp_path):
    backup = Backup(tmp_path / "backups", keep=3)
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    summary = import_harti(ConfigLoader("config/config.yaml"), raw_dir, cache, backup, 30, TODAY)
    return summary, backup


def test_new_bulletin_is_added_and_stored_rows_are_untouched(raw_dir, tmp_path, offline_harti):
    before = pd.read_csv(raw_dir / "harti_prices_detail.csv")
    summary, backup = _run_harti(raw_dir, tmp_path)

    assert offline_harti == [HARTI_URL]  # the stored 2026-11-06 bulletin isn't fetched again
    detail = pd.read_csv(raw_dir / "harti_prices_detail.csv")
    pd.testing.assert_frame_equal(detail.iloc[: len(before)].reset_index(drop=True), before)
    new = detail[detail["bulletin_date"] == "2026-11-09"]
    assert summary["dambulla_rows_added"] == len(new) > 20
    carrot = new[new["vegetable"] == "Carrot"].iloc[0]
    assert (carrot["date"], carrot["price"]) == ("2026-11-09", 615.0)  # the bulletin's own Dambulla reading
    assert carrot["source_pdf"] == HARTI_URL
    prices = pd.read_csv(raw_dir / "harti_prices.csv")
    assert list(prices.columns) == ["date", "vegetable", "market", "price"] and len(prices) == len(detail)

    panel = pd.read_csv(raw_dir / "harti_all_markets.csv")
    assert set(panel.loc[panel["published"] == "2026-11-09", "market"]) >= {"Dambulla", "Kandy", "Peliyagoda", "Meegoda"}
    assert "Brinjal" in set(panel["vegetable"]) and "Eggplant" not in set(panel["vegetable"])  # training's name
    assert not panel.duplicated(["published", "market", "vegetable"]).any()

    log = pd.read_csv(raw_dir / "harti_import_log.csv")
    assert log.iloc[-1]["status"] == "imported" and len(log.iloc[-1]["sha256"]) == 64
    assert set(backup.saved) >= {"harti_prices_detail.csv", "harti_prices.csv", "harti_all_markets.csv"}
    pd.testing.assert_frame_equal(pd.read_csv(backup.folder / "harti_prices_detail.csv"), before)


def test_running_twice_adds_nothing_the_second_time(raw_dir, tmp_path, offline_harti):
    _run_harti(raw_dir, tmp_path)
    first = (raw_dir / "harti_prices_detail.csv").read_bytes(), (raw_dir / "harti_all_markets.csv").read_bytes()
    summary, _ = _run_harti(raw_dir, tmp_path)
    assert summary["bulletins_read"] == 0 and summary["dambulla_rows_added"] == 0
    assert ((raw_dir / "harti_prices_detail.csv").read_bytes(), (raw_dir / "harti_all_markets.csv").read_bytes()) == first


def test_an_unreadable_bulletin_is_reported_and_nothing_is_written(raw_dir, tmp_path, monkeypatch, offline_harti):
    def broken(url, path, timeout, attempts=3):
        path.write_bytes(b"%PDF-1.4 broken")
        return b"%PDF-1.4 broken"

    monkeypatch.setattr(daily_update, "download_pdf", broken)
    before = (raw_dir / "harti_prices_detail.csv").read_bytes()
    summary, _ = _run_harti(raw_dir, tmp_path)
    assert summary["unreadable"] == ["2026-11-09"]
    assert (raw_dir / "harti_prices_detail.csv").read_bytes() == before
    assert pd.read_csv(raw_dir / "harti_import_log.csv").iloc[-1]["status"] == "unreadable"


def test_validation_rejects_impossible_rows():
    bulletin = pd.Timestamp("2026-11-09")
    good = {"vegetable": "Carrot", "low": 500.0, "high": 600.0, "raw_cell": "500 - 600", "date": bulletin}
    assert validate_dambulla_rows([good], bulletin) == []
    assert validate_dambulla_rows([{**good, "low": 700.0}], bulletin)
    assert validate_dambulla_rows([{**good, "low": 0.0}], bulletin)
    assert validate_dambulla_rows([{**good, "date": pd.Timestamp("2026-11-10")}], bulletin)  # after its bulletin
    assert validate_dambulla_rows([{**good, "date": pd.Timestamp("2026-10-01")}], bulletin)  # far too old


def test_cbsl_import_adds_published_days_and_settles_missing_ones(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = (FIXTURES / "price_report_20261109_e.pdf").read_bytes()

    def fake_fetch(day, session):
        if day == TODAY:
            return pdf, daily_update.CBSL_BASE + "price_report_20261109_e.pdf"
        return None  # a holiday, or not uploaded yet

    monkeypatch.setattr(daily_update, "fetch_cbsl_report", fake_fetch)
    backup = Backup(tmp_path / "backups", keep=3)
    summary = import_cbsl(raw, tmp_path / "cbsl_cache", backup, pd.Timestamp("2026-11-02"), TODAY)

    table = pd.read_csv(raw / "cbsl_daily_prices.csv")
    assert summary["days_added"] == ["2026-11-09"]
    assert set(table["date"]) == {"2026-11-09"}
    assert set(table["price_type"]) == {"wholesale", "retail"}
    assert table["source_url"].str.endswith("price_report_20261109_e.pdf").all()
    state = json.loads((raw / "cbsl_import_state.json").read_text())
    # More than 3 days old and still missing: settled as not published. Recent days are retried.
    assert "2026-11-02" in state["not_published"] and "2026-11-06" not in state["not_published"]

    again = import_cbsl(raw, tmp_path / "cbsl_cache", backup, pd.Timestamp("2026-11-02"), TODAY)
    assert again["days_added"] == [] and len(pd.read_csv(raw / "cbsl_daily_prices.csv")) == len(table)


def test_dcs_skips_links_known_to_have_no_pdf_and_rebuilds_only_on_news(tmp_path, monkeypatch):
    raw, cache = tmp_path / "raw", tmp_path / "dcs"
    raw.mkdir()
    cache.mkdir()
    (cache / "listing.json").write_text(json.dumps([{"link": "https://x/retail/DCSB-WRP-2019-12-W3", "error": "not a PDF"}]))
    (cache / "DCSB-WRP-2026-11-W1.pdf").write_bytes(b"%PDF")
    listing = [{"link": f"https://x/retail/{name}"} for name in ("DCSB-WRP-2019-12-W3", "DCSB-WRP-2026-11-W1", "DCSB-WRP-2026-11-W2")]
    requested, rebuilt = [], []
    monkeypatch.setattr(daily_update, "dcs_listing", lambda: listing)
    monkeypatch.setattr(daily_update, "download_new_dcs", lambda items, cache_dir, delay: requested.extend(i["link"] for i in items) or [])
    monkeypatch.setattr(daily_update, "rebuild_dcs", lambda cache_dir, raw_dir, items: rebuilt.append(True) or (raw_dir / "dcs_weekly_retail_prices.csv").write_text("x"))
    monkeypatch.setattr(daily_update, "_resolve", lambda path: cache)
    backup = Backup(tmp_path / "backups", keep=3)

    summary = daily_update.import_dcs(ConfigLoader("config/config.yaml"), raw, backup)
    assert requested == ["https://x/retail/DCSB-WRP-2026-11-W2"]  # neither the cached nor the known-missing one
    assert summary["new_bulletins"] == 1 and rebuilt == [True]
    assert "dcs_weekly_retail_prices.csv" in backup.changed  # a brand-new file still counts as a change

    requested.clear()
    (cache / "DCSB-WRP-2026-11-W2.pdf").write_bytes(b"%PDF")
    summary = daily_update.import_dcs(ConfigLoader("config/config.yaml"), raw, Backup(tmp_path / "backups", keep=3))
    assert requested == [] and summary["new_bulletins"] == 0 and rebuilt == [True]  # nothing new: no rebuild


def test_only_one_update_runs_at_a_time(tmp_path):
    lock = tmp_path / ".update.lock"
    with single_run(lock):
        with pytest.raises(UpdateError):
            with single_run(lock):
                pass
    assert not lock.exists()


def test_backups_keep_only_the_newest_folders(tmp_path):
    for stamp in ("20260101_000000", "20260102_000000", "20260103_000000"):
        (tmp_path / "daily" / stamp).mkdir(parents=True)
    backup = Backup(tmp_path, keep=2)
    backup.prune()
    assert sorted(p.name for p in (tmp_path / "daily").iterdir()) == ["20260102_000000", "20260103_000000"]


def test_a_failed_run_is_recorded_and_the_data_is_left_alone(tmp_path, monkeypatch):
    """E.g. harti.gov.lk unreachable: exit code 1 and a 'failed' log line, never a fake success."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "update_log.jsonl").write_text(json.dumps({"status": "ok", "finished_at": "2026-11-15T11:48:03"}) + "\n", encoding="utf-8")
    (raw / "harti_prices.csv").write_text("date,vegetable,market,price\n2026-11-12,Beans,Dambulla,430.0\n", encoding="utf-8")
    before = (raw / "harti_prices.csv").read_bytes()

    def unreachable(args):
        raise ConnectionError("harti.gov.lk unreachable")

    monkeypatch.setattr(daily_update, "run", unreachable)
    monkeypatch.setattr(daily_update, "_resolve", lambda path: raw)
    assert daily_update.main([]) == 1
    lines = [json.loads(line) for line in (raw / "update_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["status"] == "ok"
    assert lines[-1]["status"] == "failed" and "unreachable" in lines[-1]["error"]
    assert (raw / "harti_prices.csv").read_bytes() == before


def test_a_run_that_is_already_in_progress_is_not_recorded_as_failed(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(daily_update, "_resolve", lambda path: raw)
    with single_run(raw / ".update.lock"):
        assert daily_update.main([]) == 3
    assert not (raw / "update_log.jsonl").exists()


def test_cbsl_network_errors_are_reported_not_mistaken_for_unpublished_days(tmp_path, monkeypatch):
    import requests

    raw = tmp_path / "raw"
    raw.mkdir()

    def offline(day, session):
        raise requests.exceptions.ConnectionError("www.cbsl.gov.lk unreachable")

    monkeypatch.setattr(daily_update, "fetch_cbsl_report", offline)
    summary = import_cbsl(raw, tmp_path / "cache", Backup(tmp_path / "backups", keep=3), pd.Timestamp("2026-11-02"), TODAY)
    assert summary["days_added"] == [] and summary["not_published_yet"] == []
    assert len(summary["request_errors"]) == 6 and "unreachable" in summary["request_errors"]["2026-11-09"]
    # Nothing settled as "not published", so every day is asked again next time.
    assert json.loads((raw / "cbsl_import_state.json").read_text())["not_published"] == []


def test_newest_dcs_week_is_read_from_the_listed_file_names():
    listing = [
        {"link": "https://www.statistics.gov.lk/InflationAndPrices/StaticalInformation/retail/DCSB-WRP-2026-10-W4"},
        {"link": "https://www.statistics.gov.lk/InflationAndPrices/StaticalInformation/retail/DCSB-WRP-2026-11-W1"},
        {"link": "https://www.statistics.gov.lk/InflationAndPrices/StaticalInformation/retail/DCSB-WRP-2026-9-W3"},
        {"link": "https://example/other.pdf"},
    ]
    assert daily_update.newest_dcs_week(listing) == "2026-11 week 1"
    assert daily_update.newest_dcs_week([{"link": "https://example/other.pdf"}]) is None


def _isolated_run(tmp_path, monkeypatch, harti, cbsl, dcs):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(daily_update, "_resolve", lambda path: raw)
    monkeypatch.setattr(daily_update, "RETRY_WAITS_SECONDS", (0, 0))
    monkeypatch.setattr(daily_update, "import_harti", harti)
    monkeypatch.setattr(daily_update, "import_cbsl", cbsl)
    monkeypatch.setattr(daily_update, "import_dcs", dcs)
    monkeypatch.setattr(daily_update, "topup_weather", lambda *a: {})
    monkeypatch.setattr(daily_update, "topup_fuel", lambda *a: "already current")
    code = daily_update.main(["--today", "2026-11-15"])
    entry = json.loads((raw / "update_log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    return code, entry


def _cbsl_ok(*args):
    return {"days_added": ["2026-11-13"], "not_published_yet": [], "unreadable": {}, "request_errors": {}, "newest_report": "2026-11-13"}


def _dcs_ok(*args):
    return {"listed": 736, "new_bulletins": 0, "unavailable": 0, "newest_listed": "2026-11 week 1"}


def test_one_unreachable_source_does_not_stop_the_others(tmp_path, monkeypatch):
    def harti_down(*args):
        raise ConnectionError("www.harti.gov.lk unreachable")

    code, entry = _isolated_run(tmp_path, monkeypatch, harti_down, _cbsl_ok, _dcs_ok)
    assert code == 4 and entry["status"] == "partial"
    sources = entry["sources"]
    assert sources["harti"]["status"] == "failed" and "unreachable" in sources["harti"]["error"]
    assert sources["cbsl"] == {**sources["cbsl"], "status": "ok", "newest_published": "2026-11-13", "new_records": 1}
    assert sources["dcs"]["status"] == "ok" and sources["dcs"]["newest_published"] == "2026-11 week 1"


def test_every_source_reported_when_all_are_checked(tmp_path, monkeypatch):
    def harti_ok(*args):
        return {"bulletins_listed": 3149, "newest_listed": "2026-11-12", "bulletins_read": 1, "dambulla_rows_added": 30,
                "all_market_rows_added": 223, "unreadable": [], "download_failed": []}

    code, entry = _isolated_run(tmp_path, monkeypatch, harti_ok, _cbsl_ok, _dcs_ok)
    assert code == 0 and entry["status"] == "ok"
    assert entry["sources"]["harti"]["newest_published"] == "2026-11-12" and entry["sources"]["harti"]["new_records"] == 253
    assert {name: s["status"] for name, s in entry["sources"].items()} == {"harti": "ok", "cbsl": "ok", "weather": "ok", "fuel": "ok", "dcs": "ok"}
    assert all("checked_at" in s for s in entry["sources"].values())


# --- interrupted runs, retries and the manifest (2026-11-17 audit) ---------------------------------
# Found in the audit: four scheduled runs were stopped part-way (window closed, Ctrl+C, PC shut down).
# Each left its lock behind, so the next runs refused to start for hours, and none was logged.


def _dead_pid() -> int:
    """The id of a process that has already exited."""
    import subprocess
    import sys

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_process_running_tells_a_live_process_from_a_finished_one():
    import os

    assert daily_update.process_running(os.getpid())
    assert not daily_update.process_running(_dead_pid())
    assert not daily_update.process_running(0)


def test_a_lock_left_by_an_interrupted_run_is_taken_over_and_logged(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    lock = raw / ".update.lock"
    pid = _dead_pid()
    lock.write_text(f"{pid} 2026-11-17T19:00:12.339993", encoding="utf-8")  # minutes old, but its process is gone
    with single_run(lock):
        assert lock.read_text(encoding="utf-8").split()[0] != str(pid)  # this run holds it now
    assert not lock.exists()
    entry = json.loads((raw / "update_log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert entry["status"] == "interrupted" and entry["started_at"] == "2026-11-17T19:00:12.339993"
    assert str(pid) in entry["error"]


def test_a_lock_held_by_a_running_update_still_blocks(tmp_path):
    import os

    lock = tmp_path / ".update.lock"
    lock.write_text(f"{os.getpid()} 2026-11-17T19:00:12", encoding="utf-8")
    with pytest.raises(UpdateError), single_run(lock):
        pass
    assert lock.exists() and not (tmp_path / "update_log.jsonl").exists()


def test_an_unreadable_young_lock_is_treated_as_running(tmp_path):
    lock = tmp_path / ".update.lock"
    lock.write_text("not a pid", encoding="utf-8")
    with pytest.raises(UpdateError), single_run(lock):
        pass


def test_ctrl_c_is_logged_as_interrupted_and_the_lock_released(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()

    def stopped(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(daily_update, "run", stopped)
    monkeypatch.setattr(daily_update, "_resolve", lambda path: raw)
    assert daily_update.main([]) == 130
    assert not (raw / ".update.lock").exists()
    entry = json.loads((raw / "update_log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert entry["status"] == "interrupted"


def test_the_manifest_is_rewritten_when_a_dataset_is_newer(tmp_path):
    import os
    import time

    raw = tmp_path / "raw"
    raw.mkdir()
    assert daily_update.manifest_out_of_date(raw)  # missing
    (raw / "cbsl_daily_prices.csv").write_text("date\n2026-11-17\n", encoding="utf-8")
    manifest = raw / "data_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    assert not daily_update.manifest_out_of_date(raw)
    later = time.time() + 60
    os.utime(raw / "cbsl_daily_prices.csv", (later, later))  # written by a run that stopped before the manifest
    assert daily_update.manifest_out_of_date(raw)


def test_an_unreachable_source_is_retried_before_it_is_called_failed(tmp_path, monkeypatch):
    import requests

    calls = []

    def harti_flaky(*args):
        calls.append(1)
        if len(calls) < 3:
            raise requests.ConnectionError("www.harti.gov.lk didn't answer")
        return {"bulletins_listed": 3154, "newest_listed": "2026-11-17", "bulletins_read": 0, "dambulla_rows_added": 0,
                "all_market_rows_added": 0, "unreadable": [], "download_failed": []}

    code, entry = _isolated_run(tmp_path, monkeypatch, harti_flaky, _cbsl_ok, _dcs_ok)
    assert len(calls) == 3 and code == 0
    assert entry["sources"]["harti"]["status"] == "ok" and entry["sources"]["harti"]["attempts"] == 3


def test_a_source_that_stays_unreachable_is_recorded_after_its_retries(tmp_path, monkeypatch):
    calls = []

    def harti_down(*args):
        calls.append(1)
        raise ConnectionError("www.harti.gov.lk unreachable")

    code, entry = _isolated_run(tmp_path, monkeypatch, harti_down, _cbsl_ok, _dcs_ok)
    assert len(calls) == 1 + len(daily_update.RETRY_WAITS_SECONDS) and code == 4
    assert entry["sources"]["harti"]["status"] == "failed" and entry["sources"]["harti"]["attempts"] == 3


def test_a_page_that_cant_be_read_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def harti_broken(*args):
        calls.append(1)
        raise ValueError("the archive page layout changed")

    code, entry = _isolated_run(tmp_path, monkeypatch, harti_broken, _cbsl_ok, _dcs_ok)
    assert len(calls) == 1 and entry["sources"]["harti"]["status"] == "failed"
