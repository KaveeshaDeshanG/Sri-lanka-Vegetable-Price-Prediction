"""What the Predict page says about the latest checks of the official sources."""

import json

import pandas as pd

from app.catalogue import UPDATE_OVERDUE_HOURS, latest_update, update_notice, update_status

NOW = pd.Timestamp("2026-11-15 12:00")


def _log(tmp_path, *entries):
    path = tmp_path / "update_log.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in entries) + "not json\n", encoding="utf-8")
    return path


OLD_STYLE = {"finished_at": "2026-11-13T20:07:41", "harti": {"newest_listed": "2026-11-12"}}  # before runs had a status
OK = {"status": "ok", "finished_at": "2026-11-15T11:48:03", "harti": {"newest_listed": "2026-11-12"}}
FAILED = {"status": "failed", "finished_at": "2026-11-15T14:00:00", "error": "ConnectionError: harti.gov.lk unreachable"}


def test_lines_without_a_status_count_as_completed_runs(tmp_path):
    status = update_status(_log(tmp_path, OLD_STYLE), now=NOW)
    assert status.last_success == pd.Timestamp("2026-11-13 20:07:41") and not status.last_attempt_failed


def test_a_failed_attempt_never_replaces_the_last_success(tmp_path):
    path = _log(tmp_path, OK, FAILED)
    assert latest_update(path)["finished_at"] == OK["finished_at"]
    status = update_status(path, now=NOW)
    assert status.last_success == pd.Timestamp("2026-11-15 11:48:03")
    assert status.last_attempt_failed and "unreachable" in status.last_error
    level, message = update_notice(status, "2026-11-12", now=NOW)
    assert level == "warning" and "failed" in message and "2026-11-12" in message


def test_current_data_after_a_recent_check_is_not_a_warning(tmp_path):
    level, message = update_notice(update_status(_log(tmp_path, OK), now=NOW), "2026-11-12", now=NOW)
    assert level == "info" and "2026-11-15 11:48" in message


def test_no_recent_check_is_flagged_as_overdue(tmp_path):
    later = NOW + pd.Timedelta(hours=UPDATE_OVERDUE_HOURS + 1)
    status = update_status(_log(tmp_path, OK), now=later)
    assert status.overdue
    level, message = update_notice(status, "2026-11-12", now=later)
    assert level == "warning" and "haven't been checked" in message


def test_old_data_when_the_source_has_nothing_newer_says_so(tmp_path):
    now = pd.Timestamp("2026-11-17 12:00")
    checked = {**OK, "finished_at": "2026-11-17T11:00:00"}
    level, message = update_notice(update_status(_log(tmp_path, checked), now=now), "2026-11-12", now=now)
    assert level == "warning" and "hasn't published a newer bulletin" in message and "5 days old" in message


def test_old_series_when_the_source_has_newer_bulletins(tmp_path):
    now = pd.Timestamp("2026-11-20 12:00")
    checked = {"status": "ok", "finished_at": "2026-11-20T11:00:00", "harti": {"newest_listed": "2026-11-19"}}
    level, message = update_notice(update_status(_log(tmp_path, checked), now=now), "2026-11-12", now=now)
    assert level == "warning" and "2026-11-19" in message and "may not include this series" in message


def test_no_log_at_all(tmp_path):
    status = update_status(tmp_path / "missing.jsonl", now=NOW)
    assert status.last_success is None and status.overdue and not status.last_attempt_failed
    level, message = update_notice(status, "2026-11-12", now=NOW)
    assert level == "warning" and "no successful check recorded" in message


def test_a_partial_run_counts_as_completed_but_a_failed_harti_check_is_a_warning(tmp_path):
    partial = {**OK, "status": "partial", "sources": {"harti": {"status": "failed", "error": "ConnectionError: unreachable"}, "cbsl": {"status": "ok"}}}
    status = update_status(_log(tmp_path, partial), now=NOW)
    assert status.last_success is not None and not status.last_attempt_failed
    assert status.sources["harti"]["status"] == "failed"
    level, message = update_notice(status, "2026-11-12", now=NOW)
    assert level == "warning" and "HARTI couldn't be checked" in message
