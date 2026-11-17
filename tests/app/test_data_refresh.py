"""The Home page's update button: runs scripts/daily_update.py and reports its outcome honestly."""

import sys

import pytest

from app import data_refresh
from app.data_refresh import EXIT_ALREADY_RUNNING, EXIT_OK, EXIT_UNREADABLE_BULLETIN, interpret_exit_code, run_data_update


@pytest.mark.parametrize(
    ("code", "level"),
    [(EXIT_OK, "success"), (EXIT_UNREADABLE_BULLETIN, "warning"), (EXIT_ALREADY_RUNNING, "warning"), (1, "error"), (None, "error")],
)
def test_each_exit_code_has_an_honest_message(code, level):
    result = interpret_exit_code(code, "line 1\nline 2")
    assert result.level == level
    if level != "success":
        # A failure never claims new data arrived.
        assert "added" not in result.message.split(".")[0] or code == EXIT_UNREADABLE_BULLETIN
    if code not in (EXIT_OK, EXIT_UNREADABLE_BULLETIN, EXIT_ALREADY_RUNNING):
        assert "kept" in result.message


def test_only_the_last_lines_of_output_are_kept():
    result = interpret_exit_code(1, "\n".join(f"line {n}" for n in range(100)))
    assert result.output_tail.splitlines() == [f"line {n}" for n in range(85, 100)]


def _fake_update_script(tmp_path, monkeypatch, body: str):
    script = tmp_path / "daily_update.py"
    script.write_text(body, encoding="utf-8")
    monkeypatch.setattr(data_refresh, "UPDATE_SCRIPT", script)
    monkeypatch.setattr(data_refresh, "PROJECT_ROOT", tmp_path)


def test_a_successful_run_is_reported(tmp_path, monkeypatch):
    _fake_update_script(tmp_path, monkeypatch, "print('HARTI: newest listed 2026-11-12')\n")
    assert run_data_update(timeout_seconds=60, python=sys.executable).level == "success"


def test_an_unreadable_bulletin_is_a_warning_not_a_success(tmp_path, monkeypatch):
    _fake_update_script(tmp_path, monkeypatch, "import sys\nprint('unreadable')\nsys.exit(2)\n")
    result = run_data_update(timeout_seconds=60, python=sys.executable)
    assert result.level == "warning"
    assert "unreadable" in result.output_tail


def test_an_unreachable_source_keeps_the_existing_data(tmp_path, monkeypatch):
    _fake_update_script(tmp_path, monkeypatch, "raise ConnectionError('harti.gov.lk unreachable')\n")
    result = run_data_update(timeout_seconds=60, python=sys.executable)
    assert result.level == "error"
    assert "existing data was kept" in result.message
    assert "unreachable" in result.output_tail


def test_a_run_that_takes_too_long_is_stopped(tmp_path, monkeypatch):
    _fake_update_script(tmp_path, monkeypatch, "import time\ntime.sleep(30)\n")
    assert run_data_update(timeout_seconds=1, python=sys.executable).level == "error"


def test_a_missing_script_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(data_refresh, "UPDATE_SCRIPT", tmp_path / "missing.py")
    monkeypatch.setattr(data_refresh, "PROJECT_ROOT", tmp_path)
    assert run_data_update().level == "error"


def test_the_real_update_script_exists():
    assert data_refresh.UPDATE_SCRIPT.exists()


def test_an_unavailable_source_is_a_warning():
    from app.data_refresh import EXIT_SOURCE_UNAVAILABLE

    result = interpret_exit_code(EXIT_SOURCE_UNAVAILABLE)
    assert result.level == "warning" and "couldn't be checked" in result.message
