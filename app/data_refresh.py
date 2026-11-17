"""Runs scripts/daily_update.py on request from the app (the Home page's update button).

The update itself is unchanged: it backs up every file before changing it, appends only bulletins
not stored yet, never rewrites earlier prices, and takes a lock so two updates can't overlap. This
module only starts it in a separate process and turns its exit code into a message for the page.
Nothing here runs on a normal page load.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.i18n import t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATE_SCRIPT = PROJECT_ROOT / "scripts" / "daily_update.py"
DEFAULT_TIMEOUT_SECONDS = 15 * 60

# Exit codes documented in scripts/daily_update.py.
EXIT_OK = 0
EXIT_UNREADABLE_BULLETIN = 2
EXIT_ALREADY_RUNNING = 3
EXIT_SOURCE_UNAVAILABLE = 4


@dataclass(frozen=True)
class RefreshResult:
    level: str  # "success", "warning" or "error"
    message: str
    output_tail: str = ""


def interpret_exit_code(code: int | None, output: str = "") -> RefreshResult:
    """The message to show for a finished (or failed) update run."""
    tail = "\n".join(output.strip().splitlines()[-15:])
    if code == EXIT_OK:
        return RefreshResult("success", t("Update finished. Any newly published official bulletins were added; existing prices were kept."), tail)
    if code == EXIT_UNREADABLE_BULLETIN:
        return RefreshResult(
            "warning",
            t(
                "Update finished, but at least one bulletin couldn't be read, so its prices are missing. "
                "Everything else that was readable was added, and existing data was kept."
            ),
            tail,
        )
    if code == EXIT_SOURCE_UNAVAILABLE:
        return RefreshResult(
            "warning",
            t(
                "Update finished, but at least one official source couldn't be checked (see the source status). "
                "The other sources were checked, and that source's stored data was kept unchanged."
            ),
            tail,
        )
    if code == EXIT_ALREADY_RUNNING:
        return RefreshResult("warning", t("An update is already running. Wait for it to finish, then reload this page."), tail)
    if code is None:
        return RefreshResult("error", t("The update took too long and was stopped. The existing data was kept; try again later."), tail)
    return RefreshResult(
        "error",
        t(
            "The update failed (exit code {code}), for example because a source website was unreachable. "
            "The existing data was kept unchanged; the app keeps showing it with its real dates.",
            code=code,
        ),
        tail,
    )


def run_data_update(timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, python: str | None = None) -> RefreshResult:
    """Run the daily update once, with the app's own Python, and report the outcome."""
    if not UPDATE_SCRIPT.exists():
        return RefreshResult("error", t("The update script is missing: {path}.", path=UPDATE_SCRIPT.relative_to(PROJECT_ROOT)))
    try:
        completed = subprocess.run(
            [python or sys.executable, str(UPDATE_SCRIPT)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        output = error.output if isinstance(error.output, str) else ""
        return interpret_exit_code(None, output)
    except OSError as error:
        return RefreshResult("error", t("The update couldn't be started: {error}. The existing data was kept.", error=error))
    return interpret_exit_code(completed.returncode, (completed.stdout or "") + (completed.stderr or ""))
