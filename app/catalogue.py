"""The master vegetable catalogue: every vegetable the project's sources mention, and what each supports.

Streamlit-free. Static facts (names, aliases, which crop a price series belongs to) live here and in
data/reference/; what changes with the data (latest observation, whether a forecast is possible) is
computed from the datasets every time, so the app never shows a stale support status.

Internal series names are the ones the trained model uses and are not renamed. Two of them are
easy to confuse, so they get display names: the model's "Brinjal" is HARTI's "Eggplant" row,
which HARTI lists separately from its "Brinjals" row.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.models.direct_forecaster import MAX_HORIZON_DAYS

# internal series name -> (display name, crop in the DOA reference, labels as printed in sources)
HARTI_SERIES: dict[str, tuple[str, str | None, str]] = {
    "Ash Plantain": ("Ash Plantain", None, "Ash Plantains"),
    "Beans": ("Beans", "Beans", "Beans"),
    "Beetroot": ("Beetroot", "Beetroot", "Beet root; Beetroot"),
    "Beetroot (Nuwara Eliya)": ("Beetroot (Nuwara Eliya)", "Beetroot", "Beet Root(N'Eliya); Beetroot(N Eliya)"),
    "Big Onion (Imported)": ("Big Onion (Imported)", None, "B'Onion Imported"),
    "Big Onion (Local)": ("Big Onion (Local)", None, "Big-onion Local"),
    "Bitter Gourd": ("Bitter Gourd", "Bitter Gourd", "Bitter Gourd"),
    "Bitter Gourd (Other)": ("Bitter Gourd (Other)", "Bitter Gourd", "Bitter Gourd (Other)"),
    "Bitter Gourd (Village)": ("Bitter Gourd (Village)", "Bitter Gourd", "Bitter Gourd (Village)"),
    "Brinjal": ("Eggplant", "Brinjal", "Eggplant"),
    "Brinjals": ("Brinjals", "Brinjal", "Brinjals"),
    "Brinjals (Other)": ("Brinjals (Other)", "Brinjal", "Brinjals (Other)"),
    "Brinjals (Village)": ("Brinjals (Village)", "Brinjal", "Brinjals (Village)"),
    "Cabbage (Kandy)": ("Cabbage (Kandy)", "Cabbage", "Cabbage (Kandy)"),
    "Cabbage (Nuwara Eliya)": ("Cabbage (Nuwara Eliya)", "Cabbage", "Cabbage (N'Eliya)"),
    "Capsicum": ("Capsicum", "Capsicum", "Capsicum"),
    "Carrot": ("Carrot", "Carrot", "Carrot"),
    "Cucumber": ("Cucumber", "Cucumber", "Cucumber"),
    "Drumstick": ("Drumstick", None, "Drumstick"),
    "Green Chillies": ("Green Chillies", None, "Green Chillies"),
    "Knol Khol": ("Knol Khol", "Knol Khol", "Knolkhol"),
    "Ladies Fingers": ("Ladies Fingers (Okra)", "Okra", "Ladies Fingers"),
    "Leeks": ("Leeks", "Leeks", "Leeks"),
    "Lime": ("Lime", None, "Lime"),
    "Long Beans": ("Long Beans (Mae)", "Long Beans (Mae)", "Long Beans"),
    "Luffa": ("Luffa", "Luffa", "Luffa"),
    "Manioc": ("Manioc (Cassava)", "Manioc", "Manioc"),
    "Potato (Imported)": ("Potato (Imported)", None, "Potato (Imported)"),
    "Potato (Nuwara Eliya)": ("Potato (Nuwara Eliya)", "Potato", "Potato (Nuwaraeliya)"),
    "Potato (Welimada)": ("Potato (Welimada)", "Potato", "Potato (Welimada)"),
    "Pumpkin": ("Pumpkin", "Pumpkin", "Pumpkin"),
    "Radish": ("Radish", "Radish", "Raddish"),
    "Snake Gourd": ("Snake Gourd", "Snake Gourd", "Snake Gourd"),
    "Sweet Potato": ("Sweet Potato", "Sweet Potato", "Sweet Potatoe"),
    # "Thithbatu" and "Dambala (Wing Beans)" are printed as rows in 2015-2023 bulletins, but no market
    # ever has a price in them (checked 2026-11-11 against the full archive), so neither has a series.
    "Thibbatu": ("Thibbatu", "Thibbatu", "Thithbatu"),
    "Tomato": ("Tomato", "Tomato", "Tomato"),
    "Wing Beans": ("Wing Beans (Dambala)", "Winged Bean", "Dambala (Wing Beans)"),
}
# Why a crop with no HARTI price series, or no DOA duration, has that gap.
NO_DURATION_REASONS = {
    "Ash Plantain": "Plantain is a perennial crop; DOA's HORDI pages publish no harvest duration for it.",
    "Big Onion (Imported)": "Imported produce: no local growing period applies.",
    "Potato (Imported)": "Imported produce: no local growing period applies.",
    "Big Onion (Local)": "No DOA HORDI crop page for big onion was found.",
    "Bitter Gourd": "DOA's bitter gourd pages give no harvest duration.",
    "Bitter Gourd (Other)": "DOA's bitter gourd pages give no harvest duration.",
    "Bitter Gourd (Village)": "DOA's bitter gourd pages give no harvest duration.",
    "Cucumber": "DOA's cucumber pages give no harvest duration.",
    "Drumstick": "Drumstick (moringa) is perennial, and DOA's moringa page currently shows the bean page's content.",
    "Green Chillies": "No DOA HORDI crop page for chilli was found.",
    "Lime": "Lime is a perennial fruit crop; no harvest duration applies.",
    "Thibbatu": "DOA's Thithbatu page gives no harvest duration.",
    "Tomato": "DOA's tomato pages give transplanting age (14-18 days after sowing) but no harvest duration.",
}


def display_name(series: str) -> str:
    """The name shown to users for an internal series name (unknown names are shown as-is)."""
    return HARTI_SERIES.get(series, (series,))[0]


def crop_for(series: str) -> str | None:
    return HARTI_SERIES.get(series, (None, None))[1]


@dataclass(frozen=True)
class Freshness:
    latest: pd.Timestamp
    age_days: int
    level: str  # "current", "ageing" or "stale"
    message: str


def freshness(latest, today=None, horizon_days: int = MAX_HORIZON_DAYS) -> Freshness:
    """How current a series is: current (<= 3 days old), ageing, or stale (older than the forecast's reach)."""
    latest = pd.Timestamp(latest).normalize()
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    age = (today - latest).days
    if age <= 3:
        return Freshness(latest, age, "current", f"Latest data {latest.date()} ({age} day(s) old).")
    if age <= horizon_days:
        return Freshness(latest, age, "ageing", f"Latest data {latest.date()} is {age} days old; forecasts start from that date, not today.")
    return Freshness(latest, age, "stale", f"Latest data {latest.date()} is {age} days old, beyond the {horizon_days}-day forecast reach, so no forecast is offered.")


def past_date_notice(target_date, latest_actual_date, today=None) -> str | None:
    """A warning when the requested date has already passed, so a forecast for it isn't read as the
    recorded price. None when the date is today or later."""
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    target, latest = pd.Timestamp(target_date).normalize(), pd.Timestamp(latest_actual_date).normalize()
    if target >= today:
        return None
    return (
        f"**{target.date()} has already passed** ({(today - target).days} day(s) ago). This is what the model "
        f"forecasts from data through {latest.date()}; it is not the recorded price for that day, which may already "
        "be published. Run the daily update (`scripts/daily_update.py`) to import newer bulletins."
    )


def series_status(history: pd.DataFrame, trained_vegetables: list[str] | None, today=None) -> pd.DataFrame:
    """One row per HARTI Dambulla series: its dates, and whether price prediction is supported (and why not)."""
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    priced = history[history["price"].notna()]
    rows = []
    for series, group in priced.groupby("vegetable"):
        fresh = freshness(group["date"].max(), today)
        if trained_vegetables is None:
            supported, reason = False, "The forecaster isn't available."
        elif series not in trained_vegetables:
            supported, reason = False, "The forecaster wasn't trained on this series (too few prices)."
        elif fresh.level == "stale":
            supported, reason = False, f"Its latest Dambulla price is {fresh.latest.date()}, {fresh.age_days} days ago."
        else:
            supported, reason = True, ""
        rows.append(
            {
                "series": series,
                "display_name": display_name(series),
                "first": group["date"].min(),
                "latest": group["date"].max(),
                "days_with_price": int(group["date"].nunique()),
                "age_days": fresh.age_days,
                "prediction_supported": supported,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values("display_name", ignore_index=True)


def load_catalogue(path: str | Path = "data/reference/vegetable_catalogue.csv") -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (ValueError, OSError, UnicodeDecodeError):
        return None


def data_version(folders: tuple[str, ...] = ("data/raw", "models")) -> str:
    """Changes whenever a dataset or saved model changes on disk.

    Pages pass it to their cached loaders, so a running app picks up a daily update or a retrained
    model on the next interaction instead of serving what it read at start-up.
    """
    stamps = []
    for folder in folders:
        for path in sorted(Path(folder).glob("*")):
            if path.suffix in (".csv", ".json", ".joblib") and path.is_file():
                stat = path.stat()
                stamps.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    import hashlib

    return hashlib.sha256("|".join(stamps).encode()).hexdigest()[:16]


def _update_runs(path: str | Path) -> list[dict]:
    """Every readable line of scripts/daily_update.py's log, oldest first."""
    import json

    try:
        lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    runs = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            runs.append(entry)
    return runs


def _succeeded(run: dict) -> bool:
    # Lines written before failures were recorded have no status: only completed runs were logged then.
    # "partial" runs completed too; the sources they couldn't check are in run["sources"].
    return run.get("status", "ok") in ("ok", "partial")


def latest_update(path: str | Path = "data/raw/update_log.jsonl") -> dict | None:
    """The last update that completed (failed attempts are skipped), if any."""
    return next((run for run in reversed(_update_runs(path)) if _succeeded(run)), None)


# The scheduled check runs every 3 hours; a day without a successful check means it isn't running.
UPDATE_OVERDUE_HOURS = 24
# HARTI publishes Dambulla prices every day (its longest gap in 2026 was 4 days).
HARTI_BEHIND_DAYS = 3


@dataclass(frozen=True)
class UpdateStatus:
    last_success: pd.Timestamp | None  # when the last completed check finished
    newest_listed: pd.Timestamp | None  # newest HARTI bulletin the source listed at that check
    last_attempt_failed: bool
    last_error: str | None
    overdue: bool  # no completed check for UPDATE_OVERDUE_HOURS
    # Per-source results of the last completed check ({} for runs logged before sources were recorded).
    sources: dict = field(default_factory=dict)


def update_status(path: str | Path = "data/raw/update_log.jsonl", now=None) -> UpdateStatus:
    """What the data-update log says about the latest checks of the official sources."""
    runs = _update_runs(path)
    now = pd.Timestamp(now or pd.Timestamp.now())
    success = next((run for run in reversed(runs) if _succeeded(run)), None)
    last = runs[-1] if runs else None
    finished = pd.Timestamp(success["finished_at"]) if success and success.get("finished_at") else None
    listed = (success or {}).get("harti", {}).get("newest_listed")
    return UpdateStatus(
        last_success=finished,
        newest_listed=pd.Timestamp(listed) if listed and listed != "unknown" else None,
        last_attempt_failed=last is not None and not _succeeded(last),
        last_error=last.get("error") if last is not None and not _succeeded(last) else None,
        overdue=finished is None or (now - finished) > pd.Timedelta(hours=UPDATE_OVERDUE_HOURS),
        sources=(success or {}).get("sources", {}),
    )


def update_notice(status: UpdateStatus, latest_observation, now=None) -> tuple[str, str]:
    """(level, message) for a page: 'warning' when the data may be behind the official source, else 'info'.

    Separates the cases honestly: the source hasn't published anything newer (the data is as
    current as it can be), the check failed, or no recent check has run at all.
    """
    now = pd.Timestamp(now or pd.Timestamp.now())
    latest = pd.Timestamp(latest_observation).normalize()
    age = (now.normalize() - latest).days
    checked = f"last successful check {status.last_success:%Y-%m-%d %H:%M}" if status.last_success is not None else "no successful check recorded"
    if status.last_attempt_failed:
        return "warning", (
            f"The latest attempt to check the official sources failed ({status.last_error}); {checked}. "
            f"Showing the stored data, newest price {latest.date()} ({age} day(s) old)."
        )
    harti = status.sources.get("harti", {})
    if harti.get("status") == "failed":
        return "warning", (
            f"HARTI couldn't be checked at the last update ({harti.get('error', 'no details')}). Showing the stored "
            f"HARTI data, newest price {latest.date()} ({age} day(s) old); newer bulletins may exist."
        )
    if status.overdue:
        return "warning", (
            f"The official sources haven't been checked for over {UPDATE_OVERDUE_HOURS} hours ({checked}), so newer "
            f"HARTI prices may exist. Newest stored price: {latest.date()} ({age} day(s) old). Use **Check for new "
            "official data** on the Home page."
        )
    if age > HARTI_BEHIND_DAYS:
        source_has_nothing_newer = status.newest_listed is not None and status.newest_listed <= latest
        if source_has_nothing_newer:
            return "warning", (
                f"The newest price is {age} days old ({latest.date()}): HARTI hasn't published a newer bulletin yet "
                f"({checked}). Forecasts start from {latest.date()}, not today."
            )
        return "warning", (
            f"The newest price is {age} days old ({latest.date()}), although HARTI lists a bulletin for "
            f"{status.newest_listed.date() if status.newest_listed is not None else 'a later date'} ({checked}). "
            "That bulletin may not include this series."
        )
    return "info", f"Data is up to date with the official source ({checked}; newest HARTI bulletin listed {status.newest_listed.date() if status.newest_listed is not None else 'unknown'})."


def load_manifest(path: str | Path = "data/raw/data_manifest.json") -> dict | None:
    import json

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
