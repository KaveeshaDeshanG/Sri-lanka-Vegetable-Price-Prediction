"""The master vegetable catalogue: every vegetable the project's sources mention, and what each supports.

Streamlit-free. Static facts (names, aliases, which crop a price series belongs to) live here and in
data/reference/; what changes with the data (latest observation, whether a forecast is possible) is
computed from the datasets every time, so the app never shows a stale support status.

Internal series names are the ones the trained model uses and are not renamed. Two of them are
easy to confuse, so they get display names: the model's "Brinjal" is HARTI's "Eggplant" row,
which HARTI lists separately from its "Brinjals" row.
"""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pandas as pd

from app.i18n import current_language, t
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
    """The name shown to users for an internal series name (unknown names are shown as-is).

    In Sinhala or Tamil the familiar English name is kept and the local name added: "Beans · බෝංචි".
    """
    name = HARTI_SERIES.get(series, (series,))[0]
    local = local_crop_name(crop_for(series) or series)
    return f"{name} · {local}" if local else name


def localized(name: str) -> str:
    """A stored vegetable or series name as shown: unchanged in English, with its local name in Sinhala or Tamil."""
    return name if current_language() == "en" else display_name(name)


def crop_label(crop: str) -> str:
    """A DOA crop name for display, with its Sinhala or Tamil name in those languages: "Carrot · කැරට්"."""
    local = local_crop_name(crop)
    return f"{crop} · {local}" if local else crop


@cache
def _local_names(path: str = "data/reference/vegetable_names.csv") -> dict[str, dict[str, str]]:
    """{crop: {"si": Sinhala name, "ta": Tamil name}} from the DOA's own published names."""
    try:
        table = pd.read_csv(Path(__file__).resolve().parents[1] / path, dtype=str, keep_default_na=False)
    except (OSError, ValueError):
        return {}
    return {row.crop: {"si": row.sinhala_name.strip(), "ta": row.tamil_name.strip()} for row in table.itertuples(index=False)}


def local_crop_name(crop: str | None) -> str | None:
    """The crop's name in the current language (None in English, or when no local name is known).

    The DOA's HORDI pages (data/reference/vegetable_names.csv) give most names; crops they don't
    cover (onions, chillies, lime...) use the translation files.
    """
    language = current_language()
    if language == "en" or not crop:
        return None
    published = _local_names().get(crop, {}).get(language)
    if published:
        return published
    translated = t(crop)
    return translated if translated != crop else None


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
        return Freshness(latest, age, "current", t("Latest data {date} ({age} day(s) old).", date=latest.date(), age=age))
    if age <= horizon_days:
        return Freshness(
            latest, age, "ageing", t("Latest data {date} is {age} days old; forecasts start from that date, not today.", date=latest.date(), age=age)
        )
    return Freshness(
        latest,
        age,
        "stale",
        t(
            "Latest data {date} is {age} days old, beyond the {days}-day forecast reach, so no forecast is offered.",
            date=latest.date(),
            age=age,
            days=horizon_days,
        ),
    )


def past_date_notice(target_date, latest_actual_date, today=None) -> str | None:
    """A warning when the requested date has already passed, so a forecast for it isn't read as the
    recorded price. None when the date is today or later."""
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    target, latest = pd.Timestamp(target_date).normalize(), pd.Timestamp(latest_actual_date).normalize()
    if target >= today:
        return None
    return t(
        "**{date} has already passed** ({days} day(s) ago). This is what the model "
        "forecasts from data through {latest}; it is not the recorded price for that day, which may already "
        "be published. Run the daily update (`scripts/daily_update.py`) to import newer bulletins.",
        date=target.date(),
        days=(today - target).days,
        latest=latest.date(),
    )


def series_status(history: pd.DataFrame, trained_vegetables: list[str] | None, today=None) -> pd.DataFrame:
    """One row per HARTI Dambulla series: its dates, and whether price prediction is supported (and why not)."""
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    priced = history[history["price"].notna()]
    rows = []
    for series, group in priced.groupby("vegetable"):
        fresh = freshness(group["date"].max(), today)
        if trained_vegetables is None:
            supported, reason = False, t("The forecaster isn't available.")
        elif series not in trained_vegetables:
            supported, reason = False, t("The forecaster wasn't trained on this series (too few prices).")
        elif fresh.level == "stale":
            supported, reason = False, t("Its latest Dambulla price is {date}, {days} days ago.", date=fresh.latest.date(), days=fresh.age_days)
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
    checked = (
        t("last successful check {when}", when=f"{status.last_success:%Y-%m-%d %H:%M}")
        if status.last_success is not None
        else t("no successful check recorded")
    )
    if status.last_attempt_failed:
        return "warning", t(
            "The latest attempt to check the official sources failed ({error}); {checked}. "
            "Showing the stored data, newest price {date} ({age} day(s) old).",
            error=status.last_error,
            checked=checked,
            date=latest.date(),
            age=age,
        )
    harti = status.sources.get("harti", {})
    if harti.get("status") == "failed":
        return "warning", t(
            "HARTI couldn't be checked at the last update ({error}). Showing the stored "
            "HARTI data, newest price {date} ({age} day(s) old); newer bulletins may exist.",
            error=harti.get("error", t("no details")),
            date=latest.date(),
            age=age,
        )
    if status.overdue:
        return "warning", t(
            "The official sources haven't been checked for over {hours} hours ({checked}), so newer "
            "HARTI prices may exist. Newest stored price: {date} ({age} day(s) old). Use **Check for new "
            "official data** on the Home page.",
            hours=UPDATE_OVERDUE_HOURS,
            checked=checked,
            date=latest.date(),
            age=age,
        )
    if age > HARTI_BEHIND_DAYS:
        source_has_nothing_newer = status.newest_listed is not None and status.newest_listed <= latest
        if source_has_nothing_newer:
            return "warning", t(
                "The newest price is {age} days old ({date}): HARTI hasn't published a newer bulletin yet "
                "({checked}). Forecasts start from {date}, not today.",
                age=age,
                date=latest.date(),
                checked=checked,
            )
        return "warning", t(
            "The newest price is {age} days old ({date}), although HARTI lists a bulletin for "
            "{listed} ({checked}). That bulletin may not include this series.",
            age=age,
            date=latest.date(),
            listed=status.newest_listed.date() if status.newest_listed is not None else t("a later date"),
            checked=checked,
        )
    return "info", t(
        "Data is up to date with the official source ({checked}; newest HARTI bulletin listed {listed}).",
        checked=checked,
        listed=status.newest_listed.date() if status.newest_listed is not None else t("unknown"),
    )


def load_manifest(path: str | Path = "data/raw/data_manifest.json") -> dict | None:
    import json

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- Data freshness of every source (all pages) ----------------------------------------------------
# One label per source, from two facts: the newest date actually stored in its file (read from the
# file, not from a log or manifest that an interrupted run may have left behind) and the update log's
# record of the last check of that source. No source is real-time: each publishes on its own schedule.
FRESHNESS_LABELS = ("Latest Available Data", "Recently Updated", "Update Pending", "Source Unavailable", "Historical Data")
RECENT_HOURS = 24  # new records added within this many hours count as "Recently Updated"
WEATHER_ARCHIVE_LAG_DAYS = 3  # scripts/daily_update.py stores observed weather up to today minus this


@dataclass(frozen=True)
class DataSource:
    key: str  # its name in update_log.jsonl's "sources"
    name: str
    category: str
    location: str
    schedule: str  # how often the source itself publishes
    url: str


DATA_SOURCES = (
    DataSource("harti", "HARTI daily bulletin", "Wholesale", "Dambulla (used by the model) and 10 other markets",
               "Daily bulletins, uploaded 1-3 days after the market day", "https://www.harti.gov.lk/daily-price.php"),
    DataSource("cbsl", "CBSL Daily Price Report", "Wholesale and retail (kept separate)", "Pettah, Dambulla, Narahenpita, Marandagahamula",
               "Each working day", "https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report"),
    DataSource("dcs", "DCS weekly retail prices", "Retail", "Colombo District",
               "Weekly, 1-3 weeks after the week", "https://www.statistics.gov.lk/"),
    DataSource("weather", "Open-Meteo weather (observed)", "Observed weather, not a forecast", "8 production regions",
               f"Daily archive; stored up to {WEATHER_ARCHIVE_LAG_DAYS} days before today, as the newest days are still forecast-based",
               "https://archive-api.open-meteo.com/v1/archive"),
    DataSource("fuel", "CEYPETCO fuel prices", "Retail fuel price (diesel, petrol 92)", "Sri Lanka",
               "When prices are revised (about monthly)", "https://ceypetco.gov.lk/marketing-sales/historical-prices"),
)


@dataclass
class SourceFreshness:
    source: DataSource
    label: str
    latest_stored: str | None  # the newest data in the local file, as the source labels it
    latest_stored_date: pd.Timestamp | None
    newest_published: str | None  # the newest publication the last check found
    last_checked: pd.Timestamp | None
    last_success: pd.Timestamp | None  # the last successful synchronisation with this source
    note: str = ""


def _latest_stored(key: str, raw_dir: Path) -> tuple[str | None, pd.Timestamp | None, str]:
    """(label, date, extra note) of the newest data stored for a source, read from its file."""
    try:
        if key == "harti":
            dates = pd.to_datetime(pd.read_csv(raw_dir / "harti_prices.csv", usecols=["date"])["date"])
        elif key == "cbsl":
            dates = pd.to_datetime(pd.read_csv(raw_dir / "cbsl_daily_prices.csv", usecols=["date"])["date"])
        elif key == "dcs":
            table = pd.read_csv(raw_dir / "dcs_weekly_retail_prices.csv", usecols=["year", "month", "week_of_month"]).dropna()
            year, month, week = (int(v) for v in table.sort_values(["year", "month", "week_of_month"]).iloc[-1])
            start = pd.Timestamp(year=year, month=month, day=1) + pd.Timedelta(days=7 * (week - 1))
            return f"{year}-{month:02d} week {week}", start, ""  # DCS's own period label, kept as published
        elif key == "weather":
            table = pd.read_csv(raw_dir / "weather_data_by_region.csv", usecols=["date", "region"], parse_dates=["date"])
            dates = table.groupby("region")["date"].max()  # the region furthest behind
            return f"{dates.min():%Y-%m-%d}", dates.min(), ""
        elif key == "fuel":
            table = pd.read_csv(raw_dir / "fuel_prices.csv", parse_dates=["date"])
            changed = table[(table["diesel_price"].diff() != 0) | (table["petrol_price"].diff() != 0)]
            revision = changed["date"].max()
            last = table.iloc[-1]
            return (
                f"{revision:%Y-%m-%d}",
                revision,
                t(
                    "latest revision: diesel Rs {diesel:,.0f}, petrol 92 Rs {petrol:,.0f} per litre, in force through {date}",
                    diesel=last["diesel_price"],
                    petrol=last["petrol_price"],
                    date=f"{table['date'].max():%Y-%m-%d}",
                ),
            )
        else:
            return None, None, ""
    except (OSError, ValueError, KeyError, IndexError):
        return None, None, t("its data file couldn't be read")
    latest = dates.max()
    return (f"{latest:%Y-%m-%d}", latest, "") if pd.notna(latest) else (None, None, "")


def _recent_new_data(key: str, run: dict) -> bool:
    if key in ("harti", "cbsl", "dcs"):
        return bool(run.get("sources", {}).get(key, {}).get("new_records"))
    if key == "weather":
        added = run.get("weather_rows_added")
        return isinstance(added, dict) and any(isinstance(v, int) and v > 0 for v in added.values())
    return False


def data_freshness(raw_dir: str | Path = "data/raw", now=None) -> list[SourceFreshness]:
    """Every source's freshness label, newest stored data, newest publication found and last checks."""
    raw_dir = Path(raw_dir)
    now = pd.Timestamp(now or pd.Timestamp.now())
    today = now.normalize()
    runs = _update_runs(raw_dir / "update_log.jsonl")
    result = []
    for source in DATA_SOURCES:
        stored, stored_date, extra = _latest_stored(source.key, raw_dir)
        checks = [(run, run["sources"][source.key]) for run in runs if isinstance(run.get("sources"), dict) and source.key in run["sources"]]
        last = checks[-1][1] if checks else None
        successes = [(run, info) for run, info in checks if info.get("status") == "ok"]
        last_success = pd.Timestamp(successes[-1][1]["checked_at"]) if successes else None
        last_checked = pd.Timestamp(last["checked_at"]) if last else None
        newest = (successes[-1][1].get("newest_published") if successes else None) or None
        newest_date = _publication_date_of(newest)
        recent = any(
            _recent_new_data(source.key, run)
            for run, info in successes
            if now - pd.Timestamp(info["checked_at"]) <= pd.Timedelta(hours=RECENT_HOURS)
        )
        if source.key == "fuel" and stored_date is not None and (today - stored_date).days <= 3:
            recent = True  # a price revision in the last few days
        if last is None:
            label, note = "Update Pending", t("Not checked by the automatic update yet.")
        elif last.get("status") == "failed":
            label = "Source Unavailable"
            note = t(
                "The last check ({when}) failed: {error}. The stored data is kept unchanged.",
                when=f"{last_checked:%Y-%m-%d %H:%M}",
                error=last.get("error", t("no details")),
            )
        elif now - last_success > pd.Timedelta(hours=UPDATE_OVERDUE_HOURS):
            label = "Update Pending"
            note = t(
                "Not checked for {hours} hours; newer data may have been published.", hours=int((now - last_success).total_seconds() // 3600)
            )
        elif newest_date is not None and stored_date is not None and newest_date > stored_date:
            label, note = "Update Pending", t("The source lists {newest}, which isn't stored yet; the next update will add it.", newest=newest)
        elif source.key == "weather" and stored_date is not None and stored_date < today - pd.Timedelta(days=WEATHER_ARCHIVE_LAG_DAYS + 1):
            label, note = "Update Pending", t("Observed weather for the last few days hasn't been stored yet.")
        elif recent:
            label, note = "Recently Updated", t("New data was added in the last {hours} hours.", hours=RECENT_HOURS)
        else:
            label, note = "Latest Available Data", t("Nothing newer has been published since the last check.")
        if extra:
            note = f"{note} {extra[0].upper()}{extra[1:]}."
        result.append(SourceFreshness(source, label, stored, stored_date, newest, last_checked, last_success, note))
    return result


def _publication_date_of(value) -> pd.Timestamp | None:
    """'2026-11-12' or DCS's '2026-11 week 1' (that week's first day) as a date."""
    import re

    if not value:
        return None
    week = re.fullmatch(r"(\d{4})-(\d{2}) week (\d)", str(value))
    if week:
        year, month, number = map(int, week.groups())
        return pd.Timestamp(year=year, month=month, day=1) + pd.Timedelta(days=7 * (number - 1))
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.normalize()


def last_run_problem(raw_dir: str | Path = "data/raw") -> str | None:
    """A sentence about the most recent update run if it didn't complete (failed or interrupted), else None."""
    runs = _update_runs(Path(raw_dir) / "update_log.jsonl")
    if not runs or _succeeded(runs[-1]):
        return None
    run = runs[-1]
    when = str(run.get("finished_at") or run.get("started_at", ""))[:16].replace("T", " ")
    if run.get("status") == "interrupted":
        return t("The most recent update ({when}) was interrupted: {error}.", when=when, error=run.get("error", t("no details")))
    return t("The most recent update ({when}) failed: {error}.", when=when, error=run.get("error", t("no details")))
