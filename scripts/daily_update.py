"""Daily incremental update: adds newly published official prices to the stored datasets.

Run it once a day, for example at 18:00 Sri Lanka time; HARTI uploads the day's bulletin in the
afternoon and weekend bulletins the next morning. scripts/run_daily_update.bat runs it for the
Windows Task Scheduler. Unlike scripts/update_data.py, which rebuilds everything from every
bulletin, this only reads what is new, so it takes about a minute.

What it does, in order:

1. HARTI daily bulletins (wholesale, Rs/kg, the source the model is trained on): lists the
   archive and reads every bulletin from the last --lookback-days that isn't stored yet. Each
   one is downloaded, read with the same code as the full build, validated, then added to
   harti_prices_detail.csv / harti_prices.csv (Dambulla) and harti_all_markets.csv (every market).
   Rows already stored are never changed. Every bulletin tried is recorded in
   harti_import_log.csv: its URL, the PDF's SHA-256, when it was retrieved and what it yielded.
2. CBSL Daily Price Report (wholesale and retail, Pettah / Dambulla / Narahenpita): reads each
   working day's report that isn't stored yet into cbsl_daily_prices.csv. CBSL collects its own
   prices, so this is a separate dataset and is never merged into a HARTI series. CBSL's
   robots.txt asks for 10 seconds between requests, which is respected.
3. Weather (Open-Meteo archive) and fuel prices (CEYPETCO): extended to the newest available day.
   Days the archive doesn't have yet are left for the next run, not estimated.
4. DCS weekly retail prices: downloads any new weekly bulletin and rebuilds
   dcs_weekly_retail_prices.csv. Links DCS lists with no PDF behind them aren't requested daily.
5. Writes data/raw/data_manifest.json and adds a line to data/raw/update_log.jsonl.

Before any file changes, the files about to change are copied to <backup_dir>/daily/<time>/; the
newest --keep-backups of those folders are kept. Files are written to a temporary name and then
renamed, so an interrupted run never leaves a half-written dataset. Nothing is estimated: a
bulletin that can't be downloaded or read stays missing and is reported.

Exit codes: 0 done; 2 a newly published HARTI bulletin couldn't be read (its layout may have
changed, so the parser needs checking); 3 another update is already running.

The model isn't retrained here; scripts/retrain_if_better.py does that, with an acceptance check.

Usage:
    .venv312/Scripts/python.exe scripts/daily_update.py
    .venv312/Scripts/python.exe scripts/daily_update.py --cbsl-since 2026-01-01   # backfill CBSL
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402
from scripts.build_harti_prices import (  # noqa: E402
    PRICE_REVIEW_MAX,
    PRICE_REVIEW_MIN,
    download_pdf,
    extract_market_rows,
    finalize,
    hash_url,
)
from scripts.build_market_prices import read_bulletin  # noqa: E402
from scripts.update_data import (  # noqa: E402
    _dcs_file_name,
    dcs_listing,
    download_new_dcs,
    harti_entries,
    rebuild_dcs,
    rebuild_fuel,
    write_manifest,
)
from src.data_collection.cbsl_price_report_parser import parse_cbsl_price_report  # noqa: E402
from src.data_collection.harti_live_client import _REQUEST_HEADERS, MAX_REPORTING_LAG_DAYS  # noqa: E402
from src.data_collection.live_weather_client import LiveWeatherError, OpenMeteoWeatherClient  # noqa: E402
from src.preprocessing.transformer import DataTransformer  # noqa: E402

logger = logging.getLogger("daily_update")

CBSL_BASE = "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/pricerpt/"
CBSL_PAGE = "https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report"
CBSL_DELAY_SECONDS = 10  # CBSL's robots.txt: Crawl-delay: 10
CBSL_COLUMNS = ["date", "item", "item_as_printed", "vegetable", "unit", "market", "price_type", "price", "source_url", "retrieved_at"]
HARTI_LOG_COLUMNS = ["bulletin_date", "url", "sha256", "retrieved_at", "status", "dambulla_rows", "all_market_rows", "detail"]
DETAIL_COLUMNS = ["date", "vegetable", "market", "price", "low", "high", "raw_cell", "bulletin_date", "source_pdf"]
LOCK_STALE_HOURS = 6
WEATHER_ARCHIVE_LAG_DAYS = 3  # Open-Meteo's archive usually lags a few days behind today


class UpdateError(RuntimeError):
    pass


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (_PROJECT_ROOT / candidate).resolve()


@contextmanager
def single_run(lock_path: Path):
    """Refuse to start while another update holds the lock (a lock older than 6 hours is stale)."""
    if lock_path.exists() and time.time() - lock_path.stat().st_mtime < LOCK_STALE_HOURS * 3600:
        raise UpdateError(f"another update is running ({lock_path} exists)")
    lock_path.write_text(f"{os.getpid()} {pd.Timestamp.now().isoformat()}", encoding="utf-8")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def write_atomically(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(path)


class Backup:
    """Copies each file once, just before it is first changed in this run."""

    def __init__(self, backup_root: Path, keep: int) -> None:
        self.folder = backup_root / "daily" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        self.keep = keep
        self.saved: list[str] = []  # files copied (they existed before this run)
        self.changed: list[str] = []  # every file this run writes, including new ones

    def before_change(self, path: Path) -> None:
        if path.name not in self.changed:
            self.changed.append(path.name)
        if path.exists() and path.name not in self.saved:
            self.folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, self.folder / path.name)
            self.saved.append(path.name)

    def prune(self) -> None:
        """Keep the newest `keep` daily backup folders (only folders this script made)."""
        folders = sorted(p for p in self.folder.parent.glob("2*_*") if p.is_dir())
        for old in folders[: max(0, len(folders) - self.keep)]:
            shutil.rmtree(old)


# ---------------------------------------------------------------------------------------------
# HARTI
# ---------------------------------------------------------------------------------------------


def validate_dambulla_rows(rows: list[dict], bulletin_date: pd.Timestamp) -> list[str]:
    """Reasons these rows can't be stored as they are (empty when they're fine)."""
    problems = []
    for row in rows:
        if not (row["low"] > 0 and row["high"] > 0):
            problems.append(f"{row['vegetable']}: non-positive price {row['raw_cell']!r}")
        if row["low"] > row["high"]:
            problems.append(f"{row['vegetable']}: low above high {row['raw_cell']!r}")
        lag = (bulletin_date - row["date"]).days
        if not 0 <= lag <= MAX_REPORTING_LAG_DAYS:
            problems.append(f"{row['vegetable']}: reading dated {row['date'].date()} for a {bulletin_date.date()} bulletin")
    return problems


def import_harti(config: ConfigLoader, raw_dir: Path, cache_dir: Path, backup: Backup, lookback_days: int, today: pd.Timestamp) -> dict:
    detail_path, prices_path = raw_dir / "harti_prices_detail.csv", raw_dir / "harti_prices.csv"
    panel_path, log_path = raw_dir / "harti_all_markets.csv", raw_dir / "harti_import_log.csv"
    detail = pd.read_csv(detail_path, parse_dates=["date", "bulletin_date"])
    panel = pd.read_csv(panel_path, parse_dates=["published", "reported"])
    log = pd.read_csv(log_path, parse_dates=["bulletin_date"]) if log_path.exists() else pd.DataFrame(columns=HARTI_LOG_COLUMNS)

    entries = harti_entries(config, offline=False, cache_dir=cache_dir)
    window = entries[(entries["date"] > today - pd.Timedelta(days=lookback_days)) & (entries["date"] <= today)]
    stored = set(detail["bulletin_date"]) | set(panel["published"])
    done = set(log.loc[log["status"] == "imported", "bulletin_date"])
    pending = window[~window["date"].isin(stored | done)]
    logger.info("HARTI lists %d bulletins; %d in the last %d days not stored yet", len(entries), len(pending), lookback_days)

    market = config.get("harti.target_market", "Dambulla")
    new_detail, new_panel, log_rows, unreadable = [], [], [], []
    for entry in pending.itertuples(index=False):
        record = {"bulletin_date": entry.date, "url": entry.url, "retrieved_at": pd.Timestamp.now().isoformat(timespec="seconds")}
        path = cache_dir / f"{entry.date:%Y-%m-%d}_{abs(hash_url(entry.url)):016x}.pdf"
        try:
            pdf = download_pdf(entry.url, path, 90, attempts=2)
        except requests.exceptions.RequestException as error:
            log_rows.append({**record, "status": "download_failed", "detail": str(error)[:200]})
            continue
        record["sha256"] = hashlib.sha256(pdf).hexdigest()
        extract = extract_market_rows(pdf, entry.date, market)
        all_markets, all_problem = read_bulletin(path)
        invalid = [] if extract.error else validate_dambulla_rows(extract.rows, entry.date)
        problems = [p for p in (extract.error, all_problem) if p] + invalid
        if extract.error and all_markets is None:
            # Nothing at all could be read: most likely a new layout. The file is removed from the
            # cache so a corrected upload is fetched on the next run.
            path.unlink(missing_ok=True)
            unreadable.append(str(entry.date.date()))
            log_rows.append({**record, "status": "unreadable", "detail": "; ".join(problems)[:500]})
            continue
        if invalid:
            unreadable.append(str(entry.date.date()))
            log_rows.append({**record, "status": "failed_validation", "detail": "; ".join(problems)[:500]})
            continue
        # Otherwise import. Dambulla may simply not have reported (extract.error), in which case only
        # the other markets' readings are added.
        rows = [] if extract.error else [{**r, "bulletin_date": entry.date, "source_pdf": entry.url} for r in extract.rows]
        new_detail.extend(rows)
        if all_markets is not None:
            new_panel.append(all_markets)
        review = [r["vegetable"] for r in rows if not PRICE_REVIEW_MIN <= r["price"] <= PRICE_REVIEW_MAX]
        log_rows.append(
            {
                **record,
                "status": "imported",
                "dambulla_rows": len(rows),
                "all_market_rows": 0 if all_markets is None else len(all_markets),
                "detail": "; ".join(problems + ([f"price outside Rs {PRICE_REVIEW_MIN}-{PRICE_REVIEW_MAX}, check: {review}"] if review else []))[:500],
            }
        )

    added_dambulla = added_panel = 0
    if new_detail:
        combined = finalize(detail[DETAIL_COLUMNS].to_dict("records") + new_detail)
        # Rows already stored are never replaced: keep every existing (date, vegetable) as it was.
        existing = set(zip(detail["date"], detail["vegetable"]))
        combined = pd.concat(
            [detail[DETAIL_COLUMNS], combined[[(d, v) not in existing for d, v in zip(combined["date"], combined["vegetable"])]]],
            ignore_index=True,
        ).sort_values(["date", "vegetable"], ignore_index=True)
        added_dambulla = len(combined) - len(detail)
        for path in (detail_path, prices_path):
            backup.before_change(path)
        write_atomically(combined, detail_path)
        write_atomically(combined[["date", "vegetable", "market", "price"]], prices_path)
    if new_panel:
        fresh = DataTransformer().standardize_vegetable_names(pd.concat(new_panel, ignore_index=True))
        merged = pd.concat([panel, fresh[panel.columns]], ignore_index=True)
        merged = merged.drop_duplicates(["published", "market", "vegetable"], keep="first").sort_values(["published", "market", "vegetable"], ignore_index=True)
        added_panel = len(merged) - len(panel)
        backup.before_change(panel_path)
        write_atomically(merged, panel_path)
    if log_rows:
        backup.before_change(log_path)
        write_atomically(pd.concat([log, pd.DataFrame(log_rows)], ignore_index=True)[HARTI_LOG_COLUMNS], log_path)

    return {
        "bulletins_listed": len(entries),
        "newest_listed": str(entries["date"].max().date()),
        "bulletins_read": len(log_rows),
        "dambulla_rows_added": int(added_dambulla),
        "all_market_rows_added": int(added_panel),
        "unreadable": unreadable,
        "download_failed": [str(r["bulletin_date"].date()) for r in log_rows if r["status"] == "download_failed"],
    }


# ---------------------------------------------------------------------------------------------
# CBSL
# ---------------------------------------------------------------------------------------------


def fetch_cbsl_report(day: pd.Timestamp, session: requests.Session) -> tuple[bytes, str] | None:
    """The report for `day` and its URL, or None when CBSL published none (weekends, holidays)."""
    for name in (f"price_report_{day:%Y%m%d}_e.pdf", f"price_report_{day:%Y%m%d}e.pdf"):
        response = session.get(CBSL_BASE + name, headers=_REQUEST_HEADERS, timeout=60)
        time.sleep(CBSL_DELAY_SECONDS)
        if response.status_code == 200 and response.content.startswith(b"%PDF"):
            return response.content, CBSL_BASE + name
        if response.status_code not in (403, 404):
            response.raise_for_status()
    return None


def import_cbsl(raw_dir: Path, cache_dir: Path, backup: Backup, since: pd.Timestamp, today: pd.Timestamp) -> dict:
    path, state_path = raw_dir / "cbsl_daily_prices.csv", raw_dir / "cbsl_import_state.json"
    table = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=CBSL_COLUMNS)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"not_published": [], "unreadable": {}}
    stored = {d.date().isoformat() for d in table["date"]}
    # A missing report is retried for 3 days (CBSL may upload late), then treated as not published.
    settled = set(state["not_published"]) | set(state["unreadable"])
    days = [d for d in pd.bdate_range(since, today) if d.date().isoformat() not in stored | settled]
    logger.info("CBSL: %d working days to check from %s", len(days), since.date())
    cache_dir.mkdir(parents=True, exist_ok=True)
    session, rows, added_days, missing = requests.Session(), [], [], []
    for day in days:
        key = day.date().isoformat()
        try:
            found = fetch_cbsl_report(day, session)
        except requests.exceptions.RequestException as error:
            logger.warning("CBSL %s: %s", key, error)
            continue
        if found is None:
            missing.append(key)
            if (today - day).days > 3:
                state["not_published"].append(key)
            continue
        pdf, url = found
        (cache_dir / url.rsplit("/", 1)[-1]).write_bytes(pdf)
        report = parse_cbsl_price_report(pdf, url.rsplit("/", 1)[-1])
        if report.error:
            state["unreadable"][key] = report.error
            continue
        retrieved = pd.Timestamp.now().isoformat(timespec="seconds")
        rows.extend({**r, "source_url": url, "retrieved_at": retrieved} for r in report.rows)
        added_days.append(key)
    if rows:
        merged = pd.concat([table, pd.DataFrame(rows)[CBSL_COLUMNS]], ignore_index=True)
        merged = merged.drop_duplicates(["date", "item", "market", "price_type"], keep="first")
        merged = merged.sort_values(["date", "item", "price_type", "market"], ignore_index=True)
        backup.before_change(path)
        write_atomically(merged, path)
    state["not_published"] = sorted(set(state["not_published"]))
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    return {"days_added": added_days, "not_published_yet": missing, "unreadable": state["unreadable"]}


# ---------------------------------------------------------------------------------------------
# DCS weekly retail prices
# ---------------------------------------------------------------------------------------------


def import_dcs(config: ConfigLoader, raw_dir: Path, backup: Backup) -> dict:
    """New DCS weekly retail bulletins, then a rebuild of dcs_weekly_retail_prices.csv if any arrived.

    Links DCS lists without a PDF behind them (recorded in the cache's listing.json, or in
    dcs_import_state.json by an earlier run) are not requested again every day.
    """
    cache = _resolve(config.get_required("data_update.dcs_pdf_cache_dir"))
    cache.mkdir(parents=True, exist_ok=True)
    state_path = raw_dir / "dcs_import_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"unavailable": {}}
    saved = cache / "listing.json"
    known = {item["link"] for item in json.loads(saved.read_text(encoding="utf-8")) if item.get("error")} if saved.exists() else set()
    known |= set(state["unavailable"])
    try:
        listing = dcs_listing()
    except requests.exceptions.RequestException as error:
        logger.warning("DCS listing unavailable: %s", error)
        return {"error": str(error)[:200]}
    new = [item for item in listing if item["link"] not in known and not (cache / _dcs_file_name(item["link"])).exists()]
    failures = download_new_dcs(new, cache, 0.3)
    for failure in failures:
        state["unavailable"][failure["link"]] = failure["reason"]
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    downloaded = len(new) - len(failures)
    output = raw_dir / "dcs_weekly_retail_prices.csv"
    if downloaded or not output.exists():
        backup.before_change(output)
        rebuild_dcs(cache, raw_dir, listing)
    return {"listed": len(listing), "new_bulletins": downloaded, "unavailable": len(failures)}


# ---------------------------------------------------------------------------------------------
# Weather and fuel
# ---------------------------------------------------------------------------------------------


def weather_locations(config: ConfigLoader, config_path: str) -> list[tuple[str, OpenMeteoWeatherClient]]:
    import yaml

    mapping = yaml.safe_load((_PROJECT_ROOT / config.get_required("paths.crop_production_regions_mapping")).read_text(encoding="utf-8"))
    locations = [("Dambulla", OpenMeteoWeatherClient.from_config(config_path))]
    for key in sorted({entry["region"] for entry in mapping["vegetable_production_regions"].values()}):
        region = mapping["regions"][key]
        locations.append((region["name"], OpenMeteoWeatherClient.for_coordinates(region["latitude"], region["longitude"], config_path=config_path)))
    return locations


def topup_weather(config: ConfigLoader, config_path: str, backup: Backup, today: pd.Timestamp) -> dict:
    regional_path = _resolve(config.get_required("paths.raw_files.weather_data_by_region"))
    dambulla_path = _resolve(config.get_required("paths.raw_files.weather_data"))
    regional = pd.read_csv(regional_path, parse_dates=["date"])
    end = today - pd.Timedelta(days=WEATHER_ARCHIVE_LAG_DAYS)
    added = {}
    frames = [regional]
    for name, client in weather_locations(config, config_path):
        last = regional.loc[regional["region"] == name, "date"].max()
        start = (last + pd.Timedelta(days=1)) if pd.notna(last) else regional["date"].min()
        if start > end:
            continue
        try:
            fetched = client.fetch_historical_daily_weather(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
        except LiveWeatherError as error:
            logger.warning("Weather for %s not updated: %s", name, error)
            continue
        fetched = fetched.dropna(subset=["rainfall_mm", "temperature_c", "humidity_pct"], how="all")  # not published yet
        frames.append(fetched.assign(region=name)[regional.columns])
        added[name] = len(fetched)
        time.sleep(1)
    if any(added.values()):
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(["region", "date"], keep="first").sort_values(["region", "date"], ignore_index=True)
        backup.before_change(regional_path)
        write_atomically(combined, regional_path)
        dambulla = combined[combined["region"] == "Dambulla"][["date", "rainfall_mm", "temperature_c", "humidity_pct"]]
        backup.before_change(dambulla_path)
        write_atomically(dambulla, dambulla_path)
    return added


def topup_fuel(config: ConfigLoader, config_path: str, backup: Backup, today: pd.Timestamp) -> str:
    path = _resolve(config.get_required("paths.raw_files.fuel_prices"))
    current = pd.read_csv(path, parse_dates=["date"])
    if current["date"].max() >= today:
        return "already current"
    backup.before_change(path)
    try:
        rebuild_fuel(config_path, f"{current['date'].min():%Y-%m-%d}", f"{today:%Y-%m-%d}")
    except Exception as error:  # noqa: BLE001 -- the old file stays in place and is reported
        shutil.copy2(backup.folder / path.name, path)
        logger.warning("Fuel prices not updated: %s", error)
        return f"not updated: {error}"
    return f"through {pd.read_csv(path)['date'].max()}"


# ---------------------------------------------------------------------------------------------


def run(args) -> int:
    config = ConfigLoader(args.config)
    raw_dir = _resolve(config.get("paths.raw_data_dir", "data/raw"))
    harti_cache = _resolve(config.get_required("data_update.harti_pdf_cache_dir"))
    cbsl_cache = _resolve(config.get("data_update.cbsl_pdf_cache_dir", "../../Sri-lanka-Vegetable-Price-Backups/cbsl_daily_price_reports"))
    backup = Backup(_resolve(config.get_required("data_update.backup_dir")), args.keep_backups)
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp.now().normalize()
    started = pd.Timestamp.now().isoformat(timespec="seconds")
    summary: dict = {"started_at": started, "today": str(today.date())}

    summary["harti"] = import_harti(config, raw_dir, harti_cache, backup, args.lookback_days, today)
    if not args.skip_cbsl:
        since = pd.Timestamp(args.cbsl_since) if args.cbsl_since else today - pd.Timedelta(days=args.lookback_days)
        summary["cbsl"] = import_cbsl(raw_dir, cbsl_cache, backup, since, today)
    if not args.skip_weather:
        summary["weather_rows_added"] = topup_weather(config, args.config, backup, today)
    if not args.skip_fuel:
        summary["fuel"] = topup_fuel(config, args.config, backup, today)

    if not args.skip_dcs:
        summary["dcs"] = import_dcs(config, raw_dir, backup)

    if backup.changed or not (raw_dir / "data_manifest.json").exists():
        manifest_path = raw_dir / "data_manifest.json"
        backup.before_change(manifest_path)
        manifest = write_manifest(raw_dir, pd.Timestamp.now().isoformat(timespec="seconds"))
        summary["latest"] = {d["file"]: d.get("latest_observation") for d in manifest["datasets"]}
    summary["backup"] = str(backup.folder) if backup.saved else None
    summary["finished_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    backup.prune()
    with (raw_dir / "update_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(summary, default=str) + "\n")

    harti = summary["harti"]
    logger.info("HARTI: newest listed %s; %d Dambulla and %d all-market rows added", harti["newest_listed"], harti["dambulla_rows_added"], harti["all_market_rows_added"])
    if harti["unreadable"]:
        logger.error("HARTI bulletins that couldn't be read (check the parser): %s", harti["unreadable"])
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--lookback-days", type=int, default=30, help="Re-check bulletins published this many days back.")
    parser.add_argument("--cbsl-since", default=None, help="First CBSL report date to fetch (default: the lookback window).")
    parser.add_argument("--keep-backups", type=int, default=14)
    parser.add_argument("--today", default=None, help=argparse.SUPPRESS)  # for tests
    parser.add_argument("--skip-cbsl", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-fuel", action="store_true")
    parser.add_argument("--skip-dcs", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
    os.chdir(_PROJECT_ROOT)
    raw_dir = _resolve(ConfigLoader(args.config).get("paths.raw_data_dir", "data/raw"))
    try:
        with single_run(raw_dir / ".update.lock"):
            return run(args)
    except UpdateError as error:
        logger.error("%s", error)
        return 3


if __name__ == "__main__":
    sys.exit(main())
