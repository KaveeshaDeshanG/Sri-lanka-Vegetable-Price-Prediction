"""Imports every published official bulletin and rebuilds the project's datasets. Re-runnable.

Run it whenever HARTI or DCS publish new bulletins; no code edits are needed. It:

1. Backs up every current dataset, model and metrics file (paths.data_update.backup_dir).
2. HARTI daily food commodity bulletins (wholesale, Rs/kg): lists the archive, downloads only the
   PDFs not yet cached (applying data/reference/harti_url_corrections.csv for links that point at
   a folder), then rebuilds harti_prices.csv (Dambulla), harti_prices_detail.csv,
   harti_prices_build_report.json and harti_all_markets.csv (every market, kept separate).
3. DCS weekly retail prices (Colombo District, retail): downloads new bulletins and rebuilds
   dcs_weekly_retail_prices.csv. Retail is never merged with HARTI's wholesale prices.
4. CEYPETCO fuel prices and Open-Meteo weather, over the same dates as the price data.
5. Writes data/raw/data_manifest.json: source URL, retrieval time, rows and the real first and
   last observation date read from inside each file.

Nothing is estimated: a bulletin that can't be downloaded or read is listed in the build report,
and its prices stay missing. The model is NOT retrained here; see scripts/train_forecaster.py.

Usage:
    .venv312/Scripts/python.exe scripts/update_data.py            # download new bulletins and rebuild
    .venv312/Scripts/python.exe scripts/update_data.py --offline  # rebuild from the cached PDFs only
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402
from scripts import build_fuel_prices  # noqa: E402
from scripts.build_harti_prices import (  # noqa: E402
    PRICE_REVIEW_MAX,
    PRICE_REVIEW_MIN,
    download_pdf,
    extract_market_rows,
    fetch_archive_entries,
    finalize,
    hash_url,
)
from scripts.build_market_prices import build_panel, default_workers, map_in_chunks  # noqa: E402
from src.data_collection.dcs_retail_parser import parse_dcs_weekly_pdf  # noqa: E402
from src.data_collection.harti_live_client import _REQUEST_HEADERS  # noqa: E402
from src.data_collection.live_weather_client import OpenMeteoWeatherClient  # noqa: E402

logger = logging.getLogger("update_data")

DCS_PAGE = "https://www.statistics.gov.lk/InflationAndPrices/StaticalInformation/RetailPrices"
URL_CORRECTIONS = "data/reference/harti_url_corrections.csv"
DCS_MARKET = "Colombo District (DCS main markets)"


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (_PROJECT_ROOT / candidate).resolve()


def backup(config: ConfigLoader, backup_root: Path) -> Path:
    """Copy every dataset, model and metrics file into a new timestamped folder."""
    target = backup_root / f"before_update_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    for folder in ("data/raw", "models", "reports/metrics"):
        for path in (_PROJECT_ROOT / folder).glob("*"):
            if path.is_file() and path.suffix in (".csv", ".json", ".joblib"):
                destination = target / folder / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    logger.info("Backed up datasets, models and metrics to %s", target)
    return target


def harti_entries(config: ConfigLoader, offline: bool, cache_dir: Path) -> pd.DataFrame:
    listing_path = cache_dir / "archive_entries_listing.csv"
    if offline:
        entries = pd.read_csv(listing_path, parse_dates=["date"])
    else:
        entries = fetch_archive_entries(config.get("harti.archive_url"), 90)
        entries.to_csv(listing_path, index=False)
    corrections = pd.read_csv(_PROJECT_ROOT / URL_CORRECTIONS, parse_dates=["bulletin_date"])
    for row in corrections.itertuples(index=False):
        hit = (entries["date"] == row.bulletin_date) & (entries["url"] == row.listed_url)
        entries.loc[hit, "url"] = row.corrected_url
    return entries.drop_duplicates().sort_values("date", ignore_index=True)


def download_new_harti(entries: pd.DataFrame, cache_dir: Path, delay: float) -> list[dict]:
    failures = []
    for entry in entries.itertuples(index=False):
        path = cache_dir / f"{entry.date:%Y-%m-%d}_{abs(hash_url(entry.url)):016x}.pdf"
        if path.exists() or entry.url.endswith("/"):
            continue
        try:
            download_pdf(entry.url, path, 90, attempts=2)
            time.sleep(delay)
        except requests.exceptions.RequestException as error:
            failures.append({"bulletin": str(entry.date.date()), "url": entry.url, "reason": str(error)[:200]})
    return failures


EXTRACT_VERSION = "4"  # bump when the PDF readers change, so cached results are re-read


def _extract_cached(job: tuple) -> tuple:
    """One cached bulletin's rows for the target market, remembered on disk next to the PDF.

    Saving each result makes a long rebuild resumable: an interrupted run continues where it stopped.
    """
    date, url, path, market = job
    memo = Path(path).parent / "_extracted" / f"{Path(path).stem}_{market}_v{EXTRACT_VERSION}.pkl"
    if memo.exists():
        return date, url, pd.read_pickle(memo)
    extract = extract_market_rows(Path(path).read_bytes(), pd.Timestamp(date), market)
    memo.parent.mkdir(exist_ok=True)
    pd.to_pickle(extract, memo)
    return date, url, extract


def build_parallel(entries: pd.DataFrame, cache_dir: Path, market: str, workers: int | None = None) -> tuple[list[dict], dict]:
    """The same rows and report as build_harti_prices.build(), reading cached PDFs (in parallel
    unless workers=1)."""
    jobs = [(e.date, e.url, str(cache_dir / f"{e.date:%Y-%m-%d}_{abs(hash_url(e.url)):016x}.pdf"), market) for e in entries.itertuples(index=False)]
    rows, skipped, unknown, mismatches, late = [], [], {}, [], 0
    results = map_in_chunks(_extract_cached, jobs, workers or default_workers(), market)
    for date, url, extract in results:
        if extract.error:
            skipped.append({"bulletin": str(date.date()), "url": url, "reason": extract.error})
        else:
            late += extract.market_date != date
            rows.extend({**row, "bulletin_date": date, "source_pdf": url} for row in extract.rows)
        for label in extract.unknown_labels:
            unknown[label] = unknown.get(label, 0) + 1
        mismatches.extend(f"{date.date()} {item}" for item in extract.cross_check_mismatches)
    report = {
        "bulletins_listed": len(entries),
        "bulletins_skipped": skipped,
        "bulletins_where_market_date_trailed_bulletin": int(late),
        "unknown_row_labels": dict(sorted(unknown.items(), key=lambda item: -item[1])),
        "cross_check_mismatch_count": len(mismatches),
        "cross_check_mismatches": mismatches[:200],
    }
    return rows, report


def rebuild_harti(config: ConfigLoader, entries: pd.DataFrame, cache_dir: Path, raw_dir: Path, workers: int | None = 1) -> dict:
    market = config.get("harti.target_market", "Dambulla")
    cached = entries.apply(lambda e: (cache_dir / f"{e.date:%Y-%m-%d}_{abs(hash_url(e.url)):016x}.pdf").exists(), axis=1)
    rows, report = build_parallel(entries[cached], cache_dir, market, workers)
    report["bulletins_listed"] = len(entries)
    report["bulletins_skipped"] = [
        {"bulletin": str(e.date.date()), "url": e.url, "reason": "not downloadable from the archive's link (HTTP 404/403 or a folder link)"}
        for e in entries[~cached].itertuples(index=False)
    ] + report["bulletins_skipped"]
    detail = finalize(rows)
    detail.assign(date=detail["date"].dt.strftime("%Y-%m-%d"), bulletin_date=detail["bulletin_date"].dt.strftime("%Y-%m-%d")).to_csv(
        raw_dir / "harti_prices_detail.csv", index=False
    )
    detail[["date", "vegetable", "market", "price"]].assign(date=detail["date"].dt.strftime("%Y-%m-%d")).to_csv(
        raw_dir / "harti_prices.csv", index=False
    )
    review = detail[(detail["price"] < PRICE_REVIEW_MIN) | (detail["price"] > PRICE_REVIEW_MAX) | (detail["low"] > detail["high"])]
    report.update(
        {
            "rows_written": len(detail),
            "date_range": [str(detail["date"].min().date()), str(detail["date"].max().date())],
            "distinct_dates": int(detail["date"].nunique()),
            "rows_per_vegetable": detail["vegetable"].value_counts().to_dict(),
            "rows_needing_review": review[["date", "vegetable", "raw_cell"]].astype(str).to_dict("records"),
        }
    )
    (raw_dir / "harti_prices_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    panel, problems = build_panel(cache_dir, workers)
    panel.to_csv(raw_dir / "harti_all_markets.csv", index=False, date_format="%Y-%m-%d")
    logger.info("HARTI: %d Dambulla rows, %d all-market rows; %d bulletins skipped", len(detail), len(panel), len(report["bulletins_skipped"]))
    return {"skipped": len(report["bulletins_skipped"]), "unreadable_for_all_markets": problems[:50]}


def dcs_listing() -> list[dict]:
    html = requests.get(DCS_PAGE, headers=_REQUEST_HEADERS, timeout=60).text
    seen, listing = set(), []
    for href, text in re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.S | re.I):
        href = urljoin(DCS_PAGE, href)
        if "/retail/" in href and ("DCSB-WRP" in href or href.lower().endswith(".pdf")) and href not in seen:
            seen.add(href)
            listing.append({"label": " ".join(re.sub(r"<[^>]+>", "", text).split()), "link": href})
    return listing


def _dcs_file_name(link: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", link.rsplit("/", 1)[-1])
    return name if name.lower().endswith(".pdf") else name + ".pdf"


def download_new_dcs(listing: list[dict], cache_dir: Path, delay: float) -> list[dict]:
    failures = []
    for item in listing:
        path = cache_dir / _dcs_file_name(item["link"])
        if path.exists():
            continue
        try:
            url = item["link"]
            if not url.lower().endswith(".pdf"):  # newer links open a viewer page holding the PDF
                viewer = requests.get(url, headers=_REQUEST_HEADERS, timeout=60).text
                frame = re.search(r'<iframe[^>]+src=["\']([^"\']+)["\']', viewer)
                if not frame:
                    raise RuntimeError("no PDF in the viewer page")
                url = urljoin(url, frame.group(1))
            response = requests.get(url, headers=_REQUEST_HEADERS, timeout=90)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise RuntimeError("the response is not a PDF")
            path.write_bytes(response.content)
            item["pdf_url"] = url
            time.sleep(delay)
        except (requests.exceptions.RequestException, RuntimeError) as error:
            failures.append({"link": item["link"], "reason": str(error)[:200]})
    return failures


def rebuild_dcs(cache_dir: Path, raw_dir: Path, listing: list[dict] | None) -> dict:
    links = {_dcs_file_name(item["link"]): item.get("pdf_url", item["link"]) for item in (listing or [])}
    saved = cache_dir / "listing.json"
    if saved.exists():
        for item in json.loads(saved.read_text(encoding="utf-8")):
            links.setdefault(item.get("file") or _dcs_file_name(item["link"]), item.get("pdf_url", item["link"]))
    rows, problems = [], []
    for path in sorted(cache_dir.glob("*.pdf")):
        bulletin = parse_dcs_weekly_pdf(path.read_bytes(), path.name)
        if bulletin.error:
            problems.append({"file": path.name, "reason": bulletin.error})
            continue
        for row in bulletin.rows:
            if row["vegetable"] is None:
                continue
            rows.append(
                {
                    "year": bulletin.year,
                    "month": bulletin.month,
                    "week_of_month": bulletin.week_of_month,
                    "period": f"{bulletin.year}-{bulletin.month:02d} week {bulletin.week_of_month}",
                    "vegetable": row["vegetable"],
                    "item": row["item"],
                    "unit": row["unit"],
                    "market": DCS_MARKET,
                    "price_type": "retail (weekly average)",
                    "price": row["this_week_avg"],
                    "range_low": row["range_low"],
                    "range_high": row["range_high"],
                    "source_file": path.name,
                    "source_url": links.get(path.name, ""),
                }
            )
    table = pd.DataFrame(rows)
    # Two bulletins for the same week (a re-issue) would be two readings of one week: keep neither silently.
    duplicate = table.duplicated(["year", "month", "week_of_month", "vegetable"], keep=False)
    if duplicate.any():
        for file_name in sorted(table.loc[duplicate, "source_file"].unique()):
            problems.append({"file": file_name, "reason": "another bulletin covers the same week; both kept out"})
        table = table[~duplicate]
    table = table.sort_values(["year", "month", "week_of_month", "vegetable"], ignore_index=True)
    table.to_csv(raw_dir / "dcs_weekly_retail_prices.csv", index=False)
    (raw_dir / "dcs_weekly_retail_build_report.json").write_text(json.dumps({"rows": len(table), "problems": problems}, indent=2), encoding="utf-8")
    logger.info("DCS: %d weekly retail rows; %d bulletins not used", len(table), len(problems))
    return {"rows": len(table), "problems": len(problems)}


def rebuild_fuel(config_path: str, start: str, end: str) -> None:
    build_fuel_prices.main(["--config", config_path, "--start", start, "--end", end])


def fetch_weather_in_pieces(client, start: str, end: str, years: int = 2, waits: tuple[int, ...] = (65, 65, 125, 245)) -> pd.DataFrame:
    """Open-Meteo's archive, over [start, end], a few years per request.

    A decade in one request exceeds Open-Meteo's per-minute allowance (HTTP 429), so the range is
    split, and a rate-limited request is retried after the wait Open-Meteo asks for. Any other
    error is raised unchanged.
    """
    from src.data_collection.live_weather_client import LiveWeatherError

    frames, piece_start, final = [], pd.Timestamp(start), pd.Timestamp(end)
    while piece_start <= final:
        piece_end = min(piece_start + pd.DateOffset(years=years) - pd.Timedelta(days=1), final)
        for attempt in range(len(waits) + 1):
            try:
                frames.append(client.fetch_historical_daily_weather(f"{piece_start:%Y-%m-%d}", f"{piece_end:%Y-%m-%d}"))
                break
            except LiveWeatherError as error:
                if "429" not in str(error) or attempt == len(waits):
                    raise
                logger.info("Open-Meteo rate limit; waiting %d s", waits[attempt])
                time.sleep(waits[attempt])
        piece_start = piece_end + pd.Timedelta(days=1)
        time.sleep(2)
    return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date", ignore_index=True)


def rebuild_weather(config: ConfigLoader, config_path: str, start: str, end: str) -> None:
    """Dambulla plus every production region, from Open-Meteo's archive, over [start, end]."""
    import yaml

    dambulla = fetch_weather_in_pieces(OpenMeteoWeatherClient.from_config(config_path), start, end)
    mapping = yaml.safe_load((_PROJECT_ROOT / config.get_required("paths.crop_production_regions_mapping")).read_text(encoding="utf-8"))
    frames = [dambulla.assign(region="Dambulla")]
    for key in sorted({entry["region"] for entry in mapping["vegetable_production_regions"].values()}):
        region = mapping["regions"][key]
        client = OpenMeteoWeatherClient.for_coordinates(region["latitude"], region["longitude"], config_path=config_path)
        frames.append(fetch_weather_in_pieces(client, start, end).assign(region=region["name"]))
    combined = pd.concat(frames, ignore_index=True)[["date", "region", "rainfall_mm", "temperature_c", "humidity_pct"]]
    combined = combined.sort_values(["region", "date"], ignore_index=True)
    # Both files are written only once every location has been fetched, so they always cover the same dates.
    dambulla.assign(date=dambulla["date"].dt.strftime("%Y-%m-%d")).to_csv(_resolve(config.get_required("paths.raw_files.weather_data")), index=False)
    combined.assign(date=combined["date"].dt.strftime("%Y-%m-%d")).to_csv(_resolve(config.get_required("paths.raw_files.weather_data_by_region")), index=False)
    logger.info("Weather: %d regional rows, %s to %s", len(combined), start, end)


def write_manifest(raw_dir: Path, retrieved_at: str) -> dict:
    """What each dataset holds, read from the files themselves."""
    def describe(name, date_column, source, source_url, price_type, market_column=None, extra=None):
        path = raw_dir / name
        if not path.exists():
            return None
        table = pd.read_csv(path)
        dates = pd.to_datetime(table[date_column]) if date_column else None
        entry = {
            "file": f"data/raw/{name}",
            "source": source,
            "source_url": source_url,
            "price_type": price_type,
            "retrieved_at": retrieved_at,
            "rows": int(len(table)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if dates is not None:
            entry.update(first_observation=str(dates.min().date()), latest_observation=str(dates.max().date()))
        if market_column:
            entry["markets"] = sorted(table[market_column].unique().tolist())
        entry.update(extra(table) if extra else {})
        return entry

    def dcs_extra(table):
        latest = table.sort_values(["year", "month", "week_of_month"]).iloc[-1]
        first = table.sort_values(["year", "month", "week_of_month"]).iloc[0]
        return {"first_observation": first["period"], "latest_observation": latest["period"], "period_type": "week of month, as published"}

    manifest = {
        "generated_at": retrieved_at,
        "datasets": [
            entry
            for entry in (
                describe("harti_prices.csv", "date", "HARTI daily food commodity bulletins", "https://www.harti.gov.lk/daily-price.php", "wholesale, Rs/kg (midpoint of the published range)", "market"),
                describe("harti_all_markets.csv", "published", "HARTI daily food commodity bulletins, every market", "https://www.harti.gov.lk/daily-price.php", "wholesale, Rs/kg (midpoint of the published range)", "market"),
                describe("cbsl_daily_prices.csv", "date", "Central Bank of Sri Lanka, Daily Price Report", "https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report", "wholesale and retail, Rs/kg (Today column), kept apart by the price_type column", "market"),
                describe("dcs_weekly_retail_prices.csv", None, "Department of Census and Statistics, weekly retail prices", DCS_PAGE, "retail, weekly average", "market", dcs_extra),
                describe("fuel_prices.csv", "date", "CEYPETCO price revisions", "https://ceypetco.gov.lk/marketing-sales/historical-prices", "retail fuel, Rs/litre"),
                describe("weather_data_by_region.csv", "date", "Open-Meteo historical weather archive", "https://open-meteo.com/en/docs/historical-weather-api", "weather", "region"),
            )
            if entry
        ],
    }
    (raw_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--offline", action="store_true", help="Don't download; rebuild from the cached PDFs.")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-fuel", action="store_true")
    parser.add_argument("--skip-dcs", action="store_true")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between downloads.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Processes reading PDFs. 1 (the default) reads them one by one: slower on a first run, but a worker "
        "pool repeatedly stopped part-way on Windows. Results are remembered, so later runs are fast either way.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")

    config = ConfigLoader(args.config)
    raw_dir = _resolve(config.get("paths.raw_data_dir", "data/raw"))
    harti_cache = _resolve(config.get_required("data_update.harti_pdf_cache_dir"))
    dcs_cache = _resolve(config.get_required("data_update.dcs_pdf_cache_dir"))
    harti_cache.mkdir(parents=True, exist_ok=True)
    dcs_cache.mkdir(parents=True, exist_ok=True)
    retrieved_at = pd.Timestamp.now().isoformat(timespec="seconds")
    summary = {"backup": str(backup(config, _resolve(config.get_required("data_update.backup_dir"))))}

    entries = harti_entries(config, args.offline, harti_cache)
    summary["harti_listed"] = len(entries)
    summary["harti_download_failures"] = [] if args.offline else download_new_harti(entries, harti_cache, args.delay)
    summary["harti"] = rebuild_harti(config, entries, harti_cache, raw_dir, args.workers)

    if not args.skip_dcs:
        listing = None if args.offline else dcs_listing()
        summary["dcs_download_failures"] = [] if args.offline else download_new_dcs(listing, dcs_cache, args.delay)
        summary["dcs"] = rebuild_dcs(dcs_cache, raw_dir, listing)

    prices = pd.read_csv(raw_dir / "harti_prices.csv", parse_dates=["date"])
    start, end = prices["date"].min().strftime("%Y-%m-%d"), prices["date"].max().strftime("%Y-%m-%d")
    if not args.skip_fuel and not args.offline:
        rebuild_fuel(args.config, start, end)
    if not args.skip_weather and not args.offline:
        weather_end = min(prices["date"].max(), pd.Timestamp.now().normalize() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        rebuild_weather(config, args.config, start, weather_end)

    summary["manifest"] = write_manifest(raw_dir, retrieved_at)
    (raw_dir / "update_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for dataset in summary["manifest"]["datasets"]:
        logger.info("%-34s %8d rows  %s .. %s", dataset["file"], dataset["rows"], dataset.get("first_observation"), dataset.get("latest_observation"))


if __name__ == "__main__":
    main()
