"""Rebuilds data/raw/harti_prices.csv from HARTI's own daily bulletin PDF archive.

Every row is a real Dambulla wholesale reading read from a published bulletin: the
midpoint of that day's "min - max" range. A vegetable with no reading ("-") gets no row,
and a bulletin that cannot be read is reported and skipped -- nothing is estimated.

Reads every year table on the archive page, and reads each cell with word-level text
extraction because PyMuPDF's Table.extract() scrambles this PDF's date cells and strips
spaces from its row labels. Each price is cross-checked against Table.extract()'s digits,
and any disagreement is counted in the build report.

Outputs (all under data/raw/, which .gitignore excludes):
    harti_prices.csv                 date, vegetable, market, price -- the pipeline's input
    harti_prices_detail.csv          the same rows plus low/high/raw cell/source PDF
    harti_prices_build_report.json   coverage, skipped bulletins, unknown labels

Downloads about 900 PDFs (roughly 700 MB) one at a time. Downloads are cached in
--cache-dir, so an interrupted run resumes where it stopped.

Usage:
    .venv312/Scripts/python.exe scripts/build_harti_prices.py --cache-dir <folder>
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pymupdf
import requests
from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402
from src.data_collection.harti_live_client import (  # noqa: E402
    _REQUEST_HEADERS,
    MAX_REPORTING_LAG_DAYS,
    cell_text,
    parse_price_range,
    resolve_vegetable,
    split_merged_label_rows,
)
from src.data_collection.harti_live_client import parse_reporting_date as _parse_any_reporting_date  # noqa: E402

logger = logging.getLogger(__name__)

# HARTI's archive (daily-price.php) starts with the 2015-06-22 bulletin. Earlier builds started at
# 2024-04-25, which is why the datasets used to begin in 2024.
DEFAULT_START_DATE = "2015-06-22"
PRICE_REVIEW_MIN = 10
PRICE_REVIEW_MAX = 5000


@dataclass
class BulletinExtract:
    """What one PDF yielded: the target market's reporting date and its priced rows."""

    market_date: pd.Timestamp | None = None
    rows: list[dict] = field(default_factory=list)
    unknown_labels: list[str] = field(default_factory=list)
    cross_check_mismatches: list[str] = field(default_factory=list)
    error: str | None = None


def fetch_archive_entries(archive_url: str, timeout_seconds: float) -> pd.DataFrame:
    """Every English bulletin listed on the archive page, across all of its year tables."""
    response = requests.get(archive_url, timeout=timeout_seconds, headers=_REQUEST_HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    entries = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3 or cells[2].find("a") is None:
                continue
            date = pd.to_datetime(cells[0].get_text(strip=True), format="%Y-%m-%d", errors="coerce")
            medium = cells[1].get_text(strip=True).lower()
            if pd.notna(date) and medium == "english":
                entries.append({"date": date, "url": urljoin(archive_url, cells[2].find("a")["href"])})
    return pd.DataFrame(entries, columns=["date", "url"]).drop_duplicates().sort_values("date", ignore_index=True)


def parse_reporting_date(text: str, bulletin_date: pd.Timestamp) -> pd.Timestamp | None:
    """Parse a market's reporting date, accepting it only if it fits this bulletin's own date."""
    parsed = _parse_any_reporting_date(text)
    if parsed is not None and 0 <= (bulletin_date - parsed).days <= MAX_REPORTING_LAG_DAYS:
        return parsed
    return None


_cell_text = cell_text


def extract_market_rows(pdf_bytes: bytes, bulletin_date: pd.Timestamp, market: str) -> BulletinExtract:
    """Read one bulletin's priced rows for `market`. Never raises for a bad PDF: reports it."""
    target = market.lower()
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:  # noqa: BLE001 -- pymupdf's exception types vary by corruption mode
        return BulletinExtract(error=f"could not open PDF: {error}")

    for page in document:
        if target not in page.get_text().lower():
            continue
        for table in page.find_tables().tables:
            extracted = table.extract()
            header_index = next(
                (
                    index
                    for index, row in enumerate(extracted[:4])
                    if any(cell and target in cell.lower() for cell in row)
                ),
                None,
            )
            if not header_index:  # also rejects a header on row 0: the date row sits above it
                continue
            column = next(i for i, cell in enumerate(extracted[header_index]) if cell and target in cell.lower())
            return _read_table(page, table, extracted, header_index, column, bulletin_date, market)
    return BulletinExtract(error=f"no table with a '{market}' column on any page")


def _read_table(page, table, extracted, header_index, column, bulletin_date, market) -> BulletinExtract:
    words = page.get_text("words")
    date_text = _cell_text(words, table.rows[header_index - 1].cells[column])
    market_date = parse_reporting_date(date_text, bulletin_date)
    if market_date is None:
        return BulletinExtract(error=f"unusable reporting date {date_text!r}")

    result = BulletinExtract(market_date=market_date)
    seen = set()
    # Row labels, with any label cell spanning several price rows split back onto those rows.
    labels = split_merged_label_rows(
        [[_cell_text(words, table.rows[i].cells[0]) or None] for i in range(header_index + 1, len(extracted))],
        words,
        table,
        header_index + 1,
    )
    for row_index in range(header_index + 1, len(extracted)):
        label = " ".join((labels[row_index - header_index - 1][0] or "").split())
        if not label:
            continue
        vegetable = resolve_vegetable(label)
        if vegetable is None:
            result.unknown_labels.append(label)
            continue
        raw = _cell_text(words, table.rows[row_index].cells[column])
        table_digits = re.findall(r"\d[\d,]*", extracted[row_index][column] or "")
        if sorted(re.findall(r"\d[\d,]*", raw)) != sorted(table_digits):
            result.cross_check_mismatches.append(f"{label}: words={raw!r} table={extracted[row_index][column]!r}")
        price_range = parse_price_range(raw)
        if price_range is None or vegetable in seen:
            continue
        seen.add(vegetable)
        low, high = price_range
        result.rows.append(
            {
                "date": market_date,
                "vegetable": vegetable,
                "market": market,
                "price": (low + high) / 2,
                "low": low,
                "high": high,
                "raw_cell": raw,
            }
        )
    if not result.rows:
        result.error = "table found, but no usable price in it"
    return result


def download_pdf(url: str, path: Path, timeout_seconds: float, attempts: int = 3) -> bytes:
    """Return the cached PDF, downloading it (with retries) only if it isn't cached yet."""
    if path.exists():
        return path.read_bytes()
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout_seconds, headers=_REQUEST_HEADERS)
            response.raise_for_status()
            if not response.content:
                raise requests.exceptions.RequestException("empty response")
            temporary_path = path.with_suffix(".part")
            temporary_path.write_bytes(response.content)
            temporary_path.replace(path)
            return response.content
        except requests.exceptions.RequestException as error:
            if attempt == attempts:
                raise
            logger.warning("Download attempt %d failed for %s: %s", attempt, url, error)
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def build(entries: pd.DataFrame, cache_dir: Path, market: str, timeout_seconds: float, delay_seconds: float):
    """Process bulletins oldest-first, returning (detail rows, build report)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped: list[dict] = []
    unknown_labels: dict[str, int] = {}
    mismatches: list[str] = []
    late_reports = 0

    for position, entry in enumerate(entries.itertuples(index=False), start=1):
        path = cache_dir / f"{entry.date:%Y-%m-%d}_{abs(hash_url(entry.url)):016x}.pdf"
        cached = path.exists()
        try:
            pdf_bytes = download_pdf(entry.url, path, timeout_seconds)
        except requests.exceptions.RequestException as error:
            skipped.append({"bulletin": str(entry.date.date()), "url": entry.url, "reason": f"download failed: {error}"})
            continue

        extract = extract_market_rows(pdf_bytes, entry.date, market)
        if extract.error:
            skipped.append({"bulletin": str(entry.date.date()), "url": entry.url, "reason": extract.error})
        else:
            late_reports += extract.market_date != entry.date
            for row in extract.rows:
                rows.append({**row, "bulletin_date": entry.date, "source_pdf": entry.url})
        for label in extract.unknown_labels:
            unknown_labels[label] = unknown_labels.get(label, 0) + 1
        mismatches.extend(f"{entry.date.date()} {item}" for item in extract.cross_check_mismatches)

        if position % 25 == 0 or position == len(entries):
            logger.info("Processed %d/%d bulletins, %d price rows, %d skipped", position, len(entries), len(rows), len(skipped))
        if not cached:
            time.sleep(delay_seconds)

    report = {
        "bulletins_listed": len(entries),
        "bulletins_skipped": skipped,
        "bulletins_where_market_date_trailed_bulletin": late_reports,
        "unknown_row_labels": dict(sorted(unknown_labels.items(), key=lambda item: -item[1])),
        "cross_check_mismatch_count": len(mismatches),
        "cross_check_mismatches": mismatches[:200],
    }
    return rows, report


def hash_url(url: str) -> int:
    """A stable hash (Python's own hash() changes between runs, breaking the resume cache)."""
    value = 0
    for character in url:
        value = (value * 131 + ord(character)) % (2**61 - 1)
    return value


def finalize(rows: list[dict]) -> pd.DataFrame:
    """One row per (date, vegetable). A bulletin dated the same as its market reading wins
    over a later bulletin merely repeating that reading."""
    detail = pd.DataFrame(rows)
    detail["_same_day"] = detail["bulletin_date"] == detail["date"]
    detail = (
        detail.sort_values(["date", "vegetable", "_same_day", "bulletin_date"], ascending=[True, True, False, True])
        .drop_duplicates(["date", "vegetable"], keep="first")
        .drop(columns="_same_day")
        .reset_index(drop=True)
    )
    return detail


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cache-dir", required=True, help="Folder for downloaded PDFs (reused on re-runs).")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="First bulletin date (default: %(default)s).")
    parser.add_argument("--end", default=None, help="Last bulletin date (default: the newest published).")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between downloads (default: %(default)s).")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth bulletin, to sample the formats.")
    parser.add_argument("--output-dir", default=None, help="Default: the folder holding paths.raw_files.harti_prices.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
    config = ConfigLoader(args.config)
    csv_path = Path(config.get_required("paths.raw_files.harti_prices"))
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    market = config.get("harti.target_market", "Dambulla")
    timeout_seconds = max(60.0, config.get("harti.request_timeout_seconds", 15))

    entries = fetch_archive_entries(config.get("harti.archive_url"), timeout_seconds)
    selected = entries[entries["date"] >= pd.Timestamp(args.start)]
    if args.end:
        selected = selected[selected["date"] <= pd.Timestamp(args.end)]
    selected = selected.iloc[:: args.every]
    logger.info("Archive lists %d English bulletins; processing %d", len(entries), len(selected))

    rows, report = build(selected, Path(args.cache_dir), market, timeout_seconds, args.delay)
    if not rows:
        raise SystemExit("No price rows could be read from any bulletin -- nothing written.")
    detail = finalize(rows)

    detail.assign(date=detail["date"].dt.strftime("%Y-%m-%d"), bulletin_date=detail["bulletin_date"].dt.strftime("%Y-%m-%d")).to_csv(
        output_dir / "harti_prices_detail.csv", index=False
    )
    detail[["date", "vegetable", "market", "price"]].assign(date=detail["date"].dt.strftime("%Y-%m-%d")).to_csv(
        output_dir / csv_path.name, index=False
    )

    out_of_range = detail[(detail["price"] < PRICE_REVIEW_MIN) | (detail["price"] > PRICE_REVIEW_MAX) | (detail["low"] > detail["high"])]
    report.update(
        {
            "rows_written": len(detail),
            "date_range": [str(detail["date"].min().date()), str(detail["date"].max().date())],
            "distinct_dates": int(detail["date"].nunique()),
            "rows_per_vegetable": detail["vegetable"].value_counts().to_dict(),
            "rows_needing_review": out_of_range[["date", "vegetable", "raw_cell"]].astype(str).to_dict("records"),
        }
    )
    (output_dir / "harti_prices_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %d rows, %s to %s. Skipped %d bulletins; %d cross-check mismatches.",
        len(detail),
        *report["date_range"],
        len(report["bulletins_skipped"]),
        report["cross_check_mismatch_count"],
    )


if __name__ == "__main__":
    main()
