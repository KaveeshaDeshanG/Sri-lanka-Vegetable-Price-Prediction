"""Builds data/raw/harti_all_markets_ranges.csv: the published min-max range behind every stored
other-market HARTI price, read again from the saved bulletin PDFs.

harti_all_markets.csv keeps only each range's midpoint (the forecaster's input). This side file adds
the published low and high for display on the Predict page. It never changes harti_all_markets.csv:
a range is kept only when its midpoint equals the price already stored for the same bulletin,
reporting date, market and vegetable, so the two files can't disagree. Rows that don't match are
counted in the report and left out, never adjusted.

Incremental: only bulletins not yet in the side file are read, so scripts/daily_update.py runs it
after each import. Each PDF's result is remembered beside it, so an interrupted full build resumes.

Usage:
    .venv312/Scripts/python.exe scripts/build_market_price_ranges.py [--workers 8]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402
from scripts.build_market_prices import default_workers, map_in_chunks  # noqa: E402
from src.data_collection.harti_live_client import HartiLiveClient, HartiLiveDataError, parse_all_markets  # noqa: E402
from src.preprocessing.transformer import DataTransformer  # noqa: E402

logger = logging.getLogger(__name__)

RANGES_FILE = "harti_all_markets_ranges.csv"
REPORT_FILE = "harti_all_markets_ranges_report.json"
KEY = ["published", "reported", "market", "vegetable"]
RANGE_VERSION = "1"


def read_ranges(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """(this PDF's readings with low and high, or None and why not); remembered beside the PDF."""
    memo = path.parent / "_extracted" / f"{path.stem}_ranges_v{RANGE_VERSION}.pkl"
    if memo.exists():
        return pd.read_pickle(memo)
    try:
        table = HartiLiveClient()._extract_vegetable_table(path.read_bytes())
        result = parse_all_markets(table, pd.Timestamp(path.name[:10]), keep_range=True), None
    except HartiLiveDataError as error:
        result = None, f"{path.name}: {error}"
    memo.parent.mkdir(exist_ok=True)
    pd.to_pickle(result, memo)
    return result


def write_atomically(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def update_ranges(raw_dir: Path, cache_dir: Path, workers: int = 1) -> dict:
    """Add the ranges of every stored other-market bulletin not yet in the side file."""
    panel = pd.read_csv(raw_dir / "harti_all_markets.csv", parse_dates=["published", "reported"])
    panel = panel[panel["market"] != "Dambulla"]
    output = raw_dir / RANGES_FILE
    existing = pd.read_csv(output, parse_dates=["published", "reported"]) if output.exists() else pd.DataFrame(columns=[*KEY, "low", "high"])
    done = set(existing["published"])
    wanted = sorted(set(panel["published"]) - done)
    pdfs = {pd.Timestamp(p.name[:10]): p for p in sorted(cache_dir.glob("????-??-??_*.pdf"))}
    paths = [pdfs[day] for day in wanted if day in pdfs]
    no_pdf = len(wanted) - len(paths)
    frames, problems = [], []
    for frame, problem in map_in_chunks(read_ranges, paths, workers, "price ranges") if paths else []:
        if frame is not None and not frame.empty:
            frames.append(frame)
        if problem:
            problems.append(problem)
    added = mismatched = 0
    if frames:
        read = DataTransformer().standardize_vegetable_names(pd.concat(frames, ignore_index=True))
        read = read.drop_duplicates(KEY)
        joined = panel.merge(read, on=KEY, how="inner", suffixes=("", "_read"))
        matches = (joined["price"] - joined["price_read"]).abs() < 0.005
        mismatched = int((~matches).sum())
        fresh = joined.loc[matches, [*KEY, "low", "high"]]
        combined = pd.concat([existing, fresh], ignore_index=True).drop_duplicates(KEY).sort_values(KEY)
        added = len(combined) - len(existing)
        write_atomically(combined, output)
    total = len(pd.read_csv(output)) if output.exists() else 0
    report = {
        "bulletins_read": len(paths),
        "bulletins_without_saved_pdf": no_pdf,
        "unreadable": problems,
        "rows_added": added,
        "rows_not_matching_stored_price": mismatched,
        "rows_total": total,
        "other_market_rows_stored": len(panel),
        "coverage_pct": round(100 * total / len(panel), 2) if len(panel) else 0.0,
    }
    (raw_dir / REPORT_FILE).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
    config = ConfigLoader(_PROJECT_ROOT / args.config)
    raw_dir = _PROJECT_ROOT / config.get("paths.raw_data_dir", "data/raw")
    cache_dir = (_PROJECT_ROOT / config.get_required("data_update.harti_pdf_cache_dir")).resolve()
    report = update_ranges(raw_dir, cache_dir, args.workers or default_workers())
    logger.info("Price ranges: %s", {k: v for k, v in report.items() if k != "unreadable"})


if __name__ == "__main__":
    main()
