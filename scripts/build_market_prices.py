"""Builds data/raw/harti_all_markets.csv: every market's readings from the cached HARTI bulletins.

Run scripts/build_harti_prices.py first, which downloads the PDFs into the cache this reads.
`published` is the bulletin's date, which is when a reading became known. The direct forecaster
uses it because some markets are always reported a day late.

Usage:
    .venv312/Scripts/python.exe scripts/build_market_prices.py --cache-dir <folder>
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402
from src.data_collection.harti_live_client import HartiLiveClient, HartiLiveDataError, parse_all_markets  # noqa: E402
from src.preprocessing.transformer import DataTransformer  # noqa: E402

logger = logging.getLogger(__name__)


EXTRACT_VERSION = "4"  # bump when the PDF readers change, so remembered results are re-read


def read_bulletin(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """(this PDF's readings, or None and why not). The cache filename starts with its date.

    The result is remembered beside the PDF, so an interrupted rebuild resumes where it stopped.
    """
    memo = path.parent / "_extracted" / f"{path.stem}_all_markets_v{EXTRACT_VERSION}.pkl"
    if memo.exists():
        return pd.read_pickle(memo)
    try:
        table = HartiLiveClient()._extract_vegetable_table(path.read_bytes())
        result = parse_all_markets(table, pd.Timestamp(path.name[:10])), None
    except HartiLiveDataError as error:
        result = None, f"{path.name}: {error}"
    memo.parent.mkdir(exist_ok=True)
    pd.to_pickle(result, memo)
    return result


def map_in_chunks(function, items: list, workers: int, label: str, chunk: int = 200):
    """function(item) for every item, in order. workers=1 reads them one by one in this process,
    which is slower but immune to a worker pool failing; otherwise a fresh pool per chunk.
    Progress is logged after each chunk; with the remembered results, a stopped run resumes."""
    results = []
    for chunk_start in range(0, len(items), chunk):
        part = items[chunk_start:chunk_start + chunk]
        if workers == 1:
            results.extend(function(item) for item in part)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results.extend(pool.map(function, part, chunksize=4))
        logger.info("Read %d/%d bulletins for %s", len(results), len(items), label)
    return results


def default_workers() -> int:
    return max(2, (os.cpu_count() or 4) - 2)


def build_panel(cache_dir: str | Path, workers: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Every market's readings from every cached bulletin, one row per (published, market, vegetable)."""
    paths = sorted(Path(cache_dir).glob("*.pdf"))
    if not paths:
        raise SystemExit(f"No PDFs found in {cache_dir}. Run scripts/build_harti_prices.py first.")
    frames, problems = [], []
    for frame, problem in map_in_chunks(read_bulletin, paths, workers or default_workers(), "every market"):
        if frame is not None:
            frames.append(frame)
        if problem:
            problems.append(problem)
    panel = pd.concat(frames, ignore_index=True)
    panel = DataTransformer().standardize_vegetable_names(panel)  # Eggplant is "Brinjal" in training
    panel = panel.drop_duplicates(["published", "market", "vegetable"]).sort_values(["published", "market", "vegetable"])
    return panel.reset_index(drop=True), problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
    panel, problems = build_panel(args.cache_dir)
    output_path = Path(ConfigLoader(args.config).get_required("paths.raw_files.harti_all_markets"))
    panel.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    logger.info("Wrote %d readings to '%s'; %d unreadable.", len(panel), output_path, len(problems))
    for problem in problems[:20]:
        logger.warning(problem)


if __name__ == "__main__":
    main()
