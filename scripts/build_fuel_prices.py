"""Rebuilds data/raw/fuel_prices.csv from CEYPETCO's own published price-revision archive.

Reads the public data behind ceypetco.gov.lk/marketing-sales/historical-prices (no login
needed; the site's public page requests the same URL) and expands each published revision
into one row per calendar day. Fuel price is an administered step price, so carrying a
revision forward until the next one is the actual price in effect, not an estimate.

Column positions were checked against independently reported revisions: index 1 is Lanka
Petrol 92 Octane and index 2 is Lanka Auto Diesel. A published date can be a day later
than the midnight the change took effect, and is used as published.

Usage:
    .venv312/Scripts/python.exe scripts/build_fuel_prices.py --end 2026-11-06
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://api.ceypetco.gov.lk/api/admin/historical-prices/active"
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; veggie-price-prediction/1.0)"}
PETROL_92_INDEX = 1
AUTO_DIESEL_INDEX = 2


def parse_revisions(records: list[dict]) -> pd.DataFrame:
    """Published fuel revisions as (date, diesel_price, petrol_price), oldest first."""
    rows = [
        {
            "date": pd.to_datetime(str(record["sortKey"]), format="%Y%m%d"),
            "diesel_price": float(record["values"][AUTO_DIESEL_INDEX]),
            "petrol_price": float(record["values"][PETROL_92_INDEX]),
        }
        for record in records
        if record.get("kind") == "fuel" and record.get("status") == "active"
    ]
    return pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date", ignore_index=True)


def expand_to_daily(revisions: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """One row per day in [start, end], each holding the latest revision on or before it."""
    days = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    daily = pd.merge_asof(days, revisions, on="date", direction="backward")
    if daily[["diesel_price", "petrol_price"]].isna().any().any():
        raise ValueError(f"No published fuel revision on or before {start}")
    return daily


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default="2015-06-22", help="First date, the first HARTI bulletin (default: %(default)s).")
    parser.add_argument("--end", default=None, help="Last date (default: today).")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")

    response = requests.get(ARCHIVE_URL, timeout=30, headers=_REQUEST_HEADERS)
    response.raise_for_status()
    revisions = parse_revisions(response.json()["data"])
    end = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")
    daily = expand_to_daily(revisions, args.start, end)

    output_path = Path(ConfigLoader(args.config).get_required("paths.raw_files.fuel_prices"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keeps the exact published response, so the daily CSV can be traced back to its source.
    (output_path.parent / "fuel_prices_source_snapshot.json").write_text(response.text, encoding="utf-8")
    daily.assign(date=daily["date"].dt.strftime("%Y-%m-%d")).to_csv(output_path, index=False)
    logger.info("Wrote %d days (%s to %s) from %d published revisions to '%s'", len(daily), args.start, end, len(revisions), output_path)


if __name__ == "__main__":
    main()
