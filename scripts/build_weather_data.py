"""Rebuilds data/raw/weather_data.csv: real daily Dambulla weather from Open-Meteo's archive.

scripts/build_historical_weather_by_region.py reads this file for its date range and its
Dambulla series, so run this first.

Usage:
    .venv312/Scripts/python.exe scripts/build_weather_data.py --end 2026-11-04
"""

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402
from src.data_collection.live_weather_client import OpenMeteoWeatherClient  # noqa: E402

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default="2015-06-22", help="First date, the first HARTI bulletin (default: %(default)s).")
    parser.add_argument("--end", required=True, help="Last date; the archive runs about 2 days behind today.")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")

    weather = OpenMeteoWeatherClient.from_config(args.config).fetch_historical_daily_weather(args.start, args.end)
    output_path = Path(ConfigLoader(args.config).get_required("paths.raw_files.weather_data"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather.assign(date=weather["date"].dt.strftime("%Y-%m-%d")).to_csv(output_path, index=False)
    logger.info("Wrote %d days of real Dambulla weather to '%s'", len(weather), output_path)


if __name__ == "__main__":
    main()
