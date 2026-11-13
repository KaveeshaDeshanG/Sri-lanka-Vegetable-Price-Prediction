"""One-time (re-runnable) data-acquisition script: fetches REAL historical weather from
Open-Meteo's archive API for every production region referenced in
config/crop_production_regions.yaml, and combines it with the existing Dambulla series
(data/raw/weather_data.csv, reused as-is -- confirmed byte-identical to a fresh archive
fetch for the same coordinates/dates, so no need to re-fetch it) into one region-keyed
CSV: data/raw/weather_data_by_region.csv (date, region, rainfall_mm, temperature_c,
humidity_pct).

This does not modify harti_prices.csv, fuel_prices.csv, or weather_data.csv -- it only
adds a new file. Every value in it is a real Open-Meteo reading; nothing is fabricated,
interpolated across regions, or estimated.

Usage:
    .venv312/Scripts/python.exe scripts/build_historical_weather_by_region.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402
from src.data_collection.live_weather_client import (  # noqa: E402
    LiveWeatherError,
    OpenMeteoWeatherClient,
)

logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/config.yaml"
MAPPING_PATH = "config/crop_production_regions.yaml"
EXISTING_DAMBULLA_WEATHER_CSV = "data/raw/weather_data.csv"
OUTPUT_CSV = "data/raw/weather_data_by_region.csv"
DAMBULLA_REGION_NAME = "Dambulla"


def main() -> None:
    config = ConfigLoader(CONFIG_PATH)

    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = yaml.safe_load(file)
    regions = mapping["regions"]
    referenced_region_keys = {
        entry["region"] for entry in mapping["vegetable_production_regions"].values()
    }
    logger.info("Regions referenced by the vegetable mapping: %s", sorted(referenced_region_keys))

    dambulla_df = pd.read_csv(EXISTING_DAMBULLA_WEATHER_CSV, parse_dates=["date"])
    start_date = dambulla_df["date"].min().strftime("%Y-%m-%d")
    end_date = dambulla_df["date"].max().strftime("%Y-%m-%d")
    logger.info("Matching existing Dambulla weather series date range: %s to %s", start_date, end_date)

    dambulla_df = dambulla_df.copy()
    dambulla_df["region"] = DAMBULLA_REGION_NAME
    frames = [dambulla_df[["date", "region", "rainfall_mm", "temperature_c", "humidity_pct"]]]

    for region_key in sorted(referenced_region_keys):
        region = regions[region_key]
        logger.info(
            "Fetching real historical weather for %s (%.5f, %.5f) from %s to %s...",
            region["name"],
            region["latitude"],
            region["longitude"],
            start_date,
            end_date,
        )
        client = OpenMeteoWeatherClient.for_coordinates(
            region["latitude"], region["longitude"], config_path=CONFIG_PATH
        )
        try:
            df = client.fetch_historical_daily_weather(start_date=start_date, end_date=end_date)
        except LiveWeatherError as error:
            logger.error("Failed to fetch weather for %s: %s", region["name"], error)
            raise
        df = df.copy()
        df["region"] = region["name"]
        frames.append(df[["date", "region", "rainfall_mm", "temperature_c", "humidity_pct"]])
        logger.info("Fetched %d day(s) of real weather for %s", len(df), region["name"])

    combined = pd.concat(frames, ignore_index=True).sort_values(["region", "date"]).reset_index(drop=True)
    logger.info(
        "Combined region-keyed weather dataset: %d row(s) across %d region(s)",
        len(combined),
        combined["region"].nunique(),
    )

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info("Saved to '%s'", output_path)


if __name__ == "__main__":
    main()
