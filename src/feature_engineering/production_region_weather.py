"""Overwrites each row's same-day weather reading with its vegetable's production
region's reading for that exact date -- so weather-lag features computed downstream
(WeatherFeatureGenerator, grouped by vegetable -- see feature_engineer.py) describe
the right location, not always the market's. The grouping matters: without it, a
date-only lookup would average different vegetables' now-different same-day readings
back together before computing the lag feature, silently erasing this correction.

Deliberately NOT wired into FeatureEngineer, TrainTestDatasetBuilder, or
PredictionPipeline themselves: those classes' existing tests use "Carrot"/"Tomato" as
generic placeholder vegetable names with no real-world regional meaning, and baking
region resolution into them would silently change behavior those tests depend on.
Instead, this is an explicit step callers (scripts/train.py for training,
app/services.py for prediction) apply to their own DataFrame BEFORE feature
engineering -- opt-in, not implicit.
"""

import logging
from pathlib import Path

import pandas as pd

from config.config_loader import ConfigLoader
from src.data_collection.production_region_resolver import ProductionRegionResolver

logger = logging.getLogger(__name__)

WEATHER_COLUMNS = ("rainfall_mm", "temperature_c", "humidity_pct")


def apply_production_region_weather(
    df: pd.DataFrame,
    vegetable_column: str = "vegetable",
    date_column: str = "date",
    region_weather_path: str | None = None,
    mapping_path: str | None = None,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Return a copy of df with rainfall_mm/temperature_c/humidity_pct overwritten by
    each row's vegetable's production-region reading for that date, where a
    sufficiently evidenced region is mapped AND that region's weather covers the
    date. Every other row (unmapped vegetable, or the region's series doesn't reach
    that date) keeps its existing (market-location) weather reading unchanged.

    Must be called AFTER the standard market+weather merge (so these columns
    already exist) and BEFORE FeatureEngineer.build_features() (so the lag/rolling
    weather features it computes read the corrected values) -- this function does
    no feature engineering itself, only corrects the same-day source columns.
    """
    if vegetable_column not in df.columns or date_column not in df.columns:
        logger.warning(
            "apply_production_region_weather: '%s'/'%s' column missing; leaving weather unchanged",
            vegetable_column,
            date_column,
        )
        return df

    missing_weather_columns = [c for c in WEATHER_COLUMNS if c not in df.columns]
    if missing_weather_columns:
        logger.warning(
            "apply_production_region_weather: weather column(s) %s missing; leaving weather unchanged",
            missing_weather_columns,
        )
        return df

    config = ConfigLoader(config_path)
    resolved_region_weather_path = region_weather_path or config.get(
        "paths.raw_files.weather_data_by_region", "data/raw/weather_data_by_region.csv"
    )
    resolved_mapping_path = mapping_path or config.get(
        "paths.crop_production_regions_mapping", "config/crop_production_regions.yaml"
    )

    region_weather_file = Path(resolved_region_weather_path)
    if not region_weather_file.exists():
        logger.warning(
            "apply_production_region_weather: region weather file '%s' not found; "
            "leaving every row's market-location weather unchanged",
            region_weather_file,
        )
        return df

    resolver = ProductionRegionResolver(mapping_path=resolved_mapping_path, config_path=config_path)
    unique_vegetables = df[vegetable_column].dropna().unique()
    vegetable_to_region = {}
    for vegetable in unique_vegetables:
        resolved = resolver.resolve(vegetable)
        if resolved.is_production_region:
            vegetable_to_region[vegetable] = resolved.region_name

    if not vegetable_to_region:
        logger.info("apply_production_region_weather: no vegetable in this data has a mapped production region")
        return df

    df = df.copy()
    df["_production_region"] = df[vegetable_column].map(vegetable_to_region)
    mapped_row_count = int(df["_production_region"].notna().sum())

    region_weather = pd.read_csv(region_weather_file, parse_dates=[date_column])
    region_weather = region_weather.rename(
        columns={"region": "_production_region", **{c: f"_region_{c}" for c in WEATHER_COLUMNS}}
    )

    merged = df.merge(
        region_weather[[date_column, "_production_region", *[f"_region_{c}" for c in WEATHER_COLUMNS]]],
        on=[date_column, "_production_region"],
        how="left",
    )

    overridden_count = 0
    for column in WEATHER_COLUMNS:
        region_column = f"_region_{column}"
        override_mask = merged["_production_region"].notna() & merged[region_column].notna()
        overridden_count = max(overridden_count, int(override_mask.sum()))
        merged.loc[override_mask, column] = merged.loc[override_mask, region_column]

    merged = merged.drop(columns=["_production_region", *[f"_region_{c}" for c in WEATHER_COLUMNS]])
    logger.info(
        "apply_production_region_weather: %d/%d row(s) had a mapped production region; "
        "%d row(s) actually overridden (region weather covers that date)",
        mapped_row_count,
        len(df),
        overridden_count,
    )
    return merged
