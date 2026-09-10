"""Top-level orchestrator for preprocessing: load -> clean -> transform -> merge -> save."""

import logging
from pathlib import Path

import pandas as pd

from config.config_loader import ConfigLoader
from src.data_collection import DataLoader

from .cleaner import DataCleaner
from .merger import DataMerger
from .transformer import DataTransformer

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Runs the full preprocessing stage and produces one ML-ready CSV.

    Each stage is also callable individually (load_data / clean_data /
    transform_data / merge_data / save_processed_data) so it can be tested
    and reasoned about in isolation, not only through run(). State is kept
    on the instance between calls, so methods must be called in that order.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config = ConfigLoader(config_path)
        self._data_loader = DataLoader(config_path)

        date_column = self._config.get("data_schema.date_column", "date")
        date_format = "%Y-%m-%d"
        self._weather_columns = self._config.get(
            "data_schema.weather_columns", ["rainfall_mm", "temperature_c", "humidity_pct"]
        )
        self._fuel_price_columns = self._config.get(
            "data_schema.fuel_price_columns", ["diesel_price", "petrol_price"]
        )
        price_column = self._config.get("data_schema.price_column", "price")
        vegetable_column = self._config.get("data_schema.vegetable_column", "vegetable")
        market_column = self._config.get("data_schema.market_column", "market")

        self._cleaner = DataCleaner(
            date_column=date_column,
            price_column=price_column,
            vegetable_column=vegetable_column,
            market_column=market_column,
            date_format=date_format,
            min_valid_year=self._config.get("preprocessing.min_valid_year", 2000),
            temperature_min_c=self._config.get("preprocessing.temperature_min_c", 0.0),
            temperature_max_c=self._config.get("preprocessing.temperature_max_c", 45.0),
            humidity_min_pct=self._config.get("preprocessing.humidity_min_pct", 0.0),
            humidity_max_pct=self._config.get("preprocessing.humidity_max_pct", 100.0),
            fuel_price_columns=self._fuel_price_columns,
        )
        self._transformer = DataTransformer(
            date_column=date_column,
            date_format=date_format,
            vegetable_column=vegetable_column,
            market_column=market_column,
            price_column=price_column,
        )
        self._merger = DataMerger(date_column=date_column)

        self._market_df: pd.DataFrame | None = None
        self._weather_df: pd.DataFrame | None = None
        self._fuel_df: pd.DataFrame | None = None
        self._merged_df: pd.DataFrame | None = None

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Load raw market, weather, and (if configured) fuel-price data via Phase 2's DataLoader."""
        logger.info("Preprocessing: loading raw data...")
        raw = self._data_loader.load_all()
        self._market_df = raw["market"]
        self._weather_df = raw["weather"]
        self._fuel_df = raw["fuel"]  # None if paths.raw_files.fuel_prices isn't configured
        return raw

    def clean_data(self) -> None:
        """Clean the loaded market, weather, and (if present) fuel-price DataFrames."""
        self._require_loaded()
        logger.info("Preprocessing: cleaning data...")
        self._market_df = self._cleaner.clean_market_data(self._market_df)
        self._weather_df = self._cleaner.clean_weather_data(self._weather_df)
        if self._fuel_df is not None:
            self._fuel_df = self._cleaner.clean_fuel_price_data(self._fuel_df)

    def transform_data(self) -> None:
        """Transform the cleaned market, weather, and (if present) fuel-price DataFrames."""
        self._require_loaded()
        logger.info("Preprocessing: transforming data...")
        self._market_df = self._transformer.transform_market_data(self._market_df)
        self._weather_df = self._transformer.transform_weather_data(
            self._weather_df, self._weather_columns
        )
        if self._fuel_df is not None:
            self._fuel_df = self._transformer.transform_fuel_price_data(
                self._fuel_df, self._fuel_price_columns
            )

    def merge_data(self) -> pd.DataFrame:
        """Merge the transformed market, weather, and (if present) fuel-price DataFrames on date."""
        self._require_loaded()
        logger.info("Preprocessing: merging data...")
        self._merged_df = self._merger.merge(self._market_df, self._weather_df)
        self._merged_df = self._merger.merge_fuel_prices(self._merged_df, self._fuel_df)
        return self._merged_df

    def save_processed_data(self, output_path: str | None = None) -> str:
        """Save the merged dataset to CSV.

        Args:
            output_path: overrides config's paths.processed_files.merged_dataset.

        Raises:
            RuntimeError: if merge_data() has not been called yet.
        """
        if self._merged_df is None:
            raise RuntimeError("merge_data() must be called before save_processed_data().")

        resolved_path = Path(
            output_path or self._config.get_required("paths.processed_files.merged_dataset")
        )
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self._merged_df.to_csv(resolved_path, index=False)
        logger.info(
            "Preprocessing: saved processed dataset to '%s' (%d rows)",
            resolved_path,
            len(self._merged_df),
        )
        return str(resolved_path)

    def run(self, output_path: str | None = None) -> pd.DataFrame:
        """Run the full pipeline: load -> clean -> transform -> merge -> save."""
        self.load_data()
        self.clean_data()
        self.transform_data()
        merged = self.merge_data()
        self.save_processed_data(output_path)
        return merged

    def _require_loaded(self) -> None:
        if self._market_df is None or self._weather_df is None:
            raise RuntimeError("load_data() must be called before this step.")
