"""Top-level facade for loading all raw data sources (market prices + weather)."""

import logging

import pandas as pd

from config.config_loader import ConfigLoader

from .fuel_price_loader import FuelPriceDataLoader
from .market_loader import MarketDataLoader
from .weather_loader import WeatherDataLoader

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads every raw data source the pipeline needs, using config.yaml for paths and schema.

    This is the only class later pipeline stages (preprocessing, features) should
    import from data_collection — MarketDataLoader and WeatherDataLoader stay as
    internal building blocks so the raw-file layout can change without changing
    the public interface.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        try:
            self._config = ConfigLoader(config_path)
        except (FileNotFoundError, ValueError) as error:
            logger.error("DataLoader could not load configuration: %s", error)
            raise

        date_column = self._config.get("data_schema.date_column", "date")
        date_format = "%Y-%m-%d"

        self._market_loader = MarketDataLoader(
            date_column=date_column,
            vegetable_column=self._config.get("data_schema.vegetable_column", "vegetable"),
            market_column=self._config.get("data_schema.market_column", "market"),
            price_column=self._config.get("data_schema.price_column", "price"),
            date_format=date_format,
        )
        self._weather_loader = WeatherDataLoader(
            date_column=date_column,
            weather_columns=self._config.get("data_schema.weather_columns"),
            date_format=date_format,
        )
        self._fuel_price_loader = FuelPriceDataLoader(
            date_column=date_column,
            fuel_price_columns=self._config.get("data_schema.fuel_price_columns"),
            date_format=date_format,
        )

    def load_market_data(self) -> pd.DataFrame:
        """Load HARTI market price data, plus a second source if one is configured.

        harti_prices is required. dambulla_prices is optional: if the config
        key is absent or null, market data is loaded from HARTI alone rather
        than raising, since HARTI's own bulletin already reports Dambulla
        (DEC) prices directly and no genuinely independent second source is
        assumed to exist unless one is actually configured.

        Raises:
            ConfigKeyError: if paths.raw_files.harti_prices is missing from config.yaml.
            DataValidationError: if a configured raw file is missing, unreadable,
                or fails validation.
        """
        harti_path = self._config.get_required("paths.raw_files.harti_prices")
        dambulla_path = self._config.get("paths.raw_files.dambulla_prices")
        return self._market_loader.load_all(harti_path, dambulla_path)

    def load_weather_data(self) -> pd.DataFrame:
        """Load weather data.

        Raises:
            ConfigKeyError: if the weather-file path is missing from config.yaml.
            DataValidationError: if the weather file is missing, unreadable, or
                fails validation.
        """
        weather_path = self._config.get_required("paths.raw_files.weather_data")
        return self._weather_loader.load(weather_path)

    def load_fuel_price_data(self) -> pd.DataFrame | None:
        """Load fuel-price data, if configured.

        paths.raw_files.fuel_prices is optional: if the config key is absent
        or null, this returns None rather than raising, the same way
        dambulla_prices is treated as an optional second market source.
        Unlike weather_data (currently required for load_all()), fuel price
        was added after the pipeline was already working without it and
        must not become a new hard requirement for configs that haven't set
        it up.

        Raises:
            DataValidationError: if a path IS configured but the file is
                missing, unreadable, or fails validation.
        """
        fuel_prices_path = self._config.get("paths.raw_files.fuel_prices")
        if fuel_prices_path is None:
            logger.info(
                "Fuel price data: paths.raw_files.fuel_prices not configured; skipping."
            )
            return None
        return self._fuel_price_loader.load(fuel_prices_path)

    def load_all(self) -> dict[str, pd.DataFrame | None]:
        """Load market, weather, and (if configured) fuel-price data.

        Returns:
            A dict with keys "market", "weather", and "fuel". "fuel" maps to
            None when paths.raw_files.fuel_prices isn't configured -- callers
            that don't yet use fuel data can safely ignore that key.
        """
        logger.info("Loading all raw data sources...")
        data = {
            "market": self.load_market_data(),
            "weather": self.load_weather_data(),
            "fuel": self.load_fuel_price_data(),
        }
        logger.info("All raw data sources loaded successfully.")
        return data
