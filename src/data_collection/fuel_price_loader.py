"""Loads and validates Sri Lankan fuel-price data used as an optional model feature."""

import logging
from pathlib import Path

import pandas as pd

from .data_validator import DataValidationError, DataValidator, normalize_column_names

logger = logging.getLogger(__name__)

DEFAULT_FUEL_PRICE_COLUMNS = ["diesel_price", "petrol_price"]


class FuelPriceDataLoader:
    """Reads a fuel-price CSV and validates it against the expected schema.

    Mirrors WeatherDataLoader's design: fuel price is one national series
    (not per vegetable/market), so this expects a single CSV with one row
    per date. config.yaml's data_schema.fuel_price_columns decides which
    columns are required -- diesel (Auto Diesel/LAD) is the economically
    primary variable for this problem (Dambulla DEC wholesale transport is
    diesel-truck-dominated); petrol is a weaker, optional secondary signal.
    A deployment that only has diesel data can configure
    fuel_price_columns: ["diesel_price"] and this loader requires nothing
    else.

    Unlike weather (currently required for DataLoader.load_all()), the raw
    file itself is optional at the DataLoader level -- see
    DataLoader.load_fuel_price_data(). This class only validates whatever
    file it's actually given; it has no opinion on whether fuel data is
    configured at all.
    """

    def __init__(
        self,
        date_column: str = "date",
        fuel_price_columns: list[str] | None = None,
        date_format: str = "%Y-%m-%d",
    ) -> None:
        self._date_column = date_column
        self._fuel_price_columns = list(fuel_price_columns) if fuel_price_columns else list(
            DEFAULT_FUEL_PRICE_COLUMNS
        )
        self._date_format = date_format
        self._required_columns = [date_column, *self._fuel_price_columns]
        self._expected_types = {column: "numeric" for column in self._fuel_price_columns}
        self._validator = DataValidator()

    def load(self, file_path: str) -> pd.DataFrame:
        """Read the fuel-price CSV and validate it.

        Args:
            file_path: path to the fuel-price CSV file.

        Returns:
            The validated fuel-price DataFrame.

        Raises:
            DataValidationError: if the file is missing, unreadable, empty,
                unparseable, or fails schema/quality checks. All read-time
                failures are normalized to this one exception type so callers
                only need to catch one thing.
        """
        path = Path(file_path)
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as error:
            message = f"Fuel price data file not found at '{path}'"
            logger.error(message)
            raise DataValidationError(message) from error
        except PermissionError as error:
            message = f"Fuel price: permission denied reading '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except UnicodeDecodeError as error:
            message = f"Fuel price: encoding error reading '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except pd.errors.EmptyDataError as error:
            message = f"Fuel price data file at '{path}' is empty or unreadable: {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except pd.errors.ParserError as error:
            message = f"Failed to parse fuel price CSV at '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error

        if df.empty:
            message = f"Fuel price data file at '{path}' contains no rows"
            logger.error(message)
            raise DataValidationError(message)

        df = normalize_column_names(df, "FuelPrice")

        self._validator.validate(
            df=df,
            required_columns=self._required_columns,
            expected_types=self._expected_types,
            date_column=self._date_column,
            source_name="FuelPrice",
            date_format=self._date_format,
        )
        self._validator.validate_date_not_missing(df, self._date_column, "FuelPrice")
        self._validator.validate_no_duplicate_dates(df, self._date_column, "FuelPrice")
        for column in self._fuel_price_columns:
            self._validator.validate_positive(df, column, "FuelPrice")

        logger.info("FuelPrice: loaded %d row(s) from '%s'", len(df), path)
        return df
