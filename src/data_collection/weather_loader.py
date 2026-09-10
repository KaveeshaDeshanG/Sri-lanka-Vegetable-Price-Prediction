"""Loads and validates Sri Lankan weather data used as model features."""

import logging
from pathlib import Path

import pandas as pd

from .data_validator import DataValidationError, DataValidator, normalize_column_names

logger = logging.getLogger(__name__)

DEFAULT_WEATHER_COLUMNS = ["rainfall_mm", "temperature_c", "humidity_pct"]


class WeatherDataLoader:
    """Reads a weather CSV and validates it against the expected schema."""

    def __init__(
        self,
        date_column: str = "date",
        weather_columns: list[str] | None = None,
        date_format: str = "%Y-%m-%d",
        rainfall_column: str = "rainfall_mm",
        temperature_column: str = "temperature_c",
        temperature_min_c: float = 0.0,
        temperature_max_c: float = 45.0,
    ) -> None:
        self._date_column = date_column
        self._weather_columns = weather_columns or list(DEFAULT_WEATHER_COLUMNS)
        self._date_format = date_format
        self._rainfall_column = rainfall_column
        self._temperature_column = temperature_column
        self._temperature_min_c = temperature_min_c
        self._temperature_max_c = temperature_max_c
        self._required_columns = [date_column, *self._weather_columns]
        self._expected_types = {column: "numeric" for column in self._weather_columns}
        self._validator = DataValidator()

    def load(self, file_path: str) -> pd.DataFrame:
        """Read the weather CSV and validate it.

        Args:
            file_path: path to the weather CSV file.

        Returns:
            The validated weather DataFrame.

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
            message = f"Weather data file not found at '{path}'"
            logger.error(message)
            raise DataValidationError(message) from error
        except PermissionError as error:
            message = f"Weather: permission denied reading '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except UnicodeDecodeError as error:
            message = f"Weather: encoding error reading '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except pd.errors.EmptyDataError as error:
            message = f"Weather data file at '{path}' is empty or unreadable: {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except pd.errors.ParserError as error:
            message = f"Failed to parse weather CSV at '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error

        if df.empty:
            message = f"Weather data file at '{path}' contains no rows"
            logger.error(message)
            raise DataValidationError(message)

        df = normalize_column_names(df, "Weather")

        self._validator.validate(
            df=df,
            required_columns=self._required_columns,
            expected_types=self._expected_types,
            date_column=self._date_column,
            source_name="Weather",
            date_format=self._date_format,
        )
        self._validator.validate_date_not_missing(df, self._date_column, "Weather")
        self._validator.validate_non_negative(df, self._rainfall_column, "Weather")
        self._validator.validate_range(
            df,
            self._temperature_column,
            "Weather",
            self._temperature_min_c,
            self._temperature_max_c,
        )

        logger.info("Weather: loaded %d row(s) from '%s'", len(df), path)
        return df
