"""Loads and validates Sri Lankan vegetable market price data (HARTI, Dambulla)."""

import logging
from pathlib import Path

import pandas as pd

from .data_validator import DataValidationError, DataValidator, normalize_column_names

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """Reads market price CSVs and validates them against a common schema.

    HARTI is the required, primary source: its own daily bulletin already
    reports Dambulla (DEC) wholesale prices directly, under the shared
    schema (date, vegetable, market, price). A second source is supported
    but optional -- see load_all() -- for the case where a genuinely
    independent second price feed (e.g. Dambulla DEC's own records, not
    derived from HARTI) becomes available later. Both are treated as two
    instances of the same schema, tagged by a `source` column, rather than
    two different formats, since that is how they are used downstream once
    merged.
    """

    def __init__(
        self,
        date_column: str = "date",
        vegetable_column: str = "vegetable",
        market_column: str = "market",
        price_column: str = "price",
        date_format: str = "%Y-%m-%d",
    ) -> None:
        self._date_column = date_column
        self._price_column = price_column
        self._date_format = date_format
        self._required_columns = [date_column, vegetable_column, market_column, price_column]
        self._expected_types = {price_column: "numeric"}
        self._validator = DataValidator()

    def load(self, file_path: str, source_name: str) -> pd.DataFrame:
        """Read one market-price CSV, validate it, and tag rows with their source.

        Args:
            file_path: path to the CSV file.
            source_name: label used in log messages and in the returned "source" column.

        Returns:
            The validated DataFrame with an added "source" column.

        Raises:
            DataValidationError: if the file is missing, unreadable, empty,
                unparseable, or fails schema/quality checks. All read-time
                failures (missing file, permission denied, bad encoding,
                empty file, malformed CSV) are normalized to this one
                exception type so callers only need to catch one thing.
        """
        path = Path(file_path)
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as error:
            message = f"{source_name}: market price file not found at '{path}'"
            logger.error(message)
            raise DataValidationError(message) from error
        except PermissionError as error:
            message = f"{source_name}: permission denied reading '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except UnicodeDecodeError as error:
            message = f"{source_name}: encoding error reading '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except pd.errors.EmptyDataError as error:
            message = f"{source_name}: file at '{path}' is empty or unreadable: {error}"
            logger.error(message)
            raise DataValidationError(message) from error
        except pd.errors.ParserError as error:
            message = f"{source_name}: failed to parse CSV at '{path}': {error}"
            logger.error(message)
            raise DataValidationError(message) from error

        if df.empty:
            message = f"{source_name}: file at '{path}' contains no rows"
            logger.error(message)
            raise DataValidationError(message)

        df = normalize_column_names(df, source_name)

        self._validator.validate(
            df=df,
            required_columns=self._required_columns,
            expected_types=self._expected_types,
            date_column=self._date_column,
            source_name=source_name,
            date_format=self._date_format,
        )
        self._validator.validate_date_not_missing(df, self._date_column, source_name)
        self._validator.validate_positive(df, self._price_column, source_name)

        df = df.copy()
        df["source"] = source_name
        logger.info("%s: loaded %d row(s) from '%s'", source_name, len(df), path)
        return df

    def load_harti_prices(self, file_path: str) -> pd.DataFrame:
        """Load HARTI wholesale/retail price bulletin data."""
        return self.load(file_path, source_name="HARTI")

    def load_dambulla_prices(self, file_path: str) -> pd.DataFrame:
        """Load Dambulla Dedicated Economic Centre price data."""
        return self.load(file_path, source_name="Dambulla")

    def load_all(self, harti_path: str, dambulla_path: str | None = None) -> pd.DataFrame:
        """Load HARTI market price data, plus a second source if one is configured.

        HARTI is required. dambulla_path is optional and defaults to None --
        pass it only when a genuinely independent second price feed exists.
        Passing the same file as harti_path "to fill in" this parameter would
        tag identical rows with a fabricated source="Dambulla" label and
        duplicate every observation; leave dambulla_path as None instead.

        Raises:
            DataValidationError: if harti_path (or dambulla_path, when given)
                is missing, unreadable, or fails validation.
        """
        harti_df = self.load_harti_prices(harti_path)
        if dambulla_path is None:
            logger.info(
                "Market data: no second source configured; using HARTI only (%d row(s))",
                len(harti_df),
            )
            return harti_df

        dambulla_df = self.load_dambulla_prices(dambulla_path)
        combined = pd.concat([harti_df, dambulla_df], ignore_index=True)
        logger.info("Combined market data: %d total row(s) from 2 source(s)", len(combined))
        return combined
