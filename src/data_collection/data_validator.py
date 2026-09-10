"""Schema, type, and date-format validation shared by market and weather loaders."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when a DataFrame fails schema, dtype, date-format, or quality validation."""


def normalize_column_names(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Strip whitespace and lowercase column headers before validation.

    Design decision: this touches column *names* only, so that harmless
    formatting differences across raw sources (" Date ", "DATE", "Price ")
    don't fail schema validation just because a source wasn't hand-typed
    consistently. It deliberately does NOT touch cell *values* -- currency
    symbols, thousands separators, or inconsistent text casing in the data
    itself are a Phase 3 (preprocessing) concern. Phase 2's job is to confirm
    the file has the right shape; fixing its contents happens later.
    """
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    columns = list(normalized.columns)
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        message = (
            f"{source_name}: column names collide after normalization "
            f"(strip + lowercase): {duplicates}"
        )
        logger.error(message)
        raise DataValidationError(message)
    return normalized


class DataValidator:
    """Validates DataFrames against an expected schema.

    Checks are separate methods so a caller can run a subset, but `validate()`
    runs all three in the order that produces the clearest error first:
    missing columns before type checks, type checks before date checks.
    """

    def validate_required_columns(
        self, df: pd.DataFrame, required_columns: list[str], source_name: str
    ) -> None:
        """Raise DataValidationError if any required column is missing from df."""
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            message = f"{source_name}: missing required column(s): {missing}"
            logger.error(message)
            raise DataValidationError(message)
        logger.info("%s: all required columns present %s", source_name, required_columns)

    def validate_column_types(
        self, df: pd.DataFrame, expected_types: dict[str, str], source_name: str
    ) -> None:
        """Raise DataValidationError if a column's values don't match its expected type.

        expected_types maps a column name to "numeric" or "string". String columns
        are not enforced further since any CSV cell is already a valid string.
        """
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                continue  # already reported by validate_required_columns

            if expected_type == "numeric":
                coerced = pd.to_numeric(df[column], errors="coerce")
                invalid_mask = coerced.isna() & df[column].notna()
                if invalid_mask.any():
                    bad_values = df.loc[invalid_mask, column].unique().tolist()
                    message = (
                        f"{source_name}: column '{column}' expected numeric values, "
                        f"found non-numeric entries, e.g. {bad_values[:5]}"
                    )
                    logger.error(message)
                    raise DataValidationError(message)
            elif expected_type != "string":
                raise ValueError(
                    f"Unsupported expected_type '{expected_type}' for column '{column}'"
                )
        logger.info("%s: all column types valid %s", source_name, expected_types)

    def validate_date_format(
        self,
        df: pd.DataFrame,
        date_column: str,
        source_name: str,
        date_format: str = "%Y-%m-%d",
    ) -> None:
        """Raise DataValidationError if any non-null value in date_column fails to parse."""
        if date_column not in df.columns:
            return  # already reported by validate_required_columns

        parsed = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
        invalid_mask = parsed.isna() & df[date_column].notna()
        if invalid_mask.any():
            bad_values = df.loc[invalid_mask, date_column].unique().tolist()
            message = (
                f"{source_name}: column '{date_column}' has {int(invalid_mask.sum())} "
                f"value(s) not matching format '{date_format}', e.g. {bad_values[:5]}"
            )
            logger.error(message)
            raise DataValidationError(message)
        logger.info("%s: date column '%s' matches format '%s'", source_name, date_column, date_format)

    def validate(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
        expected_types: dict[str, str],
        date_column: str,
        source_name: str,
        date_format: str = "%Y-%m-%d",
    ) -> None:
        """Run the structural checks against df, raising on the first failure.

        This covers shape/type/format only. Business-rule checks (positive
        price, plausible temperature, non-negative rainfall, complete dates)
        are separate methods below -- callers opt into the ones relevant to
        their own schema rather than every check running unconditionally here.
        """
        self.validate_required_columns(df, required_columns, source_name)
        self.validate_column_types(df, expected_types, source_name)
        self.validate_date_format(df, date_column, source_name, date_format)

    def validate_date_not_missing(
        self, df: pd.DataFrame, date_column: str, source_name: str
    ) -> None:
        """Raise DataValidationError if date_column contains any null values."""
        if date_column not in df.columns:
            return  # already reported by validate_required_columns

        missing_count = int(df[date_column].isna().sum())
        if missing_count > 0:
            message = (
                f"{source_name}: date column '{date_column}' has "
                f"{missing_count} missing value(s)"
            )
            logger.error(message)
            raise DataValidationError(message)
        logger.info("%s: date column '%s' has no missing values", source_name, date_column)

    def validate_no_duplicate_dates(
        self, df: pd.DataFrame, date_column: str, source_name: str
    ) -> None:
        """Raise DataValidationError if date_column contains any duplicate (non-null) value.

        Appropriate for sources with exactly one row per date (e.g. weather,
        fuel price) -- unlike market data, where two rows legitimately share
        a date because they're different vegetable/market combinations, a
        duplicate date here is unambiguously a data error.
        """
        if date_column not in df.columns:
            return  # already reported by validate_required_columns

        non_null_dates = df[date_column].dropna()
        duplicate_count = int(non_null_dates.duplicated().sum())
        if duplicate_count > 0:
            bad_values = non_null_dates[non_null_dates.duplicated(keep=False)].unique().tolist()
            message = (
                f"{source_name}: date column '{date_column}' has {duplicate_count} "
                f"duplicate date value(s), e.g. {bad_values[:5]}"
            )
            logger.error(message)
            raise DataValidationError(message)
        logger.info("%s: date column '%s' has no duplicate values", source_name, date_column)

    def validate_positive(self, df: pd.DataFrame, column: str, source_name: str) -> None:
        """Raise DataValidationError if any non-null value in column is <= 0 (e.g. price)."""
        if column not in df.columns:
            return

        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = numeric.notna() & (numeric <= 0)
        if invalid_mask.any():
            bad_values = df.loc[invalid_mask, column].unique().tolist()
            message = (
                f"{source_name}: column '{column}' must be greater than zero, "
                f"found {bad_values[:5]}"
            )
            logger.error(message)
            raise DataValidationError(message)
        logger.info("%s: column '%s' is positive everywhere", source_name, column)

    def validate_non_negative(self, df: pd.DataFrame, column: str, source_name: str) -> None:
        """Raise DataValidationError if any non-null value in column is negative (e.g. rainfall)."""
        if column not in df.columns:
            return

        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = numeric.notna() & (numeric < 0)
        if invalid_mask.any():
            bad_values = df.loc[invalid_mask, column].unique().tolist()
            message = (
                f"{source_name}: column '{column}' cannot be negative, found {bad_values[:5]}"
            )
            logger.error(message)
            raise DataValidationError(message)
        logger.info("%s: column '%s' is non-negative everywhere", source_name, column)

    def validate_range(
        self,
        df: pd.DataFrame,
        column: str,
        source_name: str,
        min_value: float,
        max_value: float,
    ) -> None:
        """Raise DataValidationError if any non-null value in column falls outside [min_value, max_value].

        Used for plausibility bounds (e.g. temperature_c in Sri Lanka), not
        strict correctness -- the goal is to catch sentinel/typo values like
        -999 or 250, not to enforce a physically exact range.
        """
        if column not in df.columns:
            return

        numeric = pd.to_numeric(df[column], errors="coerce")
        out_of_range = numeric.notna() & ((numeric < min_value) | (numeric > max_value))
        if out_of_range.any():
            bad_values = df.loc[out_of_range, column].unique().tolist()
            message = (
                f"{source_name}: column '{column}' has {int(out_of_range.sum())} value(s) "
                f"outside expected range [{min_value}, {max_value}], e.g. {bad_values[:5]}"
            )
            logger.error(message)
            raise DataValidationError(message)
        logger.info(
            "%s: column '%s' is within range [%s, %s]", source_name, column, min_value, max_value
        )
