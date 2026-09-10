"""Dtype conversion, chronological ordering, and category-name standardization."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Illustrative alias sets only -- built from common naming variants seen in
# general Sri Lankan market reporting, NOT yet validated against real
# HARTI/Dambulla exports (no real raw files exist yet for this project; see
# the dataset-research phase). Expand this table once real data reveals the
# actual variants in use. Keys are lowercase+stripped for lookup; values are
# the canonical display name.
DEFAULT_VEGETABLE_ALIASES: dict[str, str] = {
    "eggplant": "Brinjal",
    "egg plant": "Brinjal",
    "aubergine": "Brinjal",
    "big onion": "Big Onion",
    "onion (big)": "Big Onion",
    "b onion": "Big Onion",
    "red onion": "Red Onion",
    "small onion": "Red Onion",
}

DEFAULT_MARKET_ALIASES: dict[str, str] = {
    "colombo": "Pettah",
    "colombo (pettah)": "Pettah",
    "manning market": "Pettah",
    "dambulla dec": "Dambulla",
    "dambulla (dec)": "Dambulla",
    "dambulla economic centre": "Dambulla",
    "dambulla dedicated economic centre": "Dambulla",
}


class DataTransformer:
    """Converts dtypes, sorts chronologically, and standardizes category names."""

    def __init__(
        self,
        date_column: str = "date",
        date_format: str = "%Y-%m-%d",
        vegetable_column: str = "vegetable",
        market_column: str = "market",
        price_column: str = "price",
        vegetable_aliases: dict[str, str] | None = None,
        market_aliases: dict[str, str] | None = None,
    ) -> None:
        self._date_column = date_column
        self._date_format = date_format
        self._vegetable_column = vegetable_column
        self._market_column = market_column
        self._price_column = price_column
        self._vegetable_aliases = vegetable_aliases or dict(DEFAULT_VEGETABLE_ALIASES)
        self._market_aliases = market_aliases or dict(DEFAULT_MARKET_ALIASES)

    def convert_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the date column from string to an actual datetime dtype."""
        df = df.copy()
        df[self._date_column] = pd.to_datetime(df[self._date_column], format=self._date_format)
        return df

    def sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort rows by date ascending. Requires convert_date_column() to have run first."""
        return df.sort_values(self._date_column).reset_index(drop=True)

    def convert_numeric_columns(self, df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
        """Ensure the given columns are float dtype."""
        df = df.copy()
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def standardize_vegetable_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map known vegetable name aliases to one canonical name; unmapped names pass through."""
        return self._apply_alias_map(df, self._vegetable_column, self._vegetable_aliases, "vegetable")

    def standardize_market_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map known market name aliases to one canonical name; unmapped names pass through."""
        return self._apply_alias_map(df, self._market_column, self._market_aliases, "market")

    def _apply_alias_map(
        self, df: pd.DataFrame, column: str, alias_map: dict[str, str], label: str
    ) -> pd.DataFrame:
        if column not in df.columns:
            return df

        df = df.copy()

        def resolve(value):
            if pd.isna(value):
                return value
            key = str(value).strip().lower()
            return alias_map.get(key, value)

        original = df[column]
        df[column] = original.apply(resolve)

        changed_mask = original.notna() & df[column].notna() & (original != df[column])
        changed = int(changed_mask.sum())
        if changed > 0:
            logger.info("Standardized %d %s name(s) via alias map", changed, label)
        return df

    def transform_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full transformation sequence for market price data."""
        df = self.convert_date_column(df)
        df = self.standardize_vegetable_names(df)
        df = self.standardize_market_names(df)
        df = self.convert_numeric_columns(df, [self._price_column])
        df = self.sort_chronologically(df)
        return df

    def transform_weather_data(self, df: pd.DataFrame, weather_columns: list[str]) -> pd.DataFrame:
        """Run the full transformation sequence for weather data."""
        df = self.convert_date_column(df)
        df = self.convert_numeric_columns(df, weather_columns)
        df = self.sort_chronologically(df)
        return df

    def transform_fuel_price_data(self, df: pd.DataFrame, fuel_price_columns: list[str]) -> pd.DataFrame:
        """Run the full transformation sequence for fuel-price data.

        Same three generic steps as transform_weather_data(): fuel price has
        no vegetable/market category columns to standardize, so there is
        nothing fuel-specific beyond which columns get numeric conversion.
        """
        df = self.convert_date_column(df)
        df = self.convert_numeric_columns(df, fuel_price_columns)
        df = self.sort_chronologically(df)
        return df
