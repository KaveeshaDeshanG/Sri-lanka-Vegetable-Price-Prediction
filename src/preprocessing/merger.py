"""Merges cleaned, transformed market price data with weather data on date."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataMerger:
    """Left-joins market price data with weather data using date as the key.

    See merge() for the full merge strategy and reasoning.
    """

    def __init__(self, date_column: str = "date") -> None:
        self._date_column = date_column

    def merge(self, market_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Left-join market_df with weather_df on date.

        Merge strategy:
        - Market data is the LEFT/base side, because price is the prediction
          target. Every price observation must survive the merge even if its
          matching weather record is missing -- Feature Engineering (Phase 4)
          is a better place to decide how to handle a missing weather value
          (impute, exclude, flag) than silently losing a target row here.
        - Missing weather for a market date: the market row is kept; its
          weather columns become NaN.
        - Missing market data for a weather date: naturally excluded by the
          left join -- there is no price to predict for that date, so a
          weather-only row has nothing to attach to.
        - Duplicate weather dates: resolved BEFORE the join (see
          _deduplicate_weather_dates) so the join can't silently fan out
          market rows by matching more than one weather row per date.
        """
        weather_df = self._deduplicate_by_date(weather_df, "Weather")
        weather_value_columns = [c for c in weather_df.columns if c != self._date_column]

        merged = market_df.merge(weather_df, on=self._date_column, how="left")

        if weather_value_columns:
            missing_weather_mask = merged[weather_value_columns].isna().any(axis=1)
            missing_count = int(missing_weather_mask.sum())
            if missing_count > 0:
                logger.warning(
                    "%d of %d market row(s) have no matching weather data for their date "
                    "(kept, with NaN weather values).",
                    missing_count,
                    len(merged),
                )

        market_dates = set(market_df[self._date_column])
        weather_dates = set(weather_df[self._date_column])
        weather_only_dates = weather_dates - market_dates
        if weather_only_dates:
            logger.info(
                "%d weather-only date(s) had no matching market row and were excluded "
                "by the left join.",
                len(weather_only_dates),
            )

        logger.info("Merged dataset: %d row(s), %d column(s)", len(merged), merged.shape[1])
        return merged

    def merge_fuel_prices(self, df: pd.DataFrame, fuel_df: pd.DataFrame | None) -> pd.DataFrame:
        """Left-join df (the market+weather merge output) with fuel_df on date.

        fuel_df is optional: None means fuel data isn't configured (see
        DataLoader.load_fuel_price_data()), in which case df is returned
        completely unchanged -- the rest of the pipeline must keep working
        exactly as it did before fuel support existed.

        Fuel price is one national series, not per vegetable/market -- same
        design as merge()'s weather handling -- so this is a plain
        date-keyed left join, not a manual per-vegetable duplication: a
        single date's fuel price naturally attaches to however many market
        rows already exist for that date, the same many-to-one pattern
        merge() already uses for weather.

        Missing fuel data for a market date: the market row is kept; its
        fuel columns become NaN (price is the prediction target, so no row
        is ever dropped here just because fuel is unavailable for its date).
        Duplicate fuel dates are resolved before the join the same way
        duplicate weather dates are, so the join can't fan out.
        """
        if fuel_df is None:
            return df

        fuel_df = self._deduplicate_by_date(fuel_df, "FuelPrice")
        fuel_value_columns = [c for c in fuel_df.columns if c != self._date_column]

        merged = df.merge(fuel_df, on=self._date_column, how="left")

        if fuel_value_columns:
            missing_fuel_mask = merged[fuel_value_columns].isna().any(axis=1)
            missing_count = int(missing_fuel_mask.sum())
            if missing_count > 0:
                logger.warning(
                    "%d of %d market row(s) have no matching fuel-price data for their date "
                    "(kept, with NaN fuel values).",
                    missing_count,
                    len(merged),
                )

        logger.info("Merged fuel-price data: %d row(s), %d column(s)", len(merged), merged.shape[1])
        return merged

    def _deduplicate_by_date(self, source_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Collapse duplicate rows for the same date by averaging.

        Used for single-series-per-date sources (weather, fuel price), where
        exactly one row per date is expected. A duplicate here is treated as
        a data-collection artifact -- unlike price disagreement across
        sources (deliberately preserved in cleaning), there is only one true
        reading/price for a given date, so averaging is a reasonable
        reconciliation rather than a loss of information.

        Note: groupby().mean(numeric_only=True) drops any non-numeric column
        other than the date key. Both weather's and fuel's current schemas
        (date + numeric readings only) have none, but this would need
        revisiting if a non-numeric column (e.g. a station name) is ever
        added to either.
        """
        duplicate_count = int(source_df[self._date_column].duplicated().sum())
        if duplicate_count > 0:
            logger.warning(
                "%s: %d duplicate date row(s) found; averaging numeric values.",
                source_name,
                duplicate_count,
            )
            source_df = source_df.groupby(self._date_column, as_index=False).mean(numeric_only=True)
        return source_df
