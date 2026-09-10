"""Fuel-price lag features: diesel/petrol price from fuel_price_lag_days before each row's date."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FuelPriceFeatureGenerator:
    """Adds {fuel_price_column}_lag_{n}d columns: that fuel price n days before the row's date.

    Same rationale as WeatherFeatureGenerator: the same-day diesel_price/
    petrol_price columns already present after merging are the actual
    observed price for that date -- a fine training-time feature by itself,
    but including it same-day in X would be leakage-adjacent in the same
    sense weather is: at genuine prediction time for a future date, the
    pipeline can only ever have fuel-price history up to "now," never the
    target date's own value. Lagging by fuel_price_lag_days (default 1,
    since fuel price is a slow-moving step function and next-to-nothing is
    lost by looking back one day) guarantees the value used is always
    already part of the known historical record by the time a prediction
    is made. See FeatureEngineer.split_features_target(), which excludes the
    same-day fuel columns from the model-facing feature set in favor of
    these lagged ones.

    Unlike LagFeatureGenerator/RollingFeatureGenerator (price, grouped by
    vegetable/market), this is not grouped: fuel price is one national
    series per date (src/data_collection/fuel_price_loader.py has no market
    or vegetable dimension) -- same design, and same date-only lookup key,
    as WeatherFeatureGenerator.
    """

    def __init__(
        self,
        date_column: str = "date",
        fuel_price_columns: list[str] | tuple[str, ...] = ("diesel_price", "petrol_price"),
        fuel_price_lag_days: int = 1,
    ) -> None:
        self._date_column = date_column
        self._fuel_price_columns = list(fuel_price_columns)
        self._fuel_price_lag_days = fuel_price_lag_days

    def add_fuel_price_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with one {fuel_price_column}_lag_{n}d column per available column.

        Columns configured in data_schema.fuel_price_columns but absent from
        df are skipped (logged), not fabricated -- e.g. fuel data isn't
        configured at all (see DataLoader.load_fuel_price_data()), so
        nothing was merged in.
        """
        df = df.copy()
        present_columns = [c for c in self._fuel_price_columns if c in df.columns]
        if not present_columns:
            logger.warning(
                "FuelPriceFeatureGenerator: none of the configured fuel price column(s) %s "
                "are present; skipping fuel-price lag features entirely",
                self._fuel_price_columns,
            )
            return df

        missing = set(self._fuel_price_columns) - set(present_columns)
        if missing:
            logger.warning(
                "FuelPriceFeatureGenerator: fuel price column(s) %s not present; "
                "building lag features only for %s",
                sorted(missing),
                present_columns,
            )

        if self._date_column not in df.columns:
            logger.warning(
                "FuelPriceFeatureGenerator: date column '%s' not found; skipping fuel-price lag "
                "features",
                self._date_column,
            )
            return df

        # Many market/vegetable rows share the same date and (post-merge)
        # the same fuel price for that date. Deduplicating to one row per
        # date before the join -- same approach weather_features.py uses --
        # keeps this a clean many-to-one merge instead of a fan-out.
        lookup = (
            df[[self._date_column, *present_columns]]
            .groupby(self._date_column, as_index=False)[present_columns]
            .mean()
        )
        lookup[self._date_column] = lookup[self._date_column] + pd.Timedelta(
            days=self._fuel_price_lag_days
        )
        rename_map = {c: f"{c}_lag_{self._fuel_price_lag_days}d" for c in present_columns}
        lookup = lookup.rename(columns=rename_map)

        merged = df.merge(lookup, on=self._date_column, how="left")
        for original_column, lagged_column in rename_map.items():
            populated = int(merged[lagged_column].notna().sum())
            logger.info(
                "Added '%s': populated for %d/%d row(s)",
                lagged_column,
                populated,
                len(merged),
            )
        return merged
