"""Top-level orchestrator for feature engineering: adds features, then splits into a training set."""

import logging
from pathlib import Path

import pandas as pd

from config.config_loader import ConfigLoader

from .date_features import DateFeatureGenerator
from .fuel_price_features import FuelPriceFeatureGenerator
from .lag_features import LagFeatureGenerator
from .rolling_features import RollingFeatureGenerator
from .weather_features import WeatherFeatureGenerator

logger = logging.getLogger(__name__)


class FeatureEngineeringError(Exception):
    """Raised when the input DataFrame is missing a column feature engineering cannot proceed without."""


class FeatureEngineer:
    """Turns Phase 6's merged market+weather DataFrame into a model-ready training dataset.

    Two-tier validation, deliberately mirroring src/data_collection's split
    between strict-at-the-boundary and lenient-internally: build_features()
    itself validates that date/vegetable/market/price are present and raises
    FeatureEngineeringError if not, since nothing here is possible without
    them. The individual generators it calls stay lenient about their own
    narrower concerns (e.g. a missing weather column is skipped, not fatal),
    matching how cleaner.py/transformer.py already treat optional columns.

    Two-stage output, matching the "Feature Engineering -> Training Dataset"
    split in the project's data-flow diagram:
      - build_features(df): adds every engineered column, drops no rows,
        fills no NaN. A row with NaN in e.g. price_lag_30d simply doesn't
        have 30 days of prior history yet -- that is real information, not
        a defect to paper over.
      - split_features_target(df): takes build_features()'s output and
        produces the (X, y) a model would actually train on -- excluding
        the target, identifier columns, and the raw same-day weather
        columns (see weather_features.py for why), and optionally dropping
        rows whose engineered features are still incomplete.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config = ConfigLoader(config_path)

        self._date_column = self._config.get("data_schema.date_column", "date")
        self._vegetable_column = self._config.get("data_schema.vegetable_column", "vegetable")
        self._market_column = self._config.get("data_schema.market_column", "market")
        self._price_column = self._config.get("data_schema.price_column", "price")
        self._weather_columns = self._config.get(
            "data_schema.weather_columns", ["rainfall_mm", "temperature_c", "humidity_pct"]
        )
        self._fuel_price_columns = self._config.get(
            "data_schema.fuel_price_columns", ["diesel_price", "petrol_price"]
        )
        self._target_column = self._config.get("model.target_column", self._price_column)

        self._group_columns = [self._vegetable_column, self._market_column]
        self._lag_periods_days = self._config.get("feature_engineering.lag_periods_days", [1, 7, 14, 30])
        self._rolling_windows_days = self._config.get(
            "feature_engineering.rolling_windows_days", [7, 14, 30]
        )
        self._weather_lag_days = self._config.get("feature_engineering.weather_lag_days", 14)
        self._fuel_price_lag_days = self._config.get("feature_engineering.fuel_price_lag_days", 1)

        self._date_features = DateFeatureGenerator(date_column=self._date_column)
        self._lag_features = LagFeatureGenerator(
            date_column=self._date_column,
            price_column=self._price_column,
            group_columns=self._group_columns,
            lag_periods_days=self._lag_periods_days,
        )
        self._rolling_features = RollingFeatureGenerator(
            date_column=self._date_column,
            price_column=self._price_column,
            group_columns=self._group_columns,
            rolling_windows_days=self._rolling_windows_days,
        )
        self._weather_features = WeatherFeatureGenerator(
            date_column=self._date_column,
            weather_columns=self._weather_columns,
            weather_lag_days=self._weather_lag_days,
            # Grouped by vegetable (not also market: production-region weather is
            # resolved per-vegetable only, see production_region_weather.py) so a
            # vegetable's weather-lag features come from its OWN weather history --
            # e.g. after apply_production_region_weather() has corrected same-day
            # readings, Leeks' lag features are computed from Nuwara Eliya's series,
            # never blended with Beans' Badulla reading just because they share a
            # calendar date. With no production-region correction applied (weather
            # genuinely identical for every vegetable on a date), this grouping is a
            # no-op and reproduces the original date-only result exactly.
            group_columns=[self._vegetable_column],
        )
        self._fuel_price_features = FuelPriceFeatureGenerator(
            date_column=self._date_column,
            fuel_price_columns=self._fuel_price_columns,
            fuel_price_lag_days=self._fuel_price_lag_days,
        )

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add date, lag, rolling, and weather-lag features to a merged market+weather DataFrame.

        Raises:
            FeatureEngineeringError: if date/vegetable/market/price is missing.
        """
        self._validate_required_columns(df)
        logger.info("Feature engineering: building features for %d row(s)...", len(df))
        df = self._date_features.add_date_features(df)
        df = self._lag_features.add_lag_features(df)
        df = self._rolling_features.add_rolling_features(df)
        df = self._weather_features.add_weather_lag_features(df)
        df = self._fuel_price_features.add_fuel_price_lag_features(df)
        return df

    def split_features_target(
        self, df: pd.DataFrame, drop_incomplete_rows: bool = True, require_target: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split a build_features() output into (X, y) for model training.

        Excluded from X:
          - the target column itself (self._target_column) -- otherwise the
            model would be trivially predicting price from price.
          - the date column -- a raw datetime isn't a usable numeric
            feature; its predictive content is already captured by the
            day_of_week/month/... columns build_features() adds.
          - the "source" column, if present -- data-provenance (which
            bulletin a row came from), not an economically meaningful
            predictor.
          - the raw same-day weather columns -- superseded by the
            weather-lag columns; see weather_features.py's docstring for
            why the same-day reading isn't deployment-realistic as a
            feature.
          - the raw same-day fuel-price columns (diesel_price,
            petrol_price), if present -- superseded by the fuel-price-lag
            columns for the same leakage-prevention reason; see
            fuel_price_features.py's docstring.

          vegetable/market are deliberately kept in X here as raw strings,
          unencoded: this method runs on the full dataset before any
          train/test split exists, and one-hot encoding needs to be fit on
          a training split only (see categorical_features.py --
          CategoricalEncoder is intentionally a separate, later step that
          whatever builds the train/test split composes on top of this
          method's output, not something this method does itself).

        Args:
            drop_incomplete_rows: if True (default), drop rows where any
                engineered lag/rolling/weather-lag feature is still NaN
                (insufficient history) or the target is NaN. These rows
                cannot be used for supervised training; keeping them would
                mean either fabricating a value the row doesn't have, or
                training on a missing target, neither of which this method
                does. Set False to inspect the full, ungapped feature set
                instead (e.g. for EDA).
            require_target: if True (default), rows with a missing/NaN
                target are always dropped, even when drop_incomplete_rows is
                False -- a training example with no known target is never
                usable. Set False only at inference, for a row whose target
                is genuinely unknown because it's the row being predicted:
                the target column must still be *present* (build_features()
                already enforced that), but this method never reads that
                row's value -- the target column is excluded from the
                returned features regardless, and a caller doing inference
                discards the returned target/y for that row entirely.
        """
        target = df[self._target_column]

        excluded_columns = {self._target_column, self._date_column, "source"}
        excluded_columns.update(c for c in self._weather_columns if c in df.columns)
        excluded_columns.update(c for c in self._fuel_price_columns if c in df.columns)
        feature_columns = [c for c in df.columns if c not in excluded_columns]
        features = df[feature_columns].copy()

        if drop_incomplete_rows:
            engineered_columns = [c for c in self._engineered_column_names(df) if c in features.columns]
            complete_mask = features[engineered_columns].notna().all(axis=1)
            if require_target:
                complete_mask &= target.notna()
        elif require_target:
            complete_mask = target.notna()
        else:
            complete_mask = pd.Series(True, index=df.index)

        dropped = int((~complete_mask).sum())
        if dropped > 0:
            logger.info(
                "split_features_target: dropping %d/%d row(s) with incomplete feature(s) or "
                "missing target",
                dropped,
                len(df),
            )

        features = features.loc[complete_mask].reset_index(drop=True)
        target = target.loc[complete_mask].reset_index(drop=True)
        return features, target

    def save_feature_dataset(self, df: pd.DataFrame, output_path: str | None = None) -> str:
        """Save a feature-engineered DataFrame to CSV.

        Args:
            output_path: overrides config's paths.processed_files.feature_dataset.
        """
        resolved_path = Path(
            output_path or self._config.get_required("paths.processed_files.feature_dataset")
        )
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(resolved_path, index=False)
        logger.info(
            "Feature engineering: saved feature dataset to '%s' (%d rows)", resolved_path, len(df)
        )
        return str(resolved_path)

    def _engineered_column_names(self, df: pd.DataFrame) -> list[str]:
        """The exact engineered column names this instance's config produces.

        Reconstructed from config rather than returned by each generator,
        so the generators can stay simple pure-transform functions with no
        need to separately report back what they added.
        """
        names = [f"{self._price_column}_lag_{n}d" for n in self._lag_periods_days]
        names += [f"{self._price_column}_rolling_mean_{n}d" for n in self._rolling_windows_days]
        names += [f"{self._price_column}_rolling_std_{n}d" for n in self._rolling_windows_days]
        names += [
            f"{c}_lag{self._weather_lag_days}d" for c in self._weather_columns if c in df.columns
        ]
        names += [
            f"{c}_lag_{self._fuel_price_lag_days}d"
            for c in self._fuel_price_columns
            if c in df.columns
        ]
        return names

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required = [self._date_column, self._vegetable_column, self._market_column, self._price_column]
        missing = [column for column in required if column not in df.columns]
        if missing:
            message = f"Cannot build features: missing required column(s): {missing}"
            logger.error(message)
            raise FeatureEngineeringError(message)
