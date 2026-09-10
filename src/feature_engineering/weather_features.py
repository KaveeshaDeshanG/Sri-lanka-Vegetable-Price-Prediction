"""Weather-lag features: weather conditions from weather_lag_days before each row's date."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class WeatherFeatureGenerator:
    """Adds {weather_column}_lag{n}d columns: that weather reading n days before the row's date.

    config.yaml documents why weather is lagged rather than used same-day:
    "weather affects harvested price weeks later, not same-day". There is a
    second, independent reason to prefer the lagged version as the actual
    model input: the same-day weather columns already present after Phase 6's
    merge are the *actual observed* reading for that date. That's a fine
    training-time feature (the historical value is genuinely known), but it
    quietly assumes something that won't hold at real prediction time --
    predicting a near-future date's price would need that future date's
    weather, which isn't observed yet (only a forecast would be available,
    and this pipeline has no forecast source). A weather value from
    weather_lag_days ago is always already-observed by the time a prediction
    is made, so it is deployment-realistic in a way the same-day value is
    not. See FeatureEngineer.split_features_target(), which is where the
    same-day weather columns are actually excluded from the model-facing
    feature set in favor of these lagged ones.

    Grouping is configurable via `group_columns` (empty by default), mirroring
    LagFeatureGenerator/RollingFeatureGenerator's own `group_columns` parameter
    -- but unlike those, weather has no grouping key of its own; whether one is
    needed depends entirely on whether every row's same-day weather is actually
    identical. With no production-region correction applied, it always is (one
    national series per date, per src/data_collection/weather_loader.py), so
    the default (no grouping) reproduces this class's original date-only
    behavior exactly. Once src/feature_engineering/production_region_weather.py's
    apply_production_region_weather() has run, different vegetables can carry
    genuinely different same-day readings for the same date (e.g. Leeks ->
    Nuwara Eliya, Beans -> Badulla) -- FeatureEngineer passes
    group_columns=[vegetable_column] in that case so each vegetable's lag
    feature is computed from its OWN weather history, not a same-date average
    blended across every other vegetable's (possibly different) region.
    """

    def __init__(
        self,
        date_column: str = "date",
        weather_columns: list[str] | tuple[str, ...] = ("rainfall_mm", "temperature_c", "humidity_pct"),
        weather_lag_days: int = 14,
        group_columns: list[str] | tuple[str, ...] = (),
    ) -> None:
        self._date_column = date_column
        self._weather_columns = list(weather_columns)
        self._weather_lag_days = weather_lag_days
        self._group_columns = list(group_columns)

    def add_weather_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with one {weather_column}_lag{n}d column per available weather column.

        Columns configured in data_schema.weather_columns but absent from df
        are skipped (logged), not fabricated -- e.g. if weather wasn't
        merged in, or a source only provides a subset of readings.
        """
        df = df.copy()
        present_weather_columns = [c for c in self._weather_columns if c in df.columns]
        if not present_weather_columns:
            logger.warning(
                "WeatherFeatureGenerator: none of the configured weather column(s) %s "
                "are present; skipping weather-lag features entirely",
                self._weather_columns,
            )
            return df

        missing = set(self._weather_columns) - set(present_weather_columns)
        if missing:
            logger.warning(
                "WeatherFeatureGenerator: weather column(s) %s not present; "
                "building lag features only for %s",
                sorted(missing),
                present_weather_columns,
            )

        if self._date_column not in df.columns:
            logger.warning(
                "WeatherFeatureGenerator: date column '%s' not found; skipping weather-lag features",
                self._date_column,
            )
            return df

        present_group_columns = [c for c in self._group_columns if c in df.columns]
        if len(present_group_columns) < len(self._group_columns):
            missing_group_columns = sorted(set(self._group_columns) - set(present_group_columns))
            logger.warning(
                "WeatherFeatureGenerator: grouping column(s) %s not found; weather-lag features "
                "will be computed with grouping by %s",
                missing_group_columns,
                present_group_columns or "date only (national/regional series shared by every row)",
            )
        key_columns = present_group_columns + [self._date_column]

        # Many rows can share the same (group, date) key and (post-merge) the
        # same weather reading for it. Deduplicating to one row per key before
        # the join, the same way lag_features.py dedupes before its join,
        # keeps this a clean many-to-one merge instead of a fan-out; averaging
        # is a no-op when the readings already agree, which they always
        # should for genuinely duplicate weather rows sharing one key -- e.g.
        # two markets for the same vegetable and date, both already carrying
        # that vegetable's own (possibly production-region-corrected) reading.
        lookup = (
            df[key_columns + present_weather_columns]
            .groupby(key_columns, as_index=False)[present_weather_columns]
            .mean()
        )
        lookup[self._date_column] = lookup[self._date_column] + pd.Timedelta(days=self._weather_lag_days)
        rename_map = {c: f"{c}_lag{self._weather_lag_days}d" for c in present_weather_columns}
        lookup = lookup.rename(columns=rename_map)

        merged = df.merge(lookup, on=key_columns, how="left")
        for original_column, lagged_column in rename_map.items():
            populated = int(merged[lagged_column].notna().sum())
            logger.info(
                "Added '%s': populated for %d/%d row(s)",
                lagged_column,
                populated,
                len(merged),
            )
        return merged
