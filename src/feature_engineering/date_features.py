"""Calendar features derived from the date column."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DateFeatureGenerator:
    """Adds calendar-derived columns from an existing date column.

    Limited deliberately to features with a plausible link to vegetable
    pricing: day-of-week and day-of-month (market/harvest-day effects),
    month and quarter (seasonal effects), and week-of-year (finer-grained
    seasonality than month, e.g. Avurudu-week price spikes). No Sri
    Lanka-specific season boundaries (e.g. Yala/Maha cultivation seasons or
    monsoon months) are hardcoded here -- that domain knowledge isn't
    present in config/config.yaml, and guessing at exact boundaries would
    be fabricating a business rule rather than reading one.
    """

    def __init__(self, date_column: str = "date") -> None:
        self._date_column = date_column

    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with day/week/month/quarter columns added.

        Requires self._date_column to be present; coerces it to datetime if
        it isn't already (defensive -- transformer.py normally does this
        first, but this module shouldn't assume that already happened).
        """
        df = df.copy()
        if self._date_column not in df.columns:
            logger.warning(
                "DateFeatureGenerator: date column '%s' not found; skipping date features",
                self._date_column,
            )
            return df

        date_series = df[self._date_column]
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            date_series = pd.to_datetime(date_series)

        df["day_of_week"] = date_series.dt.dayofweek
        df["day_of_month"] = date_series.dt.day
        df["week_of_year"] = date_series.dt.isocalendar().week.astype("int64")
        df["month"] = date_series.dt.month
        df["quarter"] = date_series.dt.quarter

        logger.info(
            "Added date features (day_of_week, day_of_month, week_of_year, month, quarter) "
            "for %d row(s)",
            len(df),
        )
        return df
