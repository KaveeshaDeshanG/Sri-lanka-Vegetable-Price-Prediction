"""Historical price-lag features, grouped by vegetable and market."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class LagFeatureGenerator:
    """Adds price_lag_{n}d columns: the price exactly n calendar days earlier.

    Design decision -- calendar-day lag via a self-join on date, not
    pandas' positional .shift(n): the synthetic fixtures (and, per HARTI's
    publishing pattern, likely the real data too) are not sampled on a
    strict daily cadence -- tests/fixtures/market_sample.csv has rows a
    week apart. .shift(1) on such data would silently return "the previous
    *row*" (e.g. last week's price) while being labeled "lag 1 day", which
    is wrong and not caught by anything downstream. Joining on
    (group columns..., date - n days) instead means a lag feature is only
    populated when a row for that exact calendar date actually exists;
    otherwise it is NaN rather than a mislabeled value from the wrong day.

    Grouping is by vegetable + market, as specified. Note one open
    question this raises once real data is loaded: HARTI and Dambulla DEC
    are two independent reporting sources that can both report a price for
    the same (vegetable, market, date) -- src/preprocessing/cleaner.py
    deliberately keeps both rows rather than collapsing them. Grouping only
    by (vegetable, market) means a lag feature could be computed from the
    *other* source's series rather than the same source's history. This
    isn't fixed here (the task specifies grouping by vegetable/market, and
    there is no real multi-source data yet to know how much this matters in
    practice) -- flagging it rather than silently deciding either way.
    Revisit by grouping on (vegetable, market, source) once real data shows
    how much overlap actually exists.
    """

    def __init__(
        self,
        date_column: str = "date",
        price_column: str = "price",
        group_columns: list[str] | tuple[str, ...] = ("vegetable", "market"),
        lag_periods_days: list[int] | tuple[int, ...] = (1, 7, 14, 30),
    ) -> None:
        self._date_column = date_column
        self._price_column = price_column
        self._group_columns = list(group_columns)
        self._lag_periods_days = list(lag_periods_days)

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with one price_lag_{n}d column per configured lag period."""
        df = df.copy()
        present_group_columns = [c for c in self._group_columns if c in df.columns]
        if len(present_group_columns) < len(self._group_columns):
            missing = set(self._group_columns) - set(present_group_columns)
            logger.warning(
                "LagFeatureGenerator: grouping column(s) %s not found; "
                "lag features will be computed with grouping by %s only",
                sorted(missing),
                present_group_columns or "no grouping (entire dataset as one series)",
            )

        for n_days in self._lag_periods_days:
            df = self._add_single_lag(df, n_days, present_group_columns)
        return df

    def _add_single_lag(
        self, df: pd.DataFrame, n_days: int, group_columns: list[str]
    ) -> pd.DataFrame:
        feature_column = f"{self._price_column}_lag_{n_days}d"
        key_columns = group_columns + [self._date_column]

        # Deduplicate the lookup side on (group columns, date) *before* the
        # join. df itself can legitimately have two rows for the same
        # (vegetable, market, date) -- e.g. HARTI and Dambulla DEC both
        # reporting that day, deliberately kept by cleaner.py. Without this
        # groupby, the join below would match a df row against every
        # duplicate on the lookup side and silently fan out (multiply) the
        # row count. Averaging duplicates for the lookup value only -- not
        # for df's own rows, which are left untouched -- mirrors the same
        # choice merger.py already makes for duplicate weather dates.
        lookup = (
            df[key_columns + [self._price_column]]
            .groupby(key_columns, as_index=False)[self._price_column]
            .mean()
        )
        lookup[self._date_column] = lookup[self._date_column] + pd.Timedelta(days=n_days)
        lookup = lookup.rename(columns={self._price_column: feature_column})

        merged = df.merge(lookup, on=key_columns, how="left")
        populated = int(merged[feature_column].notna().sum())
        logger.info(
            "Added '%s': populated for %d/%d row(s) (the rest have no exact "
            "match %d day(s) earlier within their group)",
            feature_column,
            populated,
            len(merged),
            n_days,
        )
        return merged
