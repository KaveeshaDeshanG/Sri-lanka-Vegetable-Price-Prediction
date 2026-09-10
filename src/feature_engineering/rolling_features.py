"""Rolling-window price statistics, grouped by vegetable and market."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class RollingFeatureGenerator:
    """Adds price_rolling_mean_{n}d / price_rolling_std_{n}d columns per group.

    Leakage note (the reason this isn't a one-line pandas .rolling() call):
    a naive rolling window ending at the current row's own date would
    include the current row's own price in its mean/std -- i.e. the row's
    engineered feature would partly be built from the value it's trying to
    help predict. This uses pandas' time-based rolling with closed="left",
    which computes the window as [date - n_days, date) -- up to but
    excluding the current row's date -- so every rolling feature is built
    only from strictly earlier observations in the same group.

    Time-based (not positional/count-based) windows are used for the same
    reason LagFeatureGenerator avoids .shift(): the data isn't sampled on a
    strict daily cadence, so a count-based rolling(window=7) would mean "the
    last 7 rows" rather than "the last 7 days", and those are not the same
    thing when rows are a week apart.
    """

    def __init__(
        self,
        date_column: str = "date",
        price_column: str = "price",
        group_columns: list[str] | tuple[str, ...] = ("vegetable", "market"),
        rolling_windows_days: list[int] | tuple[int, ...] = (7, 14, 30),
    ) -> None:
        self._date_column = date_column
        self._price_column = price_column
        self._group_columns = list(group_columns)
        self._rolling_windows_days = list(rolling_windows_days)

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with rolling mean/std columns per configured window."""
        df = df.copy()
        present_group_columns = [c for c in self._group_columns if c in df.columns]
        if len(present_group_columns) < len(self._group_columns):
            missing = set(self._group_columns) - set(present_group_columns)
            logger.warning(
                "RollingFeatureGenerator: grouping column(s) %s not found; "
                "rolling features will be computed with grouping by %s only",
                sorted(missing),
                present_group_columns or "no grouping (entire dataset as one series)",
            )

        if df.empty:
            for window_days in self._rolling_windows_days:
                df[f"{self._price_column}_rolling_mean_{window_days}d"] = pd.Series(dtype="float64")
                df[f"{self._price_column}_rolling_std_{window_days}d"] = pd.Series(dtype="float64")
            return df

        for window_days in self._rolling_windows_days:
            df = self._add_single_window(df, window_days, present_group_columns)
        return df

    def _add_single_window(
        self, df: pd.DataFrame, window_days: int, group_columns: list[str]
    ) -> pd.DataFrame:
        mean_column = f"{self._price_column}_rolling_mean_{window_days}d"
        std_column = f"{self._price_column}_rolling_std_{window_days}d"
        window = f"{window_days}D"

        if group_columns:
            grouped = df.groupby(group_columns, group_keys=False, sort=False)[
                [self._date_column, self._price_column]
            ].apply(lambda group: self._rolling_for_group(group, window, mean_column, std_column))
            df = df.join(grouped[[mean_column, std_column]])
        else:
            df = self._rolling_for_group(df, window, mean_column, std_column)

        logger.info(
            "Added '%s' / '%s': %d/%d row(s) have a full trailing history before this window",
            mean_column,
            std_column,
            int(df[mean_column].notna().sum()),
            len(df),
        )
        return df

    def _rolling_for_group(
        self, group: pd.DataFrame, window: str, mean_column: str, std_column: str
    ) -> pd.DataFrame:
        group = group.sort_values(self._date_column)
        indexed = group.set_index(self._date_column)
        rolling = indexed[self._price_column].rolling(window, closed="left")
        group[mean_column] = rolling.mean().to_numpy()
        group[std_column] = rolling.std().to_numpy()
        return group
