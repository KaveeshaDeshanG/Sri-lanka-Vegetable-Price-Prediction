"""Row-level cleaning for market price and weather data.

Design note: this module does NOT assume its input already passed
src.data_collection's strict validation. Phase 2 loaders raise and reject an
entire file if any row violates a rule (fail-fast at ingestion). This module
instead *removes* offending rows and keeps the rest, so it degrades
gracefully if it is ever run on data that did not go through that gate, and
it re-checks the same business rules defensively rather than trusting an
upstream invariant it cannot see.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans market price and weather DataFrames before transformation.

    Every cleaning decision is documented on the method that makes it --
    see clean_market_data() and clean_weather_data() for the full sequence
    and reasoning.
    """

    def __init__(
        self,
        date_column: str = "date",
        price_column: str = "price",
        vegetable_column: str = "vegetable",
        market_column: str = "market",
        rainfall_column: str = "rainfall_mm",
        temperature_column: str = "temperature_c",
        humidity_column: str = "humidity_pct",
        fuel_price_columns: list[str] | tuple[str, ...] = ("diesel_price", "petrol_price"),
        date_format: str = "%Y-%m-%d",
        min_valid_year: int = 2000,
        temperature_min_c: float = 0.0,
        temperature_max_c: float = 45.0,
        humidity_min_pct: float = 0.0,
        humidity_max_pct: float = 100.0,
    ) -> None:
        self._date_column = date_column
        self._price_column = price_column
        self._vegetable_column = vegetable_column
        self._market_column = market_column
        self._rainfall_column = rainfall_column
        self._temperature_column = temperature_column
        self._humidity_column = humidity_column
        self._fuel_price_columns = list(fuel_price_columns)
        self._date_format = date_format
        self._min_valid_year = min_valid_year
        self._temperature_min_c = temperature_min_c
        self._temperature_max_c = temperature_max_c
        self._humidity_min_pct = humidity_min_pct
        self._humidity_max_pct = humidity_max_pct

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def detect_missing_values(self, df: pd.DataFrame, source_name: str) -> pd.Series:
        """Log and return per-column missing-value counts. Does not modify df."""
        missing_counts = df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            logger.info("%s: missing values detected -> %s", source_name, missing_counts.to_dict())
        else:
            logger.info("%s: no missing values detected", source_name)
        return missing_counts

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def clean_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean market price data. Order of operations and why:

        1. Detect missing values (diagnostic only, logged, nothing changed yet).
        2. Drop EXACT duplicate rows only -- rows that share date/vegetable/
           market but differ by `source` (or price) are deliberately left
           alone: they represent two sources' genuinely different reports
           for the same day, and collapsing/averaging them would destroy
           diagnostically useful disagreement between sources.
        3. Drop rows missing any critical field (date, vegetable, market,
           price) -- none of these can be sensibly imputed: a missing
           vegetable/market/date makes the row's identity ambiguous, and
           price is the prediction target, so guessing it would contaminate
           training data.
        4. Remove rows with a non-positive/non-numeric price (a defensive
           re-check -- see module docstring for why this module doesn't
           trust that upstream validation already guarantees it).
        5. Clean text columns: whitespace/casing hygiene only. Semantic
           alias resolution (e.g. "Eggplant" meaning the same thing as
           "Brinjal") is transformer.py's job, not this module's.
        6. Date consistency check: drop unparseable or implausible dates.
           Format correctness was already checked at ingestion (Phase 2);
           this adds a plausibility bound (no year before min_valid_year,
           none in the future) that Phase 2 does not check.
        """
        df = df.copy()
        self.detect_missing_values(df, "Market")
        df = self._drop_exact_duplicates(df, "Market")
        df = self._drop_missing_critical_fields(
            df,
            [self._date_column, self._vegetable_column, self._market_column, self._price_column],
            "Market",
        )
        df = self._remove_invalid_prices(df)
        df = self.clean_text_columns(df, [self._vegetable_column, self._market_column])
        df = self._check_date_consistency(df, "Market")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Weather data
    # ------------------------------------------------------------------

    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean weather data. Order of operations and why:

        1. Detect missing values (diagnostic).
        2. Drop exact duplicate rows.
        3. Drop rows with a missing date -- a missing date can't be guessed,
           unlike the numeric weather readings, so there is no gap to fill.
        4. Interpolate missing numeric weather readings (rainfall/
           temperature/humidity) linearly in chronological order. Weather is
           a continuous daily series, so a short gap is safely filled from
           its neighbors, unlike a missing categorical/target field. Any
           value interpolation can't fill (a gap at the very start or end of
           the series, with no neighbor on one side) is dropped, logged.
        5. Remove rows with physically implausible weather values (negative
           rainfall, temperature/humidity outside a realistic bound) -- a
           defensive re-check, same rationale as market prices.
        6. Date consistency check, same as market data.
        """
        df = df.copy()
        self.detect_missing_values(df, "Weather")
        df = self._drop_exact_duplicates(df, "Weather")
        df = self._drop_missing_critical_fields(df, [self._date_column], "Weather")
        df = self._interpolate_missing_weather_values(df)
        df = self._remove_invalid_weather_values(df)
        df = self._check_date_consistency(df, "Weather")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Fuel price data
    # ------------------------------------------------------------------

    def clean_fuel_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean fuel-price data. Order of operations and why:

        1. Detect missing values (diagnostic).
        2. Drop exact duplicate rows.
        3. Drop rows with a missing date -- a missing date can't be guessed.
        4. Forward-fill missing numeric fuel price readings using the last
           known officially effective price (see
           _forward_fill_missing_fuel_price_values). NOT linear
           interpolation like clean_weather_data() uses: fuel price is a
           step-valued administrative price that is genuinely constant
           between CEYPETCO revision events, so there is no true
           in-between value to interpolate toward -- carrying the last
           known price forward is the correct representation, not an
           approximation. Never backward-filled, so a later revision can
           never leak into a date before it took effect.
        5. Remove rows with a non-positive fuel price -- a defensive
           re-check, same rationale as market prices.
        6. Date consistency check, same as market/weather data.
        """
        df = df.copy()
        self.detect_missing_values(df, "FuelPrice")
        df = self._drop_exact_duplicates(df, "FuelPrice")
        df = self._drop_missing_critical_fields(df, [self._date_column], "FuelPrice")
        df = self._forward_fill_missing_fuel_price_values(df)
        df = self._remove_invalid_fuel_price_values(df)
        df = self._check_date_consistency(df, "FuelPrice")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _drop_exact_duplicates(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(keep="first")
        removed = before - len(df)
        if removed > 0:
            logger.info("%s: removed %d exact duplicate row(s)", source_name, removed)
        return df

    def _drop_missing_critical_fields(
        self, df: pd.DataFrame, columns: list[str], source_name: str
    ) -> pd.DataFrame:
        present_columns = [column for column in columns if column in df.columns]
        before = len(df)
        df = df.dropna(subset=present_columns)
        removed = before - len(df)
        if removed > 0:
            logger.info(
                "%s: dropped %d row(s) missing a critical field %s",
                source_name,
                removed,
                present_columns,
            )
        return df

    def _remove_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._price_column not in df.columns:
            return df
        numeric_price = pd.to_numeric(df[self._price_column], errors="coerce")
        valid_mask = numeric_price.notna() & (numeric_price > 0)
        removed = int((~valid_mask).sum())
        if removed > 0:
            logger.info("Market: removed %d row(s) with a non-positive or invalid price", removed)
        return df.loc[valid_mask]

    def _remove_invalid_weather_values(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_mask = pd.Series(True, index=df.index)

        if self._rainfall_column in df.columns:
            rainfall = pd.to_numeric(df[self._rainfall_column], errors="coerce")
            valid_mask &= rainfall.notna() & (rainfall >= 0)

        if self._temperature_column in df.columns:
            temperature = pd.to_numeric(df[self._temperature_column], errors="coerce")
            valid_mask &= temperature.notna() & temperature.between(
                self._temperature_min_c, self._temperature_max_c
            )

        if self._humidity_column in df.columns:
            humidity = pd.to_numeric(df[self._humidity_column], errors="coerce")
            valid_mask &= humidity.notna() & humidity.between(
                self._humidity_min_pct, self._humidity_max_pct
            )

        removed = int((~valid_mask).sum())
        if removed > 0:
            logger.info("Weather: removed %d row(s) with implausible weather value(s)", removed)
        return df.loc[valid_mask]

    def _remove_invalid_fuel_price_values(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_mask = pd.Series(True, index=df.index)

        for column in self._fuel_price_columns:
            if column in df.columns:
                numeric = pd.to_numeric(df[column], errors="coerce")
                valid_mask &= numeric.notna() & (numeric > 0)

        removed = int((~valid_mask).sum())
        if removed > 0:
            logger.info(
                "FuelPrice: removed %d row(s) with a non-positive or invalid fuel price", removed
            )
        return df.loc[valid_mask]

    def clean_text_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Strip whitespace, collapse internal spaces, and title-case text columns.

        Character-level hygiene only. Mapping known aliases to one canonical
        name (e.g. "Eggplant" -> "Brinjal") is transformer.py's job.

        Public (not just an internal step of clean_market_data()): inference
        (src/prediction) reuses this directly to normalize new prediction
        input the same way training data was normalized, WITHOUT the
        row-dropping steps around it in clean_market_data() that would
        delete the very row being predicted (its price is unknown by
        definition).
        """
        df = df.copy()
        for column in columns:
            if column not in df.columns:
                continue
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.title()
            )
        return df

    def _check_date_consistency(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Drop rows whose date is unparseable or falls outside a plausible range.

        Format correctness was already enforced at ingestion (Phase 2); this
        adds a plausibility check Phase 2 doesn't do (no date before
        min_valid_year, none in the future). Uses a temporary parsed Series
        only for filtering -- the returned date column is left as-is;
        converting its dtype is transformer.py's responsibility.
        """
        if self._date_column not in df.columns:
            return df

        parsed = pd.to_datetime(df[self._date_column], format=self._date_format, errors="coerce")
        today = pd.Timestamp.now().normalize()
        min_date = pd.Timestamp(year=self._min_valid_year, month=1, day=1)

        valid_mask = parsed.notna() & (parsed >= min_date) & (parsed <= today)
        removed = int((~valid_mask).sum())
        if removed > 0:
            logger.info(
                "%s: dropped %d row(s) with an unparseable or implausible date "
                "(outside %s to %s)",
                source_name,
                removed,
                min_date.date(),
                today.date(),
            )
        return df.loc[valid_mask]

    def _interpolate_missing_weather_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Linearly interpolate missing numeric weather readings.

        Interpolation is only meaningful in chronological order, so this
        sorts by a temporary parsed-date key first -- transformer.py's
        official dtype conversion and sort happen later in the pipeline, so
        this module cannot rely on the DataFrame already being ordered.
        Values interpolation can't fill (a gap at the very start/end of the
        series, with no neighbor on one side) are dropped afterward.
        """
        numeric_columns = [
            column
            for column in (self._rainfall_column, self._temperature_column, self._humidity_column)
            if column in df.columns
        ]
        if not numeric_columns or self._date_column not in df.columns:
            return df

        sort_key = pd.to_datetime(df[self._date_column], format=self._date_format, errors="coerce")
        df = df.assign(_sort_key=sort_key).sort_values("_sort_key")

        missing_before = int(df[numeric_columns].isna().sum().sum())
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        df[numeric_columns] = df[numeric_columns].interpolate(method="linear")

        still_missing = df[numeric_columns].isna().any(axis=1)
        dropped = int(still_missing.sum())
        if missing_before > 0:
            logger.info(
                "Weather: interpolated missing numeric value(s); %d row(s) could not be "
                "filled (gap at series edge) and were dropped",
                dropped,
            )

        return df.loc[~still_missing].drop(columns="_sort_key")

    def _forward_fill_missing_fuel_price_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing fuel price readings using the last officially effective price.

        Deliberately different from _interpolate_missing_weather_values():
        weather is a continuously varying physical quantity, so linear
        interpolation between two known readings is a reasonable estimate of
        an unmeasured in-between value. Fuel price is a step-valued
        administrative price that is genuinely CONSTANT between CEYPETCO
        revision events -- there is no true intermediate value, so
        interpolating would fabricate a gradual change that never happened.
        Forward-fill instead carries the last known official price forward,
        which is not an estimate, it's the actual value that was in effect.

        Never backward-filled: filling a leading gap from a later, newer
        price would mean a future revision governs a date before it took
        effect, which is exactly the kind of forward-looking leak this
        project's feature generators are designed to avoid elsewhere. A gap
        at the very start of the series (no earlier known price to carry
        forward) is therefore left unfillable and dropped, logged -- never
        fabricated.
        """
        numeric_columns = [column for column in self._fuel_price_columns if column in df.columns]
        if not numeric_columns or self._date_column not in df.columns:
            return df

        sort_key = pd.to_datetime(df[self._date_column], format=self._date_format, errors="coerce")
        df = df.assign(_sort_key=sort_key).sort_values("_sort_key")

        missing_before = int(df[numeric_columns].isna().sum().sum())
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        df[numeric_columns] = df[numeric_columns].ffill()  # forward-fill ONLY -- never bfill

        still_missing = df[numeric_columns].isna().any(axis=1)
        dropped = int(still_missing.sum())
        if missing_before > 0:
            logger.info(
                "FuelPrice: forward-filled missing numeric value(s); %d row(s) could not be "
                "filled (no earlier known price, e.g. a leading gap) and were dropped",
                dropped,
            )

        return df.loc[~still_missing].drop(columns="_sort_key")
