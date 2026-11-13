"""Top-level orchestrator for inference: new input -> features -> encoded -> prediction.

Runs the SAME category normalization and feature engineering training used
(DataCleaner.clean_text_columns, DataTransformer's alias/dtype steps,
FeatureEngineer.build_features/split_features_target), then reuses -- never
refits -- the training-fitted CategoricalEncoder via Predictor. There is
deliberately no train/test split anywhere in this module: inference has no
"test set", only new input to score.
"""

import logging

import pandas as pd

from config.config_loader import ConfigLoader
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.transformer import DataTransformer

from .predictor import Predictor

logger = logging.getLogger(__name__)


class PredictionInputError(Exception):
    """Raised when new prediction input is malformed or has nothing to predict."""


class PredictionPipeline:
    """Runs new input through the same feature pipeline training used, then predicts.

    Input contract for predict(): `input_df` must have the same columns as
    preprocessing's merged market+weather output (date, vegetable, market,
    price, weather columns), and must contain:
      - enough recent historical rows for the (vegetable, market) group(s)
        being predicted -- how much is "enough" depends entirely on
        config's feature_engineering settings (lag_periods_days,
        rolling_windows_days, weather_lag_days): a lag/rolling/weather-lag
        feature is only populated when a matching earlier-dated row
        actually exists in the group (see lag_features.py/
        rolling_features.py/weather_features.py's own docstrings). This
        class does not fabricate a substitute value for missing history --
        Predictor raises FeatureAlignmentError instead.
      - one row per date being predicted, with that row's price left NaN.
        A row is treated as "to predict" exactly when its price is NaN --
        there is no separate target-date parameter. That row's weather
        columns may also be NaN/absent for a genuinely future date; they
        are not used to compute that row's OWN features (weather-lag looks
        backward only), only OTHER rows' features may reference it once it
        becomes history itself.
    """

    def __init__(self, config_path: str = "config/config.yaml", predictor: Predictor | None = None) -> None:
        self._config_path = config_path
        self._feature_engineer = FeatureEngineer(config_path=config_path)
        self._predictor = predictor

        config = ConfigLoader(config_path)
        self._date_column = config.get("data_schema.date_column", "date")
        self._vegetable_column = config.get("data_schema.vegetable_column", "vegetable")
        self._market_column = config.get("data_schema.market_column", "market")
        self._price_column = config.get("data_schema.price_column", "price")
        self._weather_columns = config.get(
            "data_schema.weather_columns", ["rainfall_mm", "temperature_c", "humidity_pct"]
        )
        self._date_format = "%Y-%m-%d"  # matches src/preprocessing/pipeline.py's own hardcoded default

        self._cleaner = DataCleaner(
            date_column=self._date_column,
            price_column=self._price_column,
            vegetable_column=self._vegetable_column,
            market_column=self._market_column,
        )
        self._transformer = DataTransformer(
            date_column=self._date_column,
            date_format=self._date_format,
            vegetable_column=self._vegetable_column,
            market_column=self._market_column,
            price_column=self._price_column,
        )

    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        target_date: "pd.Timestamp | str | None" = None,
        vegetable: str | None = None,
        market: str | None = None,
    ) -> pd.DataFrame:
        """Predict for every row in input_df whose price is NaN.

        By default (target_date/vegetable/market all omitted), every row with
        an unknown (NaN) price is predicted -- unchanged from this method's
        original behavior, and what every existing caller still gets.

        target_date/vegetable/market let a caller narrow "to predict" down to
        one specific row, identified explicitly rather than inferred purely
        from a NaN price. This exists for live-weather prediction: filling
        the gap between the newest historical row and the target date (so a
        *_lag{weather_lag_days}d feature can be computed for it) requires
        inserting extra NaN-price weather-context rows into input_df -- rows
        that must contribute to feature computation (lag/rolling/weather-lag
        lookups) but must never themselves be scored. All three arguments
        must be given together, or all omitted.

        Returns:
            A DataFrame with one row per prediction, columns:
            [date_column, vegetable_column, market_column, "predicted_{price_column}"].

        Raises:
            PredictionInputError: if a date is unparseable, if only some of
                target_date/vegetable/market are given, or no row has an
                unknown (NaN) price matching what's being asked for.
            FeatureEngineeringError: if a required column (date/vegetable/
                market/price) is missing entirely -- reused as-is from
                FeatureEngineer.build_features(), not re-implemented here.
            ModelArtifactNotFoundError / TransformationArtifactNotFoundError:
                if the persisted model or feature-pipeline state can't be loaded.
            FeatureAlignmentError: if a requested row's historical context is
                insufficient to compute a required engineered feature.
        """
        identity_given = target_date is not None or vegetable is not None or market is not None
        if identity_given and (target_date is None or vegetable is None or market is None):
            raise PredictionInputError(
                "PredictionPipeline: target_date, vegetable, and market must all be provided "
                "together to select one specific row to predict, or all omitted to predict "
                "every row with an unknown price (previous default behavior)."
            )

        if identity_given:
            vegetable, market = self._normalize_identity(vegetable, market)

        normalized_df = self._normalize(input_df)
        predictor = self._require_predictor()

        featured_df = self._feature_engineer.build_features(normalized_df)
        features, target = self._feature_engineer.split_features_target(
            featured_df, drop_incomplete_rows=False, require_target=False
        )

        # A row is "to predict" exactly when its price is unknown. Safe to
        # align featured_df/target/features by position here: require_target=False
        # and drop_incomplete_rows=False together keep every row, in its
        # original order, so none of the three have been filtered relative
        # to each other.
        to_predict_mask = target.isna()

        if identity_given:
            try:
                parsed_target_date = pd.Timestamp(target_date)
            except (ValueError, TypeError) as error:
                raise PredictionInputError(
                    f"PredictionPipeline: could not parse target_date {target_date!r} as a date"
                ) from error
            identity_mask = (
                (featured_df[self._date_column] == parsed_target_date)
                & (featured_df[self._vegetable_column] == vegetable)
                & (featured_df[self._market_column] == market)
            )
            to_predict_mask = to_predict_mask & identity_mask

        if not to_predict_mask.any():
            detail = (
                f" matching date={target_date!r}, vegetable={vegetable!r}, market={market!r}"
                if identity_given
                else ""
            )
            raise PredictionInputError(
                "PredictionPipeline: nothing to predict -- no row in input_df has an unknown "
                f"(NaN) '{self._price_column}'{detail}. Add a row for the date being predicted "
                f"with its '{self._price_column}' left as NaN."
            )

        identifiers = featured_df.loc[
            to_predict_mask, [self._date_column, self._vegetable_column, self._market_column]
        ]
        predictions = predictor.predict(features.loc[to_predict_mask])

        result = identifiers.reset_index(drop=True).copy()
        result[f"predicted_{self._price_column}"] = predictions.reset_index(drop=True).to_numpy()
        logger.info("PredictionPipeline: produced %d prediction(s)", len(result))
        return result

    def get_engineered_features(
        self,
        input_df: pd.DataFrame,
        *,
        target_date: "pd.Timestamp | str",
        vegetable: str,
        market: str,
    ) -> pd.Series:
        """Return the fully engineered feature row for one (target_date, vegetable, market)
        request -- e.g. to display which weather-lag values a prediction actually used.

        Read-only and side-effect-free: runs the exact same normalize() + build_features()
        steps predict() runs internally, but never calls the model and is never called by
        predict() itself -- this exists purely so a caller (the UI) can inspect engineered
        values without duplicating FeatureEngineer's normalize/build logic, and without any
        risk of what's displayed drifting from what's actually predicted.

        Raises:
            PredictionInputError: target_date is unparseable, or no row in input_df matches
                (target_date, vegetable, market).
        """
        try:
            parsed_target_date = pd.Timestamp(target_date)
        except (ValueError, TypeError) as error:
            raise PredictionInputError(
                f"PredictionPipeline: could not parse target_date {target_date!r} as a date"
            ) from error

        vegetable, market = self._normalize_identity(vegetable, market)
        normalized_df = self._normalize(input_df)
        featured_df = self._feature_engineer.build_features(normalized_df)

        mask = (
            (featured_df[self._date_column] == parsed_target_date)
            & (featured_df[self._vegetable_column] == vegetable)
            & (featured_df[self._market_column] == market)
        )
        if not mask.any():
            raise PredictionInputError(
                f"PredictionPipeline: no row found matching date={target_date!r}, "
                f"vegetable={vegetable!r}, market={market!r} to inspect engineered features for."
            )
        return featured_df.loc[mask].iloc[0]

    def _require_predictor(self) -> Predictor:
        if self._predictor is None:
            self._predictor = Predictor.load(config_path=self._config_path)
        return self._predictor

    def _normalize_identity(self, vegetable: str, market: str) -> tuple[str, str]:
        """Apply the same text and alias steps to a requested name that the input rows get,
        so asking for "carrot" or "Eggplant" finds the row normalized to "Carrot"/"Brinjal"."""
        identity = pd.DataFrame({self._vegetable_column: [vegetable], self._market_column: [market]})
        identity = self._cleaner.clean_text_columns(identity, [self._vegetable_column, self._market_column])
        identity = self._transformer.standardize_market_names(self._transformer.standardize_vegetable_names(identity))
        return identity[self._vegetable_column].iloc[0], identity[self._market_column].iloc[0]

    def _normalize(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same non-destructive normalization training data went through.

        Deliberately reuses only the NON-row-dropping steps of
        DataCleaner/DataTransformer, in the same relative order
        PreprocessingPipeline runs them (clean, then transform):
        text hygiene -> alias standardization -> date/numeric dtype
        conversion -> chronological sort. The row-dropping steps in
        DataCleaner.clean_market_data() (missing/invalid price, implausible
        dates) are intentionally NOT reused -- they would delete the very
        row being predicted, whose price is NaN by definition.

        Without this step, a caller writing "carrot" would be encoded as an
        unseen category even though training saw "Carrot" -- the fitted
        CategoricalEncoder does exact string matching, and category
        casing/aliasing is normalized upstream of it during training.
        """
        df = self._cleaner.clean_text_columns(input_df, [self._vegetable_column, self._market_column])
        df = self._transformer.standardize_vegetable_names(df)
        df = self._transformer.standardize_market_names(df)
        try:
            df = self._transformer.convert_date_column(df)
        except (ValueError, TypeError) as error:
            raise PredictionInputError(
                f"PredictionPipeline: could not parse '{self._date_column}' as a date "
                f"(expected format {self._date_format!r}): {error}"
            ) from error
        df = self._transformer.convert_numeric_columns(df, [self._price_column, *self._weather_columns])
        df = self._transformer.sort_chronologically(df)
        return df
