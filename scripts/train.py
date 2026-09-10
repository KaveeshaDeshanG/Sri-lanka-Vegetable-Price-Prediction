"""Standalone production training entry point.

Wires already-existing, already-tested pipeline stages together against the
real raw data configured in config.yaml:

    DataLoader (via PreprocessingPipeline)
    -> PreprocessingPipeline (clean/transform/merge market+weather+fuel)
    -> TrainTestDatasetBuilder (feature engineering, chronological split,
       train-only categorical encoding)
    -> ModelTrainingPipeline (train every configured model, evaluate on the
       held-out test set, select the best by config's evaluation metric,
       persist it)
    -> save_inference_artifact (persist the fitted encoder + feature column
       order alongside the model)

This script contains no feature engineering, splitting, encoding, model, or
persistence logic of its own -- every step below is a call into an existing,
independently-tested component. Nothing here decides *how* a split is
computed, *how* a category is encoded, or *how* a model is fit; it only
decides the order to call those things in and what to report.

Usage:
    .venv/Scripts/python.exe scripts/train.py --config config/config.yaml
    .venv/Scripts/python.exe scripts/train.py --config config/config.yaml --dry-run

--dry-run runs everything up to, but not including, model fitting: it loads
real data, preprocesses it, builds features, performs the chronological
split, and fits the categorical encoder on the training split only -- then
reports the resulting shapes and feature names without ever calling
model.fit() or writing to models/.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader  # noqa: E402
from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder  # noqa: E402
from src.feature_engineering.production_region_weather import (  # noqa: E402
    apply_production_region_weather,
)
from src.models.artifact_metadata import (  # noqa: E402
    ArtifactMetadata,
    ARTIFACT_VERSION,
    new_training_run_id,
    save_artifact_metadata,
)
from src.models.metrics_report import save_metrics_report  # noqa: E402
from src.models.pipeline import ModelTrainingPipeline  # noqa: E402
from src.prediction.predictor import InferenceArtifact, save_inference_artifact  # noqa: E402
from src.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train every model in config.model.models_to_train on real raw data, "
        "evaluate, select the best by config.evaluation.best_model_metric, and persist it."
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config.yaml (default: %(default)s)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run data loading, preprocessing, feature engineering, the chronological "
        "train/test split, and train-only categorical encoding; report the resulting "
        "shapes and feature names; then stop WITHOUT fitting or persisting any model.",
    )
    return parser.parse_args(argv)


def run_training(config_path: str, dry_run: bool = False) -> dict:
    """Run the real-data training pipeline, or (if dry_run) everything short of model fitting.

    Returns a summary dict describing what happened -- this is also what
    tests assert against, so behavior can be verified without parsing log
    output.
    """
    config = ConfigLoader(config_path)

    preprocessing = PreprocessingPipeline(config_path=config_path)
    raw = preprocessing.load_data()  # DataLoader under the hood: {"market", "weather", "fuel"}
    fuel_configured = raw["fuel"] is not None

    preprocessing.clean_data()
    preprocessing.transform_data()
    merged_df = preprocessing.merge_data()

    # Corrects each row's same-day weather to its vegetable's evidence-based
    # production region (config/crop_production_regions.yaml) where one is mapped
    # -- e.g. Carrot rows get Nuwara Eliya's weather instead of the market's
    # (Dambulla's). A vegetable with no sufficiently evidenced region is left
    # completely unchanged (market-location weather, this project's original
    # behavior). Must run BEFORE feature engineering so WeatherFeatureGenerator's
    # existing, unchanged date-only lag logic computes lag features from the
    # corrected reading.
    merged_df = apply_production_region_weather(merged_df, config_path=config_path)

    # TrainTestDatasetBuilder.build() is the ONE existing component that does
    # feature engineering + chronological_train_test_split() +
    # split_features_target() + CategoricalEncoder.fit_transform(X_train)/
    # .transform(X_test) -- this script never reimplements any of that.
    builder = TrainTestDatasetBuilder(config_path=config_path)
    X_train, X_test, y_train, y_test = builder.build(merged_df)

    date_column = config.get("data_schema.date_column", "date")

    summary = {
        "market_rows": len(raw["market"]),
        "weather_rows": len(raw["weather"]),
        "fuel_configured": fuel_configured,
        "fuel_rows": len(raw["fuel"]) if fuel_configured else 0,
        "merged_rows": len(merged_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "dry_run": dry_run,
    }
    _log_pre_training_summary(summary)

    if dry_run:
        logger.info("Dry-run complete: no model was fitted or persisted.")
        return summary

    training_run_id = new_training_run_id()

    training_pipeline = ModelTrainingPipeline(config_path=config_path)
    result = training_pipeline.run(X_train, X_test, y_train, y_test)

    inference_artifact_path = save_inference_artifact(
        InferenceArtifact(builder.fitted_encoder, list(X_train.columns), training_run_id=training_run_id),
        config_path=config_path,
    )

    best_model_metric = config.get("evaluation.best_model_metric", "rmse")
    metrics_report_path = save_metrics_report(
        result.metrics,
        result.best_model_name,
        best_model_metric,
        dataset_summary={
            "market_rows": summary["market_rows"],
            "weather_rows": summary["weather_rows"],
            "fuel_configured": summary["fuel_configured"],
            "fuel_rows": summary["fuel_rows"],
            "merged_rows": summary["merged_rows"],
            "train_rows": summary["train_rows"],
            "test_rows": summary["test_rows"],
            "feature_count": summary["feature_count"],
        },
        config_path=config_path,
    )

    artifact_metadata_path = save_artifact_metadata(
        ArtifactMetadata(
            artifact_version=ARTIFACT_VERSION,
            training_run_id=training_run_id,
            trained_at=datetime.now().isoformat(timespec="seconds"),
            model_type=type(result.best_model).__name__,
            model_name=result.best_model_name,
            training_data={
                "source": "HARTI historical dataset (paths.raw_files.harti_prices, "
                "+ weather_data, + fuel_prices, + weather_data_by_region)",
                "start_date": str(merged_df[date_column].min().date()),
                "end_date": str(merged_df[date_column].max().date()),
                "row_count": summary["merged_rows"],
                "train_rows": summary["train_rows"],
                "test_rows": summary["test_rows"],
            },
            feature_count=summary["feature_count"],
            feature_columns=summary["feature_names"],
            metrics=result.metrics[result.best_model_name],
        ),
        config_path=config_path,
    )

    summary.update(
        {
            "models_trained": list(result.models.keys()),
            "metrics": result.metrics,
            "best_model_name": result.best_model_name,
            "best_model_metric": best_model_metric,
            "model_artifact_path": result.saved_model_path,
            "inference_artifact_path": inference_artifact_path,
            "metrics_report_path": metrics_report_path,
            "artifact_metadata_path": artifact_metadata_path,
            "training_run_id": training_run_id,
        }
    )
    _log_post_training_summary(summary)
    return summary


def _log_pre_training_summary(summary: dict) -> None:
    logger.info("=" * 72)
    logger.info("Raw market rows:          %d", summary["market_rows"])
    logger.info("Raw weather rows:         %d", summary["weather_rows"])
    if summary["fuel_configured"]:
        logger.info("Raw fuel-price rows:      %d", summary["fuel_rows"])
    else:
        logger.info("Raw fuel-price rows:      not configured (paths.raw_files.fuel_prices is null)")
    logger.info("Merged/feature-eng. rows: %d", summary["merged_rows"])
    logger.info("Train rows:               %d", summary["train_rows"])
    logger.info("Test rows:                %d", summary["test_rows"])
    logger.info("Final feature count:      %d", summary["feature_count"])
    logger.info("Feature names:            %s", summary["feature_names"])
    logger.info("=" * 72)


def _log_post_training_summary(summary: dict) -> None:
    logger.info("Models trained:           %s", summary["models_trained"])
    for model_name, metrics in summary["metrics"].items():
        logger.info("  %-18s %s", model_name, metrics)
    logger.info("Selected best model:      %s", summary["best_model_name"])
    logger.info("Selection metric:         %s", summary["best_model_metric"])
    logger.info("Model artifact:           %s", summary["model_artifact_path"])
    logger.info("Inference artifact:       %s", summary["inference_artifact_path"])
    logger.info("Metrics report:           %s", summary["metrics_report_path"])
    logger.info("Artifact metadata:        %s", summary["artifact_metadata_path"])
    logger.info("Training run id:          %s", summary["training_run_id"])
    logger.info("=" * 72)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ConfigLoader(args.config)
    logging.basicConfig(
        level=config.get("logging.level", "INFO"),
        format=config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )
    run_training(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
