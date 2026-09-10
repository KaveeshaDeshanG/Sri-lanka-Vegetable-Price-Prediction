"""Unit tests for src.prediction.predictor.Predictor and the inference-artifact save/load helpers."""

import pandas as pd
import pytest

from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
from src.models.artifact_metadata import ArtifactMetadata, save_artifact_metadata
from src.models.baseline import train_baseline_model
from src.models.persistence import save_model
from src.prediction.predictor import (
    ArtifactConsistencyError,
    FeatureAlignmentError,
    InferenceArtifact,
    ModelArtifactNotFoundError,
    Predictor,
    TransformationArtifactNotFoundError,
    load_inference_artifact,
    save_inference_artifact,
)


def _daily_group_df(start, n_days, vegetable, market, price_start, price_step=1.0, source="HARTI") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "vegetable": [vegetable] * n_days,
            "market": [market] * n_days,
            "source": [source] * n_days,
            "price": [price_start + price_step * i for i in range(n_days)],
            "rainfall_mm": [5.0] * n_days,
            "temperature_c": [28.0] * n_days,
            "humidity_pct": [70.0] * n_days,
        }
    )


def _train_and_persist(tmp_path, merged_df):
    """Build+train+persist via the real production pieces; return (config_path, model, encoder, feature_columns)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n')
    config_path = str(config_path)

    builder = TrainTestDatasetBuilder(config_path=config_path)
    X_train, X_test, y_train, y_test = builder.build(merged_df, test_size=0.2)
    model = train_baseline_model(X_train, y_train)
    encoder = builder.fitted_encoder
    feature_columns = list(X_train.columns)

    save_model(model, filename="best_model.joblib", config_path=config_path)
    save_inference_artifact(InferenceArtifact(encoder, feature_columns), config_path=config_path)
    return config_path, model, encoder, feature_columns


class _ColumnRecordingModel:
    """Test double that records exactly the columns/order Predictor hands to .predict()."""

    def __init__(self):
        self.received_columns = None

    def predict(self, X):
        self.received_columns = list(X.columns)
        return [0.0] * len(X)


class _FitForbiddenEncoder:
    """Wraps a real fitted encoder; fails the test if Predictor ever calls fit()/fit_transform()."""

    def __init__(self, real_encoder):
        self._real_encoder = real_encoder

    def fit(self, X, y=None):
        raise AssertionError("Predictor must never fit the encoder")

    def fit_transform(self, X, y=None):
        raise AssertionError("Predictor must never fit the encoder")

    def transform(self, X):
        return self._real_encoder.transform(X)


def _passthrough_row(feature_columns, vegetable, market, value=1.0):
    """A pre-encoding row: raw category columns + one dummy numeric value per non-one-hot feature."""
    passthrough_columns = [c for c in feature_columns if not c.startswith(("vegetable_", "market_"))]
    row = {column: value for column in passthrough_columns}
    row["vegetable"] = vegetable
    row["market"] = market
    return pd.DataFrame([row]), passthrough_columns


# 1. Loading the persisted model + artifact --------------------------------------


def test_predictor_load_succeeds_with_persisted_artifacts(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, _, _, _ = _train_and_persist(tmp_path, merged)

    predictor = Predictor.load(config_path=config_path)
    assert isinstance(predictor, Predictor)


def test_predictor_load_raises_when_model_artifact_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n')
    with pytest.raises(ModelArtifactNotFoundError):
        Predictor.load(config_path=str(config_path))


def test_predictor_load_raises_when_transformation_artifact_missing(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, _, _ = _train_and_persist(tmp_path, merged)
    # remove only the transformation artifact, leaving the model in place
    (tmp_path / "feature_pipeline_state.joblib").unlink()

    with pytest.raises(TransformationArtifactNotFoundError):
        Predictor.load(config_path=config_path)


def test_load_inference_artifact_raises_a_distinct_error_from_model_not_found(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n')
    with pytest.raises(TransformationArtifactNotFoundError):
        load_inference_artifact(config_path=str(config_path))


# 2 & 3. Successful prediction, shape/type ----------------------------------------


def test_predictor_predict_returns_a_series_with_expected_length(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)
    predictor = Predictor(model, InferenceArtifact(encoder, feature_columns))

    X, _ = _passthrough_row(feature_columns, "Carrot", "Dambulla")
    predictions = predictor.predict(X)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 1
    assert pd.api.types.is_numeric_dtype(predictions)


# 4. Deterministic prediction for identical input ---------------------------------


def test_predictor_predict_is_deterministic_for_identical_input(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)
    predictor = Predictor(model, InferenceArtifact(encoder, feature_columns))

    X, _ = _passthrough_row(feature_columns, "Carrot", "Dambulla")
    first = predictor.predict(X)
    second = predictor.predict(X)
    assert list(first) == list(second)


# 5. Exact feature-column compatibility -------------------------------------------


def test_predictor_predict_calls_model_with_exact_training_feature_columns(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, _, encoder, feature_columns = _train_and_persist(tmp_path, merged)

    stub_model = _ColumnRecordingModel()
    predictor = Predictor(stub_model, InferenceArtifact(encoder, feature_columns))
    X, _ = _passthrough_row(feature_columns, "Carrot", "Dambulla")
    predictor.predict(X)

    assert stub_model.received_columns == feature_columns


# 6. Reuse of the training encoder -- Predictor never fits it --------------------


def test_predictor_never_calls_fit_on_the_encoder(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)

    spy_encoder = _FitForbiddenEncoder(encoder)
    predictor = Predictor(model, InferenceArtifact(spy_encoder, feature_columns))
    X, _ = _passthrough_row(feature_columns, "Carrot", "Dambulla")

    predictions = predictor.predict(X)  # must succeed using transform() only
    assert len(predictions) == 1


# 7. Missing/NaN required feature raises rather than predicting silently ---------


def test_predictor_predict_raises_when_a_required_feature_is_nan(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)
    predictor = Predictor(model, InferenceArtifact(encoder, feature_columns))

    X, passthrough_columns = _passthrough_row(feature_columns, "Carrot", "Dambulla")
    missing_feature = passthrough_columns[0]
    X.loc[0, missing_feature] = float("nan")  # simulates insufficient historical context

    with pytest.raises(FeatureAlignmentError, match=missing_feature):
        predictor.predict(X)


# 8. Unknown categorical value does not crash and does not add a column ----------


def test_predictor_predict_handles_unseen_category_without_crashing(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)
    predictor = Predictor(model, InferenceArtifact(encoder, feature_columns))

    X, _ = _passthrough_row(feature_columns, "Cabbage", "Jaffna")  # never seen during training
    predictions = predictor.predict(X)  # must not raise

    assert len(predictions) == 1
    assert "vegetable_Cabbage" not in feature_columns  # confirms it was truly never learned


# 9. Artifact metadata consistency validation at load time ------------------------


def _metadata_for(model, feature_columns, training_run_id, **overrides):
    defaults = dict(
        artifact_version="1",
        training_run_id=training_run_id,
        trained_at="2026-08-20T00:00:00",
        model_type=type(model).__name__,
        model_name="baseline",
        training_data={"source": "test", "row_count": 50},
        feature_count=len(feature_columns),
        feature_columns=list(feature_columns),
        metrics={"mae": 1.0, "rmse": 1.5, "r2": 0.9},
    )
    defaults.update(overrides)
    return ArtifactMetadata(**defaults)


def test_predictor_load_succeeds_when_metadata_matches(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)
    run_id = "run-abc"
    save_inference_artifact(
        InferenceArtifact(encoder, feature_columns, training_run_id=run_id), config_path=config_path
    )
    save_artifact_metadata(_metadata_for(model, feature_columns, run_id), config_path=config_path)

    predictor = Predictor.load(config_path=config_path)
    assert isinstance(predictor, Predictor)


def test_predictor_load_raises_when_training_run_id_mismatches_metadata(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)
    save_inference_artifact(
        InferenceArtifact(encoder, feature_columns, training_run_id="run-a"), config_path=config_path
    )
    save_artifact_metadata(_metadata_for(model, feature_columns, "run-b"), config_path=config_path)

    with pytest.raises(ArtifactConsistencyError, match="training_run_id"):
        Predictor.load(config_path=config_path)


def test_predictor_load_raises_when_model_type_mismatches_metadata(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)
    # _train_and_persist's InferenceArtifact carries no training_run_id (None),
    # so only model_type/feature_columns are checked against this metadata.
    save_artifact_metadata(
        _metadata_for(model, feature_columns, "irrelevant-run-id", model_type="RandomForestRegressor"),
        config_path=config_path,
    )

    with pytest.raises(ArtifactConsistencyError, match="model_type"):
        Predictor.load(config_path=config_path)


def test_predictor_load_raises_when_feature_columns_mismatch_metadata(tmp_path):
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, model, encoder, feature_columns = _train_and_persist(tmp_path, merged)
    save_artifact_metadata(
        _metadata_for(model, ["some", "other", "columns"], "irrelevant-run-id"),
        config_path=config_path,
    )

    with pytest.raises(ArtifactConsistencyError, match="feature_columns"):
        Predictor.load(config_path=config_path)


def test_predictor_load_succeeds_when_no_metadata_file_exists(tmp_path):
    """Backward compatibility: artifacts saved before artifact_metadata.json existed
    must still load, unvalidated -- see _train_and_persist(), which never writes one."""
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    config_path, _, _, _ = _train_and_persist(tmp_path, merged)

    predictor = Predictor.load(config_path=config_path)
    assert isinstance(predictor, Predictor)


def test_predictor_load_accepts_models_dir_override(tmp_path):
    """models_dir lets a caller load a Predictor from a directory other than
    config's paths.models_dir -- used to load a sandbox/demo predictor without
    touching the production one (see app/pages/1_Predict.py)."""
    merged = _daily_group_df("2024-01-01", 50, "Carrot", "Dambulla", 100.0)
    other_dir = tmp_path / "elsewhere"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n')
    config_path = str(config_path)

    builder = TrainTestDatasetBuilder(config_path=config_path)
    X_train, X_test, y_train, y_test = builder.build(merged, test_size=0.2)
    model = train_baseline_model(X_train, y_train)

    save_model(model, filename="best_model.joblib", output_dir=str(other_dir))
    save_inference_artifact(
        InferenceArtifact(builder.fitted_encoder, list(X_train.columns)), output_dir=str(other_dir)
    )

    assert not (tmp_path / "best_model.joblib").exists()
    predictor = Predictor.load(config_path=config_path, models_dir=str(other_dir))
    assert isinstance(predictor, Predictor)
