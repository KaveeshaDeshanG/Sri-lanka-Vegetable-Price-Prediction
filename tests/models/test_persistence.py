"""Unit tests for src.models.persistence.save_model / load_model."""

import pandas as pd

from src.models.baseline import train_baseline_model
from src.models.persistence import load_model, save_model


# 1. Round trip preserves predictions ---------------------------------------------


def test_save_and_load_round_trip_preserves_predictions(tmp_path):
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([2.0, 4.0, 6.0, 8.0])
    model = train_baseline_model(X_train, y_train)

    saved_path = save_model(model, filename="test_model.joblib", output_dir=str(tmp_path))
    loaded_model = load_model(filename="test_model.joblib", input_dir=str(tmp_path))

    X_test = pd.DataFrame({"a": [5.0, 6.0]})
    assert list(model.predict(X_test)) == list(loaded_model.predict(X_test))
    assert saved_path == str(tmp_path / "test_model.joblib")


# 2. Output directory is created if missing ---------------------------------------


def test_save_model_creates_output_directory_if_missing(tmp_path):
    output_dir = tmp_path / "nested" / "models_dir"
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1.0, 2.0, 3.0])
    model = train_baseline_model(X_train, y_train)

    saved_path = save_model(model, filename="model.joblib", output_dir=str(output_dir))

    assert (output_dir / "model.joblib").exists()
    assert saved_path == str(output_dir / "model.joblib")


# 3. Falls back to config's paths.models_dir when output_dir is omitted -----------


def test_save_and_load_default_to_config_models_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n')

    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1.0, 2.0, 3.0])
    model = train_baseline_model(X_train, y_train)

    saved_path = save_model(model, filename="from_config.joblib", config_path=str(config_path))
    assert (tmp_path / "from_config.joblib").exists()
    assert saved_path == str(tmp_path / "from_config.joblib")

    loaded_model = load_model(filename="from_config.joblib", config_path=str(config_path))
    X_test = pd.DataFrame({"a": [4.0]})
    assert list(model.predict(X_test)) == list(loaded_model.predict(X_test))
