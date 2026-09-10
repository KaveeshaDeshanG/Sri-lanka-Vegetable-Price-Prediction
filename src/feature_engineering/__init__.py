"""Feature engineering package: turns the merged dataset into a model-ready training set."""

from .categorical_features import CategoricalEncoder, CategoricalFeatureEncodingError
from .dataset_builder import TrainTestDatasetBuilder
from .date_features import DateFeatureGenerator
from .feature_engineer import FeatureEngineer, FeatureEngineeringError
from .lag_features import LagFeatureGenerator
from .rolling_features import RollingFeatureGenerator
from .train_test_split import ChronologicalSplitError, chronological_train_test_split
from .weather_features import WeatherFeatureGenerator

__all__ = [
    "FeatureEngineer",
    "FeatureEngineeringError",
    "CategoricalEncoder",
    "CategoricalFeatureEncodingError",
    "ChronologicalSplitError",
    "chronological_train_test_split",
    "TrainTestDatasetBuilder",
    "DateFeatureGenerator",
    "LagFeatureGenerator",
    "RollingFeatureGenerator",
    "WeatherFeatureGenerator",
]
