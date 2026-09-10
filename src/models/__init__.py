"""ML modelling package: trains, evaluates, selects, and persists price-prediction models."""

from .baseline import build_baseline_model, train_baseline_model
from .evaluator import EvaluationError, ModelEvaluator
from .model_selection import ModelSelectionError, select_best_model
from .persistence import load_model, save_model
from .pipeline import ModelTrainingPipeline, ModelTrainingResult
from .trainer import ModelTrainer, ModelTrainingError

__all__ = [
    "build_baseline_model",
    "train_baseline_model",
    "ModelTrainer",
    "ModelTrainingError",
    "ModelEvaluator",
    "EvaluationError",
    "select_best_model",
    "ModelSelectionError",
    "save_model",
    "load_model",
    "ModelTrainingPipeline",
    "ModelTrainingResult",
]
