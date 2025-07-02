"""
Core components of IRST Library.
"""

from .base import BaseModel, BaseDataset, BaseTrainer, BaseEvaluator
from .registry import (
    Registry, MODELS, DATASETS, LOSSES, OPTIMIZERS, SCHEDULERS,
    TRANSFORMS, METRICS, register_model, register_dataset,
    get_model, get_dataset, list_models, list_datasets
)
from .detector import IRSTDetector

__all__ = [
    "BaseModel",
    "BaseDataset", 
    "BaseTrainer",
    "BaseEvaluator",
    "Registry",
    "MODELS",
    "DATASETS",
    "LOSSES",
    "OPTIMIZERS", 
    "SCHEDULERS",
    "TRANSFORMS",
    "METRICS",
    "register_model",
    "register_dataset",
    "get_model",
    "get_dataset",
    "list_models",
    "list_datasets",
    "IRSTDetector",
]
