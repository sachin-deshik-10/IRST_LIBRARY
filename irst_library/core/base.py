"""
Core base classes and interfaces for IRST Library.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from pathlib import Path


class BaseModel(nn.Module, ABC):
    """Base class for all ISTD models."""
    
    def __init__(self):
        super().__init__()
        self._name = self.__class__.__name__
    
    @property
    def name(self) -> str:
        """Get model name."""
        return self._name
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss for training.
        
        Args:
            pred: Predictions from the model
            target: Ground truth targets
            
        Returns:
            Loss tensor
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "name": self.name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 ** 2),  # Assuming float32
        }


class BaseDataset(torch.utils.data.Dataset, ABC):
    """Base class for all ISTD datasets."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Validate split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test")
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing 'image', 'mask', and 'meta' keys
        """
        pass
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.__class__.__name__,
            "root": str(self.root),
            "split": self.split,
            "num_samples": len(self),
        }


class BaseTrainer(ABC):
    """Base class for all trainers."""
    
    def __init__(
        self,
        model: BaseModel,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset] = None,
        **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = kwargs
    
    @abstractmethod
    def fit(self) -> Dict[str, Any]:
        """Train the model.
        
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model.
        
        Returns:
            Evaluation results dictionary
        """
        pass


class BaseEvaluator(ABC):
    """Base class for model evaluation."""
    
    def __init__(self, model: BaseModel, dataset: BaseDataset):
        self.model = model
        self.dataset = dataset
    
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Returns:
            Evaluation metrics dictionary
        """
        pass
    
    @abstractmethod
    def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed metrics
        """
        pass
