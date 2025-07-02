"""
Trainer classes for infrared small target detection models.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
import warnings

from .losses import get_loss_function, IRSTLoss
from .metrics import MetricsCalculator
from .callbacks import CallbackList, EarlyStopping, ModelCheckpoint, ProgressBar
from ..core.base import BaseModel


class BaseTrainer(ABC):
    """Abstract base trainer class"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "auto"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.num_epochs = 0
        self.training_history = []
        
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        pass
    
    @abstractmethod
    def fit(self, num_epochs: int, **kwargs) -> Dict[str, List[float]]:
        """Train the model"""
        pass


class IRSTTrainer(BaseTrainer):
    """
    Trainer for infrared small target detection models.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer instance or string name
        loss_function: Loss function instance or string name
        metrics: List of metric names to compute
        device: Device to use for training ('cuda', 'cpu', or 'auto')
        callbacks: List of training callbacks
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Union[optim.Optimizer, str] = "adam",
        loss_function: Union[nn.Module, str] = "irst",
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        metrics: List[str] = None,
        device: str = "auto",
        callbacks: List = None,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, device)
        
        # Setup optimizer
        if isinstance(optimizer, str):
            self.optimizer = self._get_optimizer(optimizer, **kwargs)
        else:
            self.optimizer = optimizer
        
        # Setup loss function
        if isinstance(loss_function, str):
            self.loss_function = get_loss_function(loss_function, **kwargs)
        else:
            self.loss_function = loss_function
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup metrics
        if metrics is None:
            metrics = ['precision', 'recall', 'f1', 'iou']
        self.metrics_calculator = MetricsCalculator()
        
        # Setup callbacks
        self.callbacks = CallbackList(callbacks or [])
        
        # Add default callbacks if not present
        self._add_default_callbacks()
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        print(f"  Loss function: {self.loss_function.__class__.__name__}")
    
    def _get_optimizer(self, name: str, lr: float = 1e-3, **kwargs) -> optim.Optimizer:
        """Get optimizer by name"""
        optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
        }
        
        if name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
        
        optimizer_class = optimizers[name.lower()]
        
        # Filter kwargs for the specific optimizer
        import inspect
        sig = inspect.signature(optimizer_class.__init__)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        return optimizer_class(self.model.parameters(), lr=lr, **valid_kwargs)
    
    def _add_default_callbacks(self):
        """Add default callbacks if not present"""
        callback_types = [type(cb) for cb in self.callbacks.callbacks]
        
        # Add progress bar if not present
        if ProgressBar not in callback_types:
            self.callbacks.append(ProgressBar())
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            if isinstance(self.loss_function, (IRSTLoss,)):
                loss_dict = self.loss_function(outputs, targets)
                loss = loss_dict['total']
            else:
                loss = self.loss_function(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.metrics_calculator.update(outputs.detach(), targets.detach())
            total_loss += loss.item()
            
            # Callback
            batch_logs = {'batch_loss': loss.item()}
            self.callbacks.on_batch_end(self, batch_idx, batch_logs)
        
        # Compute epoch metrics
        metrics = self.metrics_calculator.compute()
        epoch_logs = {
            'loss': total_loss / num_batches,
            **metrics['summary']
        }
        
        return epoch_logs
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move data to device
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                if isinstance(self.loss_function, (IRSTLoss,)):
                    loss_dict = self.loss_function(outputs, targets)
                    loss = loss_dict['total']
                else:
                    loss = self.loss_function(outputs, targets)
                
                # Update metrics
                self.metrics_calculator.update(outputs, targets)
                total_loss += loss.item()
        
        # Compute epoch metrics
        metrics = self.metrics_calculator.compute()
        epoch_logs = {
            'val_loss': total_loss / num_batches,
            **{f'val_{k}': v for k, v in metrics['summary'].items()}
        }
        
        return epoch_logs
    
    def fit(
        self,
        num_epochs: int,
        verbose: int = 1,
        validation_freq: int = 1,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            validation_freq: How often to run validation (every N epochs)
            
        Returns:
            Training history dictionary
        """
        self.num_epochs = num_epochs
        history = {'loss': [], 'val_loss': []}
        
        # Training callbacks
        self.callbacks.on_train_begin(self)
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Epoch callbacks
                self.callbacks.on_epoch_begin(self, epoch)
                
                # Training phase
                train_logs = self.train_epoch()
                
                # Validation phase
                val_logs = {}
                if epoch % validation_freq == 0 or epoch == num_epochs - 1:
                    self.callbacks.on_validation_begin(self)
                    val_logs = self.validate_epoch()
                    self.callbacks.on_validation_end(self, val_logs)
                
                # Combine logs
                epoch_logs = {**train_logs, **val_logs}
                epoch_logs['epoch_time'] = time.time() - epoch_start_time
                
                # Update history
                for key, value in epoch_logs.items():
                    if key not in history:
                        history[key] = []
                    history[key].append(value)
                
                # Learning rate scheduler
                if self.scheduler is not None:
                    if hasattr(self.scheduler, 'step'):
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_logs.get('val_loss', train_logs['loss']))
                        else:
                            self.scheduler.step()
                
                # Epoch callbacks
                self.callbacks.on_epoch_end(self, epoch, epoch_logs)
                
                # Check for early stopping
                if self.callbacks.should_stop_training():
                    print(f"Training stopped early at epoch {epoch + 1}")
                    break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Training end callbacks
            self.callbacks.on_train_end(self)
        
        self.training_history = history
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test dataset"""
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                if isinstance(self.loss_function, (IRSTLoss,)):
                    loss_dict = self.loss_function(outputs, targets)
                    loss = loss_dict['total']
                else:
                    loss = self.loss_function(outputs, targets)
                
                # Update metrics
                self.metrics_calculator.update(outputs, targets)
                total_loss += loss.item()
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        
        results = {
            'test_loss': total_loss / num_batches,
            **{f'test_{k}': v for k, v in metrics['summary'].items()},
            'detailed_metrics': metrics
        }
        
        return results
    
    def predict(self, data_loader: DataLoader) -> List[torch.Tensor]:
        """Generate predictions for a dataset"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                predictions.append(outputs.cpu())
        
        return predictions
    
    def save_checkpoint(self, filepath: str, **extra_data):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            **extra_data
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True, load_scheduler: bool = True):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if load_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resumed from epoch: {self.current_epoch}")


class LightningTrainer:
    """
    PyTorch Lightning-based trainer (placeholder for future implementation).
    This would provide additional features like multi-GPU training, automatic mixed precision, etc.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LightningTrainer is not yet implemented. Use IRSTTrainer instead.",
            UserWarning
        )
        raise NotImplementedError("LightningTrainer will be implemented in future versions")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    trainer_type: str = "irst",
    **kwargs
) -> BaseTrainer:
    """
    Factory function to create trainer instances.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        trainer_type: Type of trainer ('irst', 'lightning')
        **kwargs: Additional arguments for trainer
    
    Returns:
        Trainer instance
    """
    trainers = {
        'irst': IRSTTrainer,
        'lightning': LightningTrainer,
    }
    
    if trainer_type not in trainers:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    return trainers[trainer_type](model, train_loader, val_loader, **kwargs)
