"""
Training callbacks for monitoring and controlling training process.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import json
import warnings


class Callback:
    """Base callback class"""
    
    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Called at the beginning of each epoch"""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_begin(self, trainer, batch_idx: int, **kwargs):
        """Called at the beginning of each batch"""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, logs: Dict[str, Any] = None, **kwargs):
        """Called at the end of each batch"""
        pass
    
    def on_validation_begin(self, trainer, **kwargs):
        """Called at the beginning of validation"""
        pass
    
    def on_validation_end(self, trainer, logs: Dict[str, Any] = None, **kwargs):
        """Called at the end of validation"""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to stop training when validation loss stops improving"""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def on_train_begin(self, trainer, **kwargs):
        """Reset callback state"""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Check if training should stop"""
        if logs is None:
            return
        
        current_score = logs.get(self.monitor)
        if current_score is None:
            warnings.warn(f"Early stopping conditioned on metric `{self.monitor}` "
                         f"which is not available. Available metrics are: {list(logs.keys())}")
            return
        
        if self.best_score is None:
            self.best_score = current_score
            self.best_weights = trainer.model.state_dict().copy()
        elif self.monitor_op(current_score, self.best_score - self.min_delta):
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = trainer.model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best {self.monitor}: {self.best_score:.6f}")
            
            if self.restore_best_weights and self.best_weights is not None:
                trainer.model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best model weights")


class ModelCheckpoint(Callback):
    """Save model checkpoints during training"""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        save_weights_only: bool = False,
        verbose: bool = True,
        save_freq: str = 'epoch'  # 'epoch' or integer for batch frequency
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.save_freq = save_freq
        
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.inf
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Save checkpoint at end of epoch"""
        if self.save_freq == 'epoch':
            self._save_checkpoint(trainer, epoch, logs)
    
    def on_batch_end(self, trainer, batch_idx: int, logs: Dict[str, Any] = None, **kwargs):
        """Save checkpoint at batch intervals"""
        if isinstance(self.save_freq, int) and batch_idx % self.save_freq == 0:
            self._save_checkpoint(trainer, trainer.current_epoch, logs, batch_idx)
    
    def _save_checkpoint(self, trainer, epoch: int, logs: Dict[str, Any], batch_idx: Optional[int] = None):
        """Save model checkpoint"""
        if logs is None:
            logs = {}
        
        current_score = logs.get(self.monitor)
        
        if self.save_best_only and current_score is not None:
            if not self.monitor_op(current_score, self.best_score):
                return
            self.best_score = current_score
        
        # Format filename
        if batch_idx is not None:
            filepath = self.filepath.format(epoch=epoch, batch=batch_idx, **logs)
        else:
            filepath = self.filepath.format(epoch=epoch, **logs)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'logs': logs,
            'best_score': self.best_score
        }
        
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f"\nCheckpoint saved: {filepath}")
            if current_score is not None:
                print(f"{self.monitor}: {current_score:.6f}")


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when validation loss plateaus"""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.5,
        patience: int = 5,
        min_delta: float = 0.0,
        min_lr: float = 1e-8,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Check if learning rate should be reduced"""
        if logs is None:
            return
        
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        
        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score - self.min_delta):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self._reduce_lr(trainer)
            self.counter = 0
    
    def _reduce_lr(self, trainer):
        """Reduce learning rate"""
        for param_group in trainer.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"\nReducing learning rate: {old_lr:.2e} -> {new_lr:.2e}")


class CSVLogger(Callback):
    """Log training metrics to CSV file"""
    
    def __init__(self, filename: str, separator: str = ',', append: bool = False):
        self.filename = filename
        self.separator = separator
        self.append = append
        self.keys = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if not append:
            # Clear file
            with open(filename, 'w') as f:
                pass
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Log metrics to CSV"""
        if logs is None:
            return
        
        logs = logs.copy()
        logs['epoch'] = epoch
        
        if self.keys is None:
            self.keys = list(logs.keys())
            # Write header
            with open(self.filename, 'a') as f:
                f.write(self.separator.join(self.keys) + '\n')
        
        # Write data
        with open(self.filename, 'a') as f:
            values = [str(logs.get(key, '')) for key in self.keys]
            f.write(self.separator.join(values) + '\n')


class TensorBoardLogger(Callback):
    """Log metrics to TensorBoard"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            warnings.warn("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Log metrics to TensorBoard"""
        if self.writer is None or logs is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, trainer, **kwargs):
        """Close TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()


class LearningRateScheduler(Callback):
    """Custom learning rate scheduler callback"""
    
    def __init__(self, schedule: Callable[[int], float], verbose: bool = True):
        self.schedule = schedule
        self.verbose = verbose
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Update learning rate"""
        new_lr = self.schedule(epoch)
        
        for param_group in trainer.optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = new_lr
            
            if self.verbose and old_lr != new_lr:
                print(f"\nEpoch {epoch}: Learning rate changed from {old_lr:.2e} to {new_lr:.2e}")


class ProgressBar(Callback):
    """Simple progress bar for training"""
    
    def __init__(self, update_freq: int = 10):
        self.update_freq = update_freq
        self.batch_count = 0
        self.start_time = None
    
    def on_train_begin(self, trainer, **kwargs):
        """Initialize progress tracking"""
        self.start_time = time.time()
        print("Training started...")
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Reset batch count for new epoch"""
        self.batch_count = 0
        print(f"\nEpoch {epoch + 1}/{trainer.num_epochs}")
    
    def on_batch_end(self, trainer, batch_idx: int, logs: Dict[str, Any] = None, **kwargs):
        """Update progress bar"""
        self.batch_count += 1
        
        if batch_idx % self.update_freq == 0:
            if hasattr(trainer, 'train_loader'):
                total_batches = len(trainer.train_loader)
                progress = (batch_idx + 1) / total_batches
                bar_length = 30
                filled = int(bar_length * progress)
                bar = '█' * filled + '-' * (bar_length - filled)
                
                print(f'\r[{bar}] {batch_idx + 1}/{total_batches}', end='')
    
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Print epoch summary"""
        if logs:
            log_str = ' - '.join([f'{k}: {v:.4f}' for k, v in logs.items() if isinstance(v, (int, float))])
            print(f' - {log_str}')
    
    def on_train_end(self, trainer, **kwargs):
        """Print training summary"""
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\nTraining completed in {total_time:.2f} seconds")


class CallbackList:
    """Container for managing multiple callbacks"""
    
    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        """Add a callback"""
        self.callbacks.append(callback)
    
    def __call__(self, method_name: str, *args, **kwargs):
        """Call method on all callbacks"""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                getattr(callback, method_name)(*args, **kwargs)
    
    def on_train_begin(self, *args, **kwargs):
        self('on_train_begin', *args, **kwargs)
    
    def on_train_end(self, *args, **kwargs):
        self('on_train_end', *args, **kwargs)
    
    def on_epoch_begin(self, *args, **kwargs):
        self('on_epoch_begin', *args, **kwargs)
    
    def on_epoch_end(self, *args, **kwargs):
        self('on_epoch_end', *args, **kwargs)
    
    def on_batch_begin(self, *args, **kwargs):
        self('on_batch_begin', *args, **kwargs)
    
    def on_batch_end(self, *args, **kwargs):
        self('on_batch_end', *args, **kwargs)
    
    def on_validation_begin(self, *args, **kwargs):
        self('on_validation_begin', *args, **kwargs)
    
    def on_validation_end(self, *args, **kwargs):
        self('on_validation_end', *args, **kwargs)
    
    def should_stop_training(self) -> bool:
        """Check if any callback requests training to stop"""
        for callback in self.callbacks:
            if hasattr(callback, 'should_stop') and callback.should_stop:
                return True
        return False
