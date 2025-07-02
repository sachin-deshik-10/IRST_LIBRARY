"""
Continual Learning Module for IRST Library

This module implements advanced continual learning techniques to enable
infrared small target detection models to learn new tasks without forgetting
previous knowledge. Includes Elastic Weight Consolidation (EWC), Progressive
Neural Networks, and intelligent replay mechanisms.

Key Features:
- Catastrophic forgetting prevention
- Multi-task learning capabilities
- Memory-efficient knowledge retention
- Adaptive task switching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from collections import defaultdict, deque
import copy
import pickle
import logging
from abc import ABC, abstractmethod


class ContinualLearningBase(ABC):
    """Base class for continual learning strategies"""
    
    @abstractmethod
    def before_task(self, task_id: int, dataloader: DataLoader):
        """Called before training on a new task"""
        pass
    
    @abstractmethod
    def after_task(self, task_id: int, model: nn.Module):
        """Called after completing training on a task"""
        pass
    
    @abstractmethod
    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute additional regularization loss to prevent forgetting"""
        pass


class ElasticWeightConsolidation(ContinualLearningBase):
    """
    Elastic Weight Consolidation (EWC) for continual learning
    
    Protects important parameters from large changes when learning new tasks
    by computing Fisher Information Matrix importance scores.
    
    Reference: Kirkpatrick et al. "Overcoming catastrophic forgetting in 
               neural networks" PNAS 2017
    """
    
    def __init__(
        self,
        lambda_ewc: float = 1000.0,
        fisher_estimation_samples: int = 1000,
        online_ewc: bool = True,
        gamma: float = 0.9
    ):
        """
        Args:
            lambda_ewc: Regularization strength for EWC loss
            fisher_estimation_samples: Number of samples for Fisher estimation
            online_ewc: Whether to use online EWC (accumulate Fisher across tasks)
            gamma: Decay factor for online EWC
        """
        self.lambda_ewc = lambda_ewc
        self.fisher_estimation_samples = fisher_estimation_samples
        self.online_ewc = online_ewc
        self.gamma = gamma
        
        # Storage for Fisher Information and optimal parameters
        self.fisher_diagonal = {}
        self.optimal_params = {}
        self.task_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def compute_fisher_diagonal(
        self, 
        model: nn.Module, 
        dataloader: DataLoader,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal Fisher Information Matrix
        
        The Fisher Information Matrix measures the sensitivity of the likelihood
        to parameter changes. We approximate it with the diagonal for efficiency.
        """
        model.eval()
        num_samples = num_samples or self.fisher_estimation_samples
        
        fisher_diagonal = {}
        
        # Initialize Fisher diagonal for each parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_diagonal[name] = torch.zeros_like(param)
        
        samples_processed = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
                
            data, targets = data.cuda(), targets.cuda()
            batch_size = data.size(0)
            
            # Forward pass
            model.zero_grad()
            outputs = model(data)
            
            # Use log-likelihood gradient for Fisher computation
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
                
            log_likelihood = F.log_softmax(logits, dim=1)
            
            # Sample from model's predicted distribution
            predicted_labels = torch.multinomial(
                F.softmax(logits, dim=1), 
                num_samples=1
            ).squeeze()
            
            # Compute gradients with respect to sampled labels
            log_likelihood_sampled = log_likelihood.gather(
                1, predicted_labels.unsqueeze(1)
            ).squeeze()
            
            loss = -log_likelihood_sampled.sum()
            loss.backward(retain_graph=False)
            
            # Accumulate squared gradients (Fisher diagonal approximation)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diagonal[name] += (param.grad ** 2) * batch_size
            
            samples_processed += batch_size
        
        # Normalize by number of samples
        for name in fisher_diagonal:
            fisher_diagonal[name] /= samples_processed
            
        model.train()
        return fisher_diagonal
    
    def before_task(self, task_id: int, dataloader: DataLoader):
        """Prepare for new task training"""
        self.task_count = task_id + 1
        self.logger.info(f"Preparing EWC for task {task_id}")
    
    def after_task(self, task_id: int, model: nn.Module):
        """Update Fisher information and optimal parameters after task completion"""
        
        self.logger.info(f"Computing Fisher Information for task {task_id}")
        
        # Compute Fisher diagonal for current task
        fisher_current = self.compute_fisher_diagonal(
            model, 
            self._current_dataloader  # Store dataloader in before_task
        )
        
        # Store optimal parameters for current task
        optimal_params_current = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                optimal_params_current[name] = param.clone().detach()
        
        if self.online_ewc and task_id > 0:
            # Online EWC: accumulate Fisher information across tasks
            for name in fisher_current:
                if name in self.fisher_diagonal:
                    self.fisher_diagonal[name] = (
                        self.gamma * self.fisher_diagonal[name] + 
                        fisher_current[name]
                    )
                    # Update optimal parameters as running average
                    self.optimal_params[name] = (
                        self.gamma * self.optimal_params[name] + 
                        (1 - self.gamma) * optimal_params_current[name]
                    )
                else:
                    self.fisher_diagonal[name] = fisher_current[name]
                    self.optimal_params[name] = optimal_params_current[name]
        else:
            # Standard EWC: use Fisher from previous tasks
            self.fisher_diagonal.update(fisher_current)
            self.optimal_params.update(optimal_params_current)
        
        self.logger.info(f"EWC Fisher computation completed for task {task_id}")
    
    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularization loss
        
        L_EWC = λ/2 * Σ F_i * (θ_i - θ*_i)²
        where F_i is Fisher information and θ*_i are optimal parameters
        """
        if not self.fisher_diagonal or not self.optimal_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = 0.0
        
        for name, param in model.named_parameters():
            if name in self.fisher_diagonal and param.requires_grad:
                fisher = self.fisher_diagonal[name]
                optimal = self.optimal_params[name]
                
                # EWC penalty: Fisher-weighted parameter deviation
                penalty = fisher * (param - optimal) ** 2
                ewc_loss += penalty.sum()
        
        return self.lambda_ewc / 2 * ewc_loss


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Networks for continual learning
    
    Adds new network columns for each task while preserving previous columns.
    Uses lateral connections to transfer knowledge between tasks.
    
    Reference: Rusu et al. "Progressive Neural Networks" arXiv 2016
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_sizes: List[int],
        max_tasks: int = 10
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_sizes: Hidden layer sizes for each column
            output_sizes: Output size for each task
            max_tasks: Maximum number of tasks to support
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes
        self.max_tasks = max_tasks
        self.current_task = 0
        
        # Storage for network columns (one per task)
        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()  # Inter-task connections
        self.adapters = nn.ModuleList()  # Lateral connection adapters
        
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, output_size: int) -> int:
        """
        Add a new task column to the progressive network
        
        Args:
            output_size: Output dimension for the new task
            
        Returns:
            Task ID for the new task
        """
        if self.current_task >= self.max_tasks:
            raise ValueError(f"Maximum number of tasks ({self.max_tasks}) reached")
        
        task_id = self.current_task
        
        # Create new column for this task
        column_layers = nn.ModuleList()
        
        # Input layer
        column_layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_sizes)):
            column_layers.append(
                nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i])
            )
        
        # Output layer
        column_layers.append(nn.Linear(self.hidden_sizes[-1], output_size))
        
        self.columns.append(column_layers)
        
        # Add lateral connections from previous columns
        if task_id > 0:
            lateral_adapters = nn.ModuleList()
            
            for layer_idx in range(len(self.hidden_sizes)):
                # Adapters for lateral connections from all previous tasks
                task_adapters = nn.ModuleList()
                
                for prev_task in range(task_id):
                    adapter = nn.Sequential(
                        nn.Linear(self.hidden_sizes[layer_idx], self.hidden_sizes[layer_idx]),
                        nn.ReLU(),
                        nn.Linear(self.hidden_sizes[layer_idx], self.hidden_sizes[layer_idx])
                    )
                    task_adapters.append(adapter)
                
                lateral_adapters.append(task_adapters)
            
            self.adapters.append(lateral_adapters)
        
        self.current_task += 1
        self.logger.info(f"Added new task column {task_id}")
        
        return task_id
    
    def forward_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass for a specific task
        
        Args:
            x: Input tensor
            task_id: ID of the task to execute
            
        Returns:
            Output for the specified task
        """
        if task_id >= len(self.columns):
            raise ValueError(f"Task {task_id} not available. Only {len(self.columns)} tasks added.")
        
        # Storage for activations from all previous columns
        all_activations = []
        
        # Process all columns up to and including the target task
        for col_id in range(task_id + 1):
            column = self.columns[col_id]
            
            # Input layer activation
            h = F.relu(column[0](x))
            column_activations = [h]
            
            # Hidden layers with lateral connections
            for layer_idx in range(1, len(self.hidden_sizes)):
                # Standard column processing
                h = column[layer_idx](h)
                
                # Add lateral connections from previous columns
                if col_id > 0 and col_id <= task_id:
                    lateral_input = 0
                    
                    for prev_col_id in range(col_id):
                        if prev_col_id < len(all_activations):
                            prev_activation = all_activations[prev_col_id][layer_idx - 1]
                            
                            # Apply adapter
                            adapter_idx = col_id - 1  # Adapter index
                            if adapter_idx < len(self.adapters):
                                adapter = self.adapters[adapter_idx][layer_idx - 1][prev_col_id]
                                lateral_input += adapter(prev_activation)
                    
                    h = h + lateral_input
                
                h = F.relu(h)
                column_activations.append(h)
            
            all_activations.append(column_activations)
        
        # Output layer (no lateral connections)
        final_column = self.columns[task_id]
        output = final_column[-1](all_activations[task_id][-1])
        
        return output
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass - can execute specific task or all tasks
        
        Args:
            x: Input tensor
            task_id: Specific task to execute (if None, execute all tasks)
            
        Returns:
            Dictionary with task outputs
        """
        if task_id is not None:
            return {'task_{}'.format(task_id): self.forward_task(x, task_id)}
        
        # Execute all tasks
        outputs = {}
        for tid in range(len(self.columns)):
            outputs[f'task_{tid}'] = self.forward_task(x, tid)
        
        return outputs
    
    def freeze_columns(self, task_ids: List[int]):
        """Freeze parameters of specific task columns"""
        for task_id in task_ids:
            if task_id < len(self.columns):
                for param in self.columns[task_id].parameters():
                    param.requires_grad = False
                    
                self.logger.info(f"Frozen parameters for task {task_id}")


class IntelligentReplayBuffer:
    """
    Intelligent replay buffer for continual learning
    
    Stores representative samples from previous tasks and uses sophisticated
    selection strategies to maintain a diverse, informative memory.
    """
    
    def __init__(
        self,
        buffer_size: int = 5000,
        selection_strategy: str = 'gradient_episodic',
        update_strategy: str = 'ring_buffer'
    ):
        """
        Args:
            buffer_size: Maximum number of samples to store
            selection_strategy: Strategy for selecting samples ('random', 'gradient_episodic', 'uncertainty')
            update_strategy: Strategy for updating buffer ('ring_buffer', 'priority_based')
        """
        self.buffer_size = buffer_size
        self.selection_strategy = selection_strategy
        self.update_strategy = update_strategy
        
        # Storage
        self.data_buffer = []
        self.target_buffer = []
        self.task_buffer = []
        self.metadata_buffer = []  # For storing additional info (gradients, uncertainties, etc.)
        
        self.current_size = 0
        self.insert_index = 0
        
        self.logger = logging.getLogger(__name__)
    
    def add_samples(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        task_id: int,
        model: Optional[nn.Module] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add samples to the replay buffer using the configured strategy
        
        Args:
            data: Input samples
            targets: Target labels
            task_id: Task identifier
            model: Current model (needed for some selection strategies)
            metadata: Additional metadata for samples
        """
        batch_size = data.size(0)
        
        # Select which samples to add based on strategy
        if self.selection_strategy == 'random':
            indices = torch.randperm(batch_size)[:min(batch_size, self.buffer_size)]
        elif self.selection_strategy == 'gradient_episodic':
            indices = self._select_gradient_episodic(data, targets, model)
        elif self.selection_strategy == 'uncertainty':
            indices = self._select_high_uncertainty(data, targets, model)
        else:
            indices = torch.arange(batch_size)
        
        # Add selected samples
        for idx in indices:
            sample_data = data[idx].cpu()
            sample_target = targets[idx].cpu()
            sample_metadata = metadata[idx] if metadata else {}
            
            if self.current_size < self.buffer_size:
                # Buffer not full - append
                self.data_buffer.append(sample_data)
                self.target_buffer.append(sample_target)
                self.task_buffer.append(task_id)
                self.metadata_buffer.append(sample_metadata)
                self.current_size += 1
            else:
                # Buffer full - replace based on update strategy
                if self.update_strategy == 'ring_buffer':
                    replace_idx = self.insert_index % self.buffer_size
                    self.insert_index += 1
                elif self.update_strategy == 'priority_based':
                    replace_idx = self._select_replacement_index(sample_metadata)
                else:
                    replace_idx = np.random.randint(0, self.buffer_size)
                
                self.data_buffer[replace_idx] = sample_data
                self.target_buffer[replace_idx] = sample_target
                self.task_buffer[replace_idx] = task_id
                self.metadata_buffer[replace_idx] = sample_metadata
    
    def _select_gradient_episodic(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor, 
        model: nn.Module
    ) -> torch.Tensor:
        """
        Select samples that would cause large gradient changes (GEM strategy)
        
        Reference: Lopez-Paz & Ranzato "Gradient Episodic Memory for Continual Learning" NIPS 2017
        """
        if model is None or len(self.data_buffer) == 0:
            return torch.randperm(data.size(0))[:min(data.size(0), 100)]
        
        model.eval()
        
        # Compute gradients for each sample
        gradient_norms = []
        
        for i in range(data.size(0)):
            model.zero_grad()
            
            output = model(data[i:i+1])
            if isinstance(output, dict):
                output = output['logits']
            
            loss = F.cross_entropy(output, targets[i:i+1])
            loss.backward(retain_graph=False)
            
            # Compute gradient norm
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            
            gradient_norms.append(grad_norm ** 0.5)
        
        # Select samples with highest gradient norms
        gradient_norms = torch.tensor(gradient_norms)
        _, indices = torch.topk(gradient_norms, min(len(gradient_norms), 100))
        
        model.train()
        return indices
    
    def _select_high_uncertainty(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor, 
        model: nn.Module
    ) -> torch.Tensor:
        """Select samples with highest prediction uncertainty"""
        if model is None:
            return torch.randperm(data.size(0))[:min(data.size(0), 100)]
        
        model.eval()
        
        with torch.no_grad():
            outputs = model(data)
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            
            # Compute uncertainty as entropy of predictions
            probs = F.softmax(outputs, dim=1)
            uncertainties = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Select most uncertain samples
        _, indices = torch.topk(uncertainties, min(len(uncertainties), 100))
        
        model.train()
        return indices
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the replay buffer
        
        Returns:
            Tuple of (data, targets, task_ids)
        """
        if self.current_size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample indices
        indices = np.random.choice(self.current_size, size=min(batch_size, self.current_size), replace=False)
        
        # Collect samples
        batch_data = torch.stack([self.data_buffer[i] for i in indices])
        batch_targets = torch.stack([self.target_buffer[i] for i in indices])
        batch_tasks = torch.tensor([self.task_buffer[i] for i in indices])
        
        return batch_data, batch_targets, batch_tasks
    
    def get_task_samples(self, task_id: int, num_samples: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get samples from a specific task"""
        task_indices = [i for i, tid in enumerate(self.task_buffer) if tid == task_id]
        
        if not task_indices:
            raise ValueError(f"No samples found for task {task_id}")
        
        if num_samples > 0:
            task_indices = np.random.choice(task_indices, size=min(num_samples, len(task_indices)), replace=False)
        
        batch_data = torch.stack([self.data_buffer[i] for i in task_indices])
        batch_targets = torch.stack([self.target_buffer[i] for i in task_indices])
        
        return batch_data, batch_targets
    
    def save_buffer(self, filepath: str):
        """Save replay buffer to disk"""
        buffer_state = {
            'data_buffer': self.data_buffer,
            'target_buffer': self.target_buffer,
            'task_buffer': self.task_buffer,
            'metadata_buffer': self.metadata_buffer,
            'current_size': self.current_size,
            'insert_index': self.insert_index
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(buffer_state, f)
            
        self.logger.info(f"Replay buffer saved to {filepath}")
    
    def load_buffer(self, filepath: str):
        """Load replay buffer from disk"""
        with open(filepath, 'rb') as f:
            buffer_state = pickle.load(f)
        
        self.data_buffer = buffer_state['data_buffer']
        self.target_buffer = buffer_state['target_buffer']
        self.task_buffer = buffer_state['task_buffer']
        self.metadata_buffer = buffer_state['metadata_buffer']
        self.current_size = buffer_state['current_size']
        self.insert_index = buffer_state['insert_index']
        
        self.logger.info(f"Replay buffer loaded from {filepath}")


class ContinualLearningTrainer:
    """
    Comprehensive trainer for continual learning scenarios
    
    Integrates multiple continual learning strategies and provides
    unified interface for multi-task infrared target detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        continual_strategy: ContinualLearningBase,
        replay_buffer: Optional[IntelligentReplayBuffer] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Base neural network model
            continual_strategy: Continual learning strategy (EWC, etc.)
            replay_buffer: Optional replay buffer for experience replay
            device: Training device
        """
        self.model = model.to(device)
        self.continual_strategy = continual_strategy
        self.replay_buffer = replay_buffer
        self.device = device
        
        self.task_history = []
        self.performance_history = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10,
        replay_ratio: float = 0.3
    ) -> Dict[str, float]:
        """
        Train on a new task using continual learning
        
        Args:
            task_id: Identifier for the current task
            train_loader: Training data for current task
            val_loader: Validation data for current task
            optimizer: Optimizer for training
            num_epochs: Number of training epochs
            replay_ratio: Ratio of replay samples in each batch
            
        Returns:
            Training metrics dictionary
        """
        self.logger.info(f"Starting training for task {task_id}")
        
        # Prepare continual learning strategy
        self.continual_strategy.before_task(task_id, train_loader)
        
        # Training loop
        epoch_metrics = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            self.model.train()
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                batch_size = data.size(0)
                
                # Prepare training batch with replay samples
                if self.replay_buffer and self.replay_buffer.current_size > 0:
                    replay_size = int(batch_size * replay_ratio)
                    if replay_size > 0:
                        replay_data, replay_targets, _ = self.replay_buffer.sample_batch(replay_size)
                        replay_data = replay_data.to(self.device)
                        replay_targets = replay_targets.to(self.device)
                        
                        # Combine current and replay data
                        combined_data = torch.cat([data, replay_data], dim=0)
                        combined_targets = torch.cat([targets, replay_targets], dim=0)
                    else:
                        combined_data, combined_targets = data, targets
                else:
                    combined_data, combined_targets = data, targets
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(combined_data)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Task-specific loss
                task_loss = F.cross_entropy(logits, combined_targets)
                
                # Continual learning regularization
                reg_loss = self.continual_strategy.regularization_loss(self.model)
                
                # Total loss
                total_loss = task_loss + reg_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Add samples to replay buffer
                if self.replay_buffer:
                    self.replay_buffer.add_samples(
                        data.cpu(), targets.cpu(), task_id, self.model
                    )
                
                # Track metrics
                epoch_loss += total_loss.item()
                
                # Compute accuracy
                with torch.no_grad():
                    _, predicted = torch.max(logits[:batch_size], 1)  # Only current task samples
                    correct = (predicted == targets).sum().item()
                    epoch_acc += correct / batch_size
                
                num_batches += 1
            
            # Validation
            val_metrics = self.evaluate_task(task_id, val_loader)
            
            # Log epoch results
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            
            self.logger.info(
                f"Task {task_id}, Epoch {epoch+1}/{num_epochs}: "
                f"Loss = {avg_loss:.4f}, Acc = {avg_acc:.4f}, "
                f"Val_Acc = {val_metrics['accuracy']:.4f}"
            )
            
            epoch_metrics.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_accuracy': avg_acc,
                'val_accuracy': val_metrics['accuracy']
            })
        
        # Finalize task training
        self.continual_strategy.after_task(task_id, self.model)
        self.task_history.append(task_id)
        
        # Store performance
        final_metrics = epoch_metrics[-1]
        self.performance_history[task_id].append(final_metrics)
        
        self.logger.info(f"Completed training for task {task_id}")
        
        return final_metrics
    
    def evaluate_task(self, task_id: int, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance on a specific task"""
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                num_batches += 1
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / num_batches
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct_samples': total_correct,
            'total_samples': total_samples
        }
    
    def evaluate_all_tasks(self, task_dataloaders: Dict[int, DataLoader]) -> Dict[int, Dict[str, float]]:
        """Evaluate model on all previously learned tasks"""
        all_results = {}
        
        for task_id, dataloader in task_dataloaders.items():
            if task_id in self.task_history:
                results = self.evaluate_task(task_id, dataloader)
                all_results[task_id] = results
                
                self.logger.info(
                    f"Task {task_id} evaluation: "
                    f"Accuracy = {results['accuracy']:.4f}, Loss = {results['loss']:.4f}"
                )
        
        return all_results
    
    def compute_forgetting_metrics(self, task_dataloaders: Dict[int, DataLoader]) -> Dict[str, float]:
        """
        Compute catastrophic forgetting metrics
        
        Returns:
            Dictionary with backward transfer, forward transfer, and average accuracy
        """
        if len(self.task_history) < 2:
            return {'backward_transfer': 0.0, 'forward_transfer': 0.0, 'average_accuracy': 0.0}
        
        # Evaluate current performance on all tasks
        current_performance = self.evaluate_all_tasks(task_dataloaders)
        
        # Compute metrics
        backward_transfer = 0.0  # How much performance dropped on old tasks
        forward_transfer = 0.0   # How much new tasks benefited from old knowledge
        total_accuracy = 0.0
        
        num_tasks = len(self.task_history)
        
        for i, task_id in enumerate(self.task_history):
            current_acc = current_performance[task_id]['accuracy']
            total_accuracy += current_acc
            
            # Backward transfer: compare to performance when task was first learned
            if len(self.performance_history[task_id]) > 1:
                initial_acc = self.performance_history[task_id][0]['val_accuracy']
                backward_transfer += current_acc - initial_acc
        
        # Normalize metrics
        backward_transfer /= max(1, num_tasks - 1)
        average_accuracy = total_accuracy / num_tasks
        
        return {
            'backward_transfer': backward_transfer,
            'forward_transfer': forward_transfer,  # Would need baseline for proper computation
            'average_accuracy': average_accuracy,
            'forgetting': -backward_transfer if backward_transfer < 0 else 0.0
        }
    
    def save_checkpoint(self, filepath: str):
        """Save complete trainer state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'task_history': self.task_history,
            'performance_history': dict(self.performance_history),
            'continual_strategy_state': getattr(self.continual_strategy, '__dict__', {})
        }
        
        torch.save(checkpoint, filepath)
        
        if self.replay_buffer:
            buffer_path = filepath.replace('.pth', '_replay_buffer.pkl')
            self.replay_buffer.save_buffer(buffer_path)
        
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load trainer state from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.task_history = checkpoint['task_history']
        self.performance_history = defaultdict(list, checkpoint['performance_history'])
        
        # Restore continual strategy state
        if 'continual_strategy_state' in checkpoint:
            for key, value in checkpoint['continual_strategy_state'].items():
                setattr(self.continual_strategy, key, value)
        
        if self.replay_buffer:
            buffer_path = filepath.replace('.pth', '_replay_buffer.pkl')
            try:
                self.replay_buffer.load_buffer(buffer_path)
            except FileNotFoundError:
                self.logger.warning(f"Replay buffer file not found: {buffer_path}")
        
        self.logger.info(f"Checkpoint loaded from {filepath}")


# Factory functions and utilities

def create_continual_learning_setup(
    base_model: nn.Module,
    strategy: str = 'ewc',
    strategy_params: Optional[Dict] = None,
    use_replay: bool = True,
    replay_params: Optional[Dict] = None
) -> Tuple[ContinualLearningBase, Optional[IntelligentReplayBuffer]]:
    """
    Factory function to create continual learning components
    
    Args:
        base_model: Base neural network
        strategy: Continual learning strategy ('ewc', 'progressive')
        strategy_params: Parameters for the strategy
        use_replay: Whether to use replay buffer
        replay_params: Parameters for replay buffer
        
    Returns:
        Tuple of (continual_strategy, replay_buffer)
    """
    strategy_params = strategy_params or {}
    replay_params = replay_params or {}
    
    # Create continual learning strategy
    if strategy == 'ewc':
        continual_strategy = ElasticWeightConsolidation(**strategy_params)
    else:
        raise ValueError(f"Unknown continual learning strategy: {strategy}")
    
    # Create replay buffer
    replay_buffer = None
    if use_replay:
        replay_buffer = IntelligentReplayBuffer(**replay_params)
    
    return continual_strategy, replay_buffer


# Export main components
__all__ = [
    'ContinualLearningBase',
    'ElasticWeightConsolidation',
    'ProgressiveNeuralNetwork',
    'IntelligentReplayBuffer',
    'ContinualLearningTrainer',
    'create_continual_learning_setup'
]
