"""
Meta-Learning for IRST Library

Implements state-of-the-art meta-learning algorithms for few-shot ISTD:
- MAML: Model-Agnostic Meta-Learning
- Prototypical Networks
- Relation Networks  
- Meta-SGD
- Reptile
- ANIL: Almost No Inner Loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging
import copy
from collections import OrderedDict
import random

logger = logging.getLogger(__name__)


class MetaLearner(nn.Module):
    """Base class for meta-learning algorithms"""
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, 
                 inner_steps: int = 5, meta_lr: float = 0.001):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        
        # Meta optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inner loop adaptation"""
        raise NotImplementedError
    
    def meta_update(self, meta_loss: torch.Tensor):
        """Meta update step"""
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()


class MAML(MetaLearner):
    """Model-Agnostic Meta-Learning"""
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.first_order = kwargs.get('first_order', False)
        
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """MAML inner loop"""
        # Clone model parameters
        fast_weights = OrderedDict()
        for name, param in self.model.named_parameters():
            fast_weights[name] = param.clone()
        
        # Inner loop updates
        for step in range(self.inner_steps):
            # Forward pass with current weights
            logits = self._forward_with_weights(support_x, fast_weights)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, fast_weights.values(), 
                create_graph=not self.first_order,
                retain_graph=True
            )
            
            # Update fast weights
            fast_weights = OrderedDict()
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                fast_weights[name] = param - self.inner_lr * grad
        
        # Query set evaluation
        query_logits = self._forward_with_weights(query_x, fast_weights)
        query_loss = F.cross_entropy(query_logits, query_y)
        query_acc = (query_logits.argmax(dim=1) == query_y).float().mean()
        
        return {
            'query_loss': query_loss,
            'query_acc': query_acc,
            'query_logits': query_logits
        }
    
    def _forward_with_weights(self, x: torch.Tensor, weights: OrderedDict) -> torch.Tensor:
        """Forward pass with custom weights"""
        # This is a simplified implementation
        # In practice, you'd need to properly handle the forward pass
        # with custom weights for your specific model architecture
        
        # For demonstration, assume a simple feedforward network
        out = x
        for name, weight in weights.items():
            if 'weight' in name and 'fc' in name:
                out = F.linear(out, weight)
            elif 'bias' in name and 'fc' in name:
                # Add bias (this is simplified)
                pass
        
        return out


class PrototypicalNetworks(MetaLearner):
    """Prototypical Networks for Few-shot Learning"""
    
    def __init__(self, model: nn.Module, distance_metric: str = 'euclidean', **kwargs):
        super().__init__(model, **kwargs)
        self.distance_metric = distance_metric
        
    def compute_distance(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Compute distance between query and support embeddings"""
        if self.distance_metric == 'euclidean':
            return torch.cdist(query, support, p=2)
        elif self.distance_metric == 'cosine':
            query_norm = F.normalize(query, dim=-1)
            support_norm = F.normalize(support, dim=-1)
            return 1 - torch.mm(query_norm, support_norm.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prototypical Networks adaptation"""
        # Encode support and query sets
        support_embeddings = self.model(support_x)
        query_embeddings = self.model(query_x)
        
        # Compute prototypes (class centroids)
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            mask = (support_y == label)
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute distances
        distances = self.compute_distance(query_embeddings, prototypes)
        
        # Convert to logits (negative distances)
        logits = -distances
        
        # Compute loss and accuracy
        query_loss = F.cross_entropy(logits, query_y)
        query_acc = (logits.argmax(dim=1) == query_y).float().mean()
        
        return {
            'query_loss': query_loss,
            'query_acc': query_acc,
            'query_logits': logits,
            'prototypes': prototypes
        }


class RelationNetwork(MetaLearner):
    """Relation Networks for Few-shot Learning"""
    
    def __init__(self, feature_model: nn.Module, relation_model: nn.Module, **kwargs):
        super().__init__(feature_model, **kwargs)
        self.feature_model = feature_model
        self.relation_model = relation_model
        
        # Combine both models for optimization
        self.model = nn.ModuleDict({
            'feature_model': feature_model,
            'relation_model': relation_model
        })
    
    def compute_relations(self, query_features: torch.Tensor, 
                         support_features: torch.Tensor) -> torch.Tensor:
        """Compute relation scores"""
        n_query = query_features.shape[0]
        n_support = support_features.shape[0]
        
        # Expand dimensions for pairwise comparison
        query_expanded = query_features.unsqueeze(1).expand(-1, n_support, -1)
        support_expanded = support_features.unsqueeze(0).expand(n_query, -1, -1)
        
        # Concatenate features
        relation_pairs = torch.cat([query_expanded, support_expanded], dim=-1)
        
        # Reshape for relation network
        relation_pairs = relation_pairs.view(-1, relation_pairs.shape[-1])
        
        # Compute relation scores
        relations = self.relation_model(relation_pairs)
        relations = relations.view(n_query, n_support)
        
        return relations
    
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Relation Networks adaptation"""
        # Extract features
        support_features = self.feature_model(support_x)
        query_features = self.feature_model(query_x)
        
        # Compute prototypes
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            mask = (support_y == label)
            prototype = support_features[mask].mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute relation scores
        relations = self.compute_relations(query_features, prototypes)
        
        # Compute loss and accuracy
        query_loss = F.cross_entropy(relations, query_y)
        query_acc = (relations.argmax(dim=1) == query_y).float().mean()
        
        return {
            'query_loss': query_loss,
            'query_acc': query_acc,
            'query_logits': relations,
            'prototypes': prototypes
        }


class MetaSGD(MAML):
    """Meta-SGD: Learning to Learn by Gradient Descent by Gradient Descent"""
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        # Learnable learning rates for each parameter
        self.alpha = nn.ParameterDict()
        for name, param in model.named_parameters():
            self.alpha[name.replace('.', '_')] = nn.Parameter(
                torch.ones_like(param) * self.inner_lr
            )
    
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Meta-SGD inner loop with learnable learning rates"""
        # Clone model parameters
        fast_weights = OrderedDict()
        for name, param in self.model.named_parameters():
            fast_weights[name] = param.clone()
        
        # Inner loop updates with learnable learning rates
        for step in range(self.inner_steps):
            # Forward pass
            logits = self._forward_with_weights(support_x, fast_weights)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, fast_weights.values(), 
                create_graph=True,
                retain_graph=True
            )
            
            # Update with learnable learning rates
            fast_weights = OrderedDict()
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                alpha_name = name.replace('.', '_')
                lr = self.alpha[alpha_name]
                fast_weights[name] = param - lr * grad
        
        # Query evaluation
        query_logits = self._forward_with_weights(query_x, fast_weights)
        query_loss = F.cross_entropy(query_logits, query_y)
        query_acc = (query_logits.argmax(dim=1) == query_y).float().mean()
        
        return {
            'query_loss': query_loss,
            'query_acc': query_acc,
            'query_logits': query_logits
        }


class Reptile(MetaLearner):
    """Reptile: A Scalable Meta-Learning Algorithm"""
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.meta_step_size = kwargs.get('meta_step_size', 1.0)
        
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Reptile inner loop"""
        # Store original parameters
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.clone()
        
        # Create optimizer for inner loop
        inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        # Inner loop training
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            logits = self.model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            inner_optimizer.step()
        
        # Evaluate on query set
        query_logits = self.model(query_x)
        query_loss = F.cross_entropy(query_logits, query_y)
        query_acc = (query_logits.argmax(dim=1) == query_y).float().mean()
        
        # Compute meta-gradient (difference between adapted and original params)
        meta_grads = OrderedDict()
        for name, param in self.model.named_parameters():
            meta_grads[name] = original_params[name] - param
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        # Apply meta-update manually
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data -= self.meta_step_size * meta_grads[name]
        
        return {
            'query_loss': query_loss,
            'query_acc': query_acc,
            'query_logits': query_logits,
            'meta_grads': meta_grads
        }


class ANIL(MAML):
    """ANIL: Almost No Inner Loop"""
    
    def __init__(self, model: nn.Module, head_only: bool = True, **kwargs):
        super().__init__(model, **kwargs)
        self.head_only = head_only
        
        # Identify which parameters to adapt
        self.adaptation_params = []
        self.feature_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'head' in name or 'fc' in name:
                self.adaptation_params.append((name, param))
            else:
                self.feature_params.append((name, param))
    
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ANIL inner loop - only adapt head parameters"""
        if self.head_only:
            # Only adapt head parameters
            adapt_params = self.adaptation_params
        else:
            # Adapt all parameters (same as MAML)
            adapt_params = list(self.model.named_parameters())
        
        # Clone adaptation parameters
        fast_weights = OrderedDict()
        for name, param in adapt_params:
            fast_weights[name] = param.clone()
        
        # Inner loop updates
        for step in range(self.inner_steps):
            # Forward pass
            logits = self.model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients only for adaptation parameters
            adapt_tensors = [fast_weights[name] for name, _ in adapt_params]
            grads = torch.autograd.grad(
                loss, adapt_tensors,
                create_graph=True,
                retain_graph=True
            )
            
            # Update adaptation parameters
            for (name, _), grad in zip(adapt_params, grads):
                fast_weights[name] = fast_weights[name] - self.inner_lr * grad
        
        # Update model with adapted parameters
        original_state = {}
        for name, param in adapt_params:
            original_state[name] = param.data.clone()
            param.data = fast_weights[name]
        
        # Query evaluation
        query_logits = self.model(query_x)
        query_loss = F.cross_entropy(query_logits, query_y)
        query_acc = (query_logits.argmax(dim=1) == query_y).float().mean()
        
        # Restore original parameters
        for name, param in adapt_params:
            param.data = original_state[name]
        
        return {
            'query_loss': query_loss,
            'query_acc': query_acc,
            'query_logits': query_logits
        }


class FewShotDataLoader:
    """Data loader for few-shot learning tasks"""
    
    def __init__(self, dataset, n_way: int = 5, k_shot: int = 1, 
                 q_query: int = 15, num_tasks: int = 1000):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks
        
        # Group data by class
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
    
    def sample_task(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a few-shot learning task"""
        # Sample classes
        task_classes = random.sample(self.classes, self.n_way)
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for class_idx, class_label in enumerate(task_classes):
            # Sample examples from this class
            class_indices = self.class_to_indices[class_label]
            sampled_indices = random.sample(class_indices, self.k_shot + self.q_query)
            
            # Split into support and query
            support_indices = sampled_indices[:self.k_shot]
            query_indices = sampled_indices[self.k_shot:]
            
            # Add to support set
            for idx in support_indices:
                x, _ = self.dataset[idx]
                support_x.append(x)
                support_y.append(class_idx)
            
            # Add to query set
            for idx in query_indices:
                x, _ = self.dataset[idx]
                query_x.append(x)
                query_y.append(class_idx)
        
        # Convert to tensors
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)
        
        return support_x, support_y, query_x, query_y
    
    def __iter__(self):
        for _ in range(self.num_tasks):
            yield self.sample_task()
    
    def __len__(self):
        return self.num_tasks


class MetaLearningTrainer:
    """Trainer for meta-learning algorithms"""
    
    def __init__(self, meta_learner: MetaLearner, train_loader: FewShotDataLoader,
                 val_loader: FewShotDataLoader, device: str = 'cuda'):
        self.meta_learner = meta_learner
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.meta_learner.to(device)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.meta_learner.train()
        
        epoch_losses = []
        epoch_accs = []
        
        for task_idx, (support_x, support_y, query_x, query_y) in enumerate(self.train_loader):
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Inner loop adaptation
            results = self.meta_learner.inner_loop(support_x, support_y, query_x, query_y)
            
            # Meta update
            if not isinstance(self.meta_learner, Reptile):
                self.meta_learner.meta_update(results['query_loss'])
            
            epoch_losses.append(results['query_loss'].item())
            epoch_accs.append(results['query_acc'].item())
        
        return {
            'train_loss': np.mean(epoch_losses),
            'train_acc': np.mean(epoch_accs)
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate meta-learner"""
        self.meta_learner.eval()
        
        val_losses = []
        val_accs = []
        
        with torch.no_grad():
            for support_x, support_y, query_x, query_y in self.val_loader:
                # Move to device
                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)
                
                # Inner loop adaptation
                results = self.meta_learner.inner_loop(support_x, support_y, query_x, query_y)
                
                val_losses.append(results['query_loss'].item())
                val_accs.append(results['query_acc'].item())
        
        return {
            'val_loss': np.mean(val_losses),
            'val_acc': np.mean(val_accs)
        }
    
    def train(self, num_epochs: int = 100) -> Dict[str, List[float]]:
        """Train meta-learner"""
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            logger.info(f"Epoch {epoch}/{num_epochs}: "
                       f"Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Train Acc: {train_metrics['train_acc']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}, "
                       f"Val Acc: {val_metrics['val_acc']:.4f}")
            
            # Store history
            for key, value in train_metrics.items():
                history[key].append(value)
            for key, value in val_metrics.items():
                history[key].append(value)
        
        return history


class MetaLearning:
    """Main meta-learning interface"""
    
    def __init__(self, algorithm: str = 'maml', model: nn.Module = None, **kwargs):
        self.algorithm = algorithm
        self.model = model or self._create_default_model()
        
        # Initialize meta-learner
        if algorithm == 'maml':
            self.meta_learner = MAML(self.model, **kwargs)
        elif algorithm == 'prototypical':
            self.meta_learner = PrototypicalNetworks(self.model, **kwargs)
        elif algorithm == 'relation':
            # Relation networks need separate feature and relation models
            feature_model = self.model
            relation_model = self._create_relation_model(**kwargs)
            self.meta_learner = RelationNetwork(feature_model, relation_model, **kwargs)
        elif algorithm == 'meta_sgd':
            self.meta_learner = MetaSGD(self.model, **kwargs)
        elif algorithm == 'reptile':
            self.meta_learner = Reptile(self.model, **kwargs)
        elif algorithm == 'anil':
            self.meta_learner = ANIL(self.model, **kwargs)
        else:
            raise ValueError(f"Unknown meta-learning algorithm: {algorithm}")
    
    def _create_default_model(self) -> nn.Module:
        """Create default model for meta-learning"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5-way classification
        )
    
    def _create_relation_model(self, **kwargs) -> nn.Module:
        """Create relation model for relation networks"""
        feature_dim = kwargs.get('feature_dim', 64)
        return nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def train(self, train_dataset, val_dataset, n_way: int = 5, k_shot: int = 1,
              q_query: int = 15, num_tasks: int = 1000, epochs: int = 100) -> Dict[str, List[float]]:
        """Train meta-learner"""
        # Create few-shot data loaders
        train_loader = FewShotDataLoader(
            train_dataset, n_way, k_shot, q_query, num_tasks
        )
        val_loader = FewShotDataLoader(
            val_dataset, n_way, k_shot, q_query, num_tasks // 10
        )
        
        # Create trainer
        trainer = MetaLearningTrainer(self.meta_learner, train_loader, val_loader)
        
        # Train
        return trainer.train(epochs)
    
    def evaluate(self, test_dataset, n_way: int = 5, k_shot: int = 1,
                q_query: int = 15, num_tasks: int = 100) -> Dict[str, float]:
        """Evaluate meta-learner"""
        test_loader = FewShotDataLoader(
            test_dataset, n_way, k_shot, q_query, num_tasks
        )
        
        trainer = MetaLearningTrainer(self.meta_learner, test_loader, test_loader)
        return trainer.validate()
    
    def adapt_to_task(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """Adapt model to new task"""
        self.meta_learner.eval()
        
        with torch.no_grad():
            # Create dummy query set for adaptation
            query_x = support_x[:1]  # Use first support sample
            query_y = support_y[:1]
            
            # Perform adaptation
            self.meta_learner.inner_loop(support_x, support_y, query_x, query_y)
        
        return self.model
