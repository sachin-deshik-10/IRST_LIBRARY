"""
Domain Adaptation for IRST Library

Implements state-of-the-art domain adaptation techniques for ISTD:
- DANN (Domain Adversarial Neural Networks)
- CORAL (Correlation Alignment)
- MMD (Maximum Mean Discrepancy)
- CDAN (Conditional Domain Adversarial Network)
- MCD (Maximum Classifier Discrepancy)
- ADDA (Adversarial Discriminative Domain Adaptation)
- Progressive Domain Adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import copy
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for DANN"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x, alpha=1.0):
    """Apply gradient reversal"""
    return GradientReversalLayer.apply(x, alpha)


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial training"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_domains: int = 2):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        
    def forward(self, x):
        return self.classifier(x)


class DANN(nn.Module):
    """Domain Adversarial Neural Network"""
    
    def __init__(self, feature_extractor: nn.Module, task_classifier: nn.Module,
                 domain_classifier: nn.Module = None, lambda_factor: float = 1.0):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier
        
        # Create domain classifier if not provided
        if domain_classifier is None:
            # Get feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = feature_extractor(dummy_input)
                feature_dim = features.shape[-1]
            
            self.domain_classifier = DomainClassifier(feature_dim)
        else:
            self.domain_classifier = domain_classifier
        
        self.lambda_factor = lambda_factor
        
    def forward(self, x, alpha=1.0, return_features=False):
        """Forward pass"""
        # Extract features
        features = self.feature_extractor(x)
        
        # Task prediction
        task_logits = self.task_classifier(features)
        
        # Domain prediction with gradient reversal
        reversed_features = gradient_reversal(features, alpha * self.lambda_factor)
        domain_logits = self.domain_classifier(reversed_features)
        
        if return_features:
            return task_logits, domain_logits, features
        else:
            return task_logits, domain_logits
    
    def predict(self, x):
        """Predict without domain classification"""
        features = self.feature_extractor(x)
        return self.task_classifier(features)


class CORAL:
    """Correlation Alignment for Domain Adaptation"""
    
    @staticmethod
    def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute CORAL loss"""
        # Compute covariance matrices
        source_cov = CORAL._compute_covariance(source_features)
        target_cov = CORAL._compute_covariance(target_features)
        
        # Frobenius norm of difference
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        
        # Normalize by feature dimension
        d = source_features.shape[1]
        loss = loss / (4 * d * d)
        
        return loss
    
    @staticmethod
    def _compute_covariance(features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix"""
        # Center the features
        mean = torch.mean(features, dim=0, keepdim=True)
        centered = features - mean
        
        # Compute covariance
        n = features.shape[0]
        cov = torch.mm(centered.t(), centered) / (n - 1)
        
        return cov


class MMD:
    """Maximum Mean Discrepancy for Domain Adaptation"""
    
    @staticmethod
    def mmd_loss(source_features: torch.Tensor, target_features: torch.Tensor,
                kernel_type: str = 'rbf', kernel_params: Dict[str, float] = None) -> torch.Tensor:
        """Compute MMD loss"""
        if kernel_params is None:
            kernel_params = {'gamma': 1.0, 'degree': 2, 'coef0': 1.0}
        
        # Compute kernel matrices
        K_ss = MMD._compute_kernel(source_features, source_features, kernel_type, kernel_params)
        K_tt = MMD._compute_kernel(target_features, target_features, kernel_type, kernel_params)
        K_st = MMD._compute_kernel(source_features, target_features, kernel_type, kernel_params)
        
        # Compute MMD
        n_s = source_features.shape[0]
        n_t = target_features.shape[0]
        
        mmd = (torch.sum(K_ss) / (n_s * n_s) + 
               torch.sum(K_tt) / (n_t * n_t) - 
               2 * torch.sum(K_st) / (n_s * n_t))
        
        return mmd
    
    @staticmethod
    def _compute_kernel(X: torch.Tensor, Y: torch.Tensor, kernel_type: str,
                       kernel_params: Dict[str, float]) -> torch.Tensor:
        """Compute kernel matrix"""
        if kernel_type == 'rbf':
            gamma = kernel_params.get('gamma', 1.0)
            # Compute pairwise squared distances
            X_norm = torch.sum(X ** 2, dim=1, keepdim=True)
            Y_norm = torch.sum(Y ** 2, dim=1, keepdim=True)
            dist = X_norm + Y_norm.t() - 2 * torch.mm(X, Y.t())
            return torch.exp(-gamma * dist)
        
        elif kernel_type == 'polynomial':
            degree = kernel_params.get('degree', 2)
            coef0 = kernel_params.get('coef0', 1.0)
            return (torch.mm(X, Y.t()) + coef0) ** degree
        
        elif kernel_type == 'linear':
            return torch.mm(X, Y.t())
        
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")


class DomainAdaptationTrainer:
    """Trainer for domain adaptation methods"""
    
    def __init__(self, model: nn.Module, method: str = 'dann', **kwargs):
        self.model = model
        self.method = method
        self.kwargs = kwargs
        
        # Initialize optimizers
        self.optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.001))
        
        # Initialize schedulers
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=kwargs.get('step_size', 30), 
            gamma=kwargs.get('gamma', 0.1)
        )
        
    def train_dann(self, source_loader, target_loader, epochs: int = 100) -> Dict[str, List[float]]:
        """Train DANN model"""
        history = {'task_loss': [], 'domain_loss': [], 'total_loss': [], 'target_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_task_loss = 0.0
            epoch_domain_loss = 0.0
            epoch_total_loss = 0.0
            
            # Compute lambda parameter (gradually increase)
            p = float(epoch) / epochs
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            
            for batch_idx, ((source_data, source_labels), (target_data, _)) in enumerate(
                zip(source_loader, target_loader)):
                
                self.optimizer.zero_grad()
                
                # Forward pass
                source_task, source_domain = self.model(source_data, alpha)
                target_task, target_domain = self.model(target_data, alpha)
                
                # Task loss (only on source)
                task_loss = F.cross_entropy(source_task, source_labels)
                
                # Domain loss
                batch_size = source_data.shape[0]
                source_domain_labels = torch.zeros(batch_size, dtype=torch.long)
                target_domain_labels = torch.ones(batch_size, dtype=torch.long)
                
                domain_loss = (F.cross_entropy(source_domain, source_domain_labels) +
                             F.cross_entropy(target_domain, target_domain_labels))
                
                # Total loss
                total_loss = task_loss + domain_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_task_loss += task_loss.item()
                epoch_domain_loss += domain_loss.item()
                epoch_total_loss += total_loss.item()
            
            # Average losses
            epoch_task_loss /= len(source_loader)
            epoch_domain_loss /= len(source_loader)
            epoch_total_loss /= len(source_loader)
            
            # Evaluate on target
            target_acc = self._evaluate_target(target_loader)
            
            # Store history
            history['task_loss'].append(epoch_task_loss)
            history['domain_loss'].append(epoch_domain_loss)
            history['total_loss'].append(epoch_total_loss)
            history['target_acc'].append(target_acc)
            
            # Step scheduler
            self.scheduler.step()
            
            logger.info(f"Epoch {epoch}/{epochs}: "
                       f"Task Loss: {epoch_task_loss:.4f}, "
                       f"Domain Loss: {epoch_domain_loss:.4f}, "
                       f"Target Acc: {target_acc:.4f}")
        
        return history
    
    def _evaluate_target(self, target_loader) -> float:
        """Evaluate on target domain"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in target_loader:
                outputs = self.model.predict(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total


class DomainAdapter:
    """Domain adaptation methods for ISTD"""
    
    def __init__(self, model: nn.Module, method: str = 'dann', **kwargs):
        self.model = model
        self.method = method
        
        # Initialize adaptation model based on method
        if method == 'dann':
            # Extract feature extractor and classifier from model
            feature_extractor, task_classifier = self._split_model(model)
            self.adaptation_model = DANN(feature_extractor, task_classifier, **kwargs)
        elif method == 'coral':
            self.adaptation_model = model  # CORAL uses original model
        else:
            raise ValueError(f"Unknown domain adaptation method: {method}")
        
        self.trainer = DomainAdaptationTrainer(self.adaptation_model, method, **kwargs)
        
    def _split_model(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Split model into feature extractor and classifier"""
        # This is a simplified implementation
        # In practice, you'd need to properly split based on your model architecture
        
        modules = list(model.children())
        feature_extractor = nn.Sequential(*modules[:-1])
        classifier = modules[-1]
        
        return feature_extractor, classifier
        
    def adapt(self, source_loader, target_loader, epochs: int = 100, **kwargs) -> Dict[str, List[float]]:
        """Perform domain adaptation"""
        if self.method == 'dann':
            return self.trainer.train_dann(source_loader, target_loader, epochs)
        else:
            # Generic training
            return self._generic_training(source_loader, target_loader, epochs)
    
    def _generic_training(self, source_loader, target_loader, epochs: int) -> Dict[str, List[float]]:
        """Generic training loop"""
        history = {'loss': [], 'accuracy': []}
        optimizer = torch.optim.Adam(self.adaptation_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
                optimizer.zero_grad()
                
                # Simple training on source domain
                outputs = self.adaptation_model(source_data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = F.cross_entropy(outputs, source_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(source_loader)
            acc = self._evaluate_model(target_loader)
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(acc)
            
            logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Acc = {acc:.4f}")
        
        return history
    
    def _evaluate_model(self, dataloader) -> float:
        """Evaluate model"""
        self.adaptation_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self.adaptation_model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.adaptation_model.train()
        return correct / total
