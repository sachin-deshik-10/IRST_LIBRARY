"""
Active Learning for IRST Library

This module implements advanced active learning strategies for efficient ISTD training.
It includes uncertainty sampling, query-by-committee, expected model change,
diversity-based sampling, and ensemble-based active learning methods.

Key Features:
- Uncertainty-based sampling (entropy, margin, least confidence)
- Query-by-committee with ensemble disagreement
- Expected model change and gradient-based selection
- Diversity-aware sampling with clustering
- Multi-criteria active learning with Pareto optimization
- Continual active learning for streaming data
- Budget-aware sample selection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import logging
from dataclasses import dataclass, field
from enum import Enum
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Active learning sampling strategies"""
    UNCERTAINTY = "uncertainty"
    MARGIN = "margin"
    ENTROPY = "entropy"
    COMMITTEE = "committee"
    EXPECTED_CHANGE = "expected_change"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"
    PARETO = "pareto"


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning"""
    strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY
    batch_size: int = 32
    budget: int = 1000
    uncertainty_threshold: float = 0.1
    diversity_weight: float = 0.3
    committee_size: int = 5
    clustering_method: str = "kmeans"
    feature_extractor: Optional[str] = None
    temperature: float = 1.0
    gradient_embedding_dim: int = 512
    pareto_objectives: List[str] = field(default_factory=lambda: ["uncertainty", "diversity"])
    continual_learning: bool = False
    memory_size: int = 10000
    

class UncertaintySampler:
    """Uncertainty-based active learning sampler"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        
    def compute_uncertainty(self, predictions: torch.Tensor, method: str = "entropy") -> torch.Tensor:
        """Compute uncertainty scores for predictions"""
        if method == "entropy":
            # Shannon entropy
            probs = F.softmax(predictions, dim=-1)
            log_probs = F.log_softmax(predictions, dim=-1)
            uncertainty = -(probs * log_probs).sum(dim=-1)
        elif method == "margin":
            # Margin sampling (difference between top two predictions)
            probs = F.softmax(predictions, dim=-1)
            sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
            uncertainty = 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        elif method == "least_confidence":
            # Least confidence sampling
            probs = F.softmax(predictions, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            uncertainty = 1.0 - max_probs
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
        
        return uncertainty
    
    def select_samples(self, model: nn.Module, unlabeled_data: torch.Tensor, 
                      batch_size: int) -> List[int]:
        """Select samples based on uncertainty"""
        model.eval()
        with torch.no_grad():
            predictions = model(unlabeled_data)
            uncertainties = self.compute_uncertainty(predictions, 
                                                   self.config.strategy.value)
        
        # Select top uncertain samples
        _, indices = torch.topk(uncertainties, batch_size)
        return indices.tolist()


class CommitteeSampler:
    """Query-by-committee active learning sampler"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.committee = []
        
    def train_committee(self, base_model: nn.Module, training_data: torch.Tensor,
                       training_labels: torch.Tensor):
        """Train committee of models with different initializations"""
        self.committee = []
        
        for i in range(self.config.committee_size):
            # Create model copy with different initialization
            model = type(base_model)(**base_model.config)
            model.load_state_dict(base_model.state_dict())
            
            # Add noise to weights for diversity
            with torch.no_grad():
                for param in model.parameters():
                    param += torch.randn_like(param) * 0.01
            
            # Fine-tune on bootstrap sample
            bootstrap_indices = torch.randint(0, len(training_data), 
                                            (len(training_data),))
            bootstrap_data = training_data[bootstrap_indices]
            bootstrap_labels = training_labels[bootstrap_indices]
            
            # Simple training loop (in practice, would use proper training)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(bootstrap_data)
                loss = F.cross_entropy(outputs, bootstrap_labels)
                loss.backward()
                optimizer.step()
            
            self.committee.append(model)
    
    def compute_disagreement(self, unlabeled_data: torch.Tensor) -> torch.Tensor:
        """Compute disagreement among committee members"""
        committee_predictions = []
        
        for model in self.committee:
            model.eval()
            with torch.no_grad():
                predictions = F.softmax(model(unlabeled_data), dim=-1)
                committee_predictions.append(predictions)
        
        # Stack predictions and compute variance
        committee_predictions = torch.stack(committee_predictions)
        mean_predictions = committee_predictions.mean(dim=0)
        
        # Vote entropy (disagreement measure)
        vote_entropy = -(mean_predictions * torch.log(mean_predictions + 1e-8)).sum(dim=-1)
        
        return vote_entropy
    
    def select_samples(self, unlabeled_data: torch.Tensor, batch_size: int) -> List[int]:
        """Select samples based on committee disagreement"""
        disagreements = self.compute_disagreement(unlabeled_data)
        _, indices = torch.topk(disagreements, batch_size)
        return indices.tolist()


class DiversitySampler:
    """Diversity-based active learning sampler"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        
    def extract_features(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Extract features from model for diversity computation"""
        model.eval()
        with torch.no_grad():
            # Get features from penultimate layer
            features = model.feature_extractor(data)
            return features.flatten(start_dim=1)
    
    def compute_diversity_scores(self, features: torch.Tensor, 
                                selected_indices: List[int] = None) -> torch.Tensor:
        """Compute diversity scores based on feature distances"""
        if selected_indices is None:
            # Use k-means clustering for diversity
            n_clusters = min(self.config.batch_size, len(features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features.cpu().numpy())
            
            # Compute distances to cluster centers
            distances = torch.tensor([
                np.linalg.norm(features[i].cpu().numpy() - kmeans.cluster_centers_[cluster_labels[i]])
                for i in range(len(features))
            ])
            
            return distances
        else:
            # Compute minimum distance to already selected samples
            selected_features = features[selected_indices]
            distances = torch.cdist(features, selected_features)
            min_distances, _ = torch.min(distances, dim=1)
            return min_distances
    
    def select_samples(self, model: nn.Module, unlabeled_data: torch.Tensor,
                      batch_size: int, selected_indices: List[int] = None) -> List[int]:
        """Select diverse samples"""
        features = self.extract_features(model, unlabeled_data)
        diversity_scores = self.compute_diversity_scores(features, selected_indices)
        
        _, indices = torch.topk(diversity_scores, batch_size)
        return indices.tolist()


class HybridSampler:
    """Hybrid active learning combining multiple strategies"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.uncertainty_sampler = UncertaintySampler(config)
        self.diversity_sampler = DiversitySampler(config)
        
    def select_samples(self, model: nn.Module, unlabeled_data: torch.Tensor,
                      batch_size: int) -> List[int]:
        """Select samples using hybrid strategy"""
        # Get uncertainty scores
        model.eval()
        with torch.no_grad():
            predictions = model(unlabeled_data)
            uncertainty_scores = self.uncertainty_sampler.compute_uncertainty(predictions)
        
        # Get diversity scores
        features = self.diversity_sampler.extract_features(model, unlabeled_data)
        diversity_scores = self.diversity_sampler.compute_diversity_scores(features)
        
        # Combine scores
        uncertainty_weight = 1.0 - self.config.diversity_weight
        combined_scores = (uncertainty_weight * uncertainty_scores + 
                          self.config.diversity_weight * diversity_scores)
        
        _, indices = torch.topk(combined_scores, batch_size)
        return indices.tolist()


class ExpectedModelChangeSampler:
    """Expected model change active learning sampler"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        
    def compute_gradient_embeddings(self, model: nn.Module, data: torch.Tensor,
                                  pseudo_labels: torch.Tensor) -> torch.Tensor:
        """Compute gradient embeddings for expected model change"""
        model.train()
        embeddings = []
        
        for i in range(len(data)):
            model.zero_grad()
            
            # Forward pass
            output = model(data[i:i+1])
            loss = F.cross_entropy(output, pseudo_labels[i:i+1])
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            grad_vector = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vector.append(param.grad.flatten())
            
            if grad_vector:
                embedding = torch.cat(grad_vector)
                if len(embedding) > self.config.gradient_embedding_dim:
                    # Dimensionality reduction
                    embedding = embedding[:self.config.gradient_embedding_dim]
                embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def select_samples(self, model: nn.Module, unlabeled_data: torch.Tensor,
                      batch_size: int) -> List[int]:
        """Select samples based on expected model change"""
        # Generate pseudo-labels
        model.eval()
        with torch.no_grad():
            predictions = model(unlabeled_data)
            pseudo_labels = torch.argmax(predictions, dim=-1)
        
        # Compute gradient embeddings
        gradient_embeddings = self.compute_gradient_embeddings(
            model, unlabeled_data, pseudo_labels
        )
        
        # Compute gradient magnitudes as proxy for model change
        gradient_magnitudes = torch.norm(gradient_embeddings, dim=-1)
        
        _, indices = torch.topk(gradient_magnitudes, batch_size)
        return indices.tolist()


class ParetoActiveLearner:
    """Multi-objective active learning with Pareto optimization"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.uncertainty_sampler = UncertaintySampler(config)
        self.diversity_sampler = DiversitySampler(config)
        
    def compute_objectives(self, model: nn.Module, unlabeled_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute multiple objectives for Pareto optimization"""
        objectives = {}
        
        if "uncertainty" in self.config.pareto_objectives:
            model.eval()
            with torch.no_grad():
                predictions = model(unlabeled_data)
                objectives["uncertainty"] = self.uncertainty_sampler.compute_uncertainty(predictions)
        
        if "diversity" in self.config.pareto_objectives:
            features = self.diversity_sampler.extract_features(model, unlabeled_data)
            objectives["diversity"] = self.diversity_sampler.compute_diversity_scores(features)
        
        return objectives
    
    def find_pareto_front(self, objectives: Dict[str, torch.Tensor]) -> List[int]:
        """Find Pareto front for multi-objective optimization"""
        # Convert to numpy for easier manipulation
        objective_matrix = torch.stack(list(objectives.values())).T.cpu().numpy()
        
        # Find Pareto front
        pareto_indices = []
        n_samples = len(objective_matrix)
        
        for i in range(n_samples):
            is_pareto = True
            for j in range(n_samples):
                if i != j:
                    # Check if j dominates i
                    if all(objective_matrix[j] >= objective_matrix[i]) and \
                       any(objective_matrix[j] > objective_matrix[i]):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def select_samples(self, model: nn.Module, unlabeled_data: torch.Tensor,
                      batch_size: int) -> List[int]:
        """Select samples using Pareto optimization"""
        objectives = self.compute_objectives(model, unlabeled_data)
        pareto_indices = self.find_pareto_front(objectives)
        
        # If Pareto front has fewer samples than needed, add more based on distance
        if len(pareto_indices) >= batch_size:
            return pareto_indices[:batch_size]
        else:
            # Add remaining samples based on combined objectives
            remaining_indices = [i for i in range(len(unlabeled_data)) 
                               if i not in pareto_indices]
            
            # Compute combined scores for remaining samples
            combined_scores = torch.zeros(len(remaining_indices))
            for obj_name, obj_scores in objectives.items():
                combined_scores += obj_scores[remaining_indices]
            
            _, top_indices = torch.topk(combined_scores, 
                                      batch_size - len(pareto_indices))
            additional_indices = [remaining_indices[i] for i in top_indices.tolist()]
            
            return pareto_indices + additional_indices


class ActiveLearner:
    """Main active learning interface for ISTD"""
    
    def __init__(self, config: ActiveLearningConfig = None):
        self.config = config or ActiveLearningConfig()
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.performance_history = []
        
        # Initialize samplers
        self._init_samplers()
        
        # Continual learning memory
        if self.config.continual_learning:
            self.memory_buffer = []
            
        logger.info(f"Initialized ActiveLearner with strategy: {self.config.strategy}")
    
    def _init_samplers(self):
        """Initialize sampling strategies"""
        self.samplers = {
            SamplingStrategy.UNCERTAINTY: UncertaintySampler(self.config),
            SamplingStrategy.MARGIN: UncertaintySampler(self.config),
            SamplingStrategy.ENTROPY: UncertaintySampler(self.config),
            SamplingStrategy.COMMITTEE: CommitteeSampler(self.config),
            SamplingStrategy.DIVERSITY: DiversitySampler(self.config),
            SamplingStrategy.HYBRID: HybridSampler(self.config),
            SamplingStrategy.EXPECTED_CHANGE: ExpectedModelChangeSampler(self.config),
            SamplingStrategy.PARETO: ParetoActiveLearner(self.config)
        }
    
    def initialize_pool(self, total_samples: int, initial_labeled: int = 100):
        """Initialize labeled/unlabeled pools"""
        self.labeled_indices = list(range(initial_labeled))
        self.unlabeled_indices = list(range(initial_labeled, total_samples))
        
        logger.info(f"Initialized with {len(self.labeled_indices)} labeled and "
                   f"{len(self.unlabeled_indices)} unlabeled samples")
    
    def select_samples(self, model: nn.Module, dataset: torch.utils.data.Dataset,
                      batch_size: int = None) -> List[int]:
        """Select most informative samples for annotation"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if not self.unlabeled_indices:
            logger.warning("No unlabeled samples available")
            return []
        
        # Get unlabeled data
        unlabeled_data = torch.stack([dataset[i][0] for i in self.unlabeled_indices])
        
        # Select sampler and get indices
        sampler = self.samplers[self.config.strategy]
        
        if self.config.strategy == SamplingStrategy.COMMITTEE:
            # Committee needs training first
            labeled_data = torch.stack([dataset[i][0] for i in self.labeled_indices])
            labeled_labels = torch.tensor([dataset[i][1] for i in self.labeled_indices])
            sampler.train_committee(model, labeled_data, labeled_labels)
        
        selected_relative_indices = sampler.select_samples(model, unlabeled_data, batch_size)
        
        # Convert relative indices to absolute indices
        selected_indices = [self.unlabeled_indices[i] for i in selected_relative_indices]
        
        # Update pools
        self.labeled_indices.extend(selected_indices)
        for idx in selected_indices:
            self.unlabeled_indices.remove(idx)
        
        # Update memory buffer for continual learning
        if self.config.continual_learning:
            self._update_memory_buffer(selected_indices, dataset)
        
        logger.info(f"Selected {len(selected_indices)} samples for annotation")
        return selected_indices
    
    def _update_memory_buffer(self, new_indices: List[int], dataset: torch.utils.data.Dataset):
        """Update memory buffer for continual learning"""
        for idx in new_indices:
            self.memory_buffer.append((dataset[idx][0], dataset[idx][1]))
        
        # Keep only recent samples if buffer exceeds size
        if len(self.memory_buffer) > self.config.memory_size:
            self.memory_buffer = self.memory_buffer[-self.config.memory_size:]
    
    def evaluate_selection_quality(self, model: nn.Module, selected_indices: List[int],
                                 dataset: torch.utils.data.Dataset) -> Dict[str, float]:
        """Evaluate quality of selected samples"""
        if not selected_indices:
            return {}
        
        # Get selected data
        selected_data = torch.stack([dataset[i][0] for i in selected_indices])
        
        # Compute diversity metrics
        model.eval()
        with torch.no_grad():
            features = model.feature_extractor(selected_data)
            features_flat = features.flatten(start_dim=1)
            
            # Compute pairwise distances
            distances = torch.cdist(features_flat, features_flat)
            avg_distance = distances.mean().item()
            min_distance = distances[distances > 0].min().item()
            
            # Compute predictions for uncertainty
            predictions = model(selected_data)
            uncertainties = F.softmax(predictions, dim=-1)
            entropy = -(uncertainties * torch.log(uncertainties + 1e-8)).sum(dim=-1)
            avg_uncertainty = entropy.mean().item()
        
        metrics = {
            "avg_pairwise_distance": avg_distance,
            "min_pairwise_distance": min_distance,
            "avg_uncertainty": avg_uncertainty,
            "selection_size": len(selected_indices)
        }
        
        return metrics
    
    def update_performance(self, accuracy: float, loss: float):
        """Update performance history"""
        self.performance_history.append({
            "accuracy": accuracy,
            "loss": loss,
            "labeled_samples": len(self.labeled_indices),
            "unlabeled_samples": len(self.unlabeled_indices)
        })
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning curve data"""
        if not self.performance_history:
            return {}
        
        return {
            "labeled_samples": [h["labeled_samples"] for h in self.performance_history],
            "accuracy": [h["accuracy"] for h in self.performance_history],
            "loss": [h["loss"] for h in self.performance_history]
        }
    
    def save_state(self, filepath: str):
        """Save active learning state"""
        state = {
            "config": self.config,
            "labeled_indices": self.labeled_indices,
            "unlabeled_indices": self.unlabeled_indices,
            "performance_history": self.performance_history,
            "memory_buffer": self.memory_buffer if self.config.continual_learning else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved active learning state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load active learning state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.labeled_indices = state["labeled_indices"]
        self.unlabeled_indices = state["unlabeled_indices"]
        self.performance_history = state["performance_history"]
        
        if state["memory_buffer"] is not None:
            self.memory_buffer = state["memory_buffer"]
        
        # Reinitialize samplers
        self._init_samplers()
        
        logger.info(f"Loaded active learning state from {filepath}")


class StreamingActiveLearner(ActiveLearner):
    """Active learner for streaming/online scenarios"""
    
    def __init__(self, config: ActiveLearningConfig = None):
        super().__init__(config)
        self.stream_buffer = []
        self.annotation_budget_used = 0
        
    def process_stream_batch(self, model: nn.Module, new_data: torch.Tensor,
                           max_annotations: int = None) -> Tuple[List[int], torch.Tensor]:
        """Process new streaming data batch"""
        if max_annotations is None:
            max_annotations = min(self.config.batch_size, 
                                len(new_data), 
                                self.config.budget - self.annotation_budget_used)
        
        if max_annotations <= 0:
            return [], new_data
        
        # Compute uncertainties for all new samples
        model.eval()
        with torch.no_grad():
            predictions = model(new_data)
            uncertainties = self.samplers[SamplingStrategy.UNCERTAINTY].compute_uncertainty(predictions)
        
        # Select most uncertain samples within budget
        _, selected_indices = torch.topk(uncertainties, min(max_annotations, len(uncertainties)))
        
        # Update budget
        self.annotation_budget_used += len(selected_indices)
        
        # Return selected indices and remaining data
        remaining_mask = torch.ones(len(new_data), dtype=torch.bool)
        remaining_mask[selected_indices] = False
        remaining_data = new_data[remaining_mask]
        
        return selected_indices.tolist(), remaining_data
    
    def reset_budget(self):
        """Reset annotation budget"""
        self.annotation_budget_used = 0
        logger.info("Reset annotation budget")


# Utility functions
def create_active_learning_experiment(strategy: str, dataset_name: str, 
                                   model_name: str, **kwargs) -> ActiveLearner:
    """Create active learning experiment configuration"""
    config = ActiveLearningConfig(
        strategy=SamplingStrategy(strategy),
        **kwargs
    )
    
    learner = ActiveLearner(config)
    
    logger.info(f"Created active learning experiment: {strategy} on {dataset_name} with {model_name}")
    return learner


def benchmark_active_learning_strategies(strategies: List[str], model: nn.Module,
                                       dataset: torch.utils.data.Dataset,
                                       n_rounds: int = 10) -> Dict[str, Dict]:
    """Benchmark different active learning strategies"""
    results = {}
    
    for strategy in strategies:
        logger.info(f"Benchmarking strategy: {strategy}")
        
        config = ActiveLearningConfig(strategy=SamplingStrategy(strategy))
        learner = ActiveLearner(config)
        learner.initialize_pool(len(dataset))
        
        # Simulate active learning rounds
        for round_idx in range(n_rounds):
            selected_indices = learner.select_samples(model, dataset)
            
            # Simulate training and evaluation (placeholder)
            accuracy = 0.5 + round_idx * 0.05  # Dummy accuracy increase
            loss = 1.0 - round_idx * 0.1  # Dummy loss decrease
            
            learner.update_performance(accuracy, loss)
        
        results[strategy] = {
            "learning_curve": learner.get_learning_curve(),
            "final_accuracy": learner.performance_history[-1]["accuracy"],
            "samples_used": len(learner.labeled_indices)
        }
    
    return results
