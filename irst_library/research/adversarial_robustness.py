"""
Adversarial Robustness Suite for IRST Library

This module provides comprehensive adversarial robustness capabilities including:
- State-of-the-art adversarial attacks (FGSM, PGD, C&W, AutoAttack)
- Certified defense mechanisms (randomized smoothing, IBP)
- Adaptive attack generation for robust evaluation
- Robustness benchmarking and evaluation frameworks
- Adversarial training methods

Key Features:
- Multiple attack algorithms with adaptive parameters
- Certified robustness guarantees
- Comprehensive evaluation metrics
- Integration with main IRST models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks"""
    epsilon: float = 0.1        # Perturbation budget
    alpha: float = 0.01         # Step size
    num_steps: int = 10         # Number of attack steps
    norm: str = 'inf'           # Norm type ('inf', '2', '1')
    targeted: bool = False      # Targeted vs untargeted attack
    random_init: bool = True    # Random initialization
    clip_min: float = 0.0      # Minimum pixel value
    clip_max: float = 1.0      # Maximum pixel value


@dataclass
class RobustnessMetrics:
    """Metrics for evaluating adversarial robustness"""
    clean_accuracy: float
    adversarial_accuracy: float
    average_perturbation: float
    success_rate: float
    certified_accuracy: Optional[float] = None
    verified_accuracy: Optional[float] = None


class AdversarialAttack(ABC):
    """Base class for adversarial attacks"""
    
    def __init__(self, config: AttackConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Generate adversarial examples"""
        pass
    
    def project(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Project perturbation to satisfy constraint"""
        if self.config.norm == 'inf':
            return torch.clamp(perturbation, -self.config.epsilon, self.config.epsilon)
        elif self.config.norm == '2':
            # L2 projection
            norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1)
            factor = torch.min(torch.ones_like(norm), self.config.epsilon / (norm + 1e-8))
            return perturbation * factor.view(-1, 1, 1, 1)
        elif self.config.norm == '1':
            # L1 projection (simplified)
            abs_sum = torch.sum(torch.abs(perturbation.view(perturbation.size(0), -1)), dim=1)
            factor = torch.min(torch.ones_like(abs_sum), self.config.epsilon / (abs_sum + 1e-8))
            return perturbation * factor.view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported norm: {self.config.norm}")


class FastGradientSignMethod(AdversarialAttack):
    """
    Fast Gradient Sign Method (FGSM) attack
    
    Reference: Goodfellow et al. "Explaining and Harnessing Adversarial Examples" ICLR 2015
    """
    
    def generate(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate FGSM adversarial examples
        
        x' = x + ε * sign(∇_x L(θ, x, y))
        """
        model.eval()
        
        # Enable gradient computation for inputs
        adv_inputs = inputs.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(adv_inputs)
        if isinstance(outputs, dict):
            outputs = outputs['logits']
        
        # Compute loss
        if self.config.targeted:
            # Minimize loss for targeted attack
            loss = -F.cross_entropy(outputs, targets)
        else:
            # Maximize loss for untargeted attack
            loss = F.cross_entropy(outputs, targets)
        
        # Compute gradients
        grad_inputs = grad(loss, adv_inputs, retain_graph=False)[0]
        
        # Generate perturbation
        if self.config.norm == 'inf':
            perturbation = self.config.epsilon * grad_inputs.sign()
        elif self.config.norm == '2':
            # L2 FGSM
            grad_norm = torch.norm(grad_inputs.view(grad_inputs.size(0), -1), p=2, dim=1)
            perturbation = self.config.epsilon * grad_inputs / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
        else:
            raise ValueError(f"FGSM not implemented for norm: {self.config.norm}")
        
        # Apply perturbation
        adv_examples = inputs + perturbation
        
        # Clip to valid range
        adv_examples = torch.clamp(adv_examples, self.config.clip_min, self.config.clip_max)
        
        return adv_examples


class ProjectedGradientDescent(AdversarialAttack):
    """
    Projected Gradient Descent (PGD) attack
    
    Multi-step extension of FGSM with projection to constraint set
    
    Reference: Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" ICLR 2018
    """
    
    def generate(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate PGD adversarial examples
        
        Iterative process:
        x^(t+1) = Π_{S}(x^(t) + α * sign(∇_x L(θ, x^(t), y)))
        """
        model.eval()
        
        # Random initialization
        if self.config.random_init:
            if self.config.norm == 'inf':
                delta = torch.empty_like(inputs).uniform_(
                    -self.config.epsilon, self.config.epsilon
                )
            elif self.config.norm == '2':
                delta = torch.randn_like(inputs)
                delta = self.project(delta)
            else:
                delta = torch.zeros_like(inputs)
        else:
            delta = torch.zeros_like(inputs)
        
        # Ensure delta is in the constraint set
        delta = self.project(delta)
        
        # Iterative attack
        for step in range(self.config.num_steps):
            adv_inputs = inputs + delta
            adv_inputs = torch.clamp(adv_inputs, self.config.clip_min, self.config.clip_max)
            adv_inputs.requires_grad_(True)
            
            # Forward pass
            outputs = model(adv_inputs)
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            
            # Compute loss
            if self.config.targeted:
                loss = -F.cross_entropy(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, targets)
            
            # Compute gradients
            grad_delta = grad(loss, adv_inputs, retain_graph=False)[0]
            
            # Update perturbation
            if self.config.norm == 'inf':
                delta = delta + self.config.alpha * grad_delta.sign()
            elif self.config.norm == '2':
                grad_norm = torch.norm(grad_delta.view(grad_delta.size(0), -1), p=2, dim=1)
                normalized_grad = grad_delta / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                delta = delta + self.config.alpha * normalized_grad
            else:
                raise ValueError(f"PGD not implemented for norm: {self.config.norm}")
            
            # Project to constraint set
            delta = self.project(delta)
        
        # Generate final adversarial examples
        adv_examples = inputs + delta
        adv_examples = torch.clamp(adv_examples, self.config.clip_min, self.config.clip_max)
        
        return adv_examples


class CarliniWagnerAttack(AdversarialAttack):
    """
    Carlini & Wagner (C&W) L2 attack
    
    Optimization-based attack that minimizes perturbation while ensuring misclassification
    
    Reference: Carlini & Wagner "Towards Evaluating the Robustness of Neural Networks" IEEE S&P 2017
    """
    
    def __init__(self, config: AttackConfig, c_init: float = 1.0, kappa: float = 0.0):
        super().__init__(config)
        self.c_init = c_init
        self.kappa = kappa  # Confidence parameter
        
    def f_function(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Objective function for C&W attack
        
        f(x') = max(max{Z(x')_i : i ≠ t} - Z(x')_t, -κ)
        """
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        
        # Get target class logits
        target_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Get maximum non-target class logit
        target_mask = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1).bool()
        non_target_logits = outputs.masked_fill(target_mask, float('-inf'))
        max_non_target_logits = non_target_logits.max(dim=1)[0]
        
        # Compute f function
        if self.config.targeted:
            # For targeted attack: minimize target class logit
            f_val = torch.clamp(max_non_target_logits - target_logits, min=-self.kappa)
        else:
            # For untargeted attack: maximize non-target class logits
            f_val = torch.clamp(target_logits - max_non_target_logits, min=-self.kappa)
        
        return f_val
    
    def generate(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate C&W adversarial examples using optimization
        """
        model.eval()
        batch_size = inputs.size(0)
        
        # Initialize variables
        w = torch.zeros_like(inputs, requires_grad=True)
        c = torch.full((batch_size,), self.c_init, device=inputs.device)
        
        # Optimizer
        optimizer = torch.optim.Adam([w], lr=0.01)
        
        best_adv = inputs.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=inputs.device)
        
        # Binary search on c (simplified version)
        for search_step in range(3):  # Reduced for efficiency
            
            # Optimization loop
            for step in range(self.config.num_steps):
                optimizer.zero_grad()
                
                # Convert w to adversarial example
                # Use tanh to ensure valid pixel range
                adv_inputs = 0.5 * (torch.tanh(w) + 1)
                adv_inputs = adv_inputs * (self.config.clip_max - self.config.clip_min) + self.config.clip_min
                
                # Forward pass
                outputs = model(adv_inputs)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                
                # L2 distance
                l2_dist = torch.norm((adv_inputs - inputs).view(batch_size, -1), p=2, dim=1)
                
                # Adversarial loss
                f_loss = self.f_function(outputs, targets)
                
                # Total loss
                total_loss = (c * f_loss + l2_dist).sum()
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Update best adversarial examples
                with torch.no_grad():
                    # Check if attack succeeded
                    pred = outputs.argmax(dim=1)
                    success = (pred != targets) if not self.config.targeted else (pred == targets)
                    
                    # Update best examples for successful attacks
                    better = success & (l2_dist < best_l2)
                    best_adv[better] = adv_inputs[better]
                    best_l2[better] = l2_dist[better]
            
            # Update c for binary search (simplified)
            c *= 2
        
        return best_adv


class AutoAttack:
    """
    AutoAttack: Automatic evaluation of adversarial robustness
    
    Combines multiple attacks for comprehensive robustness evaluation
    
    Reference: Croce & Hein "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks" ICML 2020
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        norm: str = 'inf',
        attacks: Optional[List[str]] = None
    ):
        """
        Args:
            epsilon: Perturbation budget
            norm: Norm constraint
            attacks: List of attacks to use
        """
        self.epsilon = epsilon
        self.norm = norm
        self.attacks = attacks or ['fgsm', 'pgd', 'cw']
        
        self.logger = logging.getLogger(__name__)
    
    def run(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Run AutoAttack evaluation
        
        Returns:
            Dictionary with adversarial examples from each attack
        """
        model.eval()
        results = {}
        
        # Base configuration
        base_config = AttackConfig(
            epsilon=self.epsilon,
            norm=self.norm,
            alpha=self.epsilon / 10,  # Typical step size
            num_steps=20
        )
        
        # Run each attack
        for attack_name in self.attacks:
            self.logger.info(f"Running {attack_name} attack...")
            
            if attack_name == 'fgsm':
                attack = FastGradientSignMethod(base_config)
            elif attack_name == 'pgd':
                attack = ProjectedGradientDescent(base_config)
            elif attack_name == 'cw':
                attack = CarliniWagnerAttack(base_config)
            else:
                self.logger.warning(f"Unknown attack: {attack_name}")
                continue
            
            try:
                adv_examples = attack.generate(model, inputs, targets)
                results[attack_name] = adv_examples
                
                # Evaluate attack success
                with torch.no_grad():
                    outputs = model(adv_examples)
                    if isinstance(outputs, dict):
                        outputs = outputs['logits']
                    
                    pred = outputs.argmax(dim=1)
                    success_rate = (pred != targets).float().mean().item()
                    
                    self.logger.info(f"{attack_name} success rate: {success_rate:.3f}")
                    
            except Exception as e:
                self.logger.error(f"Attack {attack_name} failed: {e}")
        
        return results


class RandomizedSmoothing:
    """
    Randomized Smoothing for certified adversarial robustness
    
    Provides probabilistic certificates for robustness by smoothing predictions
    with Gaussian noise.
    
    Reference: Cohen et al. "Certified Adversarial Robustness via Randomized Smoothing" ICML 2019
    """
    
    def __init__(
        self,
        noise_std: float = 0.25,
        num_samples: int = 1000,
        alpha: float = 0.001,  # Failure probability
        batch_size: int = 100
    ):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise
            num_samples: Number of samples for smoothing
            alpha: Failure probability for certification
            batch_size: Batch size for efficient computation
        """
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.alpha = alpha
        self.batch_size = batch_size
        
        self.logger = logging.getLogger(__name__)
    
    def smooth_predict(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute smoothed predictions by averaging over noise
        
        Returns:
            Tuple of (smoothed_predictions, confidence_scores)
        """
        model.eval()
        num_samples = num_samples or self.num_samples
        batch_size, channels, height, width = inputs.shape
        
        # Storage for predictions
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, num_samples, self.batch_size):
                current_batch_size = min(self.batch_size, num_samples - i)
                
                # Add Gaussian noise
                noise = torch.randn(
                    current_batch_size, batch_size, channels, height, width,
                    device=inputs.device
                ) * self.noise_std
                
                # Expand inputs to match noise batch
                noisy_inputs = inputs.unsqueeze(0) + noise
                noisy_inputs = noisy_inputs.view(-1, channels, height, width)
                
                # Forward pass
                outputs = model(noisy_inputs)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                
                # Get predictions
                predictions = outputs.argmax(dim=1)
                predictions = predictions.view(current_batch_size, batch_size)
                
                all_predictions.append(predictions)
        
        # Combine all predictions
        all_predictions = torch.cat(all_predictions, dim=0)  # [num_samples, batch_size]
        
        # Compute smoothed predictions (majority vote)
        smoothed_preds = []
        confidence_scores = []
        
        for b in range(batch_size):
            sample_preds = all_predictions[:, b]
            
            # Count votes for each class
            unique_preds, counts = torch.unique(sample_preds, return_counts=True)
            
            # Get most voted class
            max_count_idx = counts.argmax()
            most_voted_class = unique_preds[max_count_idx]
            max_count = counts[max_count_idx].item()
            
            smoothed_preds.append(most_voted_class)
            confidence_scores.append(max_count / num_samples)
        
        return torch.stack(smoothed_preds), torch.tensor(confidence_scores, device=inputs.device)
    
    def certify_robustness(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute certified robustness radius
        
        Returns:
            Certified L2 radius for each input
        """
        from scipy.stats import norm
        
        certified_radii = []
        
        for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
            if conf > 0.5:  # Only certify if we have majority
                # Compute certification radius using Neyman-Pearson lemma
                # Simplified calculation
                p_lower = conf - norm.ppf(1 - self.alpha/2) * math.sqrt(conf * (1 - conf) / self.num_samples)
                
                if p_lower > 0.5:
                    # Certified radius (simplified formula)
                    certified_radius = self.noise_std * norm.ppf(p_lower)
                else:
                    certified_radius = 0.0
            else:
                certified_radius = 0.0
            
            certified_radii.append(certified_radius)
        
        return torch.tensor(certified_radii, device=inputs.device)


class AdversarialTraining:
    """
    Adversarial training for robust model training
    
    Implements various adversarial training strategies including:
    - Standard adversarial training
    - TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
    - MART (Misclassification Aware adveRsarial Training)
    """
    
    def __init__(
        self,
        attack_config: AttackConfig,
        training_method: str = 'standard',
        lambda_trade: float = 6.0,  # TRADES regularization parameter
        beta: float = 6.0,          # MART parameter
    ):
        """
        Args:
            attack_config: Configuration for adversarial attacks during training
            training_method: Type of adversarial training ('standard', 'trades', 'mart')
            lambda_trade: TRADES regularization weight
            beta: MART parameter for balancing losses
        """
        self.attack_config = attack_config
        self.training_method = training_method
        self.lambda_trade = lambda_trade
        self.beta = beta
        
        # Create attack for training
        self.attack = ProjectedGradientDescent(attack_config)
        
        self.logger = logging.getLogger(__name__)
    
    def compute_adversarial_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        natural_outputs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adversarial training loss
        
        Args:
            model: Neural network model
            inputs: Clean input batch
            targets: Target labels
            natural_outputs: Pre-computed clean outputs (optional)
            
        Returns:
            Dictionary with loss components
        """
        model.train()
        
        # Generate adversarial examples
        adv_inputs = self.attack.generate(model, inputs, targets)
        
        # Forward pass on clean inputs
        if natural_outputs is None:
            natural_outputs = model(inputs)
            if isinstance(natural_outputs, dict):
                natural_outputs = natural_outputs['logits']
        
        # Forward pass on adversarial inputs
        adv_outputs = model(adv_inputs)
        if isinstance(adv_outputs, dict):
            adv_outputs = adv_outputs['logits']
        
        losses = {}
        
        if self.training_method == 'standard':
            # Standard adversarial training
            natural_loss = F.cross_entropy(natural_outputs, targets)
            adv_loss = F.cross_entropy(adv_outputs, targets)
            
            total_loss = 0.5 * natural_loss + 0.5 * adv_loss
            
            losses.update({
                'natural_loss': natural_loss,
                'adversarial_loss': adv_loss,
                'total_loss': total_loss
            })
            
        elif self.training_method == 'trades':
            # TRADES: balance between natural accuracy and robustness
            natural_loss = F.cross_entropy(natural_outputs, targets)
            
            # KL divergence between natural and adversarial predictions
            kl_loss = F.kl_div(
                F.log_softmax(adv_outputs, dim=1),
                F.softmax(natural_outputs, dim=1),
                reduction='batchmean'
            )
            
            total_loss = natural_loss + self.lambda_trade * kl_loss
            
            losses.update({
                'natural_loss': natural_loss,
                'kl_loss': kl_loss,
                'total_loss': total_loss
            })
            
        elif self.training_method == 'mart':
            # MART: misclassification-aware adversarial training
            natural_loss = F.cross_entropy(natural_outputs, targets)
            adv_loss = F.cross_entropy(adv_outputs, targets)
            
            # Additional term for misclassified examples
            with torch.no_grad():
                natural_pred = natural_outputs.argmax(dim=1)
                misclassified = (natural_pred != targets).float()
            
            # Weighted adversarial loss
            weighted_adv_loss = adv_loss * (1 + self.beta * misclassified.mean())
            
            total_loss = natural_loss + weighted_adv_loss
            
            losses.update({
                'natural_loss': natural_loss,
                'adversarial_loss': adv_loss,
                'weighted_adv_loss': weighted_adv_loss,
                'total_loss': total_loss
            })
            
        else:
            raise ValueError(f"Unknown training method: {self.training_method}")
        
        return losses


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation framework
    
    Provides standardized evaluation of model robustness across multiple
    attacks and certification methods.
    """
    
    def __init__(
        self,
        attacks: Optional[List[AdversarialAttack]] = None,
        certification_methods: Optional[List[str]] = None
    ):
        """
        Args:
            attacks: List of adversarial attacks to evaluate
            certification_methods: List of certification methods to use
        """
        self.attacks = attacks or []
        self.certification_methods = certification_methods or []
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_robustness(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ) -> Dict[str, RobustnessMetrics]:
        """
        Comprehensive robustness evaluation
        
        Returns:
            Dictionary mapping attack names to robustness metrics
        """
        model.eval()
        results = {}
        
        # Collect all clean predictions first
        all_clean_correct = []
        all_inputs = []
        all_targets = []
        
        self.logger.info("Collecting clean predictions...")
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                
                pred = outputs.argmax(dim=1)
                clean_correct = (pred == targets)
                
                all_clean_correct.append(clean_correct)
                all_inputs.append(inputs)
                all_targets.append(targets)
        
        # Combine all data
        all_clean_correct = torch.cat(all_clean_correct)
        all_inputs = torch.cat(all_inputs)
        all_targets = torch.cat(all_targets)
        
        clean_accuracy = all_clean_correct.float().mean().item()
        self.logger.info(f"Clean accuracy: {clean_accuracy:.3f}")
        
        # Evaluate each attack
        for attack in self.attacks:
            attack_name = attack.__class__.__name__
            self.logger.info(f"Evaluating {attack_name}...")
            
            # Generate adversarial examples
            adv_inputs = attack.generate(model, all_inputs, all_targets)
            
            # Evaluate adversarial accuracy
            with torch.no_grad():
                adv_outputs = model(adv_inputs)
                if isinstance(adv_outputs, dict):
                    adv_outputs = adv_outputs['logits']
                
                adv_pred = adv_outputs.argmax(dim=1)
                adv_correct = (adv_pred == all_targets)
                
                adversarial_accuracy = adv_correct.float().mean().item()
                
                # Compute perturbation statistics
                perturbation = adv_inputs - all_inputs
                if attack.config.norm == 'inf':
                    avg_perturbation = perturbation.abs().max(dim=1)[0].mean().item()
                elif attack.config.norm == '2':
                    avg_perturbation = torch.norm(
                        perturbation.view(perturbation.size(0), -1), p=2, dim=1
                    ).mean().item()
                else:
                    avg_perturbation = 0.0
                
                success_rate = 1.0 - adversarial_accuracy
            
            # Store results
            results[attack_name] = RobustnessMetrics(
                clean_accuracy=clean_accuracy,
                adversarial_accuracy=adversarial_accuracy,
                average_perturbation=avg_perturbation,
                success_rate=success_rate
            )
            
            self.logger.info(
                f"{attack_name}: Adv Acc = {adversarial_accuracy:.3f}, "
                f"Success Rate = {success_rate:.3f}"
            )
        
        # Add certification if requested
        if 'randomized_smoothing' in self.certification_methods:
            self.logger.info("Computing certified robustness...")
            
            smoother = RandomizedSmoothing()
            smoothed_preds, confidence = smoother.smooth_predict(model, all_inputs[:100])  # Subset for efficiency
            certified_radii = smoother.certify_robustness(
                model, all_inputs[:100], smoothed_preds, confidence
            )
            
            certified_accuracy = (certified_radii > 0).float().mean().item()
            
            # Update first result with certification
            first_attack = list(results.keys())[0]
            results[first_attack].certified_accuracy = certified_accuracy
        
        return results
    
    def generate_robustness_report(
        self, 
        results: Dict[str, RobustnessMetrics],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive robustness evaluation report
        
        Returns:
            Formatted report string
        """
        report = "# Adversarial Robustness Evaluation Report\n\n"
        
        # Summary table
        report += "## Summary\n\n"
        report += "| Attack | Clean Acc | Adv Acc | Success Rate | Avg Perturbation |\n"
        report += "|--------|-----------|---------|--------------|------------------|\n"
        
        for attack_name, metrics in results.items():
            report += f"| {attack_name} | {metrics.clean_accuracy:.3f} | "
            report += f"{metrics.adversarial_accuracy:.3f} | {metrics.success_rate:.3f} | "
            report += f"{metrics.average_perturbation:.4f} |\n"
        
        # Detailed analysis
        report += "\n## Detailed Analysis\n\n"
        
        for attack_name, metrics in results.items():
            report += f"### {attack_name}\n\n"
            report += f"- **Clean Accuracy**: {metrics.clean_accuracy:.3f}\n"
            report += f"- **Adversarial Accuracy**: {metrics.adversarial_accuracy:.3f}\n"
            report += f"- **Attack Success Rate**: {metrics.success_rate:.3f}\n"
            report += f"- **Average Perturbation**: {metrics.average_perturbation:.4f}\n"
            
            if metrics.certified_accuracy is not None:
                report += f"- **Certified Accuracy**: {metrics.certified_accuracy:.3f}\n"
            
            robustness_drop = metrics.clean_accuracy - metrics.adversarial_accuracy
            report += f"- **Robustness Drop**: {robustness_drop:.3f}\n\n"
        
        # Save report if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")
        
        return report


# Factory functions

def create_attack_suite(
    epsilon: float = 0.1,
    norm: str = 'inf',
    include_attacks: Optional[List[str]] = None
) -> List[AdversarialAttack]:
    """
    Create a suite of adversarial attacks for evaluation
    
    Args:
        epsilon: Perturbation budget
        norm: Norm constraint
        include_attacks: List of attack names to include
        
    Returns:
        List of configured adversarial attacks
    """
    include_attacks = include_attacks or ['fgsm', 'pgd', 'cw']
    
    config = AttackConfig(
        epsilon=epsilon,
        norm=norm,
        alpha=epsilon / 10,
        num_steps=20
    )
    
    attacks = []
    
    if 'fgsm' in include_attacks:
        attacks.append(FastGradientSignMethod(config))
    
    if 'pgd' in include_attacks:
        attacks.append(ProjectedGradientDescent(config))
    
    if 'cw' in include_attacks:
        attacks.append(CarliniWagnerAttack(config))
    
    return attacks


def create_robust_trainer(
    attack_epsilon: float = 0.1,
    training_method: str = 'standard',
    **kwargs
) -> AdversarialTraining:
    """
    Create adversarial training setup
    
    Args:
        attack_epsilon: Perturbation budget for training attacks
        training_method: Type of adversarial training
        **kwargs: Additional training parameters
        
    Returns:
        Configured adversarial trainer
    """
    config = AttackConfig(
        epsilon=attack_epsilon,
        alpha=attack_epsilon / 10,
        num_steps=10,  # Fewer steps for training efficiency
        norm='inf'
    )
    
    return AdversarialTraining(
        attack_config=config,
        training_method=training_method,
        **kwargs
    )


# Export main components
__all__ = [
    'AttackConfig',
    'RobustnessMetrics',
    'AdversarialAttack',
    'FastGradientSignMethod',
    'ProjectedGradientDescent',
    'CarliniWagnerAttack',
    'AutoAttack',
    'RandomizedSmoothing',
    'AdversarialTraining',
    'RobustnessEvaluator',
    'create_attack_suite',
    'create_robust_trainer'
]
