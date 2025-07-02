"""
Quantum-Inspired Neural Networks for Infrared Small Target Detection

This module implements quantum-inspired architectures that leverage quantum computing
principles for enhanced feature extraction and processing of infrared data.

References:
- Quantum Convolutional Neural Networks (QCNN)
- Variational Quantum Classifiers (VQC)
- Hybrid Classical-Quantum Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

try:
    import cirq
    import tensorflow_quantum as tfq
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


class QuantumConvLayer(nn.Module):
    """
    Quantum-inspired convolutional layer using parameterized quantum circuits
    
    This layer applies quantum gates to process local patches of the infrared image,
    potentially capturing quantum correlations in thermal data.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str = 'cpu'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical preprocessing
        self.pre_conv = nn.Conv2d(in_channels, n_qubits, kernel_size, bias=False)
        
        # Quantum circuit parameters
        self.quantum_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1  # RX, RY, RZ rotations
        )
        
        # Classical postprocessing
        self.post_conv = nn.Conv2d(n_qubits, out_channels, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
    def quantum_circuit_simulation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate quantum circuit processing on classical hardware
        
        Uses matrix representations of quantum gates for classical simulation
        """
        batch_size, channels, height, width = x.shape
        
        # Flatten spatial dimensions for processing
        x_flat = x.view(batch_size, channels, -1)
        
        # Apply quantum-inspired transformations
        for layer in range(self.n_layers):
            # Pauli rotations (quantum gates)
            for qubit in range(self.n_qubits):
                theta_x, theta_y, theta_z = self.quantum_params[layer, qubit]
                
                # Simulate RX gate effect
                cos_x, sin_x = torch.cos(theta_x/2), torch.sin(theta_x/2)
                rx_real = cos_x * x_flat[:, qubit:qubit+1, :]
                rx_imag = -sin_x * x_flat[:, qubit:qubit+1, :]
                
                # Simulate RY gate effect  
                cos_y, sin_y = torch.cos(theta_y/2), torch.sin(theta_y/2)
                ry_transform = cos_y * rx_real + sin_y * rx_imag
                
                # Simulate RZ gate effect (phase rotation)
                phase = torch.exp(1j * theta_z)
                x_flat[:, qubit:qubit+1, :] = ry_transform * phase.real
            
            # Entangling operations (simplified CNOT-like coupling)
            if self.n_qubits > 1:
                for i in range(0, self.n_qubits-1, 2):
                    control = x_flat[:, i:i+1, :]
                    target = x_flat[:, i+1:i+2, :]
                    
                    # Simplified entanglement
                    entangled = 0.5 * (control + target)
                    x_flat[:, i:i+1, :] = entangled
                    x_flat[:, i+1:i+2, :] = entangled
        
        return x_flat.view(batch_size, channels, height, width)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical preprocessing
        x = self.pre_conv(x)
        
        # Quantum-inspired processing
        x = self.quantum_circuit_simulation(x)
        
        # Classical postprocessing
        x = self.post_conv(x)
        x = self.batch_norm(x)
        
        return F.relu(x)


class VariationalQuantumClassifier(nn.Module):
    """
    Variational Quantum Classifier for infrared target classification
    
    Implements a variational quantum circuit for binary classification
    of infrared targets vs. background
    """
    
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 8,
        n_layers: int = 3,
        classical_hidden: int = 128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Feature encoder to quantum state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, classical_hidden),
            nn.ReLU(),
            nn.Linear(classical_hidden, n_qubits),
            nn.Tanh()  # Bound inputs for angle encoding
        )
        
        # Variational quantum circuit parameters
        self.vqc_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )
        
        # Measurement and output
        self.decoder = nn.Linear(n_qubits, 1)
        
    def variational_quantum_circuit(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Apply variational quantum circuit to encoded features
        """
        batch_size, n_features = encoded_features.shape
        
        # Initialize quantum state (all qubits in |0⟩)
        quantum_state = torch.zeros_like(encoded_features)
        
        # Angle encoding of classical data
        quantum_state = torch.sin(encoded_features * np.pi)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Parameterized rotation gates
            for qubit in range(self.n_qubits):
                theta, phi = self.vqc_params[layer, qubit]
                
                # Apply rotation
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
                
                quantum_state[:, qubit] = (
                    cos_theta * quantum_state[:, qubit] + 
                    sin_theta * torch.cos(phi)
                )
            
            # Entangling layer (simplified)
            if layer < self.n_layers - 1:
                for i in range(0, self.n_qubits-1, 2):
                    # CNOT-like entanglement
                    control = quantum_state[:, i]
                    target = quantum_state[:, i+1]
                    
                    new_target = control * 0.1 + target * 0.9
                    quantum_state[:, i+1] = new_target
        
        return quantum_state
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode classical features to quantum representation
        encoded = self.encoder(x)
        
        # Apply variational quantum circuit
        quantum_output = self.variational_quantum_circuit(encoded)
        
        # Measure and decode
        output = self.decoder(quantum_output)
        
        return torch.sigmoid(output)


class HybridQuantumClassicalNet(nn.Module):
    """
    Hybrid network combining classical CNN features with quantum processing
    
    This architecture uses classical convolutions for spatial feature extraction
    and quantum layers for high-level reasoning and classification.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        classical_features: int = 256,
        quantum_qubits: int = 8
    ):
        super().__init__()
        
        # Classical feature extractor
        self.classical_backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, classical_features),
            nn.ReLU()
        )
        
        # Quantum processing layers
        self.quantum_layers = nn.ModuleList([
            QuantumConvLayer(32, 32, kernel_size=3, n_qubits=4),
            QuantumConvLayer(64, 64, kernel_size=3, n_qubits=6),
        ])
        
        # Quantum classifier
        self.quantum_classifier = VariationalQuantumClassifier(
            input_dim=classical_features,
            n_qubits=quantum_qubits
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(classical_features + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Classical feature extraction
        classical_features = self.classical_backbone(x)
        
        # Quantum classification
        quantum_output = self.quantum_classifier(classical_features)
        
        # Fusion
        combined = torch.cat([classical_features, quantum_output], dim=1)
        final_output = self.fusion(combined)
        
        return {
            'logits': final_output,
            'classical_features': classical_features,
            'quantum_output': quantum_output,
            'probabilities': F.softmax(final_output, dim=1)
        }


class QuantumInspiredLoss(nn.Module):
    """
    Quantum-inspired loss function incorporating quantum fidelity measures
    
    Uses quantum information theory concepts to create more robust loss functions
    for infrared target detection.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Weight for classical loss
        self.beta = beta    # Weight for quantum fidelity term
        self.ce_loss = nn.CrossEntropyLoss()
    
    def quantum_fidelity_loss(
        self, 
        pred_probs: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantum fidelity-based loss term
        
        Quantum fidelity measures the overlap between quantum states,
        here adapted for probability distributions.
        """
        # Convert targets to one-hot
        target_one_hot = F.one_hot(target, num_classes=pred_probs.size(1)).float()
        
        # Compute quantum fidelity (simplified)
        # F(ρ,σ) = Tr(√(√ρ σ √ρ)) for quantum states ρ,σ
        # Simplified as sqrt of probability overlap
        sqrt_pred = torch.sqrt(pred_probs + 1e-8)
        sqrt_target = torch.sqrt(target_one_hot + 1e-8)
        
        fidelity = torch.sum(sqrt_pred * sqrt_target, dim=1)
        
        # Convert to loss (1 - fidelity)
        fidelity_loss = 1.0 - fidelity
        
        return fidelity_loss.mean()
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        quantum_output: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Classical cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Quantum fidelity loss
        probs = F.softmax(logits, dim=1)
        fidelity_loss = self.quantum_fidelity_loss(probs, targets)
        
        # Quantum coherence penalty (if quantum output available)
        coherence_loss = 0.0
        if quantum_output is not None:
            # Penalize low coherence in quantum features
            coherence = torch.var(quantum_output, dim=1).mean()
            coherence_loss = -self.beta * coherence  # Negative to encourage coherence
        
        # Combined loss
        total_loss = (
            self.alpha * ce_loss + 
            (1 - self.alpha) * fidelity_loss + 
            coherence_loss
        )
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'fidelity_loss': fidelity_loss,
            'coherence_loss': coherence_loss
        }


# Utility functions for quantum-inspired processing

def quantum_amplitude_encoding(data: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """
    Encode classical data into quantum amplitude encoding
    
    Maps classical data to quantum state amplitudes with normalization
    """
    batch_size = data.size(0)
    n_amplitudes = 2 ** n_qubits
    
    # Flatten and pad/truncate data to fit amplitude encoding
    flat_data = data.view(batch_size, -1)
    
    if flat_data.size(1) > n_amplitudes:
        # Truncate if too large
        flat_data = flat_data[:, :n_amplitudes]
    elif flat_data.size(1) < n_amplitudes:
        # Pad if too small
        padding = n_amplitudes - flat_data.size(1)
        flat_data = F.pad(flat_data, (0, padding))
    
    # Normalize to unit vector (quantum state constraint)
    normalized = F.normalize(flat_data, p=2, dim=1)
    
    return normalized


def quantum_measurement_expectation(
    quantum_state: torch.Tensor,
    observable: torch.Tensor
) -> torch.Tensor:
    """
    Compute expectation value of observable on quantum state
    
    ⟨ψ|O|ψ⟩ where |ψ⟩ is the quantum state and O is the observable
    """
    # Ensure observable is hermitian (for physical observables)
    hermitian_obs = 0.5 * (observable + observable.transpose(-2, -1))
    
    # Compute expectation value
    expectation = torch.sum(
        quantum_state.unsqueeze(-1) * 
        torch.matmul(hermitian_obs, quantum_state.unsqueeze(-1)),
        dim=-2
    ).squeeze(-1)
    
    return expectation


# Example usage and training utilities

class QuantumTrainer:
    """
    Specialized trainer for quantum-inspired models
    
    Handles quantum-specific optimization challenges and monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: QuantumInspiredLoss,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.quantum_metrics = {
            'coherence_history': [],
            'fidelity_history': [],
            'entanglement_entropy': []
        }
    
    def compute_quantum_metrics(self, model_output: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute quantum-specific metrics for monitoring
        """
        metrics = {}
        
        if 'quantum_output' in model_output:
            quantum_out = model_output['quantum_output']
            
            # Quantum coherence measure
            coherence = torch.var(quantum_out, dim=1).mean().item()
            metrics['coherence'] = coherence
            
            # Pseudo-entanglement entropy (simplified)
            probs = F.softmax(quantum_out, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
            metrics['entanglement_entropy'] = entropy
        
        return metrics
    
    def train_step(
        self, 
        batch_data: torch.Tensor, 
        batch_targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step with quantum-specific monitoring
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        model_output = self.model(batch_data)
        
        # Compute loss
        loss_dict = self.loss_fn(
            model_output['logits'],
            batch_targets,
            model_output.get('quantum_output')
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Gradient clipping for quantum stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute quantum metrics
        quantum_metrics = self.compute_quantum_metrics(model_output)
        
        # Combine all metrics
        step_metrics = {
            'loss': loss_dict['total_loss'].item(),
            'ce_loss': loss_dict['ce_loss'].item(),
            'fidelity_loss': loss_dict['fidelity_loss'].item(),
            **quantum_metrics
        }
        
        return step_metrics


# Factory function for creating quantum models

def create_quantum_irst_model(
    model_type: str = 'hybrid',
    input_channels: int = 1,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create quantum-inspired IRST models
    
    Args:
        model_type: Type of quantum model ('hybrid', 'pure_quantum', 'quantum_conv')
        input_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional model-specific parameters
    
    Returns:
        Quantum-inspired neural network model
    """
    
    if not QUANTUM_AVAILABLE and model_type == 'pure_quantum':
        raise ImportError(
            "Quantum libraries (cirq, tensorflow-quantum) not available. "
            "Install with: pip install cirq tensorflow-quantum"
        )
    
    if model_type == 'hybrid':
        return HybridQuantumClassicalNet(
            input_channels=input_channels,
            num_classes=num_classes,
            **kwargs
        )
    
    elif model_type == 'quantum_conv':
        # Simple quantum convolutional model
        return nn.Sequential(
            QuantumConvLayer(input_channels, 32, **kwargs),
            QuantumConvLayer(32, 64, **kwargs),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, num_classes)
        )
    
    elif model_type == 'vqc':
        # Variational quantum classifier only
        return VariationalQuantumClassifier(
            input_dim=input_channels * 256 * 256,  # Assuming 256x256 input
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown quantum model type: {model_type}")


# Export main components
__all__ = [
    'QuantumConvLayer',
    'VariationalQuantumClassifier', 
    'HybridQuantumClassicalNet',
    'QuantumInspiredLoss',
    'QuantumTrainer',
    'create_quantum_irst_model',
    'quantum_amplitude_encoding',
    'quantum_measurement_expectation'
]
