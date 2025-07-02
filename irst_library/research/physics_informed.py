"""
Physics-Informed Neural Networks (PINNs) for Infrared Small Target Detection

This module implements physics-informed neural networks that incorporate
physical laws and constraints into the learning process for infrared target
detection. Includes atmospheric propagation modeling, heat transfer equations,
and infrared physics constraints.

Key Features:
- Atmospheric propagation physics integration
- Heat transfer equation constraints
- Infrared radiation physics modeling
- Temperature distribution analysis
- Multi-scale physics modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import math
from abc import ABC, abstractmethod


class PhysicsLaw(ABC):
    """Base class for physics laws and constraints"""
    
    @abstractmethod
    def compute_physics_loss(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        outputs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute physics-based loss term"""
        pass
    
    @abstractmethod
    def get_loss_weight(self) -> float:
        """Get the weight for this physics loss term"""
        pass


class AtmosphericPropagationLaw(PhysicsLaw):
    """
    Atmospheric propagation physics for infrared radiation
    
    Models how infrared radiation propagates through atmosphere:
    - Beer-Lambert law for atmospheric absorption
    - Scattering effects (Rayleigh and Mie)
    - Temperature and humidity dependencies
    - Range-dependent attenuation
    """
    
    def __init__(
        self,
        wavelength: float = 10e-6,  # 10 μm (LWIR)
        loss_weight: float = 0.1,
        atmospheric_model: str = 'exponential'
    ):
        """
        Args:
            wavelength: Infrared wavelength in meters
            loss_weight: Weight for atmospheric physics loss
            atmospheric_model: Type of atmospheric model ('exponential', 'lowtran')
        """
        self.wavelength = wavelength
        self.loss_weight = loss_weight
        self.atmospheric_model = atmospheric_model
        
        # Atmospheric absorption coefficients (simplified)
        self.alpha_absorption = 0.1  # km⁻¹
        self.alpha_scattering = 0.05  # km⁻¹
    
    def compute_atmospheric_transmission(
        self, 
        range_km: torch.Tensor,
        temperature: torch.Tensor = None,
        humidity: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute atmospheric transmission using Beer-Lambert law
        
        T(R) = exp(-α * R)
        where α is the extinction coefficient and R is range
        """
        if self.atmospheric_model == 'exponential':
            # Simple exponential decay
            extinction_coeff = self.alpha_absorption + self.alpha_scattering
            
            # Temperature and humidity corrections
            if temperature is not None:
                # Absorption increases with temperature (water vapor)
                temp_factor = 1.0 + 0.01 * (temperature - 273.15)  # Celsius adjustment
                extinction_coeff *= temp_factor
            
            if humidity is not None:
                # Humidity increases absorption
                humidity_factor = 1.0 + 0.05 * humidity  # Relative humidity factor
                extinction_coeff *= humidity_factor
            
            transmission = torch.exp(-extinction_coeff * range_km)
            
        elif self.atmospheric_model == 'lowtran':
            # More complex LOWTRAN-based model (simplified)
            # Wavelength-dependent absorption
            wavelength_um = self.wavelength * 1e6  # Convert to μm
            
            # Water vapor absorption (simplified)
            water_absorption = 0.1 * torch.exp(-((wavelength_um - 10.0) / 2.0) ** 2)
            
            # CO2 absorption
            co2_absorption = 0.05 * torch.exp(-((wavelength_um - 4.3) / 0.5) ** 2)
            
            total_extinction = water_absorption + co2_absorption + self.alpha_scattering
            transmission = torch.exp(-total_extinction * range_km)
        
        else:
            raise ValueError(f"Unknown atmospheric model: {self.atmospheric_model}")
        
        return transmission
    
    def compute_physics_loss(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        outputs: torch.Tensor,
        range_data: Optional[torch.Tensor] = None,
        atmospheric_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute atmospheric propagation physics loss
        
        Enforces that detected intensity follows atmospheric attenuation laws
        """
        if range_data is None:
            # Assume range can be estimated from model or is part of input
            return torch.tensor(0.0, device=inputs.device)
        
        # Extract atmospheric parameters
        temperature = atmospheric_data.get('temperature') if atmospheric_data else None
        humidity = atmospheric_data.get('humidity') if atmospheric_data else None
        
        # Compute expected atmospheric transmission
        expected_transmission = self.compute_atmospheric_transmission(
            range_data, temperature, humidity
        )
        
        # Extract model predictions (assuming outputs contain intensity predictions)
        if isinstance(outputs, dict):
            predicted_intensity = outputs.get('intensity', outputs.get('logits'))
        else:
            predicted_intensity = outputs
        
        # Convert to intensity if logits
        if predicted_intensity.dim() > 1 and predicted_intensity.size(1) > 1:
            # Classification logits - convert to intensity proxy
            predicted_intensity = F.softmax(predicted_intensity, dim=1)[:, 1]  # Target class probability
        
        # Physics constraint: intensity should decrease with range according to atmospheric model
        # Assume initial intensity is 1.0 at range 0
        expected_intensity = expected_transmission
        
        # L2 loss between predicted and physics-expected intensity
        physics_loss = F.mse_loss(predicted_intensity.squeeze(), expected_intensity.squeeze())
        
        return physics_loss
    
    def get_loss_weight(self) -> float:
        return self.loss_weight


class HeatTransferLaw(PhysicsLaw):
    """
    Heat transfer physics for infrared target detection
    
    Models heat conduction, convection, and radiation:
    - Heat diffusion equation: ∂T/∂t = α∇²T
    - Stefan-Boltzmann law for thermal radiation
    - Temperature-dependent material properties
    - Thermal boundary conditions
    """
    
    def __init__(
        self,
        thermal_diffusivity: float = 1e-6,  # m²/s (typical for metals)
        emissivity: float = 0.8,
        stefan_boltzmann: float = 5.67e-8,  # W/(m²·K⁴)
        loss_weight: float = 0.2
    ):
        """
        Args:
            thermal_diffusivity: Material thermal diffusivity
            emissivity: Surface emissivity
            stefan_boltzmann: Stefan-Boltzmann constant
            loss_weight: Weight for heat transfer physics loss
        """
        self.thermal_diffusivity = thermal_diffusivity
        self.emissivity = emissivity
        self.stefan_boltzmann = stefan_boltzmann
        self.loss_weight = loss_weight
    
    def compute_heat_diffusion_residual(
        self,
        temperature: torch.Tensor,
        coordinates: torch.Tensor,
        time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute residual of heat diffusion equation
        
        Heat equation: ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
        """
        # Ensure gradients can be computed
        temperature.requires_grad_(True)
        coordinates.requires_grad_(True)
        
        # Compute spatial gradients
        grad_T = grad(
            outputs=temperature.sum(),
            inputs=coordinates,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # First-order spatial derivatives
        dT_dx = grad_T[:, 0]
        dT_dy = grad_T[:, 1]
        
        # Second-order spatial derivatives (Laplacian)
        d2T_dx2 = grad(
            outputs=dT_dx.sum(),
            inputs=coordinates,
            create_graph=True,
            retain_graph=True
        )[0][:, 0]
        
        d2T_dy2 = grad(
            outputs=dT_dy.sum(),
            inputs=coordinates,
            create_graph=True,
            retain_graph=True
        )[0][:, 1]
        
        laplacian_T = d2T_dx2 + d2T_dy2
        
        # Time derivative (if available)
        if time is not None:
            time.requires_grad_(True)
            dT_dt = grad(
                outputs=temperature.sum(),
                inputs=time,
                create_graph=True,
                retain_graph=True
            )[0]
        else:
            # Steady-state assumption: ∂T/∂t = 0
            dT_dt = torch.zeros_like(temperature)
        
        # Heat equation residual
        residual = dT_dt - self.thermal_diffusivity * laplacian_T
        
        return residual
    
    def compute_stefan_boltzmann_constraint(
        self,
        temperature: torch.Tensor,
        radiated_power: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce Stefan-Boltzmann law: P = εσAT⁴
        """
        # Convert temperature to Kelvin if necessary
        T_kelvin = temperature + 273.15 if temperature.mean() < 100 else temperature
        
        # Theoretical radiated power (per unit area)
        theoretical_power = self.emissivity * self.stefan_boltzmann * (T_kelvin ** 4)
        
        # Loss between predicted and theoretical power
        power_loss = F.mse_loss(radiated_power, theoretical_power)
        
        return power_loss
    
    def compute_physics_loss(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        outputs: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute heat transfer physics loss
        """
        total_loss = 0.0
        
        # Extract temperature predictions from model
        if isinstance(outputs, dict):
            temperature = outputs.get('temperature')
            radiated_power = outputs.get('radiated_power')
        else:
            # Assume outputs represent temperature
            temperature = outputs
            radiated_power = None
        
        if temperature is None:
            return torch.tensor(0.0, device=inputs.device)
        
        # Heat diffusion equation constraint
        if coordinates is not None:
            diffusion_residual = self.compute_heat_diffusion_residual(
                temperature, coordinates, time
            )
            diffusion_loss = torch.mean(diffusion_residual ** 2)
            total_loss += diffusion_loss
        
        # Stefan-Boltzmann radiation constraint
        if radiated_power is not None:
            radiation_loss = self.compute_stefan_boltzmann_constraint(
                temperature, radiated_power
            )
            total_loss += radiation_loss
        
        return total_loss
    
    def get_loss_weight(self) -> float:
        return self.loss_weight


class InfraredRadiationLaw(PhysicsLaw):
    """
    Infrared radiation physics for target detection
    
    Models Planck's law and Wien's displacement law:
    - Planck blackbody radiation formula
    - Wien's displacement law for peak wavelength
    - Emissivity corrections for real materials
    - Atmospheric window considerations
    """
    
    def __init__(
        self,
        planck_h: float = 6.626e-34,  # Planck constant
        speed_light: float = 3e8,     # Speed of light
        boltzmann_k: float = 1.381e-23,  # Boltzmann constant
        loss_weight: float = 0.15
    ):
        """
        Args:
            planck_h: Planck constant
            speed_light: Speed of light
            boltzmann_k: Boltzmann constant
            loss_weight: Weight for IR physics loss
        """
        self.planck_h = planck_h
        self.speed_light = speed_light
        self.boltzmann_k = boltzmann_k
        self.loss_weight = loss_weight
        
        # Wien's displacement constant
        self.wien_constant = 2.898e-3  # m·K
    
    def planck_blackbody_radiance(
        self,
        wavelength: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Planck blackbody radiance
        
        B(λ,T) = (2hc²/λ⁵) * 1/(exp(hc/λkT) - 1)
        """
        # Constants
        c1 = 2 * self.planck_h * (self.speed_light ** 2)  # W·m²
        c2 = (self.planck_h * self.speed_light) / self.boltzmann_k  # m·K
        
        # Planck function
        numerator = c1 / (wavelength ** 5)
        denominator = torch.exp(c2 / (wavelength * temperature)) - 1
        
        radiance = numerator / denominator
        
        return radiance
    
    def wien_displacement_constraint(
        self,
        temperature: torch.Tensor,
        peak_wavelength: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce Wien's displacement law: λ_max = b/T
        """
        theoretical_peak = self.wien_constant / temperature
        
        wien_loss = F.mse_loss(peak_wavelength, theoretical_peak)
        
        return wien_loss
    
    def compute_physics_loss(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        outputs: torch.Tensor,
        wavelength: float = 10e-6,  # Default LWIR wavelength
        **kwargs
    ) -> torch.Tensor:
        """
        Compute infrared radiation physics loss
        """
        total_loss = 0.0
        
        # Extract relevant predictions
        if isinstance(outputs, dict):
            temperature = outputs.get('temperature')
            radiance = outputs.get('radiance')
            peak_wavelength = outputs.get('peak_wavelength')
        else:
            # Try to infer from output shape
            if outputs.size(1) >= 3:  # Multi-output model
                temperature = outputs[:, 0]
                radiance = outputs[:, 1]
                peak_wavelength = outputs[:, 2] if outputs.size(1) > 2 else None
            else:
                temperature = None
                radiance = None
                peak_wavelength = None
        
        # Planck's law constraint
        if temperature is not None and radiance is not None:
            wavelength_tensor = torch.full_like(temperature, wavelength)
            theoretical_radiance = self.planck_blackbody_radiance(
                wavelength_tensor, temperature + 273.15  # Convert to Kelvin
            )
            
            planck_loss = F.mse_loss(radiance, theoretical_radiance)
            total_loss += planck_loss
        
        # Wien's displacement law constraint
        if temperature is not None and peak_wavelength is not None:
            wien_loss = self.wien_displacement_constraint(
                temperature + 273.15, peak_wavelength
            )
            total_loss += wien_loss
        
        return total_loss
    
    def get_loss_weight(self) -> float:
        return self.loss_weight


class PhysicsInformedIRSTNet(nn.Module):
    """
    Physics-Informed Neural Network for Infrared Small Target Detection
    
    Integrates multiple physics laws and constraints into the learning process
    to improve generalization and physical consistency of predictions.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        hidden_dims: List[int] = [64, 128, 256],
        physics_laws: Optional[List[PhysicsLaw]] = None,
        predict_physics: bool = True
    ):
        """
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions
            physics_laws: List of physics laws to enforce
            predict_physics: Whether to predict physics quantities
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.physics_laws = physics_laws or []
        self.predict_physics = predict_physics
        
        # Feature extraction backbone
        self.feature_extractor = self._build_feature_extractor(hidden_dims)
        
        # Task-specific heads
        self.classification_head = nn.Linear(hidden_dims[-1], num_classes)
        
        if predict_physics:
            # Physics prediction heads
            self.temperature_head = nn.Linear(hidden_dims[-1], 1)
            self.radiance_head = nn.Linear(hidden_dims[-1], 1)
            self.intensity_head = nn.Linear(hidden_dims[-1], 1)
            
            # Optional additional physics quantities
            self.emissivity_head = nn.Linear(hidden_dims[-1], 1)
            self.atmospheric_transmission_head = nn.Linear(hidden_dims[-1], 1)
    
    def _build_feature_extractor(self, hidden_dims: List[int]) -> nn.Module:
        """Build convolutional feature extractor"""
        layers = []
        
        # Initial convolution
        layers.extend([
            nn.Conv2d(self.input_channels, hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ])
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.extend([
                nn.Conv2d(hidden_dims[i-1], hidden_dims[i], 3, padding=1),
                nn.BatchNorm2d(hidden_dims[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
        
        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        coordinates: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        return_physics: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics-informed predictions
        
        Args:
            x: Input infrared images
            coordinates: Spatial coordinates for physics equations
            time: Time coordinate for temporal physics
            return_physics: Whether to return physics predictions
            
        Returns:
            Dictionary with predictions and physics quantities
        """
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Classification prediction
        logits = self.classification_head(features)
        
        outputs = {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1)
        }
        
        if self.predict_physics and return_physics:
            # Physics quantity predictions
            temperature = self.temperature_head(features)
            radiance = self.radiance_head(features)
            intensity = self.intensity_head(features)
            
            outputs.update({
                'temperature': temperature.squeeze(-1),
                'radiance': F.relu(radiance.squeeze(-1)),  # Radiance must be positive
                'intensity': F.relu(intensity.squeeze(-1)),  # Intensity must be positive
            })
            
            # Additional physics quantities
            if hasattr(self, 'emissivity_head'):
                emissivity = torch.sigmoid(self.emissivity_head(features))  # 0-1 range
                outputs['emissivity'] = emissivity.squeeze(-1)
            
            if hasattr(self, 'atmospheric_transmission_head'):
                transmission = torch.sigmoid(self.atmospheric_transmission_head(features))  # 0-1 range
                outputs['atmospheric_transmission'] = transmission.squeeze(-1)
        
        return outputs
    
    def compute_physics_loss(
        self, 
        inputs: torch.Tensor, 
        outputs: Dict[str, torch.Tensor],
        **physics_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss terms
        
        Returns:
            Dictionary with individual and total physics losses
        """
        physics_losses = {}
        total_physics_loss = 0.0
        
        for i, physics_law in enumerate(self.physics_laws):
            law_name = physics_law.__class__.__name__
            
            try:
                law_loss = physics_law.compute_physics_loss(
                    self, inputs, outputs, **physics_kwargs
                )
                
                weighted_loss = physics_law.get_loss_weight() * law_loss
                physics_losses[f'{law_name}_loss'] = law_loss
                physics_losses[f'{law_name}_weighted'] = weighted_loss
                
                total_physics_loss += weighted_loss
                
            except Exception as e:
                # Log warning but don't fail training
                print(f"Warning: Physics law {law_name} failed: {e}")
                physics_losses[f'{law_name}_loss'] = torch.tensor(0.0, device=inputs.device)
        
        physics_losses['total_physics_loss'] = total_physics_loss
        
        return physics_losses


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss function for physics-informed IRST networks
    
    Balances task-specific losses with physics constraints
    """
    
    def __init__(
        self,
        task_loss_weight: float = 1.0,
        physics_loss_weight: float = 0.1,
        temperature_loss_weight: float = 0.05,
        adaptive_weighting: bool = True
    ):
        """
        Args:
            task_loss_weight: Weight for primary task loss (classification)
            physics_loss_weight: Global weight for physics losses
            temperature_loss_weight: Weight for temperature prediction loss
            adaptive_weighting: Whether to adaptively adjust loss weights
        """
        super().__init__()
        
        self.task_loss_weight = task_loss_weight
        self.physics_loss_weight = physics_loss_weight
        self.temperature_loss_weight = temperature_loss_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Task losses
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # Adaptive weighting parameters
        if adaptive_weighting:
            self.loss_weights = nn.Parameter(
                torch.ones(3),  # [task, physics, temperature]
                requires_grad=True
            )
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        physics_losses: Dict[str, torch.Tensor],
        temperature_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with physics constraints
        
        Args:
            model_outputs: Model predictions
            targets: Classification targets
            physics_losses: Physics constraint losses
            temperature_targets: Optional temperature ground truth
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Primary task loss (classification)
        task_loss = self.classification_loss(model_outputs['logits'], targets)
        losses['task_loss'] = task_loss
        
        # Temperature prediction loss (if available)
        temp_loss = torch.tensor(0.0, device=task_loss.device)
        if 'temperature' in model_outputs and temperature_targets is not None:
            temp_loss = self.regression_loss(
                model_outputs['temperature'], 
                temperature_targets
            )
        losses['temperature_loss'] = temp_loss
        
        # Physics losses
        total_physics_loss = physics_losses.get('total_physics_loss', 
                                              torch.tensor(0.0, device=task_loss.device))
        losses['physics_loss'] = total_physics_loss
        
        # Add individual physics losses for monitoring
        for key, value in physics_losses.items():
            if key != 'total_physics_loss':
                losses[f'physics_{key}'] = value
        
        # Compute total loss
        if self.adaptive_weighting:
            # Use learnable weights with uncertainty weighting
            weights = F.softmax(self.loss_weights, dim=0)
            total_loss = (
                weights[0] * task_loss +
                weights[1] * total_physics_loss +
                weights[2] * temp_loss
            )
            
            losses['adaptive_weights'] = weights
        else:
            # Fixed weighting
            total_loss = (
                self.task_loss_weight * task_loss +
                self.physics_loss_weight * total_physics_loss +
                self.temperature_loss_weight * temp_loss
            )
        
        losses['total_loss'] = total_loss
        
        return losses


# Factory functions and utilities

def create_physics_informed_model(
    model_type: str = 'standard',
    input_channels: int = 1,
    num_classes: int = 2,
    physics_laws: Optional[List[str]] = None,
    **kwargs
) -> PhysicsInformedIRSTNet:
    """
    Factory function to create physics-informed IRST models
    
    Args:
        model_type: Type of model architecture
        input_channels: Number of input channels
        num_classes: Number of output classes
        physics_laws: List of physics law names to include
        **kwargs: Additional model parameters
        
    Returns:
        Physics-informed neural network
    """
    physics_laws = physics_laws or ['atmospheric', 'heat_transfer', 'infrared']
    
    # Create physics law instances
    law_instances = []
    
    for law_name in physics_laws:
        if law_name == 'atmospheric':
            law_instances.append(AtmosphericPropagationLaw())
        elif law_name == 'heat_transfer':
            law_instances.append(HeatTransferLaw())
        elif law_name == 'infrared':
            law_instances.append(InfraredRadiationLaw())
        else:
            print(f"Warning: Unknown physics law '{law_name}' ignored")
    
    # Create model
    model = PhysicsInformedIRSTNet(
        input_channels=input_channels,
        num_classes=num_classes,
        physics_laws=law_instances,
        **kwargs
    )
    
    return model


def create_physics_informed_loss(
    physics_weight: float = 0.1,
    **kwargs
) -> PhysicsInformedLoss:
    """
    Factory function to create physics-informed loss
    
    Args:
        physics_weight: Global weight for physics losses
        **kwargs: Additional loss parameters
        
    Returns:
        Physics-informed loss function
    """
    return PhysicsInformedLoss(
        physics_loss_weight=physics_weight,
        **kwargs
    )


# Export main components
__all__ = [
    'PhysicsLaw',
    'AtmosphericPropagationLaw',
    'HeatTransferLaw', 
    'InfraredRadiationLaw',
    'PhysicsInformedIRSTNet',
    'PhysicsInformedLoss',
    'create_physics_informed_model',
    'create_physics_informed_loss'
]
