"""
Advanced Synthetic Data Generation for IRST Library

This module provides cutting-edge synthetic data generation capabilities for
infrared small target detection, including GANs, physics-based rendering,
domain randomization, and procedural scene generation.

Key Features:
- GAN-based infrared scene synthesis
- Physics-based thermal rendering
- Domain randomization for robustness
- Procedural background generation
- Synthetic target placement
- Data augmentation pipelines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import numpy as np
import cv2
import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    image_size: Tuple[int, int] = (256, 256)
    num_targets: Tuple[int, int] = (1, 3)      # Min, max targets per image
    target_size_range: Tuple[int, int] = (3, 15)  # Target size in pixels
    temperature_range: Tuple[float, float] = (300.0, 400.0)  # Kelvin
    background_temp: Tuple[float, float] = (250.0, 300.0)    # Background temp
    noise_level: float = 0.05
    atmospheric_effects: bool = True
    domain_randomization: bool = True


class SyntheticDataGenerator(ABC):
    """Base class for synthetic data generators"""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Generate a batch of synthetic data"""
        pass
    
    @abstractmethod
    def generate_single(self) -> Dict[str, torch.Tensor]:
        """Generate a single synthetic sample"""
        pass


class IRTargetGAN(nn.Module):
    """
    Generative Adversarial Network for Infrared Target Generation
    
    Specialized GAN architecture for generating realistic infrared small targets
    with proper thermal characteristics and background integration.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        image_size: int = 256,
        num_channels: int = 1,
        generator_features: int = 64,
        discriminator_features: int = 64
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_channels = num_channels
        
        # Generator network
        self.generator = self._build_generator(generator_features)
        
        # Discriminator network
        self.discriminator = self._build_discriminator(discriminator_features)
        
        # Temperature-aware components
        self.temperature_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Normalize to 0-1, will scale later
        )
    
    def _build_generator(self, features: int) -> nn.Module:
        """Build generator network with progressive upsampling"""
        
        # Calculate number of upsampling layers needed
        num_layers = int(math.log2(self.image_size)) - 2  # Start from 4x4
        
        layers = []
        
        # Initial dense layer
        initial_size = 4
        layers.extend([
            nn.Linear(self.latent_dim, features * 8 * initial_size * initial_size),
            nn.ReLU(),
            nn.Unflatten(1, (features * 8, initial_size, initial_size))
        ])
        
        # Progressive upsampling blocks
        current_features = features * 8
        
        for i in range(num_layers):
            next_features = current_features // 2
            
            layers.extend([
                nn.ConvTranspose2d(
                    current_features, next_features, 
                    kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(next_features),
                nn.ReLU(inplace=True)
            ])
            
            current_features = next_features
        
        # Final output layer
        layers.extend([
            nn.ConvTranspose2d(
                current_features, self.num_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()  # Output in [-1, 1] range
        ])
        
        return nn.Sequential(*layers)
    
    def _build_discriminator(self, features: int) -> nn.Module:
        """Build discriminator network with progressive downsampling"""
        
        layers = []
        
        # Initial convolution
        layers.extend([
            nn.Conv2d(self.num_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Progressive downsampling
        current_features = features
        
        while self.image_size // (2 ** len([l for l in layers if isinstance(l, nn.Conv2d)])) > 4:
            next_features = min(current_features * 2, 512)
            
            layers.extend([
                nn.Conv2d(current_features, next_features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_features),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            current_features = next_features
        
        # Final classification layer
        layers.extend([
            nn.Conv2d(current_features, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        z: torch.Tensor, 
        mode: str = 'generate'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the GAN
        
        Args:
            z: Latent vectors
            mode: 'generate', 'discriminate', or 'both'
            
        Returns:
            Dictionary with generated images and/or discriminator outputs
        """
        results = {}
        
        if mode in ['generate', 'both']:
            # Generate images
            generated_images = self.generator(z)
            
            # Predict temperature values
            temperature_logits = self.temperature_predictor(z)
            
            results.update({
                'generated_images': generated_images,
                'temperature_logits': temperature_logits
            })
        
        if mode in ['discriminate', 'both']:
            # Discriminate real vs fake
            if 'generated_images' not in results:
                generated_images = self.generator(z)
            else:
                generated_images = results['generated_images']
            
            discriminator_output = self.discriminator(generated_images)
            results['discriminator_output'] = discriminator_output
        
        return results


class PhysicsBasedRenderer:
    """
    Physics-based infrared scene renderer
    
    Simulates realistic infrared imagery using physical principles:
    - Stefan-Boltzmann radiation law
    - Atmospheric propagation
    - Thermal emission modeling
    - Material property simulation
    """
    
    def __init__(
        self,
        config: SyntheticDataConfig,
        wavelength: float = 10e-6,  # 10 μm (LWIR)
        atmospheric_transmission: float = 0.8
    ):
        self.config = config
        self.wavelength = wavelength
        self.atmospheric_transmission = atmospheric_transmission
        
        # Physical constants
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.planck_h = 6.626e-34
        self.speed_light = 3e8
        self.boltzmann_k = 1.381e-23
        
        self.logger = logging.getLogger(__name__)
    
    def blackbody_radiance(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute blackbody radiance using Planck's law
        
        B(λ,T) = (2hc²/λ⁵) * 1/(exp(hc/λkT) - 1)
        """
        # Constants for the wavelength
        c1 = 2 * self.planck_h * (self.speed_light ** 2)  # W·m²
        c2 = (self.planck_h * self.speed_light) / self.boltzmann_k  # m·K
        
        # Planck function
        numerator = c1 / (self.wavelength ** 5)
        exponent = c2 / (self.wavelength * temperature)
        denominator = torch.exp(exponent) - 1
        
        radiance = numerator / denominator
        
        return radiance
    
    def generate_background(self, batch_size: int) -> torch.Tensor:
        """
        Generate realistic infrared backgrounds with spatial temperature variation
        """
        height, width = self.config.image_size
        
        # Base temperature field
        temp_min, temp_max = self.config.background_temp
        base_temp = torch.rand(batch_size, 1, height, width) * (temp_max - temp_min) + temp_min
        
        # Add spatial correlations using Gaussian filtering
        if self.config.domain_randomization:
            # Random kernel sizes for different correlation lengths
            kernel_sizes = [5, 9, 15, 21]
            selected_kernel = random.choice(kernel_sizes)
            
            # Create Gaussian kernel
            sigma = selected_kernel / 6.0
            kernel = self._create_gaussian_kernel(selected_kernel, sigma)
            kernel = kernel.expand(1, 1, -1, -1).to(base_temp.device)
            
            # Apply smoothing
            padding = selected_kernel // 2
            smoothed_temp = F.conv2d(
                base_temp, kernel, padding=padding, groups=1
            )
            
            base_temp = 0.7 * base_temp + 0.3 * smoothed_temp
        
        # Add atmospheric effects
        if self.config.atmospheric_effects:
            # Random atmospheric gradients
            y_coords = torch.linspace(0, 1, height).view(1, 1, -1, 1)
            x_coords = torch.linspace(0, 1, width).view(1, 1, 1, -1)
            
            # Random gradient directions
            for b in range(batch_size):
                gradient_strength = torch.rand(1) * 5.0  # Max 5K gradient
                gradient_direction = torch.rand(2) * 2 - 1  # Random direction
                
                gradient = (
                    gradient_direction[0] * x_coords + 
                    gradient_direction[1] * y_coords
                ) * gradient_strength
                
                base_temp[b] += gradient
        
        # Convert temperature to radiance
        radiance = self.blackbody_radiance(base_temp)
        
        # Normalize to image values (0-1 range)
        normalized_radiance = self._normalize_radiance(radiance)
        
        return normalized_radiance
    
    def generate_targets(
        self, 
        batch_size: int,
        backgrounds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Generate synthetic targets with realistic thermal signatures
        
        Returns:
            Tuple of (images_with_targets, binary_masks, target_metadata)
        """
        height, width = self.config.image_size
        
        images_with_targets = backgrounds.clone()
        binary_masks = torch.zeros_like(backgrounds)
        target_metadata = []
        
        for b in range(batch_size):
            batch_metadata = []
            
            # Random number of targets
            num_targets = random.randint(*self.config.num_targets)
            
            for t in range(num_targets):
                # Random target properties
                target_size = random.randint(*self.config.target_size_range)
                target_temp = random.uniform(*self.config.temperature_range)
                
                # Random position (ensure target fits in image)
                margin = target_size // 2 + 1
                x = random.randint(margin, width - margin - 1)
                y = random.randint(margin, height - margin - 1)
                
                # Target shape (circular or elliptical)
                if random.random() < 0.7:  # 70% circular
                    target_mask = self._create_circular_target(target_size)
                else:  # 30% elliptical
                    aspect_ratio = random.uniform(0.5, 2.0)
                    target_mask = self._create_elliptical_target(target_size, aspect_ratio)
                
                # Place target in image
                target_radiance = self.blackbody_radiance(torch.tensor(target_temp))
                normalized_target_radiance = self._normalize_radiance(target_radiance)
                
                # Extract region
                y1, y2 = y - target_size//2, y + target_size//2 + 1
                x1, x2 = x - target_size//2, x + target_size//2 + 1
                
                # Ensure we don't go out of bounds
                y1, y2 = max(0, y1), min(height, y2)
                x1, x2 = max(0, x1), min(width, x2)
                
                # Adjust mask size if needed
                mask_h, mask_w = y2 - y1, x2 - x1
                if target_mask.shape != (mask_h, mask_w):
                    target_mask = F.interpolate(
                        target_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(mask_h, mask_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().bool()
                
                # Blend target with background
                background_region = images_with_targets[b, 0, y1:y2, x1:x2]
                blended_region = torch.where(
                    target_mask,
                    normalized_target_radiance.item(),
                    background_region
                )
                
                images_with_targets[b, 0, y1:y2, x1:x2] = blended_region
                binary_masks[b, 0, y1:y2, x1:x2] = torch.where(
                    target_mask,
                    torch.ones_like(background_region),
                    binary_masks[b, 0, y1:y2, x1:x2]
                )
                
                # Store metadata
                batch_metadata.append({
                    'x': x,
                    'y': y,
                    'size': target_size,
                    'temperature': target_temp,
                    'bbox': [x1, y1, x2, y2],
                    'shape': 'circular' if target_mask.sum() / (target_size ** 2) > 0.6 else 'elliptical'
                })
            
            target_metadata.append(batch_metadata)
        
        # Add atmospheric attenuation
        if self.config.atmospheric_effects:
            images_with_targets *= self.atmospheric_transmission
        
        # Add noise
        if self.config.noise_level > 0:
            noise = torch.randn_like(images_with_targets) * self.config.noise_level
            images_with_targets += noise
        
        # Ensure values remain in valid range
        images_with_targets = torch.clamp(images_with_targets, 0, 1)
        
        return images_with_targets, binary_masks, target_metadata
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g2d = g1d[None, :] * g1d[:, None]
        return g2d / g2d.sum()
    
    def _create_circular_target(self, size: int) -> torch.Tensor:
        """Create circular target mask"""
        center = size // 2
        y, x = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing='ij'
        )
        
        distance = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        radius = size // 2
        
        return distance <= radius
    
    def _create_elliptical_target(self, size: int, aspect_ratio: float) -> torch.Tensor:
        """Create elliptical target mask"""
        center = size // 2
        y, x = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing='ij'
        )
        
        # Ellipse parameters
        a = size // 2  # Semi-major axis
        b = a / aspect_ratio  # Semi-minor axis
        
        # Ellipse equation: (x-cx)²/a² + (y-cy)²/b² <= 1
        ellipse_eq = ((x - center) ** 2) / (a ** 2) + ((y - center) ** 2) / (b ** 2)
        
        return ellipse_eq <= 1
    
    def _normalize_radiance(self, radiance: torch.Tensor) -> torch.Tensor:
        """Normalize radiance values to [0, 1] range"""
        # Use percentile-based normalization for better contrast
        percentile_5 = torch.quantile(radiance, 0.05)
        percentile_95 = torch.quantile(radiance, 0.95)
        
        normalized = (radiance - percentile_5) / (percentile_95 - percentile_5)
        return torch.clamp(normalized, 0, 1)


class DomainRandomizationEngine:
    """
    Domain randomization engine for robust synthetic data generation
    
    Applies systematic variations to synthetic data to improve model
    generalization across different environmental conditions.
    """
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def randomize_atmospheric_conditions(
        self, 
        images: torch.Tensor,
        metadata: List[Dict]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Apply random atmospheric effects"""
        
        batch_size = images.size(0)
        randomized_images = images.clone()
        
        for b in range(batch_size):
            # Random atmospheric transmission
            transmission = random.uniform(0.6, 0.95)
            
            # Random haze/fog effect
            if random.random() < 0.3:  # 30% chance of haze
                haze_strength = random.uniform(0.1, 0.4)
                haze_pattern = torch.rand_like(images[b])
                randomized_images[b] = (
                    randomized_images[b] * (1 - haze_strength) + 
                    haze_pattern * haze_strength
                )
            
            # Apply transmission
            randomized_images[b] *= transmission
            
            # Update metadata
            for target_meta in metadata[b]:
                target_meta['atmospheric_transmission'] = transmission
        
        return randomized_images, metadata
    
    def randomize_sensor_characteristics(
        self, 
        images: torch.Tensor
    ) -> torch.Tensor:
        """Simulate different sensor characteristics"""
        
        batch_size = images.size(0)
        randomized_images = images.clone()
        
        for b in range(batch_size):
            # Random sensor noise
            noise_type = random.choice(['gaussian', 'salt_pepper', 'thermal'])
            
            if noise_type == 'gaussian':
                noise_std = random.uniform(0.01, 0.05)
                noise = torch.randn_like(images[b]) * noise_std
                randomized_images[b] += noise
                
            elif noise_type == 'salt_pepper':
                salt_pepper_prob = random.uniform(0.001, 0.01)
                mask = torch.rand_like(images[b]) < salt_pepper_prob
                salt_mask = torch.rand_like(images[b]) < 0.5
                
                randomized_images[b][mask & salt_mask] = 1.0  # Salt
                randomized_images[b][mask & ~salt_mask] = 0.0  # Pepper
                
            elif noise_type == 'thermal':
                # Thermal sensor noise (temperature-dependent)
                thermal_noise = torch.randn_like(images[b]) * images[b] * 0.02
                randomized_images[b] += thermal_noise
            
            # Random gain and offset
            gain = random.uniform(0.8, 1.2)
            offset = random.uniform(-0.1, 0.1)
            
            randomized_images[b] = randomized_images[b] * gain + offset
        
        # Ensure valid range
        randomized_images = torch.clamp(randomized_images, 0, 1)
        
        return randomized_images
    
    def randomize_geometric_properties(
        self, 
        images: torch.Tensor,
        masks: torch.Tensor,
        metadata: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """Apply random geometric transformations"""
        
        batch_size = images.size(0)
        height, width = self.config.image_size
        
        randomized_images = []
        randomized_masks = []
        updated_metadata = []
        
        for b in range(batch_size):
            # Random rotation
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)  # degrees
                
                # Create rotation matrix
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Apply rotation to image and mask
                img_np = images[b, 0].numpy()
                mask_np = masks[b, 0].numpy()
                
                rotated_img = cv2.warpAffine(img_np, rotation_matrix, (width, height))
                rotated_mask = cv2.warpAffine(mask_np, rotation_matrix, (width, height))
                
                randomized_images.append(torch.from_numpy(rotated_img).unsqueeze(0))
                randomized_masks.append(torch.from_numpy(rotated_mask).unsqueeze(0))
            else:
                randomized_images.append(images[b])
                randomized_masks.append(masks[b])
            
            # Update metadata (simplified - would need more complex transformation for accurate bbox updates)
            updated_metadata.append(metadata[b])
        
        randomized_images = torch.stack(randomized_images)
        randomized_masks = torch.stack(randomized_masks)
        
        return randomized_images, randomized_masks, updated_metadata


class ProceduralSyntheticDataset(Dataset):
    """
    Procedural synthetic dataset for infrared small target detection
    
    Generates infinite synthetic data on-the-fly using physics-based rendering
    and domain randomization techniques.
    """
    
    def __init__(
        self,
        config: SyntheticDataConfig,
        dataset_size: int = 10000,
        use_gan: bool = False,
        gan_model: Optional[IRTargetGAN] = None
    ):
        """
        Args:
            config: Synthetic data configuration
            dataset_size: Virtual dataset size (for iteration purposes)
            use_gan: Whether to use GAN for generation
            gan_model: Pre-trained GAN model (if use_gan=True)
        """
        self.config = config
        self.dataset_size = dataset_size
        self.use_gan = use_gan
        self.gan_model = gan_model
        
        # Initialize generators
        self.physics_renderer = PhysicsBasedRenderer(config)
        self.domain_randomizer = DomainRandomizationEngine(config)
        
        if use_gan and gan_model is not None:
            self.gan_model.eval()
        
        self.logger = logging.getLogger(__name__)
    
    def __len__(self) -> int:
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a single synthetic sample
        
        Returns:
            Dictionary with image, mask, and metadata
        """
        
        if self.use_gan and self.gan_model is not None:
            return self._generate_gan_sample()
        else:
            return self._generate_physics_sample()
    
    def _generate_physics_sample(self) -> Dict[str, torch.Tensor]:
        """Generate sample using physics-based rendering"""
        
        # Generate background
        background = self.physics_renderer.generate_background(batch_size=1)
        
        # Generate targets
        image_with_targets, mask, metadata = self.physics_renderer.generate_targets(
            batch_size=1, backgrounds=background
        )
        
        # Apply domain randomization
        if self.config.domain_randomization:
            # Atmospheric randomization
            image_with_targets, metadata = self.domain_randomizer.randomize_atmospheric_conditions(
                image_with_targets, metadata
            )
            
            # Sensor randomization
            image_with_targets = self.domain_randomizer.randomize_sensor_characteristics(
                image_with_targets
            )
            
            # Geometric randomization
            image_with_targets, mask, metadata = self.domain_randomizer.randomize_geometric_properties(
                image_with_targets, mask, metadata
            )
        
        # Convert to binary classification target
        has_target = len(metadata[0]) > 0
        classification_target = torch.tensor(1 if has_target else 0, dtype=torch.long)
        
        return {
            'image': image_with_targets.squeeze(0),  # Remove batch dimension
            'mask': mask.squeeze(0),
            'classification_target': classification_target,
            'metadata': metadata[0]
        }
    
    def _generate_gan_sample(self) -> Dict[str, torch.Tensor]:
        """Generate sample using pre-trained GAN"""
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(1, self.gan_model.latent_dim)
            
            # Generate image
            gan_output = self.gan_model(z, mode='generate')
            generated_image = gan_output['generated_images']
            temperature_pred = gan_output['temperature_logits']
            
            # Convert from [-1, 1] to [0, 1] range
            generated_image = (generated_image + 1) / 2
            
            # Create simple binary mask (threshold-based target detection)
            # This is a placeholder - in practice, you'd want a more sophisticated method
            threshold = 0.7
            binary_mask = (generated_image > threshold).float()
            
            # Classification target based on mask content
            has_target = binary_mask.sum() > 10  # Arbitrary threshold
            classification_target = torch.tensor(1 if has_target else 0, dtype=torch.long)
            
            # Simple metadata
            metadata = {
                'generated_by': 'gan',
                'temperature_pred': temperature_pred.item(),
                'latent_vector': z.numpy()
            }
        
        return {
            'image': generated_image.squeeze(0),
            'mask': binary_mask.squeeze(0),
            'classification_target': classification_target,
            'metadata': metadata
        }


class SyntheticDataPipeline:
    """
    Complete pipeline for synthetic data generation and augmentation
    
    Orchestrates multiple generation methods and provides unified interface
    for creating large-scale synthetic datasets.
    """
    
    def __init__(
        self,
        config: SyntheticDataConfig,
        output_dir: str,
        methods: List[str] = ['physics', 'gan'],
        mixing_ratios: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            config: Synthetic data configuration
            output_dir: Directory to save generated data
            methods: List of generation methods to use
            mixing_ratios: Ratios for mixing different generation methods
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.methods = methods
        self.mixing_ratios = mixing_ratios or {'physics': 0.7, 'gan': 0.3}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize generators
        self.generators = {}
        
        if 'physics' in methods:
            self.generators['physics'] = PhysicsBasedRenderer(config)
            self.generators['domain_randomizer'] = DomainRandomizationEngine(config)
        
        if 'gan' in methods:
            # Note: GAN would need to be pre-trained
            self.logger.warning("GAN generation requested but no pre-trained model provided")
        
        self.logger = logging.getLogger(__name__)
    
    def generate_dataset(
        self,
        num_samples: int,
        batch_size: int = 32,
        save_format: str = 'pytorch'  # 'pytorch', 'numpy', 'images'
    ) -> str:
        """
        Generate complete synthetic dataset
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size for generation
            save_format: Format to save data in
            
        Returns:
            Path to generated dataset
        """
        self.logger.info(f"Generating {num_samples} synthetic samples...")
        
        # Calculate samples per method
        samples_per_method = {}
        remaining_samples = num_samples
        
        for method, ratio in self.mixing_ratios.items():
            if method in self.methods:
                method_samples = int(num_samples * ratio)
                samples_per_method[method] = method_samples
                remaining_samples -= method_samples
        
        # Distribute remaining samples
        if remaining_samples > 0:
            first_method = next(iter(samples_per_method.keys()))
            samples_per_method[first_method] += remaining_samples
        
        # Generate data for each method
        all_data = {
            'images': [],
            'masks': [],
            'targets': [],
            'metadata': []
        }
        
        for method, num_method_samples in samples_per_method.items():
            self.logger.info(f"Generating {num_method_samples} samples using {method}...")
            
            if method == 'physics':
                method_data = self._generate_physics_batch(num_method_samples, batch_size)
            elif method == 'gan':
                method_data = self._generate_gan_batch(num_method_samples, batch_size)
            else:
                continue
            
            # Accumulate data
            for key in all_data.keys():
                all_data[key].extend(method_data[key])
        
        # Save dataset
        dataset_path = self._save_dataset(all_data, save_format)
        
        self.logger.info(f"Dataset generation complete. Saved to: {dataset_path}")
        
        return str(dataset_path)
    
    def _generate_physics_batch(
        self, 
        num_samples: int, 
        batch_size: int
    ) -> Dict[str, List]:
        """Generate batch using physics-based rendering"""
        
        physics_renderer = self.generators['physics']
        domain_randomizer = self.generators['domain_randomizer']
        
        data = {
            'images': [],
            'masks': [],
            'targets': [],
            'metadata': []
        }
        
        for start_idx in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start_idx)
            
            # Generate backgrounds
            backgrounds = physics_renderer.generate_background(current_batch_size)
            
            # Generate targets
            images, masks, metadata = physics_renderer.generate_targets(
                current_batch_size, backgrounds
            )
            
            # Apply domain randomization
            if self.config.domain_randomization:
                images, metadata = domain_randomizer.randomize_atmospheric_conditions(
                    images, metadata
                )
                images = domain_randomizer.randomize_sensor_characteristics(images)
                images, masks, metadata = domain_randomizer.randomize_geometric_properties(
                    images, masks, metadata
                )
            
            # Convert to classification targets
            targets = []
            for batch_metadata in metadata:
                has_target = len(batch_metadata) > 0
                targets.append(1 if has_target else 0)
            
            # Store batch data
            data['images'].extend(images.unbind(0))
            data['masks'].extend(masks.unbind(0))
            data['targets'].extend(targets)
            data['metadata'].extend(metadata)
        
        return data
    
    def _generate_gan_batch(self, num_samples: int, batch_size: int) -> Dict[str, List]:
        """Generate batch using GAN (placeholder implementation)"""
        
        # This would require a pre-trained GAN model
        self.logger.warning("GAN generation not implemented - using placeholder")
        
        # Return empty data for now
        return {
            'images': [],
            'masks': [],
            'targets': [],
            'metadata': []
        }
    
    def _save_dataset(
        self, 
        data: Dict[str, List], 
        save_format: str
    ) -> Path:
        """Save generated dataset in specified format"""
        
        if save_format == 'pytorch':
            # Save as PyTorch tensors
            dataset_file = self.output_dir / 'synthetic_dataset.pt'
            
            torch_data = {
                'images': torch.stack(data['images']),
                'masks': torch.stack(data['masks']),
                'targets': torch.tensor(data['targets']),
                'metadata': data['metadata'],
                'config': self.config
            }
            
            torch.save(torch_data, dataset_file)
            return dataset_file
        
        elif save_format == 'numpy':
            # Save as NumPy arrays
            dataset_file = self.output_dir / 'synthetic_dataset.npz'
            
            np.savez(
                dataset_file,
                images=torch.stack(data['images']).numpy(),
                masks=torch.stack(data['masks']).numpy(),
                targets=np.array(data['targets']),
                metadata=data['metadata']
            )
            
            return dataset_file
        
        elif save_format == 'images':
            # Save as individual image files
            images_dir = self.output_dir / 'images'
            masks_dir = self.output_dir / 'masks'
            
            images_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
            
            for i, (image, mask, target) in enumerate(zip(
                data['images'], data['masks'], data['targets']
            )):
                # Convert to numpy and scale to [0, 255]
                img_np = (image.squeeze().numpy() * 255).astype(np.uint8)
                mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)
                
                # Save images
                cv2.imwrite(str(images_dir / f'image_{i:06d}.png'), img_np)
                cv2.imwrite(str(masks_dir / f'mask_{i:06d}.png'), mask_np)
            
            # Save metadata and targets
            metadata_file = self.output_dir / 'metadata.json'
            import json
            with open(metadata_file, 'w') as f:
                json.dump({
                    'targets': data['targets'],
                    'metadata': data['metadata']
                }, f, indent=2)
            
            return self.output_dir
        
        else:
            raise ValueError(f"Unsupported save format: {save_format}")


# Factory functions

def create_synthetic_dataset(
    config: Optional[SyntheticDataConfig] = None,
    dataset_size: int = 10000,
    **kwargs
) -> ProceduralSyntheticDataset:
    """
    Factory function to create synthetic dataset
    
    Args:
        config: Synthetic data configuration
        dataset_size: Number of samples in virtual dataset
        **kwargs: Additional dataset parameters
        
    Returns:
        Configured synthetic dataset
    """
    if config is None:
        config = SyntheticDataConfig()
    
    return ProceduralSyntheticDataset(
        config=config,
        dataset_size=dataset_size,
        **kwargs
    )


def create_synthetic_pipeline(
    output_dir: str,
    config: Optional[SyntheticDataConfig] = None,
    **kwargs
) -> SyntheticDataPipeline:
    """
    Factory function to create synthetic data pipeline
    
    Args:
        output_dir: Directory to save generated data
        config: Synthetic data configuration
        **kwargs: Additional pipeline parameters
        
    Returns:
        Configured synthetic data pipeline
    """
    if config is None:
        config = SyntheticDataConfig()
    
    return SyntheticDataPipeline(
        config=config,
        output_dir=output_dir,
        **kwargs
    )


# Export main components
__all__ = [
    'SyntheticDataConfig',
    'SyntheticDataGenerator',
    'IRTargetGAN',
    'PhysicsBasedRenderer',
    'DomainRandomizationEngine',
    'ProceduralSyntheticDataset',
    'SyntheticDataPipeline',
    'create_synthetic_dataset',
    'create_synthetic_pipeline'
]
