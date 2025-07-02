"""
NUAA-SIRST Dataset implementation.
Paper: Dense Nested Attention Network for Infrared Small Target Detection
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import albumentations as A

from ...core.base import BaseDataset
from ...core.registry import DATASET_REGISTRY
from ...utils.image import load_image, normalize_image


@DATASET_REGISTRY.register("nuaa_sirst")
class NUAASIRSTDataset(BaseDataset):
    """
    NUAA-SIRST Dataset for infrared small target detection.
    
    Dataset contains:
    - 427 images with various small targets
    - Images size: 256x256 pixels
    - Targets: aircraft, ships, vehicles
    - Annotations: binary masks
    
    Args:
        root_dir: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Albumentations transform pipeline
        target_transform: Transform for target masks
        return_paths: Whether to return image paths
        cache_in_memory: Whether to cache images in memory
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_transform: Optional[Callable] = None,
        return_paths: bool = False,
        cache_in_memory: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.cache_in_memory = cache_in_memory
        
        # Dataset paths
        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"
        
        # Load file lists
        self.image_files, self.mask_files = self._load_file_lists()
        
        # Cache for in-memory loading
        self._image_cache = {} if cache_in_memory else None
        self._mask_cache = {} if cache_in_memory else None
        
        # Dataset info
        self.dataset_info = {
            "name": "NUAA-SIRST",
            "total_images": 427,
            "image_size": (256, 256),
            "num_classes": 1,
            "target_types": ["aircraft", "ships", "vehicles"],
            "paper": "Dense Nested Attention Network for Infrared Small Target Detection"
        }
        
        print(f"Loaded {len(self.image_files)} images for {split} split")
    
    def _load_file_lists(self) -> Tuple[List[str], List[str]]:
        """Load image and mask file lists"""
        
        # Check if split files exist
        split_file = self.root_dir / f"{self.split}.txt"
        
        if split_file.exists():
            # Load from split file
            with open(split_file, 'r') as f:
                filenames = [line.strip() for line in f.readlines()]
        else:
            # Create splits automatically
            all_images = list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg"))
            all_images.sort()
            
            # Default split ratios
            total = len(all_images)
            if self.split == "train":
                start_idx, end_idx = 0, int(0.7 * total)
            elif self.split == "val":
                start_idx, end_idx = int(0.7 * total), int(0.85 * total)
            else:  # test
                start_idx, end_idx = int(0.85 * total), total
            
            filenames = [img.stem for img in all_images[start_idx:end_idx]]
        
        # Build full paths
        image_files = [str(self.images_dir / f"{name}.png") for name in filenames]
        mask_files = [str(self.masks_dir / f"{name}.png") for name in filenames]
        
        # Check if files exist
        image_files = [f for f in image_files if os.path.exists(f)]
        mask_files = [f for f in mask_files if os.path.exists(f)]
        
        return image_files, mask_files
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """Get a single sample"""
        
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # Load from cache or disk
        if self.cache_in_memory and idx in self._image_cache:
            image = self._image_cache[idx]
            mask = self._mask_cache[idx]
        else:
            # Load image and mask
            image = load_image(image_path, grayscale=True)
            mask = load_image(mask_path, grayscale=True)
            
            # Ensure mask is binary
            mask = (mask > 128).astype(np.uint8) * 255
            
            # Cache if needed
            if self.cache_in_memory:
                self._image_cache[idx] = image.copy()
                self._mask_cache[idx] = mask.copy()
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if len(image.shape) == 2:
                image = image.unsqueeze(0)  # Add channel dimension
            elif len(image.shape) == 3:
                image = image.permute(2, 0, 1)
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Add channel dimension
            elif len(mask.shape) == 3:
                mask = mask.permute(2, 0, 1)
        
        # Normalize image
        image = normalize_image(image)
        
        # Normalize mask to [0, 1]
        mask = mask / 255.0
        
        # Apply target transform
        if self.target_transform:
            mask = self.target_transform(mask)
        
        # Prepare output
        sample = {
            "image": image,
            "mask": mask,
            "target": mask,  # Alias for compatibility
        }
        
        if self.return_paths:
            sample["image_path"] = image_path
            sample["mask_path"] = mask_path
        
        return sample
    
    @staticmethod
    def get_default_transforms(
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = True
    ) -> A.Compose:
        """Get default data augmentation transforms"""
        
        if is_training:
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ])
        else:
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
            ])
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling class imbalance"""
        
        total_pixels = 0
        positive_pixels = 0
        
        print("Calculating class weights...")
        for idx in range(len(self)):
            sample = self.__getitem__(idx)
            mask = sample["mask"]
            
            total_pixels += mask.numel()
            positive_pixels += (mask > 0.5).sum().item()
        
        negative_pixels = total_pixels - positive_pixels
        
        # Avoid division by zero
        if positive_pixels == 0:
            pos_weight = 1.0
        else:
            pos_weight = negative_pixels / positive_pixels
        
        return torch.tensor([pos_weight], dtype=torch.float32)
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate dataset statistics"""
        
        pixel_values = []
        target_counts = []
        target_sizes = []
        
        print("Calculating dataset statistics...")
        for idx in range(min(100, len(self))):  # Sample subset for speed
            sample = self.__getitem__(idx)
            image = sample["image"]
            mask = sample["mask"]
            
            # Image statistics
            pixel_values.extend(image.flatten().tolist())
            
            # Target statistics
            binary_mask = (mask > 0.5).float()
            num_targets = self._count_connected_components(binary_mask.squeeze().numpy())
            target_counts.append(num_targets)
            
            if num_targets > 0:
                target_size = binary_mask.sum().item()
                target_sizes.append(target_size)
        
        stats = {
            "mean_pixel_value": float(np.mean(pixel_values)),
            "std_pixel_value": float(np.std(pixel_values)),
            "mean_targets_per_image": float(np.mean(target_counts)),
            "std_targets_per_image": float(np.std(target_counts)),
            "mean_target_size": float(np.mean(target_sizes)) if target_sizes else 0.0,
            "std_target_size": float(np.std(target_sizes)) if target_sizes else 0.0,
        }
        
        return stats
    
    def _count_connected_components(self, binary_mask: np.ndarray) -> int:
        """Count connected components (targets) in binary mask"""
        
        binary_mask = (binary_mask * 255).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(binary_mask)
        return num_labels - 1  # Subtract background
    
    def visualize_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Visualize a sample"""
        
        sample = self.__getitem__(idx)
        image = sample["image"].squeeze().numpy()
        mask = sample["mask"].squeeze().numpy()
        
        # Denormalize image for visualization
        image = (image * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)
        
        # Create overlay
        overlay = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            0.7,
            overlay,
            0.3,
            0
        )
        
        return {
            "image": image,
            "mask": mask,
            "overlay": overlay
        }
