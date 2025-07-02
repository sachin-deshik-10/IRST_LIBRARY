"""
SIRST Dataset implementation.
"""

import os
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

from ..core.base import BaseDataset
from ..core.registry import register_dataset


@register_dataset("sirst")
class SIRSTDataset(BaseDataset):
    """SIRST (Single Infrared Small Target) Dataset.
    
    Dataset structure:
    root/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/
        ├── val/
        └── test/
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(root, split, transform)
        self.target_transform = target_transform
        
        # Set up paths
        self.images_dir = self.root / "images" / split
        self.masks_dir = self.root / "masks" / split
        
        # Validate directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        
        # Load file lists
        self.image_files = self._get_file_list(self.images_dir)
        self.mask_files = self._get_file_list(self.masks_dir)
        
        # Ensure equal number of images and masks
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(
                f"Number of images ({len(self.image_files)}) != "
                f"number of masks ({len(self.mask_files)})"
            )
        
        # Sort to ensure correspondence
        self.image_files.sort()
        self.mask_files.sort()
    
    def _get_file_list(self, directory: Path) -> list:
        """Get list of image files in directory."""
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        files = []
        
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        
        return files
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load image and mask
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform:
            # If using albumentations, pass both image and mask
            if hasattr(self.transform, 'replay'):
                transformed = self.transform(image=image, mask=mask)
                image, mask = transformed["image"], transformed["mask"]
            else:
                # Standard torchvision transforms
                image = self.transform(image)
                if self.target_transform:
                    mask = self.target_transform(mask)
        
        # Ensure proper tensor format
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return {
            "image": image,
            "mask": mask,
            "meta": {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_name": image_path.name,
                "index": idx,
            }
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file."""
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        
        return image.astype(np.float32)
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask from file."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {path}")
        
        # Ensure binary mask (0 or 1)
        mask = (mask > 127).astype(np.float32)
        
        return mask
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if len(self) == 0:
            return {}
        
        # Sample a few images to compute stats
        sample_size = min(100, len(self))
        indices = np.random.choice(len(self), sample_size, replace=False)
        
        pixel_values = []
        target_counts = []
        image_sizes = []
        
        for idx in indices:
            sample = self.__getitem__(idx)
            image = sample["image"]
            mask = sample["mask"]
            
            pixel_values.extend(image.flatten().tolist())
            target_counts.append(int(mask.sum().item()))
            image_sizes.append(image.shape[-2:] if len(image.shape) > 2 else image.shape)
        
        pixel_values = np.array(pixel_values)
        
        return {
            "num_samples": len(self),
            "pixel_mean": float(pixel_values.mean()),
            "pixel_std": float(pixel_values.std()),
            "pixel_min": float(pixel_values.min()),
            "pixel_max": float(pixel_values.max()),
            "avg_targets_per_image": float(np.mean(target_counts)),
            "max_targets_per_image": int(np.max(target_counts)),
            "common_image_size": max(set(map(tuple, image_sizes)), key=image_sizes.count),
        }


@register_dataset("nudt_sirst")
class NUDTSIRSTDataset(SIRSTDataset):
    """NUDT-SIRST Dataset.
    
    Similar structure to SIRST but potentially different file organization.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        **kwargs
    ):
        # NUDT-SIRST might have different directory structure
        super().__init__(root, split, transform, **kwargs)


@register_dataset("irstd_1k")
class IRSTD1KDataset(BaseDataset):
    """IRSTD-1K Dataset implementation.
    
    Dataset structure might be different from SIRST.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(root, split, transform)
        
        # Set up paths - IRSTD-1K might have different structure
        split_file = self.root / f"{split}.txt"
        
        if split_file.exists():
            # Load from split file
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            # Fallback to directory-based structure
            self.images_dir = self.root / "images" / split
            self.masks_dir = self.root / "masks" / split
            self.image_files = self._get_file_list(self.images_dir)
            self.mask_files = self._get_file_list(self.masks_dir)
    
    def _get_file_list(self, directory: Path) -> list:
        """Get list of files."""
        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        files = []
        
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(files)
    
    def __len__(self) -> int:
        if hasattr(self, 'file_list'):
            return len(self.file_list)
        else:
            return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if hasattr(self, 'file_list'):
            # Load based on file list
            filename = self.file_list[idx]
            image_path = self.root / "images" / f"{filename}.png"
            mask_path = self.root / "masks" / f"{filename}.png"
        else:
            # Load based on directory structure
            image_path = self.image_files[idx]
            mask_path = self.mask_files[idx]
        
        # Load image and mask
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform:
            if hasattr(self.transform, 'replay'):
                transformed = self.transform(image=image, mask=mask)
                image, mask = transformed["image"], transformed["mask"]
            else:
                image = self.transform(image)
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return {
            "image": image,
            "mask": mask,
            "meta": {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "index": idx,
            }
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file."""
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        return image.astype(np.float32)
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask from file."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {path}")
        return (mask > 127).astype(np.float32)
