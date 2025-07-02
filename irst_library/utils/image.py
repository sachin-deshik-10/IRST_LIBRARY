"""
Image processing utilities for IRST detection.
"""

from typing import Tuple, Union, Optional, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def preprocess_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Preprocess image for model inference.
    
    Args:
        image: Input image
        target_size: Target size (H, W) for resizing
        normalize: Whether to normalize pixel values
        device: Target device
        
    Returns:
        Preprocessed image tensor of shape (1, 1, H, W)
    """
    # Convert to numpy array
    if isinstance(image, Image.Image):
        if image.mode != "L":
            image = image.convert("L")
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Ensure grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 1:  # Already grayscale
            image = image.squeeze(-1)
    
    # Resize if target size specified
    if target_size is not None:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Convert to tensor
    tensor = torch.from_numpy(image).float()
    
    # Add batch and channel dimensions
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)
    
    # Normalize
    if normalize:
        tensor = tensor / 255.0
    
    return tensor.to(device)


def postprocess_results(
    binary_mask: torch.Tensor,
    confidence_map: torch.Tensor,
    nms_threshold: float = 0.4,
    image_shape: Optional[Tuple[int, int]] = None,
    min_area: int = 5,
) -> Dict[str, Any]:
    """Post-process model output to extract detection results.
    
    Args:
        binary_mask: Binary prediction mask (B, 1, H, W)
        confidence_map: Confidence scores (B, 1, H, W)
        nms_threshold: NMS threshold for overlapping detections
        image_shape: Original image shape (H, W)
        min_area: Minimum area for valid detections
        
    Returns:
        Dictionary containing detection results
    """
    results = {
        "boxes": [],
        "scores": [],
        "masks": [],
        "centers": [],
    }
    
    # Process each image in batch
    for i in range(binary_mask.shape[0]):
        mask = binary_mask[i, 0].cpu().numpy().astype(np.uint8)
        conf = confidence_map[i, 0].cpu().numpy()
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        boxes = []
        scores = []
        centers = []
        masks = []
        
        for label in range(1, num_labels):  # Skip background (label 0)
            # Get component stats
            area = stats[label, cv2.CC_STAT_AREA]
            
            # Filter by minimum area
            if area < min_area:
                continue
            
            # Get bounding box
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            
            # Get mask for this component
            component_mask = (labels == label).astype(np.uint8)
            
            # Calculate confidence score (mean confidence in the region)
            score = float(conf[component_mask == 1].mean())
            
            # Get center point
            center = centroids[label]
            
            boxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2]
            scores.append(score)
            centers.append([center[0], center[1]])  # [cx, cy]
            masks.append(component_mask)
        
        # Apply NMS if we have detections
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            
            # Convert to torch tensors for NMS
            boxes_tensor = torch.from_numpy(boxes).float()
            scores_tensor = torch.from_numpy(scores).float()
            
            # Apply NMS
            keep_indices = torch.ops.torchvision.nms(
                boxes_tensor, scores_tensor, nms_threshold
            )
            
            # Filter results
            boxes = boxes[keep_indices.cpu().numpy()]
            scores = scores[keep_indices.cpu().numpy()]
            centers = [centers[i] for i in keep_indices.cpu().numpy()]
            masks = [masks[i] for i in keep_indices.cpu().numpy()]
        
        # Convert to tensors
        results["boxes"].append(torch.from_numpy(np.array(boxes)).float() if len(boxes) > 0 else torch.empty(0, 4))
        results["scores"].append(torch.from_numpy(np.array(scores)).float() if len(scores) > 0 else torch.empty(0))
        results["centers"].append(torch.from_numpy(np.array(centers)).float() if len(centers) > 0 else torch.empty(0, 2))
        results["masks"].append(masks)
    
    # If single image, return without batch dimension
    if len(results["boxes"]) == 1:
        return {
            "boxes": results["boxes"][0],
            "scores": results["scores"][0],
            "centers": results["centers"][0],
            "masks": results["masks"][0],
        }
    
    return results


def resize_image_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Resize image while maintaining aspect ratio with padding.
    
    Args:
        image: Input image array
        target_size: Target size (H, W)
        pad_value: Value to use for padding
        
    Returns:
        Resized and padded image, padding info dict
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new size
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # Add padding
    if len(image.shape) == 2:  # Grayscale
        padded = np.pad(
            resized,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
    else:  # Color
        padded = np.pad(
            resized,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
    
    padding_info = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "original_size": (h, w),
        "resized_size": (new_h, new_w),
        "padded_size": target_size,
    }
    
    return padded, padding_info


def normalize_image(
    image: np.ndarray,
    mean: Optional[Union[float, Tuple[float, ...]]] = None,
    std: Optional[Union[float, Tuple[float, ...]]] = None,
) -> np.ndarray:
    """Normalize image with mean and standard deviation.
    
    Args:
        image: Input image array (0-255 range)
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image array
    """
    # Convert to float
    image = image.astype(np.float32) / 255.0
    
    # Apply normalization if provided
    if mean is not None or std is not None:
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0
        
        image = (image - mean) / std
    
    return image


def denormalize_image(
    image: np.ndarray,
    mean: Optional[Union[float, Tuple[float, ...]]] = None,
    std: Optional[Union[float, Tuple[float, ...]]] = None,
) -> np.ndarray:
    """Denormalize image back to 0-255 range.
    
    Args:
        image: Normalized image array
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized image array (0-255 range)
    """
    # Reverse normalization if provided
    if mean is not None or std is not None:
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0
        
        image = image * std + mean
    
    # Convert back to 0-255 range
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    return image
