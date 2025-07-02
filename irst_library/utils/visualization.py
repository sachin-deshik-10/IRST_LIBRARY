"""
Visualization utilities for IRST detection results.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import torch


def visualize_detections(
    image: Union[np.ndarray, torch.Tensor],
    results: Dict[str, Any],
    show_boxes: bool = True,
    show_masks: bool = True,
    show_centers: bool = True,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    mask_color: Tuple[int, int, int] = (255, 0, 0),
    center_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    alpha: float = 0.3,
) -> np.ndarray:
    """Visualize detection results on image.
    
    Args:
        image: Input image
        results: Detection results dictionary
        show_boxes: Whether to show bounding boxes
        show_masks: Whether to show segmentation masks
        show_centers: Whether to show center points
        box_color: Color for bounding boxes (B, G, R)
        mask_color: Color for masks (B, G, R)
        center_color: Color for center points (B, G, R)
        thickness: Line thickness for drawing
        alpha: Transparency for mask overlay
        
    Returns:
        Visualization image
    """
    # Convert image to numpy if tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if len(image.shape) == 4:  # (B, C, H, W)
            image = image[0]
        if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
            image = image.transpose(1, 2, 0)
    
    # Ensure uint8 format
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert grayscale to BGR for visualization
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        vis_image = cv2.cvtColor(image.squeeze(-1), cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Get detection data
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    centers = results.get("centers", [])
    masks = results.get("masks", [])
    
    # Convert tensors to numpy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(centers, torch.Tensor):
        centers = centers.cpu().numpy()
    
    # Draw masks first (so they appear behind boxes)
    if show_masks and masks:
        mask_overlay = vis_image.copy()
        for i, mask in enumerate(masks):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Create colored mask
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = mask_color
            
            # Blend with image
            mask_overlay = cv2.addWeighted(mask_overlay, 1 - alpha, colored_mask, alpha, 0)
        
        vis_image = mask_overlay
    
    # Draw bounding boxes
    if show_boxes and len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), box_color, thickness)
            
            # Add score text if available
            if len(scores) > i:
                score = scores[i]
                text = f"{score:.3f}"
                
                # Get text size for background rectangle
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - text_h - baseline - 5),
                    (x1 + text_w, y1),
                    box_color,
                    -1,
                )
                
                # Draw text
                cv2.putText(
                    vis_image,
                    text,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
    
    # Draw center points
    if show_centers and len(centers) > 0:
        for center in centers:
            cx, cy = center.astype(int)
            cv2.circle(vis_image, (cx, cy), 3, center_color, -1)
            cv2.circle(vis_image, (cx, cy), 5, center_color, 1)
    
    return vis_image


def plot_detection_results(
    image: Union[np.ndarray, torch.Tensor],
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot detection results using matplotlib.
    
    Args:
        image: Input image
        results: Detection results dictionary
        figsize: Figure size
        save_path: Path to save the plot
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Convert image for display
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if len(image.shape) == 4:
            image = image[0]
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image[0]
    
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Plot original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot image with detections
    axes[1].imshow(image, cmap="gray")
    
    # Get detection data
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    centers = results.get("centers", [])
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(centers, torch.Tensor):
        centers = centers.cpu().numpy()
    
    # Draw bounding boxes
    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            axes[1].add_patch(rect)
            
            # Add score text if available
            if len(scores) > i:
                score = scores[i]
                axes[1].text(
                    x1, y1 - 5,
                    f"{score:.3f}",
                    fontsize=10,
                    color="green",
                    weight="bold",
                )
    
    # Draw center points
    if len(centers) > 0:
        for center in centers:
            cx, cy = center
            axes[1].plot(cx, cy, "ro", markersize=5)
    
    axes[1].set_title(f"Detections ({len(boxes)} targets)")
    axes[1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def create_heatmap(
    confidence_map: Union[np.ndarray, torch.Tensor],
    image: Optional[Union[np.ndarray, torch.Tensor]] = None,
    alpha: float = 0.6,
    colormap: str = "jet",
) -> np.ndarray:
    """Create confidence heatmap visualization.
    
    Args:
        confidence_map: Confidence/probability map
        image: Background image (optional)
        alpha: Transparency of heatmap overlay
        colormap: Colormap to use for heatmap
        
    Returns:
        Heatmap visualization
    """
    # Convert to numpy
    if isinstance(confidence_map, torch.Tensor):
        confidence_map = confidence_map.cpu().numpy()
    
    # Remove batch and channel dimensions if present
    while len(confidence_map.shape) > 2:
        confidence_map = confidence_map.squeeze()
    
    # Normalize to 0-1 range
    conf_min, conf_max = confidence_map.min(), confidence_map.max()
    if conf_max > conf_min:
        confidence_map = (confidence_map - conf_min) / (conf_max - conf_min)
    
    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    heatmap = cmap(confidence_map)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel
    
    # Overlay on image if provided
    if image is not None:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image.squeeze(-1), cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match image size if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Blend images
        result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    else:
        result = heatmap
    
    return result


def save_detection_grid(
    images: List[np.ndarray],
    results_list: List[Dict[str, Any]],
    save_path: str,
    grid_size: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (15, 10),
):
    """Save a grid of detection results.
    
    Args:
        images: List of input images
        results_list: List of detection results for each image
        save_path: Path to save the grid
        grid_size: Grid size (rows, cols). If None, auto-determine
        figsize: Figure size
    """
    n_images = len(images)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i in range(n_images):
        # Visualize detections
        vis_image = visualize_detections(images[i], results_list[i])
        
        # Convert BGR to RGB for matplotlib
        if len(vis_image.shape) == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(vis_image)
        axes[i].set_title(f"Image {i+1} ({len(results_list[i].get('boxes', []))} targets)")
        axes[i].axis("off")
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_attention_maps(
    attention_maps: List[torch.Tensor],
    input_image: Union[np.ndarray, torch.Tensor],
    save_path: Optional[str] = None,
    layer_names: Optional[List[str]] = None,
) -> plt.Figure:
    """Visualize attention maps from model layers.
    
    Args:
        attention_maps: List of attention map tensors
        input_image: Original input image
        save_path: Path to save visualization
        layer_names: Names for each attention map
        
    Returns:
        Matplotlib figure
    """
    n_maps = len(attention_maps)
    cols = min(4, n_maps + 1)  # +1 for original image
    rows = int(np.ceil((n_maps + 1) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Show original image
    if isinstance(input_image, torch.Tensor):
        input_image = input_image.cpu().numpy()
    
    if len(input_image.shape) == 4:
        input_image = input_image[0]
    if len(input_image.shape) == 3 and input_image.shape[0] == 1:
        input_image = input_image[0]
    
    axes[0].imshow(input_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    # Show attention maps
    for i, attn_map in enumerate(attention_maps):
        if isinstance(attn_map, torch.Tensor):
            attn_map = attn_map.cpu().numpy()
        
        # Average across channels if multi-channel
        if len(attn_map.shape) > 2:
            attn_map = attn_map.mean(axis=0) if attn_map.shape[0] < attn_map.shape[-1] else attn_map.mean(axis=-1)
        
        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        
        axes[i + 1].imshow(attn_map, cmap="hot")
        
        title = layer_names[i] if layer_names and i < len(layer_names) else f"Layer {i+1}"
        axes[i + 1].set_title(title)
        axes[i + 1].axis("off")
    
    # Hide empty subplots
    for i in range(n_maps + 1, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Plot training history with loss and metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot loss
    if 'loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot F1 score
    if 'f1' in history and 'val_f1' in history:
        axes[0, 1].plot(history['f1'], label='Training F1')
        axes[0, 1].plot(history['val_f1'], label='Validation F1')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot IoU
    if 'iou' in history and 'val_iou' in history:
        axes[1, 0].plot(history['iou'], label='Training IoU')
        axes[1, 0].plot(history['val_iou'], label='Validation IoU')
        axes[1, 0].set_title('IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot learning rate if available
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    elif 'precision' in history and 'val_precision' in history:
        # Plot precision as alternative
        axes[1, 1].plot(history['precision'], label='Training Precision')
        axes[1, 1].plot(history['val_precision'], label='Validation Precision')
        axes[1, 1].set_title('Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'hot'
) -> np.ndarray:
    """
    Create overlay of image and mask.
    
    Args:
        image: Input grayscale image
        mask: Binary mask or prediction map
        alpha: Overlay transparency
        colormap: Colormap for mask
        
    Returns:
        Overlay image
    """
    # Ensure image is 3-channel
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Apply colormap to mask
    if colormap == 'hot':
        mask_colored = cv2.applyColorMap((mask).astype(np.uint8), cv2.COLORMAP_HOT)
    elif colormap == 'jet':
        mask_colored = cv2.applyColorMap((mask).astype(np.uint8), cv2.COLORMAP_JET)
    else:
        mask_colored = cv2.applyColorMap((mask).astype(np.uint8), cv2.COLORMAP_HOT)
    
    # Create overlay
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)
    
    return overlay
