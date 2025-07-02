"""
Main detector class for easy inference.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
from PIL import Image

from .base import BaseModel
from .registry import get_model
from ..utils.image import preprocess_image, postprocess_results
from ..utils.visualization import visualize_detections


class IRSTDetector:
    """Main detector class for infrared small target detection."""
    
    def __init__(
        self,
        model: Union[BaseModel, str],
        device: str = "auto",
        threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        """Initialize IRST detector.
        
        Args:
            model: Model instance or model name
            device: Device to run inference on
            threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
        """
        self.device = self._get_device(device)
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        
        # Load model
        if isinstance(model, str):
            self.model = self._load_model_from_name(model)
        else:
            self.model = model
        
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> "IRSTDetector":
        """Create detector from pretrained model.
        
        Args:
            model_name: Name of the pretrained model
            checkpoint_path: Path to model checkpoint
            **kwargs: Additional arguments
            
        Returns:
            Initialized IRSTDetector
        """
        # Load model architecture
        model_cls = get_model(model_name)
        model = model_cls()
        
        # Load pretrained weights
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        
        return cls(model=model, **kwargs)
    
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, str, Path],
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """Detect small targets in infrared image.
        
        Args:
            image: Input image (tensor, array, or path)
            return_raw: Whether to return raw model output
            
        Returns:
            Detection results dictionary containing:
                - boxes: Bounding boxes (N, 4)
                - scores: Confidence scores (N,)
                - masks: Segmentation masks (N, H, W)
                - raw_output: Raw model output (if return_raw=True)
        """
        # Preprocess image
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        processed_image = preprocess_image(image, device=self.device)
        
        # Run inference
        with torch.no_grad():
            raw_output = self.model(processed_image)
        
        # Post-process results
        results = self._postprocess_output(raw_output, processed_image.shape[-2:])
        
        if return_raw:
            results["raw_output"] = raw_output
        
        return results
    
    def detect_batch(
        self,
        images: List[Union[torch.Tensor, np.ndarray, str, Path]],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """Detect targets in a batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            
        Returns:
            List of detection results for each image
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Process batch
            processed_batch = []
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    img = self._load_image(img)
                processed_batch.append(preprocess_image(img, device=self.device))
            
            batch_tensor = torch.cat(processed_batch, dim=0)
            
            # Run inference
            with torch.no_grad():
                batch_output = self.model(batch_tensor)
            
            # Post-process each image in batch
            for j, output in enumerate(batch_output):
                image_results = self._postprocess_output(
                    output.unsqueeze(0),
                    batch_tensor.shape[-2:]
                )
                results.append(image_results)
        
        return results
    
    def visualize(
        self,
        image: Union[torch.Tensor, np.ndarray, str, Path],
        results: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> np.ndarray:
        """Visualize detection results.
        
        Args:
            image: Input image
            results: Detection results (if None, will run detection)
            save_path: Path to save visualization
            show: Whether to display the image
            
        Returns:
            Visualization image as numpy array
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        # Run detection if results not provided
        if results is None:
            results = self.detect(image)
        
        # Create visualization
        viz_image = visualize_detections(image, results)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, viz_image)
        
        # Show if requested
        if show:
            cv2.imshow("IRST Detection Results", viz_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return viz_image
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model_from_name(self, model_name: str) -> BaseModel:
        """Load model from registered name."""
        model_cls = get_model(model_name)
        return model_cls()
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with PIL and convert to numpy
        pil_image = Image.open(image_path)
        if pil_image.mode != "L":  # Convert to grayscale if needed
            pil_image = pil_image.convert("L")
        
        return np.array(pil_image)
    
    def _postprocess_output(
        self,
        raw_output: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Post-process model output to get detection results."""
        # Apply sigmoid if not already applied
        if raw_output.min() < 0 or raw_output.max() > 1:
            output = torch.sigmoid(raw_output)
        else:
            output = raw_output
        
        # Threshold predictions
        binary_mask = (output > self.threshold).float()
        
        # Find connected components for bounding boxes
        results = postprocess_results(
            binary_mask,
            output,
            nms_threshold=self.nms_threshold,
            image_shape=image_shape,
        )
        
        return results
    
    def set_threshold(self, threshold: float):
        """Set detection threshold."""
        self.threshold = threshold
    
    def set_nms_threshold(self, nms_threshold: float):
        """Set NMS threshold."""
        self.nms_threshold = nms_threshold
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model.get_model_info()
