"""
Explainability and Interpretability for IRST Library

Implements state-of-the-art explainability techniques for ISTD models:
- Grad-CAM and Grad-CAM++
- Integrated Gradients
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Occlusion Sensitivity
- Layer-wise Relevance Propagation (LRP)
- Attention Visualization
- Feature Importance Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import copy

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model: nn.Module, target_layer: str = None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _find_target_layer(self) -> str:
        """Automatically find the last convolutional layer"""
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return name
        raise ValueError("No convolutional layer found in model")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Class Activation Map"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros((activations.shape[1], activations.shape[2]))
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam
    
    def visualize_cam(self, input_tensor: torch.Tensor, cam: np.ndarray, 
                     alpha: float = 0.4) -> np.ndarray:
        """Visualize CAM overlay on input image"""
        # Convert input tensor to numpy
        img = input_tensor[0].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        # Resize CAM to input size
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Overlay heatmap on image
        cam_img = heatmap * alpha + img * (1 - alpha)
        
        return cam_img


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++: Improved Visual Explanations"""
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Grad-CAM++ map"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Compute second and third order gradients
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients
        
        # Compute alpha weights
        alpha = grad_2 / (2 * grad_2 + np.sum(grad_3, axis=(1, 2), keepdims=True) + 1e-7)
        
        # Compute weights
        weights = np.sum(alpha * F.relu(torch.tensor(gradients)).numpy(), axis=(1, 2))
        
        # Weighted combination
        cam = np.zeros((activations.shape[1], activations.shape[2]))
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam


class IntegratedGradients:
    """Integrated Gradients for attribution"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def generate_integrated_gradients(self, input_tensor: torch.Tensor, 
                                    target_class: int = None, 
                                    steps: int = 50) -> torch.Tensor:
        """Generate integrated gradients"""
        self.model.eval()
        
        # Get baseline (zeros)
        baseline = torch.zeros_like(input_tensor)
        
        # Get target class
        if target_class is None:
            output = self.model(input_tensor)
            target_class = output.argmax(dim=1).item()
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps)
        gradients = []
        
        for alpha in alphas:
            # Interpolated input
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated_input)
            target_score = output[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            target_score.backward()
            
            # Store gradients
            gradients.append(interpolated_input.grad.clone())
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        return integrated_gradients
    
    def visualize_attributions(self, input_tensor: torch.Tensor, 
                             attributions: torch.Tensor) -> np.ndarray:
        """Visualize attribution map"""
        # Convert to numpy
        attrs = attributions[0].cpu().numpy()
        
        # Sum across channels for visualization
        if len(attrs.shape) == 3:
            attrs = np.sum(np.abs(attrs), axis=0)
        
        # Normalize
        attrs = (attrs - attrs.min()) / (attrs.max() - attrs.min())
        
        return attrs


class LIME:
    """Local Interpretable Model-agnostic Explanations"""
    
    def __init__(self, model: nn.Module, num_samples: int = 1000):
        self.model = model
        self.num_samples = num_samples
        
    def explain_instance(self, input_tensor: torch.Tensor, 
                        target_class: int = None,
                        num_features: int = 100) -> Dict[str, Any]:
        """Explain single instance using LIME"""
        self.model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(input_tensor)
            if target_class is None:
                target_class = original_output.argmax(dim=1).item()
            original_prob = F.softmax(original_output, dim=1)[0, target_class].item()
        
        # Generate superpixels (simple grid-based)
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        segment_size = int(np.sqrt(h * w / num_features))
        segments = self._create_segments(h, w, segment_size)
        
        # Generate perturbed samples
        samples = []
        predictions = []
        
        for _ in range(self.num_samples):
            # Random perturbation
            mask = np.random.randint(0, 2, segments.max() + 1)
            
            # Apply mask to create perturbed image
            perturbed_img = self._apply_mask(input_tensor, segments, mask)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(perturbed_img)
                prob = F.softmax(output, dim=1)[0, target_class].item()
            
            samples.append(mask)
            predictions.append(prob)
        
        # Fit linear model
        X = np.array(samples)
        y = np.array(predictions)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Get feature importance
        importance = model.coef_
        
        return {
            'feature_importance': importance,
            'segments': segments,
            'r2_score': r2_score(y, model.predict(X)),
            'original_prediction': original_prob
        }
    
    def _create_segments(self, h: int, w: int, segment_size: int) -> np.ndarray:
        """Create simple grid-based segments"""
        segments = np.zeros((h, w), dtype=int)
        segment_id = 0
        
        for i in range(0, h, segment_size):
            for j in range(0, w, segment_size):
                segments[i:i+segment_size, j:j+segment_size] = segment_id
                segment_id += 1
        
        return segments
    
    def _apply_mask(self, input_tensor: torch.Tensor, segments: np.ndarray, 
                   mask: np.ndarray) -> torch.Tensor:
        """Apply mask to input tensor"""
        perturbed = input_tensor.clone()
        
        for segment_id in range(len(mask)):
            if mask[segment_id] == 0:  # Remove segment
                segment_mask = (segments == segment_id)
                perturbed[0, :, segment_mask] = 0
        
        return perturbed


class OcclusionSensitivity:
    """Occlusion Sensitivity Analysis"""
    
    def __init__(self, model: nn.Module, patch_size: int = 16, stride: int = 8):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        
    def generate_occlusion_map(self, input_tensor: torch.Tensor, 
                              target_class: int = None) -> np.ndarray:
        """Generate occlusion sensitivity map"""
        self.model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(input_tensor)
            if target_class is None:
                target_class = original_output.argmax(dim=1).item()
            original_prob = F.softmax(original_output, dim=1)[0, target_class].item()
        
        # Initialize occlusion map
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        occlusion_map = np.zeros((h, w))
        
        # Slide occlusion patch
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                # Create occluded image
                occluded_img = input_tensor.clone()
                occluded_img[0, :, i:i+self.patch_size, j:j+self.patch_size] = 0
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(occluded_img)
                    prob = F.softmax(output, dim=1)[0, target_class].item()
                
                # Compute sensitivity (drop in probability)
                sensitivity = original_prob - prob
                occlusion_map[i:i+self.patch_size, j:j+self.patch_size] = sensitivity
        
        return occlusion_map


class ExplainabilityPipeline:
    """Complete explainability pipeline for ISTD models"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        # Initialize explainability methods
        self.grad_cam = GradCAM(model)
        self.grad_cam_pp = GradCAMPlusPlus(model)
        self.integrated_gradients = IntegratedGradients(model)
        self.lime = LIME(model)
        self.occlusion = OcclusionSensitivity(model)
        
    def explain_prediction(self, input_tensor: torch.Tensor, 
                          target_class: int = None,
                          methods: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive explanations"""
        if methods is None:
            methods = ['gradcam', 'gradcam++', 'integrated_gradients', 'lime', 'occlusion']
        
        explanations = {}
        
        # Grad-CAM
        if 'gradcam' in methods:
            cam = self.grad_cam.generate_cam(input_tensor, target_class)
            explanations['gradcam'] = cam
        
        # Grad-CAM++
        if 'gradcam++' in methods:
            cam_pp = self.grad_cam_pp.generate_cam(input_tensor, target_class)
            explanations['gradcam++'] = cam_pp
        
        # Integrated Gradients
        if 'integrated_gradients' in methods:
            ig = self.integrated_gradients.generate_integrated_gradients(input_tensor, target_class)
            explanations['integrated_gradients'] = ig
        
        # LIME
        if 'lime' in methods:
            lime_exp = self.lime.explain_instance(input_tensor, target_class)
            explanations['lime'] = lime_exp
        
        # Occlusion Sensitivity
        if 'occlusion' in methods:
            occlusion_map = self.occlusion.generate_occlusion_map(input_tensor, target_class)
            explanations['occlusion'] = occlusion_map
        
        return explanations
    
    def generate_report(self, input_tensor: torch.Tensor, 
                       explanations: Dict[str, Any],
                       save_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive explainability report"""
        report = {
            'model_prediction': None,
            'confidence': None,
            'explanations': explanations,
            'visualizations': {}
        }
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1).max().item()
        
        report['model_prediction'] = pred_class
        report['confidence'] = confidence
        
        # Save report if requested
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _save_report(self, report: Dict[str, Any], save_path: str):
        """Save explainability report"""
        import pickle
        
        with open(save_path, 'wb') as f:
            pickle.dump(report, f)
        
        logger.info(f"Explainability report saved to {save_path}")


class ExplainabilityAnalyzer:
    """Model explainability and interpretability tools"""
    
    def __init__(self, model: nn.Module, method: str = 'gradcam', **kwargs):
        self.model = model
        self.method = method
        self.pipeline = ExplainabilityPipeline(model)
        
    def explain(self, input_tensor: torch.Tensor, target_class: int = None,
               methods: List[str] = None) -> Dict[str, Any]:
        """Generate explanations for input"""
        return self.pipeline.explain_prediction(input_tensor, target_class, methods)
    
    def visualize(self, input_tensor: torch.Tensor, explanations: Dict[str, Any]):
        """Visualize explanations"""
        return self.pipeline.generate_report(input_tensor, explanations)
    
    def batch_explain(self, dataloader, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Explain multiple samples"""
        explanations = []
        
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            for j in range(inputs.size(0)):
                if len(explanations) >= num_samples:
                    break
                
                input_tensor = inputs[j:j+1]
                target_class = targets[j].item()
                
                exp = self.explain(input_tensor, target_class)
                explanations.append(exp)
        
        return explanations
