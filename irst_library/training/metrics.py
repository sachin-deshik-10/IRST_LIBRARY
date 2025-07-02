"""
Evaluation metrics for infrared small target detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import cv2


class BinaryMetrics:
    """Binary classification metrics for ISTD"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-7):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.predictions = []
        self.targets = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with batch predictions and targets"""
        
        # Convert to binary predictions
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)
        
        # Update confusion matrix
        self.tp += torch.sum((pred_flat == 1) & (target_flat == 1)).item()
        self.fp += torch.sum((pred_flat == 1) & (target_flat == 0)).item()
        self.tn += torch.sum((pred_flat == 0) & (target_flat == 0)).item()
        self.fn += torch.sum((pred_flat == 0) & (target_flat == 1)).item()
        
        # Store for AUC calculation
        self.predictions.extend(pred.view(-1).cpu().numpy())
        self.targets.extend(target.view(-1).cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        
        # Basic metrics
        precision = self.tp / (self.tp + self.fp + self.smooth)
        recall = self.tp / (self.tp + self.fn + self.smooth)
        specificity = self.tn / (self.tn + self.fp + self.smooth)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + self.smooth)
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
        
        # IoU (Jaccard index)
        iou = self.tp / (self.tp + self.fp + self.fn + self.smooth)
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(self.targets, self.predictions)
        except:
            auc_roc = 0.0
        
        # AUC-PR
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(self.targets, self.predictions)
            auc_pr = auc(recall_curve, precision_curve)
        except:
            auc_pr = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1': f1,
            'iou': iou,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn
        }


class PixelLevelMetrics:
    """Pixel-level evaluation metrics"""
    
    def __init__(self, num_classes: int = 2, smooth: float = 1e-7):
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
        """Update confusion matrix"""
        
        pred_binary = (pred > threshold).long().cpu().numpy()
        target_binary = target.long().cpu().numpy()
        
        # Flatten
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()
        
        # Update confusion matrix
        for t, p in zip(target_flat, pred_flat):
            self.confusion_matrix[t, p] += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute pixel-level metrics"""
        
        # Per-class metrics
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Avoid division by zero
        precision = tp / (tp + fp + self.smooth)
        recall = tp / (tp + fn + self.smooth)
        f1 = 2 * precision * recall / (precision + recall + self.smooth)
        
        # Overall metrics
        accuracy = tp.sum() / self.confusion_matrix.sum()
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        mean_f1 = f1.mean()
        
        # Mean IoU
        iou = tp / (tp + fp + fn + self.smooth)
        mean_iou = iou.mean()
        
        return {
            'pixel_accuracy': accuracy,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_iou': mean_iou,
            'class_precision': precision.tolist(),
            'class_recall': recall.tolist(),
            'class_f1': f1.tolist(),
            'class_iou': iou.tolist()
        }


class ObjectLevelMetrics:
    """Object-level (connected component) evaluation metrics"""
    
    def __init__(self, min_area: int = 10, distance_threshold: float = 3.0):
        self.min_area = min_area
        self.distance_threshold = distance_threshold
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.true_detections = 0
        self.false_detections = 0
        self.missed_targets = 0
        self.total_targets = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
        """Update object-level metrics"""
        
        batch_size = pred.size(0)
        
        for i in range(batch_size):
            pred_mask = (pred[i] > threshold).squeeze().cpu().numpy().astype(np.uint8)
            target_mask = target[i].squeeze().cpu().numpy().astype(np.uint8)
            
            # Find connected components
            pred_objects = self._find_objects(pred_mask)
            target_objects = self._find_objects(target_mask)
            
            self.total_targets += len(target_objects)
            
            # Match predicted objects to ground truth
            matched_targets = set()
            for pred_obj in pred_objects:
                matched = False
                for j, target_obj in enumerate(target_objects):
                    if j not in matched_targets:
                        distance = self._compute_distance(pred_obj, target_obj)
                        if distance <= self.distance_threshold:
                            matched_targets.add(j)
                            matched = True
                            break
                
                if matched:
                    self.true_detections += 1
                else:
                    self.false_detections += 1
            
            # Count missed targets
            self.missed_targets += len(target_objects) - len(matched_targets)
    
    def _find_objects(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        """Find object centroids in binary mask"""
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask)
        
        objects = []
        for label in range(1, num_labels):  # Skip background (label 0)
            component_mask = (labels == label)
            area = np.sum(component_mask)
            
            if area >= self.min_area:
                # Compute centroid
                y_coords, x_coords = np.where(component_mask)
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                objects.append((centroid_x, centroid_y))
        
        return objects
    
    def _compute_distance(self, obj1: Tuple[float, float], obj2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between object centroids"""
        return np.sqrt((obj1[0] - obj2[0])**2 + (obj1[1] - obj2[1])**2)
    
    def compute(self) -> Dict[str, float]:
        """Compute object-level metrics"""
        
        precision = self.true_detections / (self.true_detections + self.false_detections + 1e-7)
        recall = self.true_detections / (self.true_detections + self.missed_targets + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            'object_precision': precision,
            'object_recall': recall,
            'object_f1': f1,
            'true_detections': self.true_detections,
            'false_detections': self.false_detections,
            'missed_targets': self.missed_targets,
            'total_targets': self.total_targets
        }


class MetricsCalculator:
    """Comprehensive metrics calculator for ISTD"""
    
    def __init__(
        self,
        thresholds: List[float] = None,
        min_object_area: int = 10,
        distance_threshold: float = 3.0
    ):
        if thresholds is None:
            thresholds = [0.5]
        
        self.thresholds = thresholds
        self.binary_metrics = {t: BinaryMetrics(threshold=t) for t in thresholds}
        self.pixel_metrics = PixelLevelMetrics()
        self.object_metrics = ObjectLevelMetrics(min_object_area, distance_threshold)
    
    def reset(self):
        """Reset all metrics"""
        for metrics in self.binary_metrics.values():
            metrics.reset()
        self.pixel_metrics.reset()
        self.object_metrics.reset()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update all metrics"""
        
        # Update binary metrics for all thresholds
        for threshold, metrics in self.binary_metrics.items():
            metrics.update(pred, target)
        
        # Update pixel-level metrics
        self.pixel_metrics.update(pred, target)
        
        # Update object-level metrics
        self.object_metrics.update(pred, target)
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute all metrics"""
        
        results = {}
        
        # Binary metrics for each threshold
        for threshold, metrics in self.binary_metrics.items():
            results[f'binary_t{threshold}'] = metrics.compute()
        
        # Pixel-level metrics
        results['pixel'] = self.pixel_metrics.compute()
        
        # Object-level metrics
        results['object'] = self.object_metrics.compute()
        
        # Summary metrics (using default threshold 0.5)
        if 0.5 in self.binary_metrics:
            default_metrics = self.binary_metrics[0.5].compute()
            results['summary'] = {
                'precision': default_metrics['precision'],
                'recall': default_metrics['recall'],
                'f1': default_metrics['f1'],
                'iou': default_metrics['iou'],
                'auc_roc': default_metrics['auc_roc'],
                'pixel_accuracy': results['pixel']['pixel_accuracy'],
                'object_f1': results['object']['object_f1']
            }
        
        return results


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute Dice coefficient"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute IoU score"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def precision_recall_at_thresholds(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: List[float]
) -> Tuple[List[float], List[float]]:
    """Compute precision and recall at multiple thresholds"""
    
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        tp = torch.sum((pred_binary == 1) & (target_binary == 1)).item()
        fp = torch.sum((pred_binary == 1) & (target_binary == 0)).item()
        fn = torch.sum((pred_binary == 0) & (target_binary == 1)).item()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls
