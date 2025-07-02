"""
Loss functions for infrared small target detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class IoULoss(nn.Module):
    """IoU (Jaccard) Loss for binary segmentation"""
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss"""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous()
        target = target.contiguous()
        
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss"""
    
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            weight = torch.tensor([self.pos_weight], device=pred.device)
            return F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)
        else:
            return F.binary_cross_entropy(pred, target)


class CombinedLoss(nn.Module):
    """Combined loss function with multiple loss terms"""
    
    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Dict[str, float],
        reduction: str = 'mean'
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            weighted_loss = self.weights.get(name, 1.0) * loss_value
            
            loss_dict[name] = loss_value
            loss_dict[f"weighted_{name}"] = weighted_loss
            total_loss += weighted_loss
        
        loss_dict['total'] = total_loss
        return loss_dict


class IRSTLoss(nn.Module):
    """Specialized loss for infrared small target detection"""
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        iou_weight: float = 0.5,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        smooth: float = 1e-5
    ):
        super().__init__()
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou_loss = IoULoss(smooth=smooth)
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        iou = self.iou_loss(pred, target)
        
        total_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.iou_weight * iou
        )
        
        return {
            'dice': dice,
            'focal': focal,
            'iou': iou,
            'total': total_loss
        }


class AdaptiveWeightedLoss(nn.Module):
    """Adaptive weighted loss that adjusts weights during training"""
    
    def __init__(
        self,
        losses: Dict[str, nn.Module],
        initial_weights: Dict[str, float],
        adaptation_rate: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 2.0
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight, dtype=torch.float32))
            for name, weight in initial_weights.items()
        })
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Track loss history for adaptation
        self.loss_history = {name: [] for name in losses.keys()}
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            weight = torch.clamp(self.weights[name], self.min_weight, self.max_weight)
            weighted_loss = weight * loss_value
            
            loss_dict[name] = loss_value
            loss_dict[f"weighted_{name}"] = weighted_loss
            loss_dict[f"weight_{name}"] = weight
            
            total_loss += weighted_loss
            
            # Update loss history
            self.loss_history[name].append(loss_value.item())
            if len(self.loss_history[name]) > 100:
                self.loss_history[name].pop(0)
        
        loss_dict['total'] = total_loss
        return loss_dict
    
    def adapt_weights(self):
        """Adapt weights based on loss history"""
        if not self.training:
            return
        
        for name in self.losses.keys():
            if len(self.loss_history[name]) >= 10:
                recent_losses = self.loss_history[name][-10:]
                loss_trend = np.mean(recent_losses[-5:]) - np.mean(recent_losses[:5])
                
                # If loss is increasing, increase weight
                if loss_trend > 0:
                    self.weights[name].data += self.adaptation_rate
                # If loss is decreasing well, slightly decrease weight
                elif loss_trend < -0.01:
                    self.weights[name].data -= self.adaptation_rate * 0.5
                
                # Clamp weights
                self.weights[name].data.clamp_(self.min_weight, self.max_weight)


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Factory function to get loss function by name"""
    
    loss_functions = {
        'bce': nn.BCELoss,
        'dice': DiceLoss,
        'iou': IoULoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'weighted_bce': WeightedBCELoss,
        'irst': IRSTLoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)
