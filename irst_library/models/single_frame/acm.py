"""
ACM (Asymmetric Contextual Modulation) Network for Infrared Small Target Detection
Paper: ACM: Asymmetric Contextual Modulation for Infrared Small Target Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional

from ...core.base import BaseModel
from ...core.registry import MODEL_REGISTRY


class AsymmetricContextualModule(nn.Module):
    """Asymmetric Contextual Modulation Module"""
    
    def __init__(self, channels: int, kernel_sizes: List[int] = [5, 9, 17]):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        
        # Branch convolutions with different kernel sizes
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            padding = ks // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, ks, padding=padding, groups=channels),
                    nn.Conv2d(channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * (len(kernel_sizes) + 1), channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale branches
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # Global branch
        global_out = self.global_branch(x)
        global_out = F.interpolate(global_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        branch_outputs.append(global_out)
        
        # Concatenate and fuse
        fused = torch.cat(branch_outputs, dim=1)
        fused = self.fusion(fused)
        
        # Channel attention
        attention = self.channel_attention(fused)
        output = fused * attention
        
        return output + x  # Residual connection


class DenseAsymmetricBlock(nn.Module):
    """Dense Asymmetric Block with multiple ACM modules"""
    
    def __init__(self, channels: int, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        
        # ACM layers
        self.acm_layers = nn.ModuleList([
            AsymmetricContextualModule(channels) for _ in range(num_layers)
        ])
        
        # Transition layers
        self.transitions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * (i + 2), channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_layers - 1)
        ])
        
        # Final transition
        self.final_transition = nn.Sequential(
            nn.Conv2d(channels * (num_layers + 1), channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        
        for i, acm in enumerate(self.acm_layers):
            # Concatenate all previous features
            concat_features = torch.cat(features, dim=1)
            
            # Apply ACM
            new_feature = acm(features[-1])
            features.append(new_feature)
            
            # Apply transition (except for the last layer)
            if i < self.num_layers - 1:
                concat_features = torch.cat(features, dim=1)
                features = [features[0], self.transitions[i](concat_features)]
        
        # Final concatenation and transition
        final_concat = torch.cat(features, dim=1)
        output = self.final_transition(final_concat)
        
        return output


class UpSamplingBlock(nn.Module):
    """Up-sampling block with skip connections"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.acm = AsymmetricContextualModule(out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Process skip connection
        skip = self.skip_conv(skip)
        
        # Concatenate and convolve
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        # Apply ACM
        x = self.acm(x)
        
        return x


@MODEL_REGISTRY.register("acm")
class ACMNet(BaseModel):
    """
    ACM Network for infrared small target detection.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale IR images)
        num_classes: Number of output classes (default: 1 for binary segmentation)
        base_channels: Base number of channels (default: 64)
        num_dense_layers: Number of layers in dense blocks (default: 4)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        num_dense_layers: int = 4,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_dense_layers = num_dense_layers
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder - Down-sampling path
        self.down1 = nn.Sequential(
            DenseAsymmetricBlock(base_channels, num_dense_layers),
            nn.MaxPool2d(2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            DenseAsymmetricBlock(base_channels * 2, num_dense_layers),
            nn.MaxPool2d(2)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            DenseAsymmetricBlock(base_channels * 4, num_dense_layers),
            nn.MaxPool2d(2)
        )
        
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            DenseAsymmetricBlock(base_channels * 8, num_dense_layers),
            nn.MaxPool2d(2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            DenseAsymmetricBlock(base_channels * 16, num_dense_layers)
        )
        
        # Decoder - Up-sampling path
        self.up4 = UpSamplingBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up3 = UpSamplingBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpSamplingBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpSamplingBlock(base_channels * 2, base_channels, base_channels)
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Initial convolution
        x0 = self.initial_conv(x)
        
        # Encoder path - save skip connections
        x1 = self.down1(x0)
        skip1 = x0
        
        x2 = self.down2(x1)
        skip2 = x1
        
        x3 = self.down3(x2)
        skip3 = x2
        
        x4 = self.down4(x3)
        skip4 = x3
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder path
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        
        # Final output
        x = self.final_conv(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "ACMNet",
            "paper": "ACM: Asymmetric Contextual Modulation for Infrared Small Target Detection",
            "year": 2021,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "num_dense_layers": self.num_dense_layers,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# Alias for compatibility
ACMDetector = ACMNet
