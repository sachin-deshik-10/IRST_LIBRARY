"""
SERANKNet - Searching for the Ranking Modules for Real-time Infrared Small Target Detection
Paper: https://arxiv.org/abs/2110.06373
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple

from ...core.base import BaseModel
from ...core.registry import MODEL_REGISTRY


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SearchRankModule(nn.Module):
    """Search and Ranking Module"""
    
    def __init__(self, channels: int, num_searches: int = 16):
        super().__init__()
        self.num_searches = num_searches
        
        # Search branches
        self.search_branches = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
            for _ in range(num_searches)
        ])
        
        # Ranking network
        self.ranking_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_searches, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion
        self.fusion = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply search branches
        search_results = []
        for branch in self.search_branches:
            search_results.append(branch(x))
        
        # Stack search results
        search_stack = torch.stack(search_results, dim=2)  # [B, C, N, H, W]
        
        # Get ranking weights
        ranking_weights = self.ranking_net(x)  # [B, N, 1, 1]
        ranking_weights = ranking_weights.unsqueeze(1)  # [B, 1, N, 1, 1]
        
        # Apply ranking
        ranked_features = (search_stack * ranking_weights).sum(dim=2)  # [B, C, H, W]
        
        # Fusion
        output = self.fusion(ranked_features)
        return output + x  # Residual connection


class EncoderBlock(nn.Module):
    """Encoder block with CBAM attention"""
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
        
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        self.search_rank = SearchRankModule(out_channels)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Attention
        x = self.attention(x)
        
        # Search and rank
        x = self.search_rank(x)
        
        # Skip connection and pooling
        skip = x
        x = self.pool(x)
        
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.attention = CBAM(out_channels)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Attention
        x = self.attention(x)
        
        return x


@MODEL_REGISTRY.register("serank")
class SERANKNet(BaseModel):
    """
    SERANKNet for infrared small target detection.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale IR images)
        num_classes: Number of output classes (default: 1 for binary segmentation)
        base_channels: Base number of channels (default: 64)
        use_attention: Whether to use CBAM attention (default: True)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        use_attention: bool = True,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.use_attention = use_attention
        
        # Encoder
        self.encoder1 = EncoderBlock(in_channels, base_channels, use_attention)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2, use_attention)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4, use_attention)
        self.encoder4 = EncoderBlock(base_channels * 4, base_channels * 8, use_attention)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 16, 3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            CBAM(base_channels * 16) if use_attention else nn.Identity(),
            SearchRankModule(base_channels * 16)
        )
        
        # Decoder
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        
        # Final classification
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        
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
        # Encoder path
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder path
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        
        # Final output
        x = self.final_conv(x)
        
        return torch.sigmoid(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "SERANKNet",
            "paper": "Searching for the Ranking of Neural Architecture for Infrared Small Target Detection",
            "year": 2021,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "use_attention": self.use_attention,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# Alias for compatibility
SERANKDetector = SERANKNet
