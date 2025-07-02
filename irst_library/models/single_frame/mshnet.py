"""
Example implementation of MSHNet for infrared small target detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..core.base import BaseModel
from ..core.registry import register_model


class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Different scale branches
        self.branch_1x1 = nn.Conv2d(in_channels, out_channels // 4, 1, padding=0)
        self.branch_3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.branch_5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1, padding=0)
        )
        
        self.conv_concat = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch_1x1(x)
        branch2 = self.branch_3x3(x)
        branch3 = self.branch_5x5(x)
        branch4 = self.branch_pool(x)
        
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.conv_concat(out)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class AttentionModule(nn.Module):
    """Channel and spatial attention module."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x


@register_model("mshnet")
class MSHNet(BaseModel):
    """Multi-Scale Hierarchical Network for infrared small target detection."""
    
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 1,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Build backbone
        self.backbone = self._build_backbone(backbone, pretrained)
        
        # Get backbone output channels
        backbone_channels = self._get_backbone_channels(backbone)
        
        # Multi-scale feature extraction
        self.ms_block1 = MultiScaleBlock(backbone_channels[0], 64)
        self.ms_block2 = MultiScaleBlock(backbone_channels[1], 128)
        self.ms_block3 = MultiScaleBlock(backbone_channels[2], 256)
        self.ms_block4 = MultiScaleBlock(backbone_channels[3], 512)
        
        # Attention modules
        self.attention1 = AttentionModule(64)
        self.attention2 = AttentionModule(128)
        self.attention3 = AttentionModule(256)
        self.attention4 = AttentionModule(512)
        
        # Decoder
        self.decoder4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        
        # Final classifier
        self.classifier = nn.Conv2d(32, num_classes, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_backbone(self, backbone: str, pretrained: bool = True):
        """Build backbone network."""
        if backbone == "resnet18":
            import torchvision.models as models
            resnet = models.resnet18(pretrained=pretrained)
            
            # Remove last two layers (avgpool and fc)
            modules = list(resnet.children())[:-2]
            
            # Modify first conv for single channel input
            first_conv = modules[0]
            modules[0] = nn.Conv2d(
                1, first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # Copy weights from pretrained model (repeat for single channel)
            if pretrained:
                with torch.no_grad():
                    modules[0].weight = nn.Parameter(
                        first_conv.weight.mean(dim=1, keepdim=True)
                    )
            
            # Return feature extraction layers
            return nn.ModuleDict({
                'layer0': nn.Sequential(*modules[:4]),  # conv1, bn1, relu, maxpool
                'layer1': modules[4],  # layer1
                'layer2': modules[5],  # layer2
                'layer3': modules[6],  # layer3
                'layer4': modules[7],  # layer4
            })
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
    
    def _get_backbone_channels(self, backbone: str) -> list:
        """Get number of channels for each backbone layer."""
        if backbone == "resnet18":
            return [64, 64, 128, 256, 512]
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone feature extraction
        c0 = self.backbone['layer0'](x)
        c1 = self.backbone['layer1'](c0)
        c2 = self.backbone['layer2'](c1)
        c3 = self.backbone['layer3'](c2)
        c4 = self.backbone['layer4'](c3)
        
        # Multi-scale feature extraction
        f1 = self.ms_block1(c1)
        f2 = self.ms_block2(c2)
        f3 = self.ms_block3(c3)
        f4 = self.ms_block4(c4)
        
        # Apply attention
        f1 = self.attention1(f1)
        f2 = self.attention2(f2)
        f3 = self.attention3(f3)
        f4 = self.attention4(f4)
        
        # Decoder with skip connections
        d4 = self.decoder4(f4)
        d4 = F.interpolate(d4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        d4 = d4 + f3
        
        d3 = self.decoder3(d4)
        d3 = F.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        d3 = d3 + f2
        
        d2 = self.decoder2(d3)
        d2 = F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        d2 = d2 + f1
        
        d1 = self.decoder1(d2)
        d1 = F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Final classification
        output = self.classifier(d1)
        
        return output
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = "dice_bce"
    ) -> torch.Tensor:
        """Compute loss for training."""
        if loss_type == "bce":
            return F.binary_cross_entropy_with_logits(pred, target)
        elif loss_type == "dice":
            return self._dice_loss(pred, target)
        elif loss_type == "dice_bce":
            bce_loss = F.binary_cross_entropy_with_logits(pred, target)
            dice_loss = self._dice_loss(pred, target)
            return 0.5 * bce_loss + 0.5 * dice_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        pred = torch.sigmoid(pred)
        smooth = 1e-7
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Example of a simpler model
@register_model("simple_unet")
class SimpleUNet(BaseModel):
    """Simple U-Net implementation for comparison."""
    
    def __init__(self, num_classes: int = 1, **kwargs):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(1, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Decoder
        self.dec4 = self._upconv_block(512, 256)
        self.dec3 = self._upconv_block(256, 128)
        self.dec2 = self._upconv_block(128, 64)
        
        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        """Convolution block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels: int, out_channels: int):
        """Upsampling block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder with skip connections
        d4 = self.dec4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        # Final output
        output = self.classifier(d2)
        
        return output
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy loss."""
        return F.binary_cross_entropy_with_logits(pred, target)
