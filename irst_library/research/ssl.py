"""
Self-Supervised Learning for IRST Library

Implements state-of-the-art self-supervised learning methods for ISTD:
- SimCLR: Simple Framework for Contrastive Learning
- MoCo: Momentum Contrast
- SwAV: Swapping Assignments between Views
- BYOL: Bootstrap Your Own Latent
- SimSiam: Simple Siamese Networks
- MAE: Masked Autoencoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import random
from collections import deque
import copy

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feature_dim) normalized features
            labels: (batch_size,) optional labels for supervised contrastive learning
        """
        batch_size = features.shape[0]
        
        if labels is None:
            # SimCLR style: assume consecutive pairs are positive
            labels = torch.arange(batch_size // 2).repeat(2)
            labels = labels.sort()[0]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask.fill_diagonal_(0)
        
        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Compute log probability
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ProjectionHead(nn.Module):
    """Projection head for self-supervised learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, 
                 output_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MomentumEncoder(nn.Module):
    """Momentum encoder for MoCo and BYOL"""
    
    def __init__(self, encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Create momentum encoder
        self.momentum_encoder = copy.deepcopy(encoder)
        
        # Stop gradients for momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    def update_momentum_encoder(self):
        """Update momentum encoder parameters"""
        for param_q, param_k in zip(self.encoder.parameters(), 
                                   self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    def forward(self, x: torch.Tensor, use_momentum: bool = False) -> torch.Tensor:
        if use_momentum:
            with torch.no_grad():
                return self.momentum_encoder(x)
        else:
            return self.encoder(x)


class QueueManager:
    """Queue for negative samples in MoCo"""
    
    def __init__(self, queue_size: int = 65536, feature_dim: int = 128):
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.queue = torch.randn(feature_dim, queue_size)
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0
    
    def dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue with new keys"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, "Queue size should be divisible by batch size"
        
        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # Move pointer
        
        self.queue_ptr = ptr
    
    def get_queue(self) -> torch.Tensor:
        return self.queue.clone()


class SimCLR(nn.Module):
    """SimCLR: Simple Framework for Contrastive Learning"""
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 128, 
                 temperature: float = 0.07):
        super().__init__()
        self.encoder = encoder
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            encoder_output = self.encoder(dummy_input)
            encoder_dim = encoder_output.shape[-1]
        
        self.projection_head = ProjectionHead(encoder_dim, output_dim=projection_dim)
        self.criterion = ContrastiveLoss(temperature)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with two augmented views"""
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project to contrastive space
        z1 = F.normalize(self.projection_head(h1), dim=1)
        z2 = F.normalize(self.projection_head(z2), dim=1)
        
        # Concatenate for contrastive loss
        features = torch.cat([z1, z2], dim=0)
        
        # Compute loss
        loss = self.criterion(features)
        
        return {
            'loss': loss,
            'features': features,
            'representations': torch.cat([h1, h2], dim=0)
        }


class MoCo(nn.Module):
    """MoCo: Momentum Contrast for Unsupervised Visual Representation Learning"""
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 128,
                 queue_size: int = 65536, momentum: float = 0.999, 
                 temperature: float = 0.07):
        super().__init__()
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            encoder_output = encoder(dummy_input)
            encoder_dim = encoder_output.shape[-1]
        
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        
        self.projection_q = ProjectionHead(encoder_dim, output_dim=projection_dim)
        self.projection_k = copy.deepcopy(self.projection_q)
        
        # Stop gradients for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projection_k.parameters():
            param.requires_grad = False
        
        self.momentum = momentum
        self.temperature = temperature
        self.queue_manager = QueueManager(queue_size, projection_dim)
        
        # Register queue as buffer
        self.register_buffer("queue", self.queue_manager.get_queue())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _momentum_update_key_encoder(self):
        """Momentum update of key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                   self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
        
        for param_q, param_k in zip(self.projection_q.parameters(), 
                                   self.projection_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace keys in queue
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue.shape[1]
        
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Query features
        q = self.encoder_q(im_q)
        q = F.normalize(self.projection_q(q), dim=1)
        
        # Key features (no gradients)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k = self.encoder_k(im_k)
            k = F.normalize(self.projection_k(k), dim=1)
        
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return {
            'loss': loss,
            'logits': logits,
            'targets': labels
        }


class BYOL(nn.Module):
    """BYOL: Bootstrap Your Own Latent"""
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 256,
                 hidden_dim: int = 4096, momentum: float = 0.996):
        super().__init__()
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            encoder_output = encoder(dummy_input)
            encoder_dim = encoder_output.shape[-1]
        
        # Online network
        self.online_encoder = encoder
        self.online_projector = ProjectionHead(encoder_dim, hidden_dim, projection_dim)
        self.online_predictor = ProjectionHead(projection_dim, hidden_dim, projection_dim)
        
        # Target network (momentum update)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Stop gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        self.momentum = momentum
    
    def _update_target_network(self):
        """Update target network with momentum"""
        for param_online, param_target in zip(self.online_encoder.parameters(),
                                             self.target_encoder.parameters()):
            param_target.data = param_target.data * self.momentum + \
                               param_online.data * (1.0 - self.momentum)
        
        for param_online, param_target in zip(self.online_projector.parameters(),
                                             self.target_projector.parameters()):
            param_target.data = param_target.data * self.momentum + \
                               param_online.data * (1.0 - self.momentum)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Online network
        online_proj_1 = self.online_projector(self.online_encoder(x1))
        online_proj_2 = self.online_projector(self.online_encoder(x2))
        
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)
        
        # Target network (no gradients)
        with torch.no_grad():
            self._update_target_network()
            
            target_proj_1 = self.target_projector(self.target_encoder(x1))
            target_proj_2 = self.target_projector(self.target_encoder(x2))
        
        # Compute loss (negative cosine similarity)
        loss_1 = 2 - 2 * F.cosine_similarity(online_pred_1, target_proj_2, dim=-1).mean()
        loss_2 = 2 - 2 * F.cosine_similarity(online_pred_2, target_proj_1, dim=-1).mean()
        
        loss = (loss_1 + loss_2) / 2
        
        return {
            'loss': loss,
            'online_proj': torch.cat([online_proj_1, online_proj_2], dim=0),
            'target_proj': torch.cat([target_proj_1, target_proj_2], dim=0)
        }


class SimSiam(nn.Module):
    """SimSiam: Simple Siamese Networks for Self-Supervised Learning"""
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 2048,
                 prediction_dim: int = 512):
        super().__init__()
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            encoder_output = encoder(dummy_input)
            encoder_dim = encoder_output.shape[-1]
        
        self.encoder = encoder
        self.projector = ProjectionHead(encoder_dim, projection_dim, projection_dim, 3)
        self.predictor = ProjectionHead(projection_dim, prediction_dim, projection_dim, 2)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode both views
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        # Predict
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Compute loss (negative cosine similarity)
        # Stop gradient for z (following paper)
        loss = -(F.cosine_similarity(p1, z2.detach(), dim=-1).mean() + 
                F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) * 0.5
        
        return {
            'loss': loss,
            'z1': z1,
            'z2': z2,
            'p1': p1,
            'p2': p2
        }


class MaskedAutoEncoder(nn.Module):
    """MAE: Masked Autoencoders for Self-Supervised Learning"""
    
    def __init__(self, encoder: nn.Module, decoder_dim: int = 512,
                 mask_ratio: float = 0.75, patch_size: int = 16):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            encoder_output = encoder(dummy_input)
            encoder_dim = encoder_output.shape[-1]
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, patch_size * patch_size * 3)
        )
        
        self.criterion = nn.MSELoss()
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches"""
        B, C, H, W = imgs.shape
        p = self.patch_size
        
        # Reshape to patches
        x = imgs.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        
        return x
    
    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert patches back to images"""
        B, N, _ = x.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        
        x = x.reshape(B, h, w, 3, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, 3, H, W)
        
        return x
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random masking"""
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        B, C, H, W = imgs.shape
        
        # Patchify
        patches = self.patchify(imgs)
        
        # Random masking
        patches_masked, mask, ids_restore = self.random_masking(patches)
        
        # Encode visible patches
        features = self.encoder(patches_masked)
        
        # Decode
        reconstructed = self.decoder(features)
        
        # Compute loss only on masked patches
        target = patches
        loss = self.criterion(reconstructed, target)
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches
        
        return {
            'loss': loss,
            'reconstructed': reconstructed,
            'mask': mask,
            'target': target
        }


class SelfSupervisedLearning:
    """Main SSL interface"""
    
    def __init__(self, method: str = 'simclr', encoder: nn.Module = None, **kwargs):
        self.method = method
        self.encoder = encoder or self._create_default_encoder()
        
        # Initialize SSL model based on method
        if method == 'simclr':
            self.model = SimCLR(self.encoder, **kwargs)
        elif method == 'moco':
            self.model = MoCo(self.encoder, **kwargs)
        elif method == 'byol':
            self.model = BYOL(self.encoder, **kwargs)
        elif method == 'simsiam':
            self.model = SimSiam(self.encoder, **kwargs)
        elif method == 'mae':
            self.model = MaskedAutoEncoder(self.encoder, **kwargs)
        else:
            raise ValueError(f"Unknown SSL method: {method}")
        
        self.optimizer = None
        self.scheduler = None
    
    def _create_default_encoder(self) -> nn.Module:
        """Create default encoder"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512)
        )
    
    def train(self, dataloader, epochs: int = 100, learning_rate: float = 0.001):
        """Train SSL model"""
        device = next(self.model.parameters()).device
        
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, (x1, x2) in enumerate(dataloader):
                x1, x2 = x1.to(device), x2.to(device)
                
                self.optimizer.zero_grad()
                
                if self.method == 'mae':
                    # MAE only needs one view
                    outputs = self.model(x1)
                else:
                    outputs = self.model(x1, x2)
                
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            self.scheduler.step()
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")
    
    def extract_features(self, dataloader) -> torch.Tensor:
        """Extract features using trained encoder"""
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for batch, _ in dataloader:
                if hasattr(self.model, 'encoder'):
                    feat = self.model.encoder(batch)
                else:
                    feat = self.model.online_encoder(batch)
                features.append(feat.cpu())
        
        return torch.cat(features, dim=0)
    
    def save_model(self, path: str):
        """Save SSL model"""
        torch.save({
            'method': self.method,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        
        logger.info(f"SSL model saved to {path}")
    
    def load_model(self, path: str):
        """Load SSL model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"SSL model loaded from {path}")


# Data augmentation utilities for SSL
class SSLTransforms:
    """Self-supervised learning data augmentations"""
    
    @staticmethod
    def get_simclr_transforms(size: int = 224):
        """Get SimCLR augmentations"""
        import torchvision.transforms as T
        
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_mae_transforms(size: int = 224):
        """Get MAE augmentations"""
        import torchvision.transforms as T
        
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
