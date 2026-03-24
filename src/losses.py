"""
Loss functions for semantic segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging


logger = logging.getLogger(__name__)


# ===========================
# DICE LOSS
# ===========================
class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Reference: https://arxiv.org/abs/1606.06650
    """
    def __init__(self, smooth: float = 1.0, power: int = 2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.power = power
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits or probabilities
            targets: (B, 1, H, W) binary targets
        
        Returns:
            Dice loss value
        """
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.pow(self.power).sum() + targets.pow(self.power).sum()
        
        # Dice loss
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


# ===========================
# FOCAL LOSS
# ===========================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, 1, H, W) binary targets
        
        Returns:
            Focal loss value
        """
        # Sigmoid
        p = torch.sigmoid(predictions)
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy(p, targets, reduction='none')
        
        # Focal loss
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_loss = (1 - p_t) ** self.gamma * bce_loss
        
        # Weighted by alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


# ===========================
# TVERSKY LOSS
# ===========================
class TverskyLoss(nn.Module):
    """
    Tversky Loss - weights false positives and false negatives differently
    Reference: https://arxiv.org/abs/1706.05721
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) probabilities
            targets: (B, 1, H, W) binary targets
        
        Returns:
            Tversky loss value
        """
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        # True and false positives, false negatives
        tp = (predictions * targets).sum()
        fp = (predictions * (1 - targets)).sum()
        fn = ((1 - predictions) * targets).sum()
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky


# ===========================
# JACCARD LOSS
# ===========================
class JaccardLoss(nn.Module):
    """
    Jaccard Loss (IoU Loss)
    Reference: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    def __init__(self, smooth: float = 1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) probabilities
            targets: (B, 1, H, W) binary targets
        
        Returns:
            Jaccard loss value
        """
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - jaccard


# ===========================
# COMBINED LOSS
# ===========================
class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions
    """
    def __init__(
        self,
        loss_weights: Dict[str, float] = None,
        dice_smooth: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float = 7.0
    ):
        super(CombinedLoss, self).__init__()
        
        self.loss_weights = loss_weights or {
            'dice': 0.5,
            'bce': 0.5,
            'focal': 0.0
        }
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits or probabilities
            targets: (B, 1, H, W) binary targets
        
        Returns:
            Combined loss value
        """
        loss = 0.0
        
        if self.loss_weights.get('dice', 0) > 0:
            # Dice expects probabilities
            dice = self.dice_loss(torch.sigmoid(predictions), targets)
            loss += self.loss_weights['dice'] * dice
        
        if self.loss_weights.get('bce', 0) > 0:
            # BCE expects logits
            bce = self.bce_loss(predictions, targets)
            loss += self.loss_weights['bce'] * bce
        
        if self.loss_weights.get('focal', 0) > 0:
            # Focal expects logits
            focal = self.focal_loss(predictions, targets)
            loss += self.loss_weights['focal'] * focal
        
        return loss


# ===========================
# WEIGHTED LOSSES
# ===========================
class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for handling class imbalance
    """
    def __init__(self, pos_weight: float = 7.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, 1, H, W) binary targets
        
        Returns:
            Weighted BCE loss value
        """
        return F.binary_cross_entropy_with_logits(
            predictions, targets,
            pos_weight=torch.tensor([self.pos_weight], device=predictions.device)
        )


# ===========================
# LOSS FACTORY
# ===========================
def create_loss(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """
    Create loss function based on type
    
    Args:
        loss_type: Type of loss ('dice', 'bce', 'focal', 'combined', 'tversky', 'jaccard', 'weighted_bce')
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function module
    """
    if loss_type == 'dice':
        return DiceLoss(**{k: v for k, v in kwargs.items() if k in ['smooth', 'power']})
    
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    elif loss_type == 'focal':
        return FocalLoss(**{k: v for k, v in kwargs.items() if k in ['alpha', 'gamma']})
    
    elif loss_type == 'tversky':
        return TverskyLoss(**{k: v for k, v in kwargs.items() if k in ['alpha', 'beta', 'smooth']})
    
    elif loss_type == 'jaccard':
        return JaccardLoss(**{k: v for k, v in kwargs.items() if k in ['smooth']})
    
    elif loss_type == 'weighted_bce':
        return WeightedBCELoss(**{k: v for k, v in kwargs.items() if k in ['pos_weight']})
    
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


logger.info("Loss functions module loaded")
