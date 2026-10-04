"""Loss functions for flood segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        smooth: float = 1e-5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Align target dimensions with logits (B, 1, H, W)
        if logits.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if valid_mask is not None and logits.dim() == 4 and valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)

        probs = torch.sigmoid(logits)
        bce_loss = self.bce(logits, targets.float())

        if valid_mask is not None:
            v_sum = valid_mask.sum() + 1e-8
            bce_loss = (bce_loss * valid_mask).sum() / v_sum

            # Dice calculation on valid pixels only
            probs_masked = probs * valid_mask
            targets_masked = targets.float() * valid_mask
            intersection = (probs_masked * targets_masked).sum()
            union = probs_masked.sum() + targets_masked.sum()
        else:
            bce_loss = bce_loss.mean()
            intersection = (probs * targets.float()).sum()
            union = probs.sum() + targets.float().sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if valid_mask is not None:
            return (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        return focal_loss.mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight: float = 0.5, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = 1.0 - focal_weight
        self.focal = FocalLoss(alpha, gamma)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        focal_loss = self.focal(logits, targets, valid_mask)
        
        probs = torch.sigmoid(logits)
        if valid_mask is not None:
            probs = probs * valid_mask
            targets_masked = targets.float() * valid_mask
            intersection = (probs * targets_masked).sum()
            union = probs.sum() + targets_masked.sum()
        else:
            intersection = (probs * targets.float()).sum()
            union = probs.sum() + targets.float().sum()
            
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

def get_loss_function(name: str, **kwargs) -> nn.Module:
    losses = {
        'bce_dice': BCEDiceLoss,
        'focal': FocalLoss,
        'focal_dice': FocalDiceLoss,
    }
    if name not in losses:
        raise ValueError(f"Loss {name} not found.")
    return losses[name](**kwargs)
