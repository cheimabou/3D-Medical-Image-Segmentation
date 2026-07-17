"""
Loss functions for nodule segmentation.
BCEDiceLoss:   L = alpha * BCE + (1-alpha) * DiceLoss
FocalDiceLoss: L = alpha * FocalLoss + (1-alpha) * DiceLoss 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(logits: torch.Tensor,
              targets: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    probs   = torch.sigmoid(logits)
    probs   = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    has_foreground = targets.sum(dim=1) > 0
    if has_foreground.any():
        probs   = probs[has_foreground]
        targets = targets[has_foreground]
    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()
def bce_loss(logits: torch.Tensor,
             targets: torch.Tensor,
             pos_weight: torch.Tensor = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )

class BCEDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, pos_weight: float = None):
        super().__init__()
        self.alpha      = alpha
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight],
                              device=logits.device,
                              dtype=logits.dtype)
        bce  = bce_loss(logits, targets, pos_weight=pw)
        dice = dice_loss(logits, targets)
        total = self.alpha * bce + (1.0 - self.alpha) * dice
        return total, bce.detach(), dice.detach()

