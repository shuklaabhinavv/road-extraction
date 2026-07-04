import segmentation_models_pytorch as smp
import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """Dice + BCE, the standard combination for thin-structure segmentation."""

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce = nn.BCEWithLogitsLoss()
        self.w = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w * self.dice(logits, target) + (1 - self.w) * self.bce(logits, target)
