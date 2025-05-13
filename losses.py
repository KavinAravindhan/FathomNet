# fgvc-comp-2025/losses.py
"""
Loss functions used in the FathomNet pipeline.

Currently provides:
    • FocalLoss  – class-balanced variant of cross-entropy
"""

import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """
    Focal loss for multi-class classification (one-hot targets).

    Parameters
    ----------
    gamma : float
        Focusing parameter.  gamma = 0 → plain cross-entropy.
    alpha : float
        Balancing weight for hard / easy examples.
        (Set to 0.25–0.5 for class imbalance.)
    reduction : str
        'mean' (default) or 'sum'.
    """
    def __init__(self, gamma: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                target_onehot: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor, shape [B, C]
            Raw scores from the model.
        target_onehot : Tensor, shape [B, C], float32
            One-hot (or mixed-label MixUp) targets.

        Returns
        -------
        loss : scalar Tensor
        """
        p = torch.softmax(logits, dim=1).clamp_(1e-6, 1 - 1e-6)
        ce = -(target_onehot * torch.log(p))
        focal_term = self.alpha * (1.0 - p) ** self.gamma * ce

        if self.reduction == "sum":
            return focal_term.sum()
        # default: mean over batch
        return focal_term.sum(dim=1).mean()
