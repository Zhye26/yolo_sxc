"""Composite loss: weighted sum of L1 and MSE."""
from __future__ import annotations

import torch
from torch import nn


class CompositeLoss(nn.Module):
    """Composite loss combining L1 and MSE for super-resolution."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.l1 = nn.L1Loss(reduction=reduction)
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1_weight * self.l1(pred, target) + self.mse_weight * self.mse(
            pred, target
        )


__all__ = ["CompositeLoss"]
