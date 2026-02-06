"""3D PSNR calculation."""
from __future__ import annotations

import torch


def psnr_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate Peak Signal-to-Noise Ratio for 3D volumes.

    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        data_range: Maximum value range. If None, computed from target.
        eps: Small value to avoid log(0)
        reduction: 'mean' or 'none'

    Returns:
        PSNR value(s)
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if data_range is None:
        data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = 1.0
    diff = pred - target
    if diff.dim() == 3:
        mse = diff.pow(2).mean()
        data_range_t = torch.as_tensor(data_range, device=pred.device, dtype=pred.dtype)
        return 20 * torch.log10(data_range_t) - 10 * torch.log10(mse + eps)
    if diff.dim() < 4:
        raise ValueError("psnr_3d expects at least 3 spatial dimensions")
    dims = tuple(range(1, diff.dim()))
    mse = diff.pow(2).mean(dim=dims)
    data_range_t = torch.as_tensor(data_range, device=pred.device, dtype=pred.dtype)
    psnr = 20 * torch.log10(data_range_t) - 10 * torch.log10(mse + eps)
    if reduction == "mean":
        return psnr.mean()
    if reduction == "none":
        return psnr
    raise ValueError("Unsupported reduction. Use 'mean' or 'none'")


__all__ = ["psnr_3d"]
