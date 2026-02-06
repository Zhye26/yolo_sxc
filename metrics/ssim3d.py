"""3D SSIM calculation with Gaussian window."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_5d(volume: torch.Tensor) -> torch.Tensor:
    if volume.dim() == 3:
        return volume.unsqueeze(0).unsqueeze(0)
    if volume.dim() == 4:
        return volume.unsqueeze(0)
    if volume.dim() == 5:
        return volume
    raise ValueError("ssim_3d expects a 3D/4D/5D tensor")


def _gaussian_1d(
    size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    radius = size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_window_3d(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kz = _gaussian_1d(window_size, sigma, device, dtype)
    ky = _gaussian_1d(window_size, sigma, device, dtype)
    kx = _gaussian_1d(window_size, sigma, device, dtype)
    kernel = kz[:, None, None] * ky[None, :, None] * kx[None, None, :]
    kernel = kernel / kernel.sum()
    window = kernel.view(1, 1, window_size, window_size, window_size)
    return window.repeat(channels, 1, 1, 1, 1)


def ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate Structural Similarity Index for 3D volumes.

    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        data_range: Maximum value range. If None, computed from target.
        window_size: Size of the Gaussian window (must be odd)
        sigma: Standard deviation of the Gaussian window
        reduction: 'mean' or 'none'

    Returns:
        SSIM value(s)
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if data_range is None:
        data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = 1.0
    x = _to_5d(pred)
    y = _to_5d(target)
    channels = x.shape[1]
    window = _gaussian_window_3d(window_size, sigma, channels, x.device, x.dtype)
    padding = window_size // 2
    mu_x = F.conv3d(x, window, padding=padding, groups=channels)
    mu_y = F.conv3d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv3d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv3d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv3d(x * y, window, padding=padding, groups=channels) - mu_xy
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-8)
    ssim_per = ssim_map.mean(dim=(1, 2, 3, 4))
    if reduction == "mean":
        return ssim_per.mean()
    if reduction == "none":
        return ssim_per
    raise ValueError("Unsupported reduction. Use 'mean' or 'none'")


__all__ = ["ssim_3d"]
