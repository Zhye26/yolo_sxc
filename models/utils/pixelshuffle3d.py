"""PixelShuffle3D for volumetric upsampling."""
from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn

Triple = Tuple[int, int, int]


def _to_triple(value: Union[int, Triple]) -> Triple:
    if isinstance(value, int):
        return (value, value, value)
    return value


def pixel_shuffle_3d(x: torch.Tensor, scale: Union[int, Triple]) -> torch.Tensor:
    """Rearrange channels to spatial dimensions for 3D upsampling.

    Args:
        x: Input tensor of shape (B, C*s^3, D, H, W)
        scale: Upsampling factor (int or tuple)

    Returns:
        Output tensor of shape (B, C, D*s, H*s, W*s)
    """
    if x.dim() != 5:
        raise ValueError("pixel_shuffle_3d expects a 5D tensor (B, C, D, H, W)")
    sd, sh, sw = _to_triple(scale)
    b, c, d, h, w = x.shape
    factor = sd * sh * sw
    if c % factor != 0:
        raise ValueError("Channel dimension must be divisible by scale^3")
    out_c = c // factor
    x = x.contiguous().view(b, out_c, sd, sh, sw, d, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(b, out_c, d * sd, h * sh, w * sw)


class PixelShuffle3D(nn.Module):
    """3D pixel shuffle layer for upsampling."""

    def __init__(self, scale: Union[int, Triple]) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(x, self.scale)


__all__ = ["PixelShuffle3D", "pixel_shuffle_3d"]
