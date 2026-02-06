"""SR3DNet: Lightweight 3D Super-Resolution Network."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models.blocks.rraf import RRAF
from models.utils.convs import DilatedConv3d
from models.utils.pixelshuffle3d import PixelShuffle3D


class Upsample3D(nn.Module):
    """3D upsampling module using PixelShuffle."""

    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4")
        self.scale = scale
        if scale == 2:
            self.conv = nn.Conv3d(channels, channels * 8, kernel_size=3, padding=1)
            self.shuffle = PixelShuffle3D(2)
        else:  # scale == 4
            self.conv1 = nn.Conv3d(channels, channels * 8, kernel_size=3, padding=1)
            self.shuffle1 = PixelShuffle3D(2)
            self.conv2 = nn.Conv3d(channels, channels * 8, kernel_size=3, padding=1)
            self.shuffle2 = PixelShuffle3D(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            return self.shuffle(self.conv(x))
        x = self.shuffle1(self.conv1(x))
        return self.shuffle2(self.conv2(x))


class SR3DNet(nn.Module):
    """Lightweight 3D Super-Resolution Network.

    Architecture:
        - Shallow feature extraction: Dilated Conv3d
        - Deep feature extraction: N stacked RRAF blocks
        - Global residual connection
        - PixelShuffle3D upsampling
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 48,
        num_blocks: int = 8,
        scale: int = 2,
        dilation: int = 2,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4")
        self.scale = scale

        # Shallow feature extraction
        self.head = DilatedConv3d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
        )

        # Deep feature extraction: N stacked RRAF blocks
        self.body = nn.Sequential(
            *[RRAF(channels, dilation=dilation) for _ in range(num_blocks)]
        )

        # Feature fusion
        self.fusion = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

        # Upsampling
        self.upsample = Upsample3D(channels, scale)

        # Output reconstruction
        self.out = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shallow features
        shallow = self.head(x)

        # Deep features with global residual
        deep = self.body(shallow)
        deep = self.fusion(deep)
        features = shallow + deep

        # Upsample and reconstruct
        up = self.upsample(features)
        out = self.out(up)
        return out


__all__ = ["SR3DNet", "Upsample3D"]
