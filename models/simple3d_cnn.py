"""Simple3DCNN: Basic 3D CNN baseline for comparative experiments."""
from __future__ import annotations

import torch
from torch import nn

from models.utils.pixelshuffle3d import PixelShuffle3D


class Simple3DCNN(nn.Module):
    """Plain 3D CNN baseline without attention or gating mechanisms.

    Architecture: Conv -> N x (Conv+ReLU) -> Conv -> PixelShuffle upsample
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 48,
        num_blocks: int = 8,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4")

        self.head = nn.Conv3d(in_channels, channels, 3, padding=1)

        layers = []
        for _ in range(num_blocks):
            layers += [
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)
        self.fusion = nn.Conv3d(channels, channels, 3, padding=1)

        if scale == 2:
            self.up = nn.Sequential(
                nn.Conv3d(channels, channels * 8, 3, padding=1),
                PixelShuffle3D(2),
            )
        else:
            self.up = nn.Sequential(
                nn.Conv3d(channels, channels * 8, 3, padding=1),
                PixelShuffle3D(2),
                nn.Conv3d(channels, channels * 8, 3, padding=1),
                PixelShuffle3D(2),
            )

        self.out = nn.Conv3d(channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.head(x)
        deep = self.fusion(self.body(shallow))
        features = shallow + deep
        return self.out(self.up(features))


__all__ = ["Simple3DCNN"]
