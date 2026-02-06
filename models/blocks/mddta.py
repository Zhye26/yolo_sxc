"""Multi-Depthwise Dilated Convolution Transposed Attention (MDDTA)."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.utils.convs import DepthwiseConv3d, PointwiseConv3d


class MDDTA(nn.Module):
    """Multi-scale attention using depthwise dilated convolutions.

    Captures features at multiple receptive field scales and applies
    channel attention for adaptive feature weighting.
    """

    def __init__(
        self,
        channels: int,
        dilations: Sequence[int] = (1, 2, 3),
        reduction: int = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        self.branches = nn.ModuleList([
            DepthwiseConv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                dilation=d,
                bias=bias,
            )
            for d in dilations
        ])
        self.fuse = PointwiseConv3d(
            in_channels=channels * len(dilations),
            out_channels=channels,
            bias=bias,
        )
        mid = max(channels // reduction, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        y = self.fuse(torch.cat(feats, dim=1))
        g = self.attn(y)
        return y * g


__all__ = ["MDDTA"]
