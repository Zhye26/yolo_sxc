"""Reverse Residual Attention Fusion (RRAF) block."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.blocks.gddfn import GDDFN
from models.blocks.mddta import MDDTA
from models.utils.convs import DepthwiseConv3d, PointwiseConv3d


class SimpleChannelAttn(nn.Module):
    """Ablation replacement for MDDTA: single-branch SE-style attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class SimpleFFN(nn.Module):
    """Ablation replacement for GDDFN: plain Conv-ReLU-Conv FFN."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class RRAF(nn.Module):
    """Reverse Residual Attention Fusion block.

    Combines depthwise convolution, multi-scale attention, and gated FFN
    with reverse residual connections for efficient feature extraction.

    Ablation flags:
        use_mddta: if False, replace MDDTA with SimpleChannelAttn
        use_gddfn: if False, replace GDDFN with SimpleFFN
    """

    def __init__(
        self,
        channels: int,
        dilation: int = 2,
        attn_dilations: Sequence[int] = (1, 2, 3),
        bias: bool = False,
        use_mddta: bool = True,
        use_gddfn: bool = True,
    ) -> None:
        super().__init__()
        self.dw = DepthwiseConv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
            bias=bias,
        )
        self.pw = PointwiseConv3d(channels, channels, bias=bias)
        self.act = nn.GELU()
        self.attn = (
            MDDTA(channels, dilations=attn_dilations, reduction=4, bias=bias)
            if use_mddta
            else SimpleChannelAttn(channels)
        )
        self.ffn = (
            GDDFN(channels, expansion=2, dilation=dilation, bias=bias)
            if use_gddfn
            else SimpleFFN(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.dw(x))
        y = self.act(y)
        y = self.attn(y)
        y = y + x
        y = self.ffn(y)
        return y


__all__ = ["RRAF"]
