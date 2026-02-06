"""Reverse Residual Attention Fusion (RRAF) block."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.blocks.gddfn import GDDFN
from models.blocks.mddta import MDDTA
from models.utils.convs import DepthwiseConv3d, PointwiseConv3d


class RRAF(nn.Module):
    """Reverse Residual Attention Fusion block.

    Combines depthwise convolution, multi-scale attention, and gated FFN
    with reverse residual connections for efficient feature extraction.
    """

    def __init__(
        self,
        channels: int,
        dilation: int = 2,
        attn_dilations: Sequence[int] = (1, 2, 3),
        bias: bool = False,
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
        self.attn = MDDTA(channels, dilations=attn_dilations, reduction=4, bias=bias)
        self.ffn = GDDFN(channels, expansion=2, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.dw(x))
        y = self.act(y)
        y = self.attn(y)
        y = y + x  # Reverse residual: attention then add
        y = self.ffn(y)
        return y


__all__ = ["RRAF"]
