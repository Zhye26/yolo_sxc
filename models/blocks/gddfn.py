"""Gated-DDConv Feed-Forward Network (GDDFN)."""
from __future__ import annotations

import torch
from torch import nn

from models.utils.convs import DepthwiseConv3d, PointwiseConv3d


class GDDFN(nn.Module):
    """Gated feed-forward network with depthwise dilated convolutions.

    Uses split-gate mechanism for efficient feature transformation.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        dilation: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        hidden = channels * expansion
        if hidden % 2 != 0:
            raise ValueError("channels * expansion must be even for gating")

        self.expand = PointwiseConv3d(channels, hidden, bias=bias)
        self.depthwise = DepthwiseConv3d(
            in_channels=hidden // 2,
            out_channels=hidden // 2,
            kernel_size=3,
            dilation=dilation,
            bias=bias,
        )
        self.project = PointwiseConv3d(hidden // 2, channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        a, b = torch.chunk(y, 2, dim=1)
        a = self.depthwise(a)
        b = torch.sigmoid(b)
        y = a * b
        y = self.project(y)
        return y + x


__all__ = ["GDDFN"]
