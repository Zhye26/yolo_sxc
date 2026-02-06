"""Lightweight 3D convolution helpers for volumetric SR models."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn

Triple = Tuple[int, int, int]


def _to_triple(value: Union[int, Triple]) -> Triple:
    if isinstance(value, int):
        return (value, value, value)
    return value


def _same_padding(kernel_size: Triple, dilation: Triple) -> Triple:
    return tuple(((k - 1) * d) // 2 for k, d in zip(kernel_size, dilation))


class DepthwiseConv3d(nn.Module):
    """Depthwise separable 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: Union[int, Triple] = 3,
        stride: Union[int, Triple] = 1,
        dilation: Union[int, Triple] = 1,
        padding: Optional[Union[int, Triple]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        k = _to_triple(kernel_size)
        s = _to_triple(stride)
        d = _to_triple(dilation)
        p = _same_padding(k, d) if padding is None else _to_triple(padding)
        out_ch = in_channels if out_channels is None else out_channels
        if out_ch % in_channels != 0:
            raise ValueError("out_channels must be a multiple of in_channels")
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PointwiseConv3d(nn.Module):
    """1x1x1 convolution for channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DilatedConv3d(nn.Module):
    """Dilated 3D convolution for expanded receptive field."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Triple] = 3,
        stride: Union[int, Triple] = 1,
        dilation: Union[int, Triple] = 2,
        padding: Optional[Union[int, Triple]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        k = _to_triple(kernel_size)
        s = _to_triple(stride)
        d = _to_triple(dilation)
        p = _same_padding(k, d) if padding is None else _to_triple(padding)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


__all__ = ["DepthwiseConv3d", "PointwiseConv3d", "DilatedConv3d"]
