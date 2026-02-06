"""3D transforms for MRI data preprocessing."""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

Number = Union[int, float]


def _as_tuple3(value: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("Expected a sequence of length 3")
    return (int(value[0]), int(value[1]), int(value[2]))


def _as_tuple3f(value: Union[Number, Sequence[Number]]) -> Tuple[float, float, float]:
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value))
    if len(value) != 3:
        raise ValueError("Expected a sequence of length 3")
    return (float(value[0]), float(value[1]), float(value[2]))


def _pad_to_min(
    volume: Union[np.ndarray, torch.Tensor],
    min_shape: Tuple[int, int, int],
) -> Union[np.ndarray, torch.Tensor]:
    spatial = volume.shape[-3:] if volume.ndim == 4 else volume.shape
    pad_pairs = []
    for dim, min_dim in zip(spatial, min_shape):
        pad_total = max(0, min_dim - dim)
        before = pad_total // 2
        after = pad_total - before
        pad_pairs.append((before, after))
    if all(p == 0 for pair in pad_pairs for p in pair):
        return volume
    if torch.is_tensor(volume):
        pad = (
            pad_pairs[2][0], pad_pairs[2][1],
            pad_pairs[1][0], pad_pairs[1][1],
            pad_pairs[0][0], pad_pairs[0][1],
        )
        if volume.ndim == 4:
            return F.pad(volume, pad, mode="replicate")
        padded = F.pad(volume.unsqueeze(0), pad, mode="replicate")
        return padded.squeeze(0)
    return np.pad(volume, pad_pairs, mode="edge")


class RandomCrop3D:
    """Random 3D crop for data augmentation."""

    def __init__(self, size: Union[int, Sequence[int]]) -> None:
        self.size = _as_tuple3(size)

    def __call__(
        self, volume: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        if volume.ndim not in (3, 4):
            raise ValueError("RandomCrop3D expects 3D or 4D (C,D,H,W) input")
        volume = _pad_to_min(volume, self.size)
        spatial = volume.shape[-3:] if volume.ndim == 4 else volume.shape
        starts = []
        for dim, crop in zip(spatial, self.size):
            starts.append(random.randint(0, max(0, dim - crop)))
        d0, h0, w0 = starts
        d1, h1, w1 = d0 + self.size[0], h0 + self.size[1], w0 + self.size[2]
        if volume.ndim == 3:
            return volume[d0:d1, h0:h1, w0:w1]
        return volume[:, d0:d1, h0:h1, w0:w1]


class Normalize3D:
    """Normalize 3D volume with mean and std."""

    def __init__(
        self,
        mean: Union[Number, Sequence[Number]],
        std: Union[Number, Sequence[Number]],
        eps: float = 1e-8,
    ) -> None:
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(
        self, volume: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        if torch.is_tensor(volume):
            mean = torch.as_tensor(self.mean, device=volume.device, dtype=volume.dtype)
            std = torch.as_tensor(self.std, device=volume.device, dtype=volume.dtype)
            if volume.ndim == 3 or mean.numel() == 1:
                return (volume - mean) / (std + self.eps)
            shape = (mean.numel(), 1, 1, 1)
            return (volume - mean.view(shape)) / (std.view(shape) + self.eps)
        vol = volume.astype(np.float32, copy=False)
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if vol.ndim == 3 or mean.size == 1:
            return (vol - mean) / (std + self.eps)
        return (vol - mean.reshape(-1, 1, 1, 1)) / (std.reshape(-1, 1, 1, 1) + self.eps)


class ToTensor3D:
    """Convert numpy array to PyTorch tensor."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.dtype = dtype

    def __call__(self, volume: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(volume):
            tensor = volume.to(dtype=self.dtype)
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(volume)).to(dtype=self.dtype)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError("ToTensor3D expects 3D or 4D input")
        return tensor


class GaussianBlur3D:
    """Apply 3D Gaussian blur."""

    def __init__(
        self,
        sigma: Union[Number, Sequence[Number]] = 1.0,
        kernel_size: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        self.sigma = _as_tuple3f(sigma)
        self.kernel_size = _as_tuple3(kernel_size) if kernel_size is not None else None

    @staticmethod
    def _auto_kernel_size(sigma: float) -> int:
        return int(2 * math.ceil(2 * sigma) + 1)

    @staticmethod
    def _gaussian_1d(
        size: int, sigma: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if sigma <= 0:
            kernel = torch.zeros(size, device=device, dtype=dtype)
            kernel[size // 2] = 1.0
            return kernel
        radius = size // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def __call__(self, volume: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(volume):
            volume = torch.from_numpy(np.ascontiguousarray(volume))
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        if volume.ndim == 4:
            volume = volume.unsqueeze(0)
            squeeze = True
        elif volume.ndim == 5:
            squeeze = False
        else:
            raise ValueError("GaussianBlur3D expects 3D/4D/5D tensor")
        if all(s <= 0 for s in self.sigma):
            return volume.squeeze(0) if squeeze else volume
        sizes = self.kernel_size or tuple(self._auto_kernel_size(s) for s in self.sigma)
        device = volume.device
        dtype = volume.dtype
        kz = self._gaussian_1d(sizes[0], self.sigma[0], device, dtype)
        ky = self._gaussian_1d(sizes[1], self.sigma[1], device, dtype)
        kx = self._gaussian_1d(sizes[2], self.sigma[2], device, dtype)
        kernel_3d = kz[:, None, None] * ky[None, :, None] * kx[None, None, :]
        kernel_3d = kernel_3d / kernel_3d.sum()
        kernel = kernel_3d.view(1, 1, *kernel_3d.shape)
        channels = volume.shape[1]
        kernel = kernel.repeat(channels, 1, 1, 1, 1)
        pad_d = kernel_3d.shape[0] // 2
        pad_h = kernel_3d.shape[1] // 2
        pad_w = kernel_3d.shape[2] // 2
        pad = (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d)
        volume = F.pad(volume, pad, mode="replicate")
        blurred = F.conv3d(volume, kernel, groups=channels)
        return blurred.squeeze(0) if squeeze else blurred


class Downsample3D:
    """Downsample 3D volume by scale factor."""

    def __init__(
        self,
        scale: int,
        mode: str = "trilinear",
        align_corners: Optional[bool] = False,
    ) -> None:
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4")
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, volume: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(volume):
            volume = torch.from_numpy(np.ascontiguousarray(volume))
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        if volume.ndim == 4:
            volume = volume.unsqueeze(0)
            squeeze = True
        elif volume.ndim == 5:
            squeeze = False
        else:
            raise ValueError("Downsample3D expects 3D/4D/5D tensor")
        scale_factor = 1.0 / float(self.scale)
        if self.mode in ("linear", "bilinear", "trilinear"):
            out = F.interpolate(
                volume,
                scale_factor=scale_factor,
                mode=self.mode,
                align_corners=self.align_corners,
            )
        else:
            out = F.interpolate(volume, scale_factor=scale_factor, mode=self.mode)
        return out.squeeze(0) if squeeze else out


__all__ = [
    "RandomCrop3D",
    "Normalize3D",
    "ToTensor3D",
    "GaussianBlur3D",
    "Downsample3D",
]
