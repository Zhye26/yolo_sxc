"""Kirby21 NIfTI dataset loader for 3D MRI super-resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import Downsample3D, GaussianBlur3D, RandomCrop3D, ToTensor3D


def _as_tuple3(value: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("Expected a sequence of length 3")
    return (int(value[0]), int(value[1]), int(value[2]))


def _zscore_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = float(volume.mean())
    std = float(volume.std())
    return (volume - mean) / (std + eps)


def _pad_to_min(volume: np.ndarray, min_shape: Tuple[int, int, int]) -> np.ndarray:
    pad_width = []
    for dim, min_dim in zip(volume.shape, min_shape):
        pad_total = max(0, min_dim - dim)
        before = pad_total // 2
        after = pad_total - before
        pad_width.append((before, after))
    if all(p == 0 for pair in pad_width for p in pair):
        return volume
    return np.pad(volume, pad_width, mode="edge")


def _load_nifti(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    return np.ascontiguousarray(data, dtype=np.float32)


def _compute_starts(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _build_index_map(
    paths: List[Path],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> List[Tuple[int, int, int, int]]:
    index_map: List[Tuple[int, int, int, int]] = []
    for vol_idx, path in enumerate(paths):
        shape = nib.load(str(path)).shape
        if len(shape) > 3:
            shape = shape[:3]
        eff_shape = tuple(max(shape[i], patch_size[i]) for i in range(3))
        d_starts = _compute_starts(eff_shape[0], patch_size[0], stride[0])
        h_starts = _compute_starts(eff_shape[1], patch_size[1], stride[1])
        w_starts = _compute_starts(eff_shape[2], patch_size[2], stride[2])
        for d in d_starts:
            for h in h_starts:
                for w in w_starts:
                    index_map.append((vol_idx, d, h, w))
    return index_map


def _gather_nii_files(root: Path, split: str) -> List[Path]:
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(
            f"Split directory '{split}' not found at {split_root}. "
            f"Please create {split_root} or provide a file_list."
        )
    nii = list(split_root.rglob("*.nii"))
    nii_gz = list(split_root.rglob("*.nii.gz"))
    return sorted(nii + nii_gz)


class Kirby21Dataset(Dataset):
    """Kirby21 NIfTI dataset with z-score normalization and LR/HR generation."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        scale: int = 2,
        patch_size: Union[int, Sequence[int]] = 64,
        stride: Union[int, Sequence[int]] = 32,
        training: Optional[bool] = None,
        file_list: Optional[Sequence[Union[str, Path]]] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        hr_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        lr_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        blur_sigma: float = 1.2,
        cache: bool = False,
    ) -> None:
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4")
        self.scale = scale
        self.root = Path(root)
        self.split = split
        self.patch_size = _as_tuple3(patch_size)
        self.stride = _as_tuple3(stride)
        self.training = (
            training if training is not None else split.lower() in ("train", "training")
        )
        if file_list is not None:
            self.paths = [Path(p) for p in file_list]
        else:
            self.paths = _gather_nii_files(self.root, self.split)
        if not self.paths:
            raise ValueError("No NIfTI files found for the provided root/split")
        self.transform = transform
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.random_crop = RandomCrop3D(self.patch_size)
        self.to_tensor = ToTensor3D()
        self.blur = GaussianBlur3D(sigma=blur_sigma)
        self.downsample = Downsample3D(scale=self.scale)
        self.cache = cache
        self._cache: Dict[int, np.ndarray] = {}
        if self.cache:
            for idx, path in enumerate(self.paths):
                self._cache[idx] = _load_nifti(path)
        if self.training:
            self.index_map: List[Tuple[int, int, int, int]] = []
        else:
            self.index_map = _build_index_map(self.paths, self.patch_size, self.stride)

    def __len__(self) -> int:
        if self.training:
            return len(self.paths)
        return len(self.index_map)

    def _get_volume(self, index: int) -> np.ndarray:
        if index in self._cache:
            return self._cache[index]
        return _load_nifti(self.paths[index])

    def _make_lr(self, hr: torch.Tensor) -> torch.Tensor:
        if self.lr_transform is not None:
            return self.lr_transform(hr)
        return self.downsample(self.blur(hr))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.training:
            vol = self._get_volume(index).astype(np.float32, copy=True)
            vol = _zscore_normalize(vol)
            if self.transform is not None:
                vol = self.transform(vol)
            vol = self.random_crop(vol)
            hr = self.to_tensor(vol)
            if self.hr_transform is not None:
                hr = self.hr_transform(hr)
            lr = self._make_lr(hr)
            return {
                "lr": lr,
                "hr": hr,
                "path": str(self.paths[index]),
                "scale": self.scale,
            }
        vol_idx, d, h, w = self.index_map[index]
        vol = self._get_volume(vol_idx).astype(np.float32, copy=True)
        orig_shape = vol.shape
        vol = _zscore_normalize(vol)
        if self.transform is not None:
            vol = self.transform(vol)
        vol = _pad_to_min(vol, self.patch_size)
        patch = vol[
            d : d + self.patch_size[0],
            h : h + self.patch_size[1],
            w : w + self.patch_size[2],
        ]
        hr = self.to_tensor(patch)
        if self.hr_transform is not None:
            hr = self.hr_transform(hr)
        lr = self._make_lr(hr)
        return {
            "lr": lr,
            "hr": hr,
            "path": str(self.paths[vol_idx]),
            "coord": (d, h, w),
            "orig_shape": orig_shape,
            "padded_shape": vol.shape,
            "scale": self.scale,
        }


__all__ = ["Kirby21Dataset"]
