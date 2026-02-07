"""Visualization utilities for 3D MRI super-resolution results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_numpy_slice(
    vol: np.ndarray, axis: int = 0, idx: Optional[int] = None,
) -> np.ndarray:
    if idx is None:
        idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


def plot_slice_comparison(
    lr: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    save_path: str,
    axis: int = 0,
    slice_idx: Optional[int] = None,
    sr_psnr: Optional[float] = None,
    sr_ssim: Optional[float] = None,
    bl_psnr: Optional[float] = None,
    bl_ssim: Optional[float] = None,
) -> None:
    """Plot LR(upsampled) / SR / HR slices side-by-side."""
    if slice_idx is None:
        slice_idx = hr.shape[axis] // 2

    lr_s = _to_numpy_slice(lr, axis, slice_idx)
    sr_s = _to_numpy_slice(sr, axis, slice_idx)
    hr_s = _to_numpy_slice(hr, axis, slice_idx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["LR (Trilinear)", "SR (Ours)", "HR (Ground Truth)"]
    slices = [lr_s, sr_s, hr_s]

    vmin, vmax = hr_s.min(), hr_s.max()
    for ax, img, title in zip(axes, slices, titles):
        ax.imshow(img.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    info_parts = []
    if sr_psnr is not None:
        info_parts.append(f"SR PSNR={sr_psnr:.2f}")
    if sr_ssim is not None:
        info_parts.append(f"SSIM={sr_ssim:.4f}")
    if bl_psnr is not None:
        info_parts.append(f"| BL PSNR={bl_psnr:.2f}")
    if bl_ssim is not None:
        info_parts.append(f"SSIM={bl_ssim:.4f}")
    if info_parts:
        fig.suptitle("  ".join(info_parts), fontsize=11, y=0.02)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_map(
    sr: np.ndarray,
    hr: np.ndarray,
    save_path: str,
    axis: int = 0,
    slice_idx: Optional[int] = None,
) -> None:
    """Plot absolute error map |SR - HR| as a heatmap."""
    if slice_idx is None:
        slice_idx = hr.shape[axis] // 2
    diff = np.abs(sr - hr)
    diff_s = _to_numpy_slice(diff, axis, slice_idx)
    sr_s = _to_numpy_slice(sr, axis, slice_idx)
    hr_s = _to_numpy_slice(hr, axis, slice_idx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = hr_s.min(), hr_s.max()
    axes[0].imshow(sr_s.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("SR")
    axes[0].axis("off")
    axes[1].imshow(hr_s.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("HR")
    axes[1].axis("off")
    im = axes[2].imshow(diff_s.T, cmap="hot", origin="lower")
    axes[2].set_title("|SR - HR|")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    log_dir: str,
    save_path: str,
) -> None:
    """Plot training loss, PSNR, and SSIM curves from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("tensorboard not installed, skipping training curve plot")
        return

    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"No scalar data found in {log_dir}")
        return

    tag_map = {
        "train/loss": "Training Loss",
        "val/psnr": "Validation PSNR (dB)",
        "val/ssim": "Validation SSIM",
    }
    available = {k: v for k, v in tag_map.items() if k in tags}
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (tag, label) in zip(axes, available.items()):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        ax.plot(steps, vals, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = ["plot_slice_comparison", "plot_error_map", "plot_training_curves"]
