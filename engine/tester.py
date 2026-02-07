"""Test/inference pipeline for 3D MRI super-resolution."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from metrics.psnr3d import psnr_3d
from metrics.ssim3d import ssim_3d

logger = logging.getLogger(__name__)


class Tester:
    """Sliding-window inference with patch aggregation and baseline comparison."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        save_dir: str = "results",
        scale: int = 2,
        patch_size: int = 64,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir) / "test"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale
        self.patch_size = patch_size

    @staticmethod
    def _unpack_coord(batch_coord: tuple) -> Tuple[int, int, int]:
        return tuple(c.item() for c in batch_coord)

    @staticmethod
    def _unpack_shape(batch_shape: tuple) -> Tuple[int, int, int]:
        return tuple(s.item() for s in batch_shape)

    def _collect_patches(self) -> Dict[str, List[dict]]:
        """Run inference on all patches and group results by volume."""
        vol_data: Dict[str, List[dict]] = defaultdict(list)
        with torch.no_grad():
            for batch in self.test_loader:
                lr = batch["lr"].to(self.device)
                sr = self.model(lr)
                baseline = F.interpolate(
                    lr, scale_factor=self.scale,
                    mode="trilinear", align_corners=False,
                )
                vol_data[batch["path"][0]].append({
                    "sr": sr.cpu(),
                    "baseline": baseline.cpu(),
                    "hr": batch["hr"],
                    "coord": self._unpack_coord(batch["coord"]),
                    "padded_shape": self._unpack_shape(batch["padded_shape"]),
                    "orig_shape": self._unpack_shape(batch["orig_shape"]),
                })
        return vol_data

    def _aggregate(
        self, patches: List[dict], padded_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate overlapping patches into full volumes via averaging."""
        ps = self.patch_size
        sr_buf = torch.zeros(1, 1, *padded_shape)
        bl_buf = torch.zeros(1, 1, *padded_shape)
        hr_buf = torch.zeros(1, 1, *padded_shape)
        wt = torch.zeros(1, 1, *padded_shape)

        for p in patches:
            d, h, w = p["coord"]
            sr_buf[:, :, d:d+ps, h:h+ps, w:w+ps] += p["sr"]
            bl_buf[:, :, d:d+ps, h:h+ps, w:w+ps] += p["baseline"]
            hr_buf[:, :, d:d+ps, h:h+ps, w:w+ps] += p["hr"]
            wt[:, :, d:d+ps, h:h+ps, w:w+ps] += 1.0

        wt = wt.clamp(min=1.0)
        return sr_buf / wt, bl_buf / wt, hr_buf / wt

    @staticmethod
    def _crop(vol: torch.Tensor, orig_shape: Tuple[int, int, int]) -> torch.Tensor:
        pd, ph, pw = vol.shape[2:]
        d, h, w = orig_shape
        od = (pd - d) // 2
        oh = (ph - h) // 2
        ow = (pw - w) // 2
        return vol[:, :, od:od + d, oh:oh + h, ow:ow + w]

    @staticmethod
    def _save_nifti(tensor: torch.Tensor, path: Path) -> None:
        arr = tensor.squeeze().numpy().astype(np.float32)
        img = nib.Nifti1Image(arr, affine=np.eye(4))
        nib.save(img, str(path))

    def run(self) -> Dict[str, Dict[str, float]]:
        """Execute full test pipeline. Returns per-volume and average metrics."""
        logger.info("Starting test inference...")
        vol_data = self._collect_patches()
        all_metrics: Dict[str, Dict[str, float]] = {}

        for path, patches in vol_data.items():
            name = Path(path).stem.replace(".nii", "")
            padded_shape = patches[0]["padded_shape"]
            orig_shape = patches[0]["orig_shape"]

            sr_vol, bl_vol, hr_vol = self._aggregate(patches, padded_shape)
            sr_vol = self._crop(sr_vol, orig_shape)
            bl_vol = self._crop(bl_vol, orig_shape)
            hr_vol = self._crop(hr_vol, orig_shape)

            sr_psnr = psnr_3d(sr_vol, hr_vol).item()
            sr_ssim = ssim_3d(sr_vol, hr_vol).item()
            bl_psnr = psnr_3d(bl_vol, hr_vol).item()
            bl_ssim = ssim_3d(bl_vol, hr_vol).item()

            all_metrics[name] = {
                "sr_psnr": sr_psnr, "sr_ssim": sr_ssim,
                "bl_psnr": bl_psnr, "bl_ssim": bl_ssim,
            }
            logger.info(
                f"[{name}] SR PSNR={sr_psnr:.4f} SSIM={sr_ssim:.4f} | "
                f"Baseline PSNR={bl_psnr:.4f} SSIM={bl_ssim:.4f}"
            )

            self._save_nifti(sr_vol, self.save_dir / f"{name}_sr.nii.gz")
            self._save_nifti(bl_vol, self.save_dir / f"{name}_baseline.nii.gz")

            # Generate per-volume visualizations
            from visualization.visualize import plot_slice_comparison, plot_error_map
            sr_np = sr_vol.squeeze().numpy()
            bl_np = bl_vol.squeeze().numpy()
            hr_np = hr_vol.squeeze().numpy()
            plot_slice_comparison(
                bl_np, sr_np, hr_np,
                str(self.save_dir / f"{name}_comparison.png"),
                sr_psnr=sr_psnr, sr_ssim=sr_ssim,
                bl_psnr=bl_psnr, bl_ssim=bl_ssim,
            )
            plot_error_map(
                sr_np, hr_np,
                str(self.save_dir / f"{name}_error_map.png"),
            )

        n = len(all_metrics)
        if n > 0:
            avg_sr_psnr = sum(m["sr_psnr"] for m in all_metrics.values()) / n
            avg_sr_ssim = sum(m["sr_ssim"] for m in all_metrics.values()) / n
            avg_bl_psnr = sum(m["bl_psnr"] for m in all_metrics.values()) / n
            avg_bl_ssim = sum(m["bl_ssim"] for m in all_metrics.values()) / n
            logger.info(
                f"Average SR  PSNR={avg_sr_psnr:.4f} SSIM={avg_sr_ssim:.4f}"
            )
            logger.info(
                f"Average BL  PSNR={avg_bl_psnr:.4f} SSIM={avg_bl_ssim:.4f}"
            )
            all_metrics["average"] = {
                "sr_psnr": avg_sr_psnr, "sr_ssim": avg_sr_ssim,
                "bl_psnr": avg_bl_psnr, "bl_ssim": avg_bl_ssim,
            }

        # Save metrics to JSON
        metrics_path = self.save_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Results saved to {self.save_dir}")
        return all_metrics


__all__ = ["Tester"]
