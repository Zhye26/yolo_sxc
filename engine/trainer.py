"""Training loop for 3D MRI super-resolution."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from losses.composite import CompositeLoss
from metrics.psnr3d import psnr_3d
from metrics.ssim3d import ssim_3d

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with validation, checkpointing, and logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        save_dir: str = "results",
        device: str = "cuda",
        l1_weight: float = 1.0,
        mse_weight: float = 0.1,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = CompositeLoss(l1_weight=l1_weight, mse_weight=mse_weight)
        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))
        self.best_psnr = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            lr_img = batch["lr"].to(self.device)
            hr_img = batch["hr"].to(self.device)

            self.optimizer.zero_grad()
            sr_img = self.model(lr_img)
            loss = self.criterion(sr_img, hr_img)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        return {"loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        for batch in self.val_loader:
            lr_img = batch["lr"].to(self.device)
            hr_img = batch["hr"].to(self.device)

            sr_img = self.model(lr_img)
            total_psnr += psnr_3d(sr_img, hr_img).item()
            total_ssim += ssim_3d(sr_img, hr_img).item()
            count += 1

        return {
            "psnr": total_psnr / count if count > 0 else 0.0,
            "ssim": total_ssim / count if count > 0 else 0.0,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{epoch}.pth")
        if metrics.get("psnr", 0) > self.best_psnr:
            self.best_psnr = metrics["psnr"]
            torch.save(checkpoint, self.save_dir / "best_model.pth")
            logger.info(f"New best model saved with PSNR: {self.best_psnr:.4f}")

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"]

    def run(self, resume: str | None = None) -> None:
        start_epoch = 1
        if resume:
            start_epoch = self.load_checkpoint(resume) + 1
            logger.info(f"Resumed from epoch {start_epoch - 1}")
        logger.info(f"Training epochs {start_epoch}..{self.epochs}")
        for epoch in range(start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            if val_metrics:
                self.writer.add_scalar("val/psnr", val_metrics["psnr"], epoch)
                self.writer.add_scalar("val/ssim", val_metrics["ssim"], epoch)

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch}/{self.epochs} - "
                f"Loss: {train_metrics['loss']:.6f} - "
                f"PSNR: {val_metrics.get('psnr', 0):.4f} - "
                f"SSIM: {val_metrics.get('ssim', 0):.4f}"
            )

            if epoch % 10 == 0 or epoch == self.epochs:
                self.save_checkpoint(epoch, val_metrics)

        self.writer.close()
        logger.info("Training completed")


__all__ = ["Trainer"]
