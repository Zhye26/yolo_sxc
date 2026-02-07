"""CLI entry point for 3D MRI Super-Resolution."""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets.kirby21_dataset import Kirby21Dataset
from engine.trainer import Trainer
from engine.tester import Tester
from models.sr3d_net import SR3DNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(cfg: dict) -> None:
    set_seed(cfg["experiment"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = Kirby21Dataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["train_split"],
        scale=cfg["training"]["scale"],
        patch_size=cfg["data"]["patch_size"],
        stride=cfg["data"]["stride"],
        blur_sigma=cfg["data"]["blur_sigma"],
        cache=cfg["data"]["cache"],
        training=True,
    )

    val_dataset = Kirby21Dataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["val_split"],
        scale=cfg["training"]["scale"],
        patch_size=cfg["data"]["patch_size"],
        stride=cfg["data"]["stride"],
        blur_sigma=cfg["data"]["blur_sigma"],
        cache=cfg["data"]["cache"],
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # Create model
    model = SR3DNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=cfg["model"]["channels"],
        num_blocks=cfg["model"]["num_blocks"],
        scale=cfg["training"]["scale"],
        dilation=cfg["model"]["dilation"],
    )

    # Create trainer and run
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        save_dir=cfg["output"]["save_dir"],
        device=device,
        l1_weight=cfg["training"]["l1_weight"],
        mse_weight=cfg["training"]["mse_weight"],
    )

    trainer.run()


def test(cfg: dict) -> None:
    set_seed(cfg["experiment"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    test_dataset = Kirby21Dataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["test_split"],
        scale=cfg["training"]["scale"],
        patch_size=cfg["data"]["patch_size"],
        stride=cfg["data"]["stride"],
        blur_sigma=cfg["data"]["blur_sigma"],
        cache=cfg["data"]["cache"],
        training=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = SR3DNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=cfg["model"]["channels"],
        num_blocks=cfg["model"]["num_blocks"],
        scale=cfg["training"]["scale"],
        dilation=cfg["model"]["dilation"],
    )

    checkpoint_path = Path(cfg["output"]["save_dir"]) / "best_model.pth"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    tester = Tester(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=cfg["output"]["save_dir"],
        scale=cfg["training"]["scale"],
        patch_size=cfg["data"]["patch_size"],
    )

    metrics = tester.run()

    # Generate visualizations
    from visualization.visualize import plot_training_curves
    log_dir = str(Path(cfg["output"]["save_dir"]) / "logs")
    curves_path = str(Path(cfg["output"]["save_dir"]) / "test" / "training_curves.png")
    plot_training_curves(log_dir, curves_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D MRI Super-Resolution")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode: train or test",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 4],
        default=None,
        help="Override scale factor",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID",
    )

    args = parser.parse_args()

    # Set GPU device before any CUDA operations
    if args.gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Config file validation
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return

    try:
        cfg = load_config(args.config)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return

    if args.scale is not None:
        cfg["training"]["scale"] = args.scale

    if args.mode == "train":
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
