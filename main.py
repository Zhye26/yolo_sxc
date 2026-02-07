"""CLI entry point for 3D MRI Super-Resolution."""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml
from torch.utils.data import DataLoader

from data.datasets.kirby21_dataset import Kirby21Dataset
from engine.trainer import Trainer
from engine.tester import Tester
from models.sr3d_net import SR3DNet
from models.simple3d_cnn import Simple3DCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(cfg: dict, scale: int, model_name: str = "sr3dnet",
                use_mddta: bool = True, use_gddfn: bool = True) -> nn.Module:
    """Build model by name with ablation support."""
    mc = cfg["model"]
    if model_name == "simple3dcnn":
        return Simple3DCNN(
            in_channels=mc["in_channels"], out_channels=mc["out_channels"],
            channels=mc["channels"], num_blocks=mc["num_blocks"], scale=scale,
        )
    return SR3DNet(
        in_channels=mc["in_channels"], out_channels=mc["out_channels"],
        channels=mc["channels"], num_blocks=mc["num_blocks"], scale=scale,
        dilation=mc["dilation"], use_mddta=use_mddta, use_gddfn=use_gddfn,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(cfg: dict, resume: str | None = None,
          model_name: str = "sr3dnet",
          use_mddta: bool = True, use_gddfn: bool = True) -> None:
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
        stride=cfg["data"]["patch_size"],  # non-overlapping for faster validation
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
    model = build_model(cfg, cfg["training"]["scale"], model_name, use_mddta, use_gddfn)

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

    trainer.run(resume=resume)


def test(cfg: dict, checkpoint: str | None = None,
         model_name: str = "sr3dnet",
         use_mddta: bool = True, use_gddfn: bool = True) -> None:
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

    model = build_model(cfg, cfg["training"]["scale"], model_name, use_mddta, use_gddfn)

    checkpoint_path = Path(checkpoint) if checkpoint else Path(cfg["output"]["save_dir"]) / "best_model.pth"
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for testing (default: best_model.pth)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sr3dnet", "simple3dcnn"],
        default="sr3dnet",
        help="Model architecture",
    )
    parser.add_argument("--no-mddta", action="store_true", help="Ablation: disable MDDTA")
    parser.add_argument("--no-gddfn", action="store_true", help="Ablation: disable GDDFN")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Override number of RRAF blocks",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (creates subdirectory under save_dir)",
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

    # Build save_dir: base / [exp_name] / x{scale}
    save_base = Path(cfg["output"]["save_dir"])
    if args.exp_name:
        save_base = save_base / args.exp_name
    if args.scale is not None:
        save_base = save_base / f"x{args.scale}"
    cfg["output"]["save_dir"] = str(save_base)

    if args.num_blocks is not None:
        cfg["model"]["num_blocks"] = args.num_blocks

    use_mddta = not args.no_mddta
    use_gddfn = not args.no_gddfn

    if args.mode == "train":
        train(cfg, resume=args.resume, model_name=args.model,
              use_mddta=use_mddta, use_gddfn=use_gddfn)
    else:
        test(cfg, checkpoint=args.checkpoint, model_name=args.model,
             use_mddta=use_mddta, use_gddfn=use_gddfn)


if __name__ == "__main__":
    main()
