"""Generate thesis-ready results: comparison tables, summary figures, model analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.sr3d_net import SR3DNet


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: torch.nn.Module, scale: int, patch: int = 64) -> float:
    """Rough FLOPs estimate via a forward pass with hooks."""
    total_flops = 0
    hooks = []

    def conv_hook(module, inp, out):
        nonlocal total_flops
        b, c_out, *spatial = out.shape
        k = module.kernel_size if hasattr(module, "kernel_size") else (1, 1, 1)
        if isinstance(k, int):
            k = (k, k, k)
        c_in = module.in_channels
        groups = module.groups if hasattr(module, "groups") else 1
        flops = b * c_out * (c_in // groups)
        for ki in k:
            flops *= ki
        for s in spatial:
            flops *= s
        total_flops += flops

    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            hooks.append(m.register_forward_hook(conv_hook))

    lr_size = patch // scale
    x = torch.randn(1, 1, lr_size, lr_size, lr_size)
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return total_flops


def model_summary(scale: int = 2) -> dict:
    model = SR3DNet(scale=scale)
    params = count_parameters(model)
    flops = estimate_flops(model, scale)
    return {
        "scale": scale,
        "parameters": params,
        "parameters_M": f"{params / 1e6:.2f}M",
        "flops": flops,
        "flops_G": f"{flops / 1e9:.2f}G",
    }


def collect_metrics(results_dir: str) -> dict:
    """Collect test metrics from results directories."""
    base = Path(results_dir)
    summary = {}
    for scale_dir in sorted(base.iterdir()):
        if not scale_dir.is_dir():
            continue
        test_dir = scale_dir / "test"
        if not test_dir.exists():
            continue
        scale_name = scale_dir.name
        nifti_files = list(test_dir.glob("*_sr.nii.gz"))
        summary[scale_name] = {
            "num_volumes": len(nifti_files),
            "test_dir": str(test_dir),
        }
    return summary


def print_model_table():
    print("\n" + "=" * 60)
    print("Model Complexity Analysis")
    print("=" * 60)
    print(f"{'Scale':<10}{'Params':<15}{'FLOPs (64^3 input)':<20}")
    print("-" * 60)
    for s in [2, 4]:
        info = model_summary(s)
        print(f"x{s:<9}{info['parameters_M']:<15}{info['flops_G']:<20}")
    print("=" * 60)


def print_results_table(results_dir: str):
    summary = collect_metrics(results_dir)
    if not summary:
        print(f"\nNo test results found in {results_dir}")
        print("Run experiments first: bash scripts/run_experiments.sh")
        return
    print("\n" + "=" * 60)
    print("Experiment Results Summary")
    print("=" * 60)
    for scale_name, info in summary.items():
        print(f"\n[{scale_name}] {info['num_volumes']} test volumes")
        print(f"  Results: {info['test_dir']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate thesis results")
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    print_model_table()
    print_results_table(args.results_dir)


if __name__ == "__main__":
    main()
