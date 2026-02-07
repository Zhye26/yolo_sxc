"""Generate thesis-ready results: comparison tables, model analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.sr3d_net import SR3DNet
from models.simple3d_cnn import Simple3DCNN


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: torch.nn.Module, scale: int, patch: int = 64) -> float:
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


MODEL_CONFIGS = {
    "sr3dnet":          {"cls": SR3DNet,    "kw": {}},
    "simple3dcnn":      {"cls": Simple3DCNN, "kw": {}},
    "sr3dnet_no_mddta": {"cls": SR3DNet,    "kw": {"use_mddta": False}},
    "sr3dnet_no_gddfn": {"cls": SR3DNet,    "kw": {"use_mddta": True, "use_gddfn": False}},
    "sr3dnet_no_both":  {"cls": SR3DNet,    "kw": {"use_mddta": False, "use_gddfn": False}},
    "sr3dnet_n4":       {"cls": SR3DNet,    "kw": {"num_blocks": 4}},
    "sr3dnet_n12":      {"cls": SR3DNet,    "kw": {"num_blocks": 12}},
}

DISPLAY_NAMES = {
    "sr3dnet":          "SR3DNet (Ours)",
    "simple3dcnn":      "Simple 3D CNN",
    "sr3dnet_no_mddta": "w/o MDDTA",
    "sr3dnet_no_gddfn": "w/o GDDFN",
    "sr3dnet_no_both":  "w/o MDDTA & GDDFN",
    "sr3dnet_n4":       "N=4 RRAF blocks",
    "sr3dnet_n12":      "N=12 RRAF blocks",
    "trilinear":        "Trilinear Interp.",
}


def build_model_info(name: str, scale: int) -> dict:
    cfg = MODEL_CONFIGS[name]
    model = cfg["cls"](scale=scale, **cfg["kw"])
    params = count_parameters(model)
    flops = estimate_flops(model, scale)
    return {"params": params, "params_M": f"{params/1e6:.2f}M", "flops_G": f"{flops/1e9:.2f}G"}


def collect_all_metrics(results_dir: Path) -> dict:
    """Scan results/{exp_name}/x{scale}/test/metrics.json for all experiments."""
    data = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        data[exp_name] = {}
        for scale_dir in sorted(exp_dir.iterdir()):
            if not scale_dir.is_dir():
                continue
            metrics_file = scale_dir / "test" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                data[exp_name][scale_dir.name] = m.get("average", {})
    return data


def print_complexity_table():
    print("\n" + "=" * 70)
    print("Table 1: Model Complexity Comparison")
    print("=" * 70)
    header = f"{'Method':<25}{'Scale':<8}{'Params':<12}{'FLOPs':<12}"
    print(header)
    print("-" * 70)
    for name in ["sr3dnet", "simple3dcnn", "sr3dnet_no_mddta",
                  "sr3dnet_no_gddfn", "sr3dnet_no_both", "sr3dnet_n4", "sr3dnet_n12"]:
        for s in [2, 4]:
            info = build_model_info(name, s)
            label = DISPLAY_NAMES.get(name, name)
            print(f"{label:<25}x{s:<7}{info['params_M']:<12}{info['flops_G']:<12}")
    print("=" * 70)


def print_comparative_table(metrics: dict):
    print("\n" + "=" * 70)
    print("Table 2: Comparative Experiment Results (PSNR / SSIM)")
    print("=" * 70)
    comp_methods = ["sr3dnet", "simple3dcnn"]
    header = f"{'Method':<25}{'2x PSNR':<12}{'2x SSIM':<12}{'4x PSNR':<12}{'4x SSIM':<12}"
    print(header)
    print("-" * 70)

    # Trilinear baseline (from sr3dnet test results)
    for scale_key in ["x2", "x4"]:
        sr3d = metrics.get("sr3dnet", {}).get(scale_key, {})
        if sr3d:
            break
    trilinear_row = f"{'Trilinear Interp.':<25}"
    for sk in ["x2", "x4"]:
        m = metrics.get("sr3dnet", {}).get(sk, {})
        if m:
            trilinear_row += f"{m.get('bl_psnr', '-'):<12.4f}" if isinstance(m.get('bl_psnr'), float) else f"{'-':<12}"
            trilinear_row += f"{m.get('bl_ssim', '-'):<12.4f}" if isinstance(m.get('bl_ssim'), float) else f"{'-':<12}"
        else:
            trilinear_row += f"{'-':<12}{'-':<12}"
    print(trilinear_row)

    for name in comp_methods:
        label = DISPLAY_NAMES.get(name, name)
        row = f"{label:<25}"
        for sk in ["x2", "x4"]:
            m = metrics.get(name, {}).get(sk, {})
            if m and "sr_psnr" in m:
                row += f"{m['sr_psnr']:<12.4f}{m['sr_ssim']:<12.4f}"
            else:
                row += f"{'-':<12}{'-':<12}"
        print(row)
    print("=" * 70)


def print_ablation_table(metrics: dict):
    print("\n" + "=" * 70)
    print("Table 3: Ablation Study Results (PSNR / SSIM)")
    print("=" * 70)
    ablation_methods = ["sr3dnet", "sr3dnet_no_mddta", "sr3dnet_no_gddfn",
                        "sr3dnet_no_both", "sr3dnet_n4", "sr3dnet_n12"]
    header = f"{'Variant':<25}{'2x PSNR':<12}{'2x SSIM':<12}{'4x PSNR':<12}{'4x SSIM':<12}"
    print(header)
    print("-" * 70)
    for name in ablation_methods:
        label = DISPLAY_NAMES.get(name, name)
        row = f"{label:<25}"
        for sk in ["x2", "x4"]:
            m = metrics.get(name, {}).get(sk, {})
            if m and "sr_psnr" in m:
                row += f"{m['sr_psnr']:<12.4f}{m['sr_ssim']:<12.4f}"
            else:
                row += f"{'-':<12}{'-':<12}"
        print(row)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate thesis results")
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    print_complexity_table()

    results_path = Path(args.results_dir)
    metrics = collect_all_metrics(results_path)
    if metrics:
        print_comparative_table(metrics)
        print_ablation_table(metrics)
    else:
        print(f"\nNo metrics found in {results_path}. Run experiments first.")


if __name__ == "__main__":
    main()
