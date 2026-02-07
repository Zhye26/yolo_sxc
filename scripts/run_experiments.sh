#!/usr/bin/env bash
# Full experiment pipeline: comparative + ablation experiments for 2x and 4x.
# Usage: bash scripts/run_experiments.sh [GPU_ID]
set -e

GPU="${1:-0}"
CONFIG="configs/default.yaml"
PY="python3"

run_exp() {
    local name="$1" scale="$2" extra_args="$3"
    echo ""
    echo "====== [${name}] Scale ${scale}x Training ======"
    $PY main.py --mode train --config "$CONFIG" --scale "$scale" --gpu "$GPU" --exp-name "$name" $extra_args
    echo "====== [${name}] Scale ${scale}x Testing ======"
    $PY main.py --mode test  --config "$CONFIG" --scale "$scale" --gpu "$GPU" --exp-name "$name" $extra_args
}

echo "============================================"
echo "  3D MRI Super-Resolution Experiments"
echo "============================================"

# --- Comparative Experiments ---
echo ""
echo ">>> COMPARATIVE EXPERIMENTS <<<"

# 1. SR3DNet (Ours - full model)
for s in 2 4; do run_exp "sr3dnet" "$s" ""; done

# 2. Simple 3D CNN baseline
for s in 2 4; do run_exp "simple3dcnn" "$s" "--model simple3dcnn"; done

# --- Ablation Experiments ---
echo ""
echo ">>> ABLATION EXPERIMENTS <<<"

# 3. w/o MDDTA
for s in 2 4; do run_exp "sr3dnet_no_mddta" "$s" "--no-mddta"; done

# 4. w/o GDDFN
for s in 2 4; do run_exp "sr3dnet_no_gddfn" "$s" "--no-gddfn"; done

# 5. w/o MDDTA & GDDFN
for s in 2 4; do run_exp "sr3dnet_no_both" "$s" "--no-mddta --no-gddfn"; done

# 6. N=4 RRAF blocks
for s in 2 4; do run_exp "sr3dnet_n4" "$s" "--num-blocks 4"; done

# 7. N=12 RRAF blocks
for s in 2 4; do run_exp "sr3dnet_n12" "$s" "--num-blocks 12"; done

# --- Summary ---
echo ""
echo ">>> GENERATING RESULTS SUMMARY <<<"
$PY scripts/generate_results.py --results_dir ./results

echo ""
echo "All experiments completed."
