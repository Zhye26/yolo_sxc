#!/usr/bin/env bash
# Run full experiments: train + test for both 2x and 4x scales.
# Usage: bash scripts/run_experiments.sh [GPU_ID]
set -e

GPU="${1:-0}"
CONFIG="configs/default.yaml"

echo "========== Scale 2x Training =========="
python main.py --mode train --config "$CONFIG" --scale 2 --gpu "$GPU"

echo "========== Scale 2x Testing =========="
python main.py --mode test --config "$CONFIG" --scale 2 --gpu "$GPU"

echo "========== Scale 4x Training =========="
python main.py --mode train --config "$CONFIG" --scale 4 --gpu "$GPU"

echo "========== Scale 4x Testing =========="
python main.py --mode test --config "$CONFIG" --scale 4 --gpu "$GPU"

echo "========== Generating summary =========="
python scripts/generate_results.py --results_dir ./results

echo "All experiments completed."
