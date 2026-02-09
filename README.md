# SR3DNet: Lightweight 3D MRI Super-Resolution Network

A PyTorch implementation of SR3DNet for 3D MRI super-resolution reconstruction, featuring RRAF (Residual in Residual Attention Feature) blocks with MDDTA (Multi-Dimensional Dual-path Transposed Attention) and GDDFN (Gated Dual-path Depth-wise Feed-forward Network) modules.

## Project Structure

```
yolo_sxc/
├── configs/
│   └── default.yaml          # Training configuration
├── data/
│   ├── datasets/
│   │   ├── kirby21_dataset.py    # Kirby21 dataset loader
│   │   └── transforms.py         # Data augmentation
│   └── scripts/
│       └── prepare_kirby21.py    # Data preparation script
├── models/
│   ├── blocks/
│   │   ├── rraf.py           # RRAF block with MDDTA & GDDFN
│   │   ├── mddta.py          # Multi-Dimensional Dual-path Transposed Attention
│   │   └── gddfn.py          # Gated Dual-path Depth-wise FFN
│   ├── utils/
│   │   └── pixelshuffle3d.py # 3D PixelShuffle for upsampling
│   ├── sr3d_net.py           # SR3DNet main architecture
│   └── simple3d_cnn.py       # Simple3DCNN baseline
├── engine/
│   ├── trainer.py            # Training loop
│   └── tester.py             # Testing & evaluation
├── losses/
│   └── composite.py          # L1 + MSE composite loss
├── metrics/
│   ├── psnr3d.py             # 3D PSNR metric
│   └── ssim3d.py             # 3D SSIM metric
├── scripts/
│   ├── run_experiments.sh    # Batch experiment runner
│   └── generate_results.py   # Results aggregation
├── visualization/
│   └── visualize.py          # SR result visualization
└── main.py                   # Main entry point
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.8.0
- CUDA >= 10.2 (for GPU training)

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the [Kirby21 dataset](https://www.nitrc.org/projects/multimodal) containing 42 T1-weighted MPRAGE brain MRI volumes (21 subjects × 2 scan-rescan sessions).

### Data Preparation

```bash
# Download and prepare Kirby21 dataset
python data/scripts/prepare_kirby21.py --output_dir ./data/kirby21
```

Data split ratio: **Train:Val:Test = 7:2:1** (subject-level split to avoid data leakage)

## Usage

### Training

```bash
# Train SR3DNet with 2x upscaling
python main.py --mode train --model sr3dnet --scale 2 --exp-name sr3dnet_2x

# Train with 4x upscaling
python main.py --mode train --model sr3dnet --scale 4 --exp-name sr3dnet_4x

# Train Simple3DCNN baseline
python main.py --mode train --model simple3dcnn --scale 2 --exp-name simple3dcnn_2x
```

### Testing

```bash
# Test trained model
python main.py --mode test --model sr3dnet --scale 2 --exp-name sr3dnet_2x
```

### Ablation Experiments

```bash
# Without MDDTA attention
python main.py --mode train --model sr3dnet --scale 2 --no-mddta --exp-name sr3dnet_no_mddta

# Without GDDFN
python main.py --mode train --model sr3dnet --scale 2 --no-gddfn --exp-name sr3dnet_no_gddfn

# Different number of RRAF blocks
python main.py --mode train --model sr3dnet --scale 2 --num-blocks 4 --exp-name sr3dnet_n4
python main.py --mode train --model sr3dnet --scale 2 --num-blocks 12 --exp-name sr3dnet_n12
```

### Run All Experiments

```bash
bash scripts/run_experiments.sh
```

## Experimental Results

### Comparative Experiments

| Model | 2× PSNR (dB) | 2× SSIM | 4× PSNR (dB) | 4× SSIM |
|-------|-------------|---------|-------------|---------|
| Trilinear (Baseline) | 32.21 | 0.884 | 30.60 | 0.836 |
| Simple3DCNN | **36.17** | **0.958** | **31.91** | **0.886** |
| SR3DNet (Ours) | 35.64 | 0.952 | 31.90 | 0.885 |

### Ablation Study

| Configuration | 2× PSNR (dB) | 2× SSIM | 4× PSNR (dB) | 4× SSIM |
|---------------|-------------|---------|-------------|---------|
| SR3DNet (Full) | 35.64 | 0.952 | 31.90 | 0.885 |
| w/o MDDTA | 35.60 | 0.952 | 31.89 | 0.884 |
| w/o GDDFN | 35.98 | 0.956 | 31.91 | 0.886 |
| w/o Both | 36.02 | 0.957 | 31.90 | 0.885 |
| 4 RRAF Blocks | 35.58 | 0.952 | 31.88 | 0.884 |
| 12 RRAF Blocks | 35.75 | 0.954 | 31.89 | 0.885 |

## Model Architecture

```
SR3DNet
├── Shallow Feature Extraction (Conv3D)
├── Deep Feature Extraction
│   └── N × RRAF Blocks
│       ├── MDDTA (Multi-Dimensional Dual-path Transposed Attention)
│       └── GDDFN (Gated Dual-path Depth-wise FFN)
├── Feature Fusion (Conv3D)
└── Upsampling (PixelShuffle3D)
```

## Configuration

Key parameters in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 64 | Training patch size |
| `stride` | 32 | Patch extraction stride |
| `batch_size` | 2 | Batch size (for 8GB GPU) |
| `epochs` | 100 | Training epochs |
| `lr` | 0.0002 | Learning rate |
| `num_blocks` | 8 | Number of RRAF blocks |
| `channels` | 48 | Feature channels |

## License

This project is for academic research purposes.

## Acknowledgments

- Kirby21 dataset: [NITRC](https://www.nitrc.org/projects/multimodal)
- Auckland University of Technology (AUT)
