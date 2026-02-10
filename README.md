# SR3DNet: Lightweight 3D MRI Super-Resolution Network

[English](#english) | [中文](#中文)

---

## English

A PyTorch implementation of SR3DNet for 3D MRI super-resolution reconstruction, featuring RRAF (Residual in Residual Attention Feature) blocks with MDDTA (Multi-Dimensional Dual-path Transposed Attention) and GDDFN (Gated Dual-path Depth-wise Feed-forward Network) modules.

### Project Structure

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

### Requirements

- Python >= 3.8
- PyTorch >= 1.8.0
- CUDA >= 10.2 (for GPU training)

```bash
pip install -r requirements.txt
```

### Dataset

This project uses the [Kirby21 dataset](https://www.nitrc.org/projects/multimodal) containing 42 T1-weighted MPRAGE brain MRI volumes (21 subjects × 2 scan-rescan sessions).

**Data Preparation:**
```bash
python data/scripts/prepare_kirby21.py --output_dir ./data/kirby21
```

Data split ratio: **Train:Val:Test = 7:2:1** (subject-level split)

### Usage

**Training:**
```bash
# Train SR3DNet with 2x upscaling
python main.py --mode train --model sr3dnet --scale 2 --exp-name sr3dnet_2x

# Train with 4x upscaling
python main.py --mode train --model sr3dnet --scale 4 --exp-name sr3dnet_4x
```

**Testing:**
```bash
python main.py --mode test --model sr3dnet --scale 2 --exp-name sr3dnet_2x
```

**Ablation Experiments:**
```bash
# Without MDDTA attention
python main.py --mode train --model sr3dnet --scale 2 --no-mddta --exp-name sr3dnet_no_mddta

# Without GDDFN
python main.py --mode train --model sr3dnet --scale 2 --no-gddfn --exp-name sr3dnet_no_gddfn

# Different number of RRAF blocks
python main.py --mode train --model sr3dnet --scale 2 --num-blocks 4 --exp-name sr3dnet_n4
```

### Experimental Results

#### Comparative Experiments

| Model | 2× PSNR (dB) | 2× SSIM | 4× PSNR (dB) | 4× SSIM |
|-------|-------------|---------|-------------|---------|
| Trilinear (Baseline) | 32.21 | 0.884 | 30.60 | 0.836 |
| Simple3DCNN | **36.17** | **0.958** | **31.91** | **0.886** |
| SR3DNet (Ours) | 35.64 | 0.952 | 31.90 | 0.885 |

#### Ablation Study

| Configuration | 2× PSNR (dB) | 2× SSIM | 4× PSNR (dB) | 4× SSIM |
|---------------|-------------|---------|-------------|---------|
| SR3DNet (Full) | 35.64 | 0.952 | 31.90 | 0.885 |
| w/o MDDTA | 35.60 | 0.952 | 31.89 | 0.884 |
| w/o GDDFN | 35.98 | 0.956 | 31.91 | 0.886 |
| w/o Both | 36.02 | 0.957 | 31.90 | 0.885 |
| 4 RRAF Blocks | 35.58 | 0.952 | 31.88 | 0.884 |
| 12 RRAF Blocks | 35.75 | 0.954 | 31.89 | 0.885 |

### Model Architecture

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

### Configuration

Key parameters in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 64 | Training patch size |
| `stride` | 32 | Patch extraction stride |
| `batch_size` | 2 | Batch size (for 8GB GPU) |
| `epochs` | 100 | Training epochs |
| `lr` | 0.0002 | Learning rate |
| `num_blocks` | 8 | Number of RRAF blocks |

---

## 中文

基于 PyTorch 实现的 SR3DNet 三维 MRI 超分辨率重建网络，采用 RRAF（残差中的残差注意力特征）模块，集成 MDDTA（多维双路径转置注意力）和 GDDFN（门控双路径深度前馈网络）。

### 项目结构

```
yolo_sxc/
├── configs/
│   └── default.yaml          # 训练配置文件
├── data/
│   ├── datasets/
│   │   ├── kirby21_dataset.py    # Kirby21 数据集加载器
│   │   └── transforms.py         # 数据增强
│   └── scripts/
│       └── prepare_kirby21.py    # 数据准备脚本
├── models/
│   ├── blocks/
│   │   ├── rraf.py           # RRAF 模块（含 MDDTA 和 GDDFN）
│   │   ├── mddta.py          # 多维双路径转置注意力
│   │   └── gddfn.py          # 门控双路径深度前馈网络
│   ├── utils/
│   │   └── pixelshuffle3d.py # 3D 像素重排上采样
│   ├── sr3d_net.py           # SR3DNet 主网络
│   └── simple3d_cnn.py       # Simple3DCNN 基线模型
├── engine/
│   ├── trainer.py            # 训练循环
│   └── tester.py             # 测试与评估
├── losses/
│   └── composite.py          # L1 + MSE 复合损失
├── metrics/
│   ├── psnr3d.py             # 3D PSNR 指标
│   └── ssim3d.py             # 3D SSIM 指标
├── scripts/
│   ├── run_experiments.sh    # 批量实验脚本
│   └── generate_results.py   # 结果汇总
├── visualization/
│   └── visualize.py          # SR 结果可视化
└── main.py                   # 主入口
```

### 环境要求

- Python >= 3.8
- PyTorch >= 1.8.0
- CUDA >= 10.2（GPU 训练）

```bash
pip install -r requirements.txt
```

### 数据集

本项目使用 [Kirby21 数据集](https://www.nitrc.org/projects/multimodal)，包含 42 个 T1 加权 MPRAGE 脑部 MRI 体数据（21 名受试者 × 2 次扫描）。

**数据准备：**
```bash
python data/scripts/prepare_kirby21.py --output_dir ./data/kirby21
```

数据划分比例：**训练:验证:测试 = 7:2:1**（按受试者划分，避免数据泄露）

### 使用方法

**训练：**
```bash
# 训练 SR3DNet 2倍超分
python main.py --mode train --model sr3dnet --scale 2 --exp-name sr3dnet_2x

# 训练 4倍超分
python main.py --mode train --model sr3dnet --scale 4 --exp-name sr3dnet_4x
```

**测试：**
```bash
python main.py --mode test --model sr3dnet --scale 2 --exp-name sr3dnet_2x
```

**消融实验：**
```bash
# 移除 MDDTA 注意力
python main.py --mode train --model sr3dnet --scale 2 --no-mddta --exp-name sr3dnet_no_mddta

# 移除 GDDFN
python main.py --mode train --model sr3dnet --scale 2 --no-gddfn --exp-name sr3dnet_no_gddfn

# 不同 RRAF 块数量
python main.py --mode train --model sr3dnet --scale 2 --num-blocks 4 --exp-name sr3dnet_n4
```

### 实验结果

#### 对比实验

| 模型 | 2× PSNR (dB) | 2× SSIM | 4× PSNR (dB) | 4× SSIM |
|------|-------------|---------|-------------|---------|
| 三线性插值（基线） | 32.21 | 0.884 | 30.60 | 0.836 |
| Simple3DCNN | **36.17** | **0.958** | **31.91** | **0.886** |
| SR3DNet（本文） | 35.64 | 0.952 | 31.90 | 0.885 |

#### 消融实验

| 配置 | 2× PSNR (dB) | 2× SSIM | 4× PSNR (dB) | 4× SSIM |
|------|-------------|---------|-------------|---------|
| SR3DNet（完整） | 35.64 | 0.952 | 31.90 | 0.885 |
| 无 MDDTA | 35.60 | 0.952 | 31.89 | 0.884 |
| 无 GDDFN | 35.98 | 0.956 | 31.91 | 0.886 |
| 无 MDDTA+GDDFN | 36.02 | 0.957 | 31.90 | 0.885 |
| 4 个 RRAF 块 | 35.58 | 0.952 | 31.88 | 0.884 |
| 12 个 RRAF 块 | 35.75 | 0.954 | 31.89 | 0.885 |

### 模型架构

```
SR3DNet
├── 浅层特征提取 (Conv3D)
├── 深层特征提取
│   └── N × RRAF 模块
│       ├── MDDTA（多维双路径转置注意力）
│       └── GDDFN（门控双路径深度前馈网络）
├── 特征融合 (Conv3D)
└── 上采样 (PixelShuffle3D)
```

### 配置参数

`configs/default.yaml` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `patch_size` | 64 | 训练块大小 |
| `stride` | 32 | 块提取步长 |
| `batch_size` | 2 | 批大小（适配 8GB 显存） |
| `epochs` | 100 | 训练轮数 |
| `lr` | 0.0002 | 学习率 |
| `num_blocks` | 8 | RRAF 块数量 |

---

## License

This project is for academic research purposes.

本项目仅供学术研究使用。

## Acknowledgments | 致谢

- Kirby21 dataset: [NITRC](https://www.nitrc.org/projects/multimodal)
- Auckland University of Technology (AUT) | 奥克兰理工大学
