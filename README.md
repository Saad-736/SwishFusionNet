# Food Image Recognition with Adaptive Noise Injection — Final-Exam Experiments

**Course:** Topics on Intelligent Systems
**Student:** Saad Muhammad (26160024)
**Instructor:** Prof. Shin, Dong Il
**Lab:** Signal Processing and Digital Communication (SDC) Lab, Sejong University

## Overview

This repository contains the experimental code for the final-exam paper of the
*Topics on Intelligent Systems* course. It implements:

1. **Three external baselines** — re-implementations of the three papers from
   the midterm survey:
   - `alexnet` — Yanai & Kawano, ICMEW 2015
   - `mresnet50` — Abdul Kareem et al., Comp. Biol. Med. 2024
   - `vit` — Ghosh & Sazonov, IEEE EMBC 2025 (Transformer)
2. **The proposed model** — **SwishFusionNet** with **Adaptive Noise Injection (ANI)**,
   which contributes:
   - A hybrid CNN–Transformer specialized for fine-grained food images.
   - **ANI module**: per-channel learnable noise σ, gated by the model's own
     prediction entropy — a self-adaptive regularization curriculum, unlike
     NoisyViT's fixed scalar σ.
3. **A 5-variant ablation study** (V1 → V5) that turns each novel component on
   one at a time so the contribution of each is measurable.

## Models trained (8 total)

| Tag | Description |
|---|---|
| `alexnet` | External baseline — Paper 1 |
| `mresnet50` | External baseline — Paper 2 |
| `vit` | External baseline — Paper 3 |
| `v1_resnet50` | Vanilla ResNet-50 (ReLU, no transformer, no noise) |
| `v2_swish` | + Swish activations |
| `v3_transformer` | + Transformer head over 7×7 spatial tokens |
| `v4_fixed_noise` | + Fixed scalar noise (NoisyViT-style baseline for our framework) |
| **`v5_proposed`** | **+ Adaptive Noise Injection — the proposed novelty** |

## Evaluation metrics (7)

1. Top-1 accuracy
2. Top-5 accuracy
3. Macro F1-score
4. Macro Precision
5. Macro Recall
6. Inference time per image (ms, batch = 1, GPU if available)
7. Trainable parameter count (M)

## Dataset

**Food-101** (Bossard et al., ECCV 2014) — 101 categories, 1,000 images per
class. Auto-downloaded by torchvision. To make experiments tractable on a
single GPU, a configurable subset of 20 classes is used by default. Set
`num_classes=101` in CONFIG for the full dataset.

## Repository layout

```
food-recognition-experiments/
├── README.md
├── requirements.txt
├── smoke_test.py                    # validates pipeline with synthetic data
├── notebooks/
│   └── main_experiment.ipynb        # one-click Colab notebook
├── src/
│   ├── dataset.py                   # Food-101 loader with subset selection
│   ├── models.py                    # all 8 model definitions
│   ├── train.py                     # training loop
│   ├── evaluate.py                  # 7-metric evaluation
│   └── plot_results.py              # 10 figures + 2 CSV tables
└── outputs/                         # created at runtime
    ├── logs/                        # per-epoch JSON logs + eval summaries
    ├── checkpoints/                 # best model weights
    └── figures/                     # PNG + PDF of all paper figures
```

## Quick start (Colab)

1. Open Colab → set Runtime → GPU (T4).
2. Upload `food-recognition-experiments.zip`, then run:
   ```python
   !unzip -q food-recognition-experiments.zip
   %cd food-recognition-experiments
   ```
3. Open `notebooks/main_experiment.ipynb` and run all cells.

Total wall-clock on a T4 GPU with 20-class subset, 15 epochs each: **≈ 4–5 hours**.

## Quick start (local with VS Code)

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python smoke_test.py                                # 2-min pipeline check

# Train all 8 models
for m in alexnet mresnet50 vit v1_resnet50 v2_swish v3_transformer v4_fixed_noise v5_proposed; do
  python src/train.py --model $m --epochs 15
done

# Evaluate all 8 models
for m in alexnet mresnet50 vit v1_resnet50 v2_swish v3_transformer v4_fixed_noise v5_proposed; do
  python src/evaluate.py --model $m
done

# Generate all figures + CSVs
python src/plot_results.py
```

## Reproducibility

- Random seed: `42` for all data splits, model init, and dataloaders.
- Hyperparameters logged per run in `outputs/logs/<model>_config.json`.
- Pretrained weights from torchvision / timm (ImageNet-1K).

## License

MIT.
