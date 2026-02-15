# Stress-Testing CNNs on CIFAR-100

ResNet-18 trained from scratch on CIFAR-100 with failure analysis, Grad-CAM explainability, and CutMix augmentation as a constrained improvement.

## Requirements

| Dependency | Version Tested |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ (CUDA) |
| torchvision | 0.15+ |
| pytorch-grad-cam | 1.5+ |
| numpy | 1.24+ |
| matplotlib | 3.7+ |
| tqdm | 4.65+ |

**Install:**
```bash
pip install torch torchvision pytorch-grad-cam numpy matplotlib tqdm
```

## Hardware

- **GPU required.** Tested on NVIDIA L40S (CUDA 12.8). Any GPU with ≥4 GB VRAM works.
- Training takes ~8 min/model (50 epochs) on L40S; ~15 min on a T4.
- CPU-only will work but expect ~2 hours per training run.

## Configuration

| Parameter | Baseline | CutMix |
|---|---|---|
| Epochs | 50 | 50 |
| Optimizer | SGD (momentum=0.9) | SGD (momentum=0.9) |
| Learning Rate | 0.1 → 0.01 → 0.001 | 0.1 → 0.01 → 0.001 |
| LR Schedule | MultiStepLR [25, 40] | MultiStepLR [25, 40] |
| Weight Decay | 5e-4 | 5e-4 |
| Batch Size | 128 | 128 |
| Loss | CrossEntropy | CrossEntropy (smoothing=0.1) |
| Augmentation | RandomCrop + HFlip | RandomCrop + HFlip + CutMix (α=1.0, p=0.5) |
| Seed | 42 | 42 |

**Architecture:** ResNet-18 adapted for 32×32 input — first conv changed to 3×3 (stride 1, pad 1), max-pool replaced with Identity, FC layer outputs 100 classes. No pretrained weights.

## How to Run

1. **Open** `DL_Assignment01_Complete.ipynb` in Jupyter/Colab.
2. **Run all cells sequentially** (top to bottom). The notebook handles everything:
   - CIFAR-100 auto-downloads on first run
   - Checkpoints save automatically as `best_baseline_resnet18.pth` and `best_cutmix_resnet18.pth`
3. **GPU selection** (if multi-GPU): uncomment and edit the `CUDA_VISIBLE_DEVICES` line in cell 1.

> **Note:** The full notebook takes ~20 min end-to-end on a modern GPU. If restarting mid-run, you can skip training cells and load from saved checkpoints — the evaluation cells handle this.

## Notebook Pipeline

```
Setup & Seed (42)
    │
    ├── PART A: Baseline
    │   ├── Train ResNet-18 (50 epochs, standard CE)
    │   ├── Evaluate on test set → 60.69%
    │   ├── Discover top failure cases (ranked by confidence)
    │   ├── Select 5 cases + form hypotheses
    │   └── Grad-CAM on layer4 → validate hypotheses
    │
    ├── PART B: CutMix (Constrained Improvement)
    │   ├── Train fresh ResNet-18 (50 epochs, CutMix + label smoothing)
    │   ├── Evaluate on test set → 75.70%
    │   ├── Re-check original 5 failure cases
    │   └── Grad-CAM comparison on still-failing cases
    │
    └── PART C: Comparative Analysis
        ├── Full test set: corrected / worsened / both-wrong / both-right
        ├── Confidence distribution comparison
        ├── High-confidence error reduction (thresholds 50–99%)
        ├── Per-class accuracy waterfall (100 classes)
        └── Side-by-side Grad-CAM: baseline vs CutMix
```

## Outputs

All plots save as PNGs in the working directory. Key files:

| File | Description |
|---|---|
| `best_baseline_resnet18.pth` | Baseline model checkpoint |
| `best_cutmix_resnet18.pth` | CutMix model checkpoint |
| `baseline_metrics_plot.png` | Training curves (baseline) |
| `cutmix_metrics_plot.png` | Training curves (CutMix) |
| `baseline_gradcam.png` | Grad-CAM heatmaps (baseline) |
| `gradcam_side_by_side.png` | Baseline vs CutMix Grad-CAM |
| `per_class_accuracy_change.png` | Per-class accuracy delta |
| `DL_Assignment01_Report.pdf` | Final 6-page report |

## Reproducibility

Seed 42 is set globally for Python, NumPy, PyTorch, and CUDA. `torch.backends.cudnn.deterministic = True` is enabled. Results should be identical across runs on the same hardware.