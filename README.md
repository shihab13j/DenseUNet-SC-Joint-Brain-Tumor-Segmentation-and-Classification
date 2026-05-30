# DenseUNet-SC: Joint Brain Tumor Segmentation and Classification

**Paper:** *A Densely Connected Deep Learning Method for Joint Brain Tumor Segmentation and Classification*
**Venue:** Healthcare Analytics (Elsevier)

---

## Architecture Overview

```
Input MRI [B, 1, 128, 128]
        │
        ▼
┌─────────────────────────────────────────────────┐
│           DenseNet Encoder (shared)             │
│  InitConv(64) → DB1(L=4) → Tr1 → DB2(L=6) →   │
│  Tr2 → DB3(L=8) → Tr3 → Bottleneck-DB(L=4)     │
│              ↓ Z (bottleneck)                   │
└──────────────┬──────────────────────────────────┘
               │        (parallel, not sequential)
       ┌───────┴────────┐
       ▼                ▼
U-Net Decoder      Classification Head
(segmentation)     GAP → FC(256) → FC(128) → FC(3)
       ↓                ↓
  M̂ ∈ [0,1]      ŷ ∈ ℝ³  (logits)
Binary tumor mask  Tumor-type prediction
```

**Key novelty:** Dense connectivity within the shared encoder enables
encoder-mediated implicit task coupling: the encoder is jointly optimized
under a weighted multi-task loss (w_seg=0.6, w_cls=0.4), learning representations
beneficial for both tumor localization and tumor-type discrimination simultaneously.

---

## Project Structure

```
DenseUNet_SC/
├── models/
│   ├── denseunet_sc.py     ← Main model + MultiTaskLoss
│   └── baselines.py        ← ResNet-50, EfficientNet-B0, DenseNet-121
├── data/
│   └── dataset.py          ← Preprocessing, subject-level split, DataLoaders
├── experiments/
│   └── trainer.py          ← Training engine, Evaluator (subject-level)
├── outputs/                ← Generated figures + results JSON
├── run_experiment.py       ← Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## Hyperparameters (Paper Table 3)

| Parameter | Value |
|-----------|-------|
| Input size | 128×128, grayscale |
| Growth rate g | 32 |
| Dense Block layers | 4, 6, 8, 4 (+ bottleneck 4) |
| Dropout (encoder) | 0.2 |
| Dropout (classifier) | 0.5 / 0.3 |
| Transition reduction | 0.5 |
| Optimizer | Adam, lr=1e-4 |
| Weight decay | 1e-4 |
| Loss weights | seg=0.6, cls=0.4 |
| Batch size | 8 |
| Epochs | 50 |
| Early stopping | patience=15 |
| LR schedule | ReduceLROnPlateau(factor=0.5, patience=5) |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Run (synthetic data — no dataset required)

```bash
cd DenseUNet_SC
python run_experiment.py --quick
```

## Full Run

```bash
python run_experiment.py
```

Outputs saved to `./outputs/`:
- `best_hybrid_model.pth` — best checkpoint
- `DenseUNet-SC_training_curves.png` — training dynamics (Paper Fig 7-10)
- `DenseUNet-SC_perclass_confusion.png` — per-class confusion matrix
- `standalone_comparison.png` — baseline comparison
- `all_results.json` — all numeric results

---

## Model Verification

```bash
python models/denseunet_sc.py   # architecture sanity check
python models/baselines.py      # baseline models check
python data/dataset.py          # preprocessing + split check
```

---

## Classification Labels (Paper Section 3.2)

| Class | Tumor Type |
|-------|-----------|
| 0 | Meningioma |
| 1 | Glioma |
| 2 | Pituitary tumor |

**Evaluation protocol:**
- Labels defined at **subject level**
- Per-slice logits averaged per subject at test time
- All metrics computed at **subject level** (no data leakage)

---

## Binary Segmentation Design Rationale

Binary (tumor vs. background) rather than multi-class:
1. Avoids severe class imbalance in BraTS subregion annotations
2. Provides adequate spatial grounding for the shared encoder
3. Establishes a controlled multi-task coupling benchmark

Extension to multi-class subregion segmentation is future work.
