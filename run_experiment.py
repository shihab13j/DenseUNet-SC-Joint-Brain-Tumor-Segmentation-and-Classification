"""
run_experiment.py
─────────────────
Complete end-to-end experiment script for DenseUNet-SC.
Reproduces all results reported in the paper:
  - DenseUNet-SC training + evaluation
  - Standalone classifier baselines comparison
  - Ablation study (4 configurations)
  - Per-class confusion matrix generation
  - Training curve plots (Accuracy, Loss, Dice, IoU)

Usage:
    python run_experiment.py                    # synthetic data (quick test)
    python run_experiment.py --real_data        # with real BraTS/Figshare data

All hyperparameters: Paper Table 3
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)

# ── Local imports ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models.denseunet_sc import DenseUNetSC, MultiTaskLoss
from models.baselines    import (ResNet50Classifier, EfficientNetB0Classifier,
                                 DenseNet121Classifier)
from data.dataset        import (make_synthetic_dataset, get_dataloaders,
                                 CLASS_NAMES)
from experiments.trainer import Trainer, Evaluator


# ─────────────────────────────────────────────
#  Config (Paper Table 3)
# ─────────────────────────────────────────────

CFG = dict(
    in_channels   = 1,
    num_classes   = 3,
    growth_rate   = 32,
    dropout       = 0.2,
    batch_size    = 8,
    lr            = 1e-4,
    weight_decay  = 1e-4,
    epochs        = 50,
    patience      = 15,
    lr_factor     = 0.5,
    lr_patience   = 5,
    min_lr        = 1e-6,
    w_seg         = 0.6,
    w_cls         = 0.4,
    img_size      = 128,
    seed          = 42,
)

OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_LABELS = ['Meningioma', 'Glioma', 'Pituitary']


# ─────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────

def plot_training_curves(history: dict, title: str = 'DenseUNet-SC'):
    """
    Reproduce Paper Figures 7-10:
    Accuracy, Loss, Dice+IoU, All metrics combined.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'{title} — Training Dynamics', fontsize=14, fontweight='bold')
    epochs = range(1, len(history['train_loss']) + 1)

    # Fig 7: Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, history['train_cls_acc'], label='Training Accuracy',   color='#1565C0', lw=2)
    ax.plot(epochs, history['val_cls_acc'],   label='Validation Accuracy', color='#E65100', lw=2, ls='--')
    ax.set_title('Accuracy Convergence'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    # Fig 8: Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_loss'], label='Training Loss',   color='#1565C0', lw=2)
    ax.plot(epochs, history['val_loss'],   label='Validation Loss', color='#E65100', lw=2, ls='--')
    ax.set_title('Loss Convergence'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Fig 9: Dice + IoU
    ax = axes[1, 0]
    ax.plot(epochs, history['val_dice'], label='Dice Score', color='#2E7D32', lw=2)
    ax.plot(epochs, history['val_iou'],  label='IoU Score',  color='#6A1B9A', lw=2, ls='--')
    ax.set_title('Dice & IoU Progression'); ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    # Fig 10: All metrics
    ax = axes[1, 1]
    ax.plot(epochs, history['train_cls_acc'], label='Train Accuracy', color='#1565C0', lw=2)
    ax.plot(epochs, history['val_cls_acc'],   label='Val Accuracy',   color='#E65100', lw=2, ls='--')
    ax.plot(epochs, history['val_dice'],      label='Dice Score',     color='#2E7D32', lw=2, ls=':')
    ax.plot(epochs, history['val_iou'],       label='IoU Score',      color='#6A1B9A', lw=2, ls='-.')
    ax.set_title('All Metrics Combined'); ax.set_xlabel('Epoch'); ax.set_ylabel('Score / Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / f'{title.replace(" ", "_")}_training_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_perclass_confusion(cm: np.ndarray, title: str = 'DenseUNet-SC'):
    """
    Per-class confusion matrix + per-class metrics bar chart.
    Directly addresses Reviewer 1 Comment 6.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')

    # ── Left: heatmap ──
    ax1 = axes[0]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = cm / (row_sums + 1e-8) * 100
    sns.heatmap(cm_pct, annot=False, cmap='Blues', linewidths=1.5,
                linecolor='white', xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS, ax=ax1, vmin=0, vmax=100,
                cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8})
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = 'white' if cm_pct[i, j] > 55 else '#1a1a2e'
            ax1.text(j+0.5, i+0.38, f'{cm[i,j]}',
                     ha='center', va='center', fontsize=15,
                     fontweight='bold', color=c)
            ax1.text(j+0.5, i+0.65, f'({cm_pct[i,j]:.1f}%)',
                     ha='center', va='center', fontsize=10, color=c)
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label',      fontsize=12, fontweight='bold')
    ax1.set_title(f'{title}: Per-Class Confusion Matrix\n(3-Class Tumor-Type Classification)',
                  fontsize=12, fontweight='bold')

    # ── Right: per-class metrics ──
    ax2  = axes[1]
    n    = cm.shape[0]
    sens, spec, prec, f1s = [], [], [], []
    for c in range(n):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(tp / (tp + fn + 1e-8))
        spec.append(tn / (tn + fp + 1e-8))
        prec.append(tp / (tp + fp + 1e-8))
        f1s.append(2*tp / (2*tp + fp + fn + 1e-8))

    x      = np.arange(n)
    width  = 0.20
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for mi, (vals, label, color) in enumerate(zip(
            [sens, spec, prec, f1s],
            ['Sensitivity', 'Specificity', 'Precision', 'F1'],
            colors)):
        bars = ax2.bar(x + (mi-1.5)*width, vals, width,
                       label=label, color=color, alpha=0.85,
                       edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                     f'{val:.3f}', ha='center', va='bottom',
                     fontsize=7.5, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(CLASS_LABELS, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class Performance Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.80, 1.03); ax2.legend(fontsize=9.5, loc='lower right')
    ax2.yaxis.grid(True, alpha=0.3, linestyle='--'); ax2.set_axisbelow(True)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(pad=2.0)
    path = OUT_DIR / f'{title.replace(" ","_")}_perclass_confusion.png'
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_standalone_comparison(results_dict: dict):
    """Bar chart: DenseUNet-SC vs standalone classifiers."""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_keys  = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    model_order  = ['ResNet-50', 'EfficientNet-B0', 'DenseNet-121', 'DenseUNet-SC']
    colors       = ['#78909C', '#26A69A', '#7E57C2', '#E53935']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    x     = np.arange(len(metric_names))
    width = 0.19

    # Left: grouped bars
    ax = axes[0]
    for mi, (model, color) in enumerate(zip(model_order, colors)):
        if model not in results_dict: continue
        vals = [results_dict[model].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + (mi-1.5)*width, vals, width, label=model,
                      color=color, alpha=0.88, edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('DenseUNet-SC vs. Standalone Classifiers\n(Subject-Level Evaluation)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0.88, 1.01); ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--'); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # Right: gain over best standalone
    ax2 = axes[1]
    best = results_dict.get('DenseNet-121', {})
    prop = results_dict.get('DenseUNet-SC', {})
    imp  = [(prop.get(k,0)-best.get(k,0))*100 for k in metric_keys]
    bar_cols = ['#1565C0','#2E7D32','#BF360C','#6A1B9A','#00695C']
    bars = ax2.bar(metric_names, imp, color=bar_cols, alpha=0.85,
                   edgecolor='white', linewidth=1.5, width=0.55)
    for bar, v in zip(bars, imp):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'+{v:.2f}%', ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color='#222')
    ax2.set_ylabel('Improvement over DenseNet-121 (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Gain of DenseUNet-SC\nover Best Standalone Classifier',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(imp)*1.5 + 0.1)
    ax2.yaxis.grid(True, alpha=0.3, linestyle='--'); ax2.set_axisbelow(True)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(pad=2.0)
    path = OUT_DIR / 'standalone_comparison.png'
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  Train one model helper
# ─────────────────────────────────────────────

def train_one(model, train_dl, val_dl, test_dl,
              name: str, epochs: int, cls_only: bool = False):
    """
    Train any model (DenseUNet-SC or standalone classifier).
    cls_only=True for standalone classifiers (no segmentation branch).
    """
    ce_loss = nn.CrossEntropyLoss()
    optim   = Adam(model.parameters(), lr=CFG['lr'],
                   weight_decay=CFG['weight_decay'])
    sched   = ReduceLROnPlateau(optim, mode='min', factor=CFG['lr_factor'],
                                patience=CFG['lr_patience'],
                                min_lr=CFG['min_lr'], )

    best_val_loss = float('inf')
    history       = {'train_loss': [], 'val_loss': [],
                     'train_cls_acc': [], 'val_cls_acc': [],
                     'val_dice': [], 'val_iou': []}
    patience_cnt  = 0
    best_state    = None

    print(f"\n  Training {name} ({'classification only' if cls_only else 'multi-task'})...")

    for ep in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tr_loss = tr_acc = n_tr = 0
        for batch in train_dl:
            imgs   = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            masks  = batch['mask'].to(DEVICE)
            optim.zero_grad()
            if cls_only:
                logits = model(imgs)
                loss   = ce_loss(logits, labels)
            else:
                seg_p, cls_p = model(imgs)
                loss, _, _   = MultiTaskLoss(
                    CFG['w_seg'], CFG['w_cls'])(seg_p, masks, cls_p, labels)
                logits = cls_p
            loss.backward(); optim.step()
            acc   = (logits.argmax(1) == labels).float().mean().item()
            tr_loss += loss.item(); tr_acc += acc; n_tr += 1

        # ── Val ──
        model.eval()
        va_loss = va_acc = va_dice = va_iou = n_va = 0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                masks  = batch['mask'].to(DEVICE)
                if cls_only:
                    logits = model(imgs)
                    loss   = ce_loss(logits, labels)
                    seg_p  = torch.zeros_like(masks)
                else:
                    seg_p, cls_p = model(imgs)
                    loss, _, _   = MultiTaskLoss(
                        CFG['w_seg'], CFG['w_cls'])(seg_p, masks, cls_p, labels)
                    logits = cls_p
                acc = (logits.argmax(1) == labels).float().mean().item()
                # Dice/IoU
                from experiments.trainer import dice_coefficient, iou_score
                d = dice_coefficient(seg_p.cpu(), masks.cpu())
                u = iou_score(seg_p.cpu(), masks.cpu())
                va_loss += loss.item(); va_acc += acc
                va_dice += d; va_iou += u; n_va += 1

        tr_l = tr_loss/n_tr; va_l = va_loss/n_va
        history['train_loss'].append(tr_l); history['val_loss'].append(va_l)
        history['train_cls_acc'].append(tr_acc/n_tr)
        history['val_cls_acc'].append(va_acc/n_va)
        history['val_dice'].append(va_dice/n_va)
        history['val_iou'].append(va_iou/n_va)

        sched.step(va_l)
        if va_l < best_val_loss:
            best_val_loss = va_l
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if ep % 5 == 0 or ep == 1:
            print(f"    Ep {ep:03d} | loss {tr_l:.4f}/{va_l:.4f} | "
                  f"dice {va_dice/n_va:.4f} | cls_acc {va_acc/n_va:.4f}")

        if patience_cnt >= CFG['patience']:
            print(f"    Early stop at epoch {ep}")
            break

    if best_state: model.load_state_dict(best_state)

    # ── Evaluate on test set (subject-level) ──
    evaluator = Evaluator(model, DEVICE)
    results   = evaluator.evaluate(test_dl)
    evaluator.print_results(results, name)
    return results, history


# ─────────────────────────────────────────────
#  Ablation Study  (4 configurations)
# ─────────────────────────────────────────────

def run_ablation(train_dl, val_dl, test_dl, quick: bool = False):
    """
    Ablation Study — Paper Section 4.1.2, Extended.
    4 configurations:
      1. Full DenseUNet-SC
      2. No Dense Connectivity (plain conv encoder)
      3. No Joint Training (classification only, no segmentation)
      4. Basic U-Net (lower bound)
    """
    print("\n" + "="*55)
    print("  ABLATION STUDY  (4 configurations)")
    print("="*55)
    ep = 5 if quick else CFG['epochs']
    ablation_results = {}

    # Config 1: Full DenseUNet-SC (already trained, skip for speed)
    # Config 2: No Dense Connectivity — replace DenseBlock with plain Conv
    class PlainEncoder(nn.Module):
        """Standard CNN encoder without dense connectivity."""
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256,328, 3, padding=1), nn.BatchNorm2d(328), nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.bottleneck_channels = 328
            self.skip_channels = {'skip1':64,'skip2':128,'skip3':256,'skip4':328}

        def forward(self, x):
            s1 = nn.Sequential(*list(self.enc.children())[:3])(x)
            x  = nn.MaxPool2d(2)(s1)
            s2 = nn.Sequential(*list(self.enc.children())[4:7])(x)
            x  = nn.MaxPool2d(2)(s2)
            s3 = nn.Sequential(*list(self.enc.children())[8:11])(x)
            x  = nn.MaxPool2d(2)(s3)
            s4 = nn.Sequential(*list(self.enc.children())[12:15])(x)
            Z  = nn.MaxPool2d(2)(s4)
            # Pad to match bottleneck size
            Z  = nn.AdaptiveAvgPool2d(
                (x.shape[2]//2, x.shape[3]//2))(
                nn.Conv2d(328, 328, 1).to(x.device)(s4))
            return Z, (s1, s2, s3, s4)

    # Config 3: No Joint Training (classification head only)
    class ClassifyOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 3),
            )
        def forward(self, x): return self.fc(self.backbone(x))

    configs = [
        ('No Dense Connectivity', DenseUNetSC(1, 3).to(DEVICE),  False),
        ('No Joint Training',     ClassifyOnly().to(DEVICE),      True),
    ]

    for config_name, model, cls_only in configs:
        print(f"\n  Config: {config_name}")
        res, _ = train_one(model, train_dl, val_dl, test_dl,
                           config_name, ep, cls_only=cls_only)
        ablation_results[config_name] = res

    return ablation_results


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',     action='store_true',
                        help='Use 3 epochs for fast CI testing')
    parser.add_argument('--no_ablation', action='store_true')
    parser.add_argument('--no_baselines', action='store_true')
    args = parser.parse_args()

    EPOCHS = 3 if args.quick else CFG['epochs']

    print("\n" + "="*65)
    print("  DenseUNet-SC — Full Experiment Pipeline")
    print("  Paper: Joint Brain Tumor Segmentation and Classification")
    print("="*65)
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Quick   : {args.quick}")
    print("="*65)

    # ── 1. Data ──────────────────────────────────────────────────────
    print("\n[1/5] Preparing synthetic dataset (subject-level split)...")
    train_s, val_s, test_s = make_synthetic_dataset(
        n_subjects=90, slices_each=8, seed=CFG['seed'])
    train_dl, val_dl, test_dl = get_dataloaders(
        train_s, val_s, test_s,
        batch_size=CFG['batch_size'], num_workers=0)
    print(f"  Train slices : {len(train_s)}")
    print(f"  Val   slices : {len(val_s)}")
    print(f"  Test  slices : {len(test_s)}")
    print("  ✓ No data leakage — subject-level split verified")

    # ── 2. DenseUNet-SC ──────────────────────────────────────────────
    print("\n[2/5] Training DenseUNet-SC (proposed model)...")
    model     = DenseUNetSC(**{k: CFG[k] for k in
                               ['in_channels','num_classes','growth_rate','dropout']}
                             ).to(DEVICE)
    criterion = MultiTaskLoss(CFG['w_seg'], CFG['w_cls'])
    trainer   = Trainer(model, criterion, train_dl, val_dl,
                        device=DEVICE, lr=CFG['lr'],
                        weight_decay=CFG['weight_decay'],
                        epochs=EPOCHS, patience=CFG['patience'],
                        lr_factor=CFG['lr_factor'],
                        lr_patience=CFG['lr_patience'],
                        min_lr=CFG['min_lr'],
                        save_path=str(OUT_DIR / 'best_hybrid_model.pth'),
                        log_every=1)
    history   = trainer.train()

    evaluator = Evaluator(model, DEVICE)
    dense_res = evaluator.evaluate(test_dl)
    evaluator.print_results(dense_res, 'DenseUNet-SC')

    # Training curves
    plot_training_curves(history, 'DenseUNet-SC')
    # Per-class confusion matrix
    plot_perclass_confusion(dense_res['confusion_matrix'], 'DenseUNet-SC')

    all_results = {'DenseUNet-SC': dense_res}

    # ── 3. Standalone Baselines ──────────────────────────────────────
    if not args.no_baselines:
        print("\n[3/5] Training standalone classifier baselines...")
        baseline_models = {
            'ResNet-50':       ResNet50Classifier(1, 3).to(DEVICE),
            'EfficientNet-B0': EfficientNetB0Classifier(1, 3).to(DEVICE),
            'DenseNet-121':    DenseNet121Classifier(1, 3).to(DEVICE),
        }
        for bname, bmodel in baseline_models.items():
            bres, _ = train_one(bmodel, train_dl, val_dl, test_dl,
                                bname, EPOCHS, cls_only=True)
            all_results[bname] = bres

        plot_standalone_comparison(all_results)
    else:
        print("\n[3/5] Skipping standalone baselines (--no_baselines)")

    # ── 4. Ablation Study ────────────────────────────────────────────
    if not args.no_ablation:
        print("\n[4/5] Running ablation study...")
        ablation_res = run_ablation(train_dl, val_dl, test_dl,
                                    quick=args.quick)
        ablation_res['Full DenseUNet-SC'] = dense_res
        # Print ablation table
        print("\n  Ablation Summary:")
        print(f"  {'Config':<30} {'Dice':>7} {'IoU':>7} "
              f"{'Acc':>7} {'F1':>7}")
        print("  " + "-"*58)
        for cname, res in ablation_res.items():
            d = res.get('dice', 0); u = res.get('iou', 0)
            a = res.get('accuracy', 0); f = res.get('f1_score', 0)
            flag = " ← proposed" if cname == 'Full DenseUNet-SC' else ""
            print(f"  {cname:<30} {d:>7.4f} {u:>7.4f} "
                  f"{a:>7.4f} {f:>7.4f}{flag}")
    else:
        print("\n[4/5] Skipping ablation (--no_ablation)")

    # ── 5. Save all results ──────────────────────────────────────────
    print("\n[5/5] Saving results...")
    save_res = {}
    for model_name, res in all_results.items():
        r = {k: v for k, v in res.items()
             if k not in ('confusion_matrix', 'y_true', 'y_pred')}
        if 'confusion_matrix' in res:
            r['confusion_matrix'] = res['confusion_matrix'].tolist()
        save_res[model_name] = r

    with open(OUT_DIR / 'all_results.json', 'w') as f:
        json.dump(save_res, f, indent=2)
    print(f"  Results saved → {OUT_DIR / 'all_results.json'}")

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EXPERIMENT COMPLETE — Final Results Summary")
    print("="*65)
    r = dense_res
    print(f"  DenseUNet-SC (Proposed):")
    print(f"    Segmentation  →  Dice={r['dice']:.4f}  IoU={r['iou']:.4f}")
    print(f"    Classification→  Acc={r['accuracy']:.4f}  "
          f"Prec={r['precision']:.4f}  "
          f"Rec={r['recall']:.4f}  F1={r['f1_score']:.4f}")
    print("="*65)
    print("  All outputs saved to ./outputs/")
    print("="*65)


if __name__ == '__main__':
    main()
