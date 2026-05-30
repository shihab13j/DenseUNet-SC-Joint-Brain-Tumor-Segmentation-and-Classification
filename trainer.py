"""
Training Engine for DenseUNet-SC
All hyperparameters exactly from Paper Table 3:
  - Optimizer:    Adam, lr=1e-4
  - Epochs:       50, batch_size=8
  - Loss weights: seg=0.6, cls=0.4
  - Early stop:   patience=15, monitor=val_loss
  - LR schedule:  ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
  - Checkpoint:   save best val_loss → best_hybrid_model.pth
"""

import os
import time
import copy
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred_bin = (pred > threshold).float()
    inter    = (pred_bin * target).sum()
    return ((2 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred_bin = (pred > threshold).float()
    inter    = (pred_bin * target).sum()
    union    = pred_bin.sum() + target.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def compute_seg_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict:
    dice = dice_coefficient(pred, target)
    iou  = iou_score(pred, target)
    # Pixel-level accuracy
    pred_bin = (pred > 0.5).float()
    acc = (pred_bin == target).float().mean().item()
    return {'dice': dice, 'iou': iou, 'seg_acc': acc}


def compute_cls_metrics(logits: torch.Tensor,
                        targets: torch.Tensor) -> Dict:
    preds = logits.argmax(dim=1)
    acc   = (preds == targets).float().mean().item()
    return {'cls_acc': acc}


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    End-to-end training loop for DenseUNet-SC.
    Implements Paper Table 2 Steps 11-12 and Table 3 settings.
    """

    def __init__(
        self,
        model:          nn.Module,
        criterion:      nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        device:         str   = 'cpu',
        lr:             float = 1e-4,       # Paper Table 3
        weight_decay:   float = 1e-4,       # Paper Table 3
        epochs:         int   = 50,         # Paper Table 3
        patience:       int   = 15,         # Paper Table 3 (early stopping)
        lr_factor:      float = 0.5,        # Paper Table 3 (ReduceLROnPlateau)
        lr_patience:    int   = 5,          # Paper Table 3
        min_lr:         float = 1e-6,       # Paper Table 3
        save_path:      str   = 'best_hybrid_model.pth',  # Paper Table 3
        log_every:      int   = 1,
    ):
        self.model        = model.to(device)
        self.criterion    = criterion
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.patience     = patience
        self.save_path    = save_path
        self.log_every    = log_every

        # Optimizer: Adam (Paper Table 3)
        self.optimizer = Adam(model.parameters(),
                              lr=lr, weight_decay=weight_decay)

        # LR Scheduler: ReduceLROnPlateau (Paper Table 3)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=lr_factor,
            patience=lr_patience, min_lr=min_lr, )

        self.history = defaultdict(list)

    # ── Single epoch ────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict:
        self.model.train(train)
        totals = defaultdict(float)
        n_batches = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                imgs    = batch['image'].to(self.device)   # [B,1,128,128]
                masks   = batch['mask'].to(self.device)    # [B,1,128,128]
                labels  = batch['label'].to(self.device)   # [B]

                seg_pred, cls_pred = self.model(imgs)

                loss, l_seg, l_cls = self.criterion(
                    seg_pred, masks, cls_pred, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Metrics
                seg_m = compute_seg_metrics(
                    seg_pred.detach().cpu(), masks.cpu())
                cls_m = compute_cls_metrics(
                    cls_pred.detach().cpu(), labels.cpu())

                totals['loss']    += loss.item()
                totals['l_seg']   += l_seg.item()
                totals['l_cls']   += l_cls.item()
                totals['dice']    += seg_m['dice']
                totals['iou']     += seg_m['iou']
                totals['seg_acc'] += seg_m['seg_acc']
                totals['cls_acc'] += cls_m['cls_acc']
                n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    # ── Full training loop ───────────────────────────────────────────

    def train(self) -> Dict:
        """
        Train for up to self.epochs with early stopping.
        Returns training history dict.
        """
        best_val_loss  = float('inf')
        best_weights   = None
        patience_count = 0
        start_time     = time.time()

        print("=" * 65)
        print("  DenseUNet-SC Training — Paper Table 3 Hyperparameters")
        print("=" * 65)
        print(f"  Device      : {self.device}")
        print(f"  Epochs      : {self.epochs}")
        print(f"  LR          : {self.optimizer.param_groups[0]['lr']:.1e}")
        print(f"  Patience    : {self.patience}")
        print(f"  Checkpoint  : {self.save_path}")
        print("=" * 65)

        for epoch in range(1, self.epochs + 1):
            # Training pass
            tr = self._run_epoch(self.train_loader, train=True)
            # Validation pass
            va = self._run_epoch(self.val_loader,   train=False)

            # LR scheduling on val_loss (Paper Table 3)
            self.scheduler.step(va['loss'])
            cur_lr = self.optimizer.param_groups[0]['lr']

            # Log history
            for k, v in tr.items():
                self.history[f'train_{k}'].append(v)
            for k, v in va.items():
                self.history[f'val_{k}'].append(v)
            self.history['lr'].append(cur_lr)

            # Model checkpointing (Paper Table 3: save best val_loss)
            if va['loss'] < best_val_loss:
                best_val_loss  = va['loss']
                best_weights   = copy.deepcopy(self.model.state_dict())
                patience_count = 0
                torch.save({
                    'epoch':       epoch,
                    'model_state': best_weights,
                    'val_loss':    best_val_loss,
                    'history':     dict(self.history),
                }, self.save_path)
                ckpt_flag = ' ✓ saved'
            else:
                patience_count += 1
                ckpt_flag = ''

            if epoch % self.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Ep {epoch:03d}/{self.epochs} | "
                    f"Loss {tr['loss']:.4f}/{va['loss']:.4f} | "
                    f"Dice {va['dice']:.4f} | IoU {va['iou']:.4f} | "
                    f"ClsAcc {va['cls_acc']:.4f} | "
                    f"LR {cur_lr:.1e} | {elapsed:.0f}s{ckpt_flag}"
                )

            # Early stopping (Paper Table 3: patience=15)
            if patience_count >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs).")
                break

        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        elapsed = time.time() - start_time
        print(f"\n  Training complete in {elapsed:.1f}s")
        print(f"  Best val loss : {best_val_loss:.6f}")
        return dict(self.history)


# ─────────────────────────────────────────────
#  Subject-Level Evaluator
# ─────────────────────────────────────────────

class Evaluator:
    """
    Evaluate DenseUNet-SC on test set with subject-level aggregation.

    Paper Section 3.2 (Classification task definition):
    "Per-slice logits from all slices belonging to the same test subject
    are averaged (majority vote over softmax outputs), and a single
    subject-level prediction is reported."

    Returns per-slice segmentation metrics + subject-level classification metrics.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model  = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()

        # Collect per-slice results
        seg_dices, seg_ious = [], []
        subject_logits: Dict[str, List[torch.Tensor]] = defaultdict(list)
        subject_labels: Dict[str, int] = {}

        for batch in loader:
            imgs      = batch['image'].to(self.device)
            masks     = batch['mask'].to(self.device)
            labels    = batch['label']
            sub_ids   = batch['subject_id']

            seg_pred, cls_pred = self.model(imgs)
            cls_probs = torch.softmax(cls_pred.cpu(), dim=1)

            # Per-slice segmentation metrics
            for i in range(imgs.size(0)):
                seg_dices.append(
                    dice_coefficient(seg_pred[i].cpu(), masks[i].cpu()))
                seg_ious.append(
                    iou_score(seg_pred[i].cpu(), masks[i].cpu()))

            # Aggregate logits per subject (for subject-level cls)
            for i, sid in enumerate(sub_ids):
                subject_logits[sid].append(cls_probs[i])
                subject_labels[sid] = labels[i].item()

        # Subject-level classification (Paper: majority vote)
        y_true, y_pred = [], []
        for sid, logit_list in subject_logits.items():
            avg_prob  = torch.stack(logit_list).mean(dim=0)
            pred_cls  = avg_prob.argmax().item()
            true_cls  = subject_labels[sid]
            y_true.append(true_cls)
            y_pred.append(pred_cls)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Classification metrics
        accuracy = (y_true == y_pred).mean()

        from sklearn.metrics import (precision_score, recall_score,
                                     f1_score, confusion_matrix)

        precision = precision_score(y_true, y_pred, average='macro',
                                    zero_division=0)
        recall    = recall_score(y_true, y_pred,    average='macro',
                                 zero_division=0)
        f1        = f1_score(y_true, y_pred,        average='macro',
                             zero_division=0)
        cm        = confusion_matrix(y_true, y_pred)

        # Specificity per class (from confusion matrix)
        specificities = []
        n_cls = cm.shape[0]
        for c in range(n_cls):
            tn = cm.sum() - (cm[c, :].sum() + cm[:, c].sum() - cm[c, c])
            fp = cm[:, c].sum() - cm[c, c]
            specificities.append(tn / (tn + fp + 1e-8))
        specificity = np.mean(specificities)

        return {
            # Segmentation (per-slice)
            'dice':         np.mean(seg_dices),
            'iou':          np.mean(seg_ious),
            # Classification (subject-level)
            'accuracy':     float(accuracy),
            'precision':    float(precision),
            'recall':       float(recall),
            'f1_score':     float(f1),
            'specificity':  float(specificity),
            'confusion_matrix': cm,
            'y_true':       y_true,
            'y_pred':       y_pred,
            'n_subjects':   len(y_true),
        }

    def print_results(self, results: Dict, model_name: str = 'DenseUNet-SC'):
        print(f"\n{'=' * 55}")
        print(f"  Test Results — {model_name}")
        print(f"{'=' * 55}")
        print(f"  Segmentation (per-slice):")
        print(f"    Dice Score : {results['dice']:.4f}")
        print(f"    IoU Score  : {results['iou']:.4f}")
        print(f"  Classification (subject-level, n={results['n_subjects']}):")
        print(f"    Accuracy   : {results['accuracy']:.4f}")
        print(f"    Precision  : {results['precision']:.4f}")
        print(f"    Recall     : {results['recall']:.4f}")
        print(f"    F1-Score   : {results['f1_score']:.4f}")
        print(f"    Specificity: {results['specificity']:.4f}")
        print(f"{'=' * 55}")
