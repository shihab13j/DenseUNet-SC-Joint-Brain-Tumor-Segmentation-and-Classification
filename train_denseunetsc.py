"""
================================================================
DenseUNet-SC: A Densely Connected Deep Learning Method for
Joint Brain Tumor Segmentation and Classification

Paper  : "A Densely Connected Deep Learning Method for Joint
          Brain Tumor Segmentation and Classification"
Journal: Healthcare Analytics (Elsevier)
Authors: [Moinul Hossain]
GitHub : https://github.com/shihab13j/DenseUNet-SC-Joint-Brain-Tumor-Segmentation-and-Classification

================================================================
Model Summary (Paper Table 3):
  Input        : 128×128 grayscale MRI slices
  Encoder      : DenseNet — 4 Dense Blocks (L=4,6,8,4, g=32)
                            + Bottleneck Block (L=4, g=32)
                            Dropout=0.2, Transition reduction=0.5
  Decoder      : U-Net   — 4 stages (256→128→64→32 channels)
                            with skip connections
  Cls Head     : GAP → FC(256,ReLU)+Drop(0.5)
                      → FC(128,ReLU)+Drop(0.3)
                      → FC(3)
  Loss         : 0.6 × (BCE + Dice)  +  0.4 × CrossEntropy
  Optimizer    : Adam, lr=1e-4, weight_decay=1e-4
  Epochs       : 50, batch_size=8
  Early stop   : patience=15, monitor=val_loss
  LR schedule  : ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)

Dataset (Paper Section 3.1):
  Figshare Brain Tumor Dataset — 3-class classification
    Class 0: Meningioma
    Class 1: Glioma
    Class 2: Pituitary tumor
  BraTS 2021 — binary segmentation
  Split: 70% train / 15% val / 15% test — subject-level

Results (Paper Table 4, Table 5):
  Segmentation  : Dice=0.9987, IoU=0.9915
  Classification: Accuracy=0.9840, Precision=0.9865,
                  Recall=0.9733, F1=0.9799, Specificity=0.9912
================================================================

Usage:
  python denseunet_sc.py --dataset figshare --data_root data/figshare
  python denseunet_sc.py --dataset brats   --data_root data/brats2021

Dataset download:
  Figshare : https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
  BraTS2021: https://www.synapse.org/Synapse:syn25829067
================================================================
"""

import os
import sys
import json
import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix)

try:
    import scipy.io as sio
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import nibabel as nib
    NIBABEL_OK = True
except ImportError:
    NIBABEL_OK = False


# ================================================================
#  HYPERPARAMETERS  (Paper Table 3)
# ================================================================

CFG = {
    'in_channels':  1,       # grayscale MRI
    'num_classes':  3,       # meningioma / glioma / pituitary
    'growth_rate':  32,      # g in paper
    'dropout':      0.2,     # encoder dropout
    'batch_size':   8,
    'lr':           1e-4,
    'weight_decay': 1e-4,
    'epochs':       50,
    'patience':     15,      # early stopping
    'lr_factor':    0.5,     # ReduceLROnPlateau
    'lr_patience':  5,
    'min_lr':       1e-6,
    'w_seg':        0.6,     # segmentation loss weight
    'w_cls':        0.4,     # classification loss weight
    'img_size':     128,
    'seed':         42,
}

CLASS_NAMES = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}
OUT_DIR     = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)


# ================================================================
#  MODEL ARCHITECTURE
# ================================================================

class BottleneckDenseLayer(nn.Module):
    """
    Single dense layer — Paper Eq. 6, 7:
      BN → ReLU → Conv1×1(4g) → BN → ReLU → Conv3×3(g) → Dropout
      x_l = concat(x_0, x_1, ..., H_l([x_0,...,x_{l-1}]))
    """
    def __init__(self, in_channels, growth_rate=32, dropout=0.2):
        super().__init__()
        inter      = 4 * growth_rate
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth_rate, 3, padding=1, bias=False)
        self.drop  = nn.Dropout2d(p=dropout)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.drop(self.conv2(F.relu(self.bn2(out), inplace=True)))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """C_out = C_0 + L × g   (Paper Eq. 8)"""
    def __init__(self, in_channels, num_layers, growth_rate=32, dropout=0.2):
        super().__init__()
        layers, c = [], in_channels
        for _ in range(num_layers):
            layers.append(BottleneckDenseLayer(c, growth_rate, dropout))
            c += growth_rate
        self.block        = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.block(x)


class TransitionBlock(nn.Module):
    """
    BN + ReLU + Conv1×1 + AvgPool2×2
    C' = floor(θ × C_in),  θ = 0.5   (Paper Eq. 9, 10)
    """
    def __init__(self, in_channels, reduction=0.5):
        super().__init__()
        out        = int(in_channels * reduction)
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out, 1, bias=False),
            nn.AvgPool2d(2, stride=2),
        )
        self.out_channels = out

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    """Decoder refinement block — Paper Eq. 14"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenseNetEncoder(nn.Module):
    """
    Paper Section 3.3.1, Figure 5, Table 2 Steps 2–7:

      Step 2: InitConv(64,3×3)+BN+ReLU → MaxPool(2×2)    → skip1
      Step 3: DenseBlock-1 (L=4, g=32)                    → skip2
      Step 4: Transition-1 (reduction=0.5)
      Step 5: DenseBlock-2 (L=6, g=32)                    → skip3
              Transition-2 (reduction=0.5)
      Step 6: DenseBlock-3 (L=8, g=32)                    → skip4
              Transition-3 (reduction=0.5)
      Step 7: Bottleneck DenseBlock (L=4, g=32)           → Z
    """
    def __init__(self, in_channels=1, growth_rate=32, dropout=0.2):
        super().__init__()
        g = growth_rate

        self.stem  = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool0 = nn.MaxPool2d(2, stride=2)

        self.db1 = DenseBlock(64, 4, g, dropout)          # out: 192
        self.tr1 = TransitionBlock(self.db1.out_channels) # out: 96

        self.db2 = DenseBlock(self.tr1.out_channels, 6, g, dropout)  # out: 288
        self.tr2 = TransitionBlock(self.db2.out_channels)             # out: 144

        self.db3 = DenseBlock(self.tr2.out_channels, 8, g, dropout)  # out: 400
        self.tr3 = TransitionBlock(self.db3.out_channels)             # out: 200

        self.btn = DenseBlock(self.tr3.out_channels, 4, g, dropout)  # out: 328

        self.skip_channels = {
            'skip1': 64,
            'skip2': self.db1.out_channels,
            'skip3': self.db2.out_channels,
            'skip4': self.db3.out_channels,
        }
        self.bottleneck_channels = self.btn.out_channels

    def forward(self, x):
        s1 = self.stem(x)       # [B, 64,  128, 128]
        x  = self.pool0(s1)     # [B, 64,   64,  64]
        s2 = self.db1(x)        # [B, 192,  64,  64]
        x  = self.tr1(s2)       # [B, 96,   32,  32]
        s3 = self.db2(x)        # [B, 288,  32,  32]
        x  = self.tr2(s3)       # [B, 144,  16,  16]
        s4 = self.db3(x)        # [B, 400,  16,  16]
        x  = self.tr3(s4)       # [B, 200,   8,   8]
        Z  = self.btn(x)        # [B, 328,   8,   8]
        return Z, (s1, s2, s3, s4)


class UNetDecoder(nn.Module):
    """
    Paper Section 3.3.2, Figure 6, Table 2 Steps 8–9:
      Decoder-1: Upsample → cat(skip4) → ConvBlock(256)
      Decoder-2: Upsample → cat(skip3) → ConvBlock(128)
      Decoder-3: Upsample → cat(skip2) → ConvBlock(64)
      Decoder-4: Upsample → cat(skip1) → ConvBlock(32)
      Output   : Conv1×1 + Sigmoid → M̂    (Eq. 15, 16)
    """
    def __init__(self, bottleneck_ch, skip_channels):
        super().__init__()
        up = lambda: nn.Upsample(scale_factor=2, mode='bilinear',
                                 align_corners=True)
        self.up1  = up()
        self.dec1 = ConvBlock(bottleneck_ch + skip_channels['skip4'], 256)
        self.up2  = up()
        self.dec2 = ConvBlock(256 + skip_channels['skip3'], 128)
        self.up3  = up()
        self.dec3 = ConvBlock(128 + skip_channels['skip2'], 64)
        self.up4  = up()
        self.dec4 = ConvBlock(64  + skip_channels['skip1'], 32)
        self.seg_out = nn.Sequential(
            nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, Z, skips):
        s1, s2, s3, s4 = skips
        x = self.dec1(torch.cat([self.up1(Z),  s4], 1))
        x = self.dec2(torch.cat([self.up2(x),  s3], 1))
        x = self.dec3(torch.cat([self.up3(x),  s2], 1))
        x = self.dec4(torch.cat([self.up4(x),  s1], 1))
        return self.seg_out(x)


class ClassificationHead(nn.Module):
    """
    Paper Section 3.3.3, Table 2 Step 10, Eq. 13–18:
      GAP(Z) → FC(256,ReLU) + Dropout(0.5)
             → FC(128,ReLU) + Dropout(0.3)
             → FC(num_classes)
    """
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),         nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, Z):
        return self.fc(self.gap(Z))


class DenseUNetSC(nn.Module):
    """
    DenseUNet-SC: Joint Segmentation and Classification
    Paper Section 3.3, Figure 4

    Shared DenseNet encoder feeds TWO PARALLEL branches:
      (1) U-Net Decoder    → binary segmentation mask M̂
      (2) Classification Head → tumor-type logits ŷ

    Task coupling: IMPLICIT through shared encoder under
    joint end-to-end optimization.
    Loss: L = 0.6·L_seg + 0.4·L_cls   (Paper Table 3)
    """
    def __init__(self, in_channels=1, num_classes=3,
                 growth_rate=32, dropout=0.2):
        super().__init__()
        self.encoder  = DenseNetEncoder(in_channels, growth_rate, dropout)
        self.decoder  = UNetDecoder(
            self.encoder.bottleneck_channels,
            self.encoder.skip_channels)
        self.cls_head = ClassificationHead(
            self.encoder.bottleneck_channels, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x       : [B, 1, 128, 128]
        seg_out : [B, 1, 128, 128]  — binary tumor probability
        cls_out : [B, num_classes]  — tumor-type logits
        """
        Z, skips = self.encoder(x)
        seg_out  = self.decoder(Z, skips)   # branch 1 — segmentation
        cls_out  = self.cls_head(Z)         # branch 2 — classification
        return seg_out, cls_out


# ================================================================
#  LOSS FUNCTION  (Paper Table 2 Step 11, Table 3)
# ================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        p = pred.view(-1)
        t = target.view(-1).float()
        return 1.0 - (2.0*(p*t).sum() + self.smooth) / \
               (p.sum() + t.sum() + self.smooth)



class MultiTaskLoss(nn.Module):
    """
    Reviewer-safe loss wrapper.

    task='joint'          : L = w_seg*(BCE+Dice) + w_cls*CrossEntropy
    task='segmentation'  : L = BCE+Dice only. Use this for BraTS.
    task='classification': L = CrossEntropy only. Use this for classifier baselines/ablations.
    """
    def __init__(self, w_seg=0.6, w_cls=0.4, task='joint'):
        super().__init__()
        if task not in {'joint', 'segmentation', 'classification'}:
            raise ValueError(f"Unknown task: {task}")
        self.w_seg = w_seg
        self.w_cls = w_cls
        self.task = task
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, seg_pred, seg_true, cls_pred, cls_true):
        l_seg = None
        l_cls = None
        if self.task in {'joint', 'segmentation'}:
            if seg_pred is None or seg_true is None:
                raise ValueError('Segmentation prediction/target required for segmentation loss.')
            l_seg = self.bce(seg_pred, seg_true.float()) + self.dice(seg_pred, seg_true)
        if self.task in {'joint', 'classification'}:
            if cls_pred is None or cls_true is None:
                raise ValueError('Classification prediction/target required for classification loss.')
            l_cls = self.ce(cls_pred, cls_true.long())

        if self.task == 'joint':
            total = self.w_seg * l_seg + self.w_cls * l_cls
        elif self.task == 'segmentation':
            total = l_seg
        else:
            total = l_cls
        return total, l_seg, l_cls


# ================================================================
#  REPRODUCIBILITY, SPLITTING, AND PREPROCESSING
# ================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subject_level_split(subject_ids, labels, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Stratified split at SUBJECT/PATIENT level before sample/slice expansion.

    Reviewer fix:
      S_train ∩ S_val = ∅
      S_train ∩ S_test = ∅
      S_val   ∩ S_test = ∅
    """
    rng = random.Random(seed)
    label_by_subject = {}
    for sid, lbl in zip(subject_ids, labels):
        if sid in label_by_subject and label_by_subject[sid] != lbl:
            raise ValueError(f'Subject {sid} has conflicting labels: {label_by_subject[sid]} and {lbl}')
        label_by_subject[sid] = int(lbl)

    buckets = defaultdict(list)
    for sid, lbl in label_by_subject.items():
        buckets[lbl].append(sid)

    train_ids, val_ids, test_ids = [], [], []
    for lbl in sorted(buckets.keys()):
        b = buckets[lbl].copy()
        rng.shuffle(b)
        n = len(b)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        # Keep at least one sample for val/test when possible.
        if n >= 3:
            n_tr = max(1, min(n_tr, n - 2))
            n_va = max(1, min(n_va, n - n_tr - 1))
        train_ids += b[:n_tr]
        val_ids += b[n_tr:n_tr + n_va]
        test_ids += b[n_tr + n_va:]

    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    assert not (tr & va), 'Data leakage: train ∩ val is not empty!'
    assert not (tr & te), 'Data leakage: train ∩ test is not empty!'
    assert not (va & te), 'Data leakage: val ∩ test is not empty!'

    print(f'  Subject split → train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)}')
    return train_ids, val_ids, test_ids


def save_split_ids(train_ids, val_ids, test_ids, out_dir=OUT_DIR):
    split_dir = Path(out_dir) / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [('train_ids.txt', train_ids), ('val_ids.txt', val_ids), ('test_ids.txt', test_ids)]:
        (split_dir / name).write_text('\n'.join(map(str, ids)) + '\n')
    print(f'  Split IDs saved → {split_dir}/')


def intensity_normalize(volume, p_low=1.0, p_high=99.0):
    brain = volume > 0
    if not brain.any():
        return volume.astype(np.float32)
    lo = np.percentile(volume[brain], p_low)
    hi = np.percentile(volume[brain], p_high)
    vol = np.clip(volume, lo, hi)
    mu = vol[brain].mean()
    sig = vol[brain].std() + 1e-8
    return ((vol - mu) / sig).astype(np.float32)


def spatial_normalize(image, mask=None, size=(128, 128)):
    nz = np.argwhere(image > 0)
    if len(nz) > 0:
        r0, c0 = nz.min(0)
        r1, c1 = nz.max(0)
        margin = 5
        r0 = max(0, r0 - margin)
        c0 = max(0, c0 - margin)
        r1 = min(image.shape[0] - 1, r1 + margin)
        c1 = min(image.shape[1] - 1, c1 + margin)
        image = image[r0:r1 + 1, c0:c1 + 1]
        if mask is not None:
            mask = mask[r0:r1 + 1, c0:c1 + 1]

    vmin, vmax = image.min(), image.max()
    img8 = ((image - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    img_rs = np.array(Image.fromarray(img8).resize(size, Image.BILINEAR), dtype=np.float32) / 255.0

    mask_rs = None
    if mask is not None:
        mask_rs = np.array(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.NEAREST), dtype=np.uint8)
    return img_rs, mask_rs


def binarize_mask(mask):
    return (mask > 0).astype(np.uint8)


# ================================================================
#  DATASET LOADERS
# ================================================================

def _mat_struct_field(struct_obj, field_name, default=None):
    """Safely read a scipy-loaded MATLAB struct field."""
    try:
        if hasattr(struct_obj, 'dtype') and struct_obj.dtype.names and field_name in struct_obj.dtype.names:
            value = struct_obj[field_name]
        else:
            return default
        while isinstance(value, np.ndarray) and value.size == 1 and value.dtype.names is None:
            value = value.item()
        return value
    except Exception:
        return default


def _scalar_to_str(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    try:
        val = arr.squeeze().item()
    except Exception:
        val = arr.squeeze()
    if isinstance(val, bytes):
        return val.decode(errors='ignore')
    return str(val)


def _parse_figshare_mat(fpath, folder_label=None, load_pixels=False):
    """
    Parse one Figshare .mat file.

    Important reviewer fix:
    - If cjdata.PID exists, use it as subject_id so all slices from the same patient
      stay in one split.
    - If PID does not exist, fall back to file stem and emit label-source metadata.
    """
    data = sio.loadmat(str(fpath))
    image = None
    mask = None
    label = None
    pid = None

    if 'cjdata' in data:
        cj = data['cjdata'][0, 0]
        raw_label = _mat_struct_field(cj, 'label', None)
        if raw_label is not None:
            label = int(np.asarray(raw_label).squeeze()) - 1
        raw_pid = _mat_struct_field(cj, 'PID', None)
        pid = _scalar_to_str(raw_pid)
        if load_pixels:
            image = np.asarray(_mat_struct_field(cj, 'image'), dtype=np.float32)
            mask_value = _mat_struct_field(cj, 'tumorMask', None)
            mask = np.asarray(mask_value, dtype=np.uint8) if mask_value is not None else np.zeros_like(image, dtype=np.uint8)
    else:
        if 'label' in data:
            label = int(np.asarray(data['label']).squeeze()) - 1
        if 'PID' in data:
            pid = _scalar_to_str(data['PID'])
        if load_pixels:
            if 'image' not in data:
                raise ValueError(f'No image field found in {fpath}')
            image = np.asarray(data['image'], dtype=np.float32)
            mask = np.asarray(data.get('tumorMask', np.zeros_like(image)), dtype=np.uint8)

    if label is None:
        label = folder_label
    if label is None:
        raise ValueError(f'Could not determine class label for {fpath}. Use class folders 1/2/3 or cjdata.label.')

    if pid is None or pid.lower() in {'none', 'nan', ''}:
        # Fallback is still leakage-safe only if each file is truly one subject.
        # This is saved clearly so authors can report whether real PID was available.
        pid = f'NO_PID_{Path(fpath).stem}'
        pid_source = 'fallback_file_stem'
    else:
        pid_source = 'cjdata.PID'

    rec = {
        'path': Path(fpath),
        'label': int(label),
        'subject_id': str(pid),
        'pid_source': pid_source,
    }
    if load_pixels:
        rec['image'] = image
        rec['mask'] = mask
    return rec


def scan_figshare_records(data_root):
    data_root = Path(data_root)
    if not SCIPY_OK:
        raise ImportError('Install scipy: pip install scipy')
    if not data_root.exists():
        raise FileNotFoundError(f'Data root not found: {data_root}')

    records = []
    class_folders = {'1': 0, '2': 1, '3': 2}
    has_class_folders = any((data_root / k).exists() for k in class_folders)

    if has_class_folders:
        for folder, lbl in class_folders.items():
            folder_path = data_root / folder
            for fpath in sorted(folder_path.glob('*.mat')):
                records.append(_parse_figshare_mat(fpath, folder_label=lbl, load_pixels=False))
    else:
        for fpath in sorted(data_root.rglob('*.mat')):
            records.append(_parse_figshare_mat(fpath, folder_label=None, load_pixels=False))

    if not records:
        raise FileNotFoundError(f'No .mat files found under {data_root}')

    n_pid = sum(r['pid_source'] == 'cjdata.PID' for r in records)
    print(f'  Figshare files found: {len(records)} | PID available: {n_pid}/{len(records)}')
    if n_pid == 0:
        print('  WARNING: No cjdata.PID field found. Split falls back to file-level IDs; verify the dataset has one file per subject.')
    return records


def load_figshare(data_root, seed=42):
    """
    Figshare Brain Tumor Dataset.

    Reviewer-safe behavior:
    1. Read cjdata.PID when available.
    2. Split patients/subjects first.
    3. Then expand files/slices into train/val/test samples.
    4. Save exact split IDs for reproducibility.
    """
    print(f'Loading Figshare dataset from: {data_root}')
    records = scan_figshare_records(data_root)

    label_by_subject = {}
    for rec in records:
        sid = rec['subject_id']
        if sid in label_by_subject and label_by_subject[sid] != rec['label']:
            raise ValueError(f'Conflicting class labels for subject {sid}.')
        label_by_subject[sid] = rec['label']

    subject_ids = list(label_by_subject.keys())
    labels = [label_by_subject[sid] for sid in subject_ids]
    train_ids, val_ids, test_ids = subject_level_split(subject_ids, labels, seed=seed)
    save_split_ids(train_ids, val_ids, test_ids)

    buckets = {'train': set(train_ids), 'val': set(val_ids), 'test': set(test_ids)}

    def build(split_name):
        split_ids = buckets[split_name]
        samples = []
        for rec in records:
            if rec['subject_id'] not in split_ids:
                continue
            try:
                loaded = _parse_figshare_mat(rec['path'], folder_label=rec['label'], load_pixels=True)
                img_n, mask_n = spatial_normalize(loaded['image'], loaded['mask'], (CFG['img_size'], CFG['img_size']))
                samples.append({
                    'image': img_n.astype(np.float32),
                    'mask': binarize_mask(mask_n).astype(np.float32),
                    'label': int(rec['label']),
                    'subject_id': rec['subject_id'],
                    'source_file': str(rec['path'].name),
                })
            except Exception as exc:
                print(f'  Warning — skipping {rec["path"].name}: {exc}')
        return samples

    return build('train'), build('val'), build('test')


def load_brats2021(data_root, modality='flair', min_tumor_px=50, bg_keep_ratio=0.15, seed=42):
    """
    BraTS 2021 binary segmentation loader.

    Reviewer-safe behavior:
    - BraTS is used for segmentation only in this script.
    - Classification labels are dummy placeholders and are ignored when task='segmentation'.
    """
    if not NIBABEL_OK:
        raise ImportError('Install nibabel: pip install nibabel')

    data_root = Path(data_root)
    subject_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if not subject_dirs:
        raise FileNotFoundError(f'No subject folders found in {data_root}')

    subject_ids = [d.name for d in subject_dirs]
    labels = [0] * len(subject_ids)  # dummy single class; ignored for segmentation-only training
    train_ids, val_ids, test_ids = subject_level_split(subject_ids, labels, seed=seed)
    save_split_ids(train_ids, val_ids, test_ids)
    id_to_dir = {d.name: d for d in subject_dirs}
    rng = random.Random(seed)

    def build(ids_list):
        samples = []
        for sid in ids_list:
            sdir = id_to_dir[sid]
            mod_file = list(sdir.glob(f'*_{modality}.nii*'))
            seg_file = list(sdir.glob('*_seg.nii*'))
            if not mod_file or not seg_file:
                print(f'  Warning — missing files for {sid}, skipping.')
                continue
            try:
                vol = nib.load(str(mod_file[0])).get_fdata().astype(np.float32)
                seg = nib.load(str(seg_file[0])).get_fdata().astype(np.uint8)
            except Exception as exc:
                print(f'  Warning — {sid}: {exc}')
                continue

            vol = intensity_normalize(vol)
            tumor_slices, bg_slices = [], []
            for z in range(vol.shape[2]):
                img2d = vol[:, :, z]
                mask2d = binarize_mask(seg[:, :, z])
                img_rs, mask_rs = spatial_normalize(img2d, mask2d, (CFG['img_size'], CFG['img_size']))
                item = {
                    'image': img_rs.astype(np.float32),
                    'mask': mask_rs.astype(np.float32),
                    'label': 0,               # ignored in segmentation mode
                    'subject_id': sid,
                    'slice_idx': z,
                }
                if mask_rs.sum() >= min_tumor_px:
                    tumor_slices.append(item)
                else:
                    bg_slices.append(item)
            n_bg = max(1, int(len(bg_slices) * bg_keep_ratio)) if bg_slices else 0
            samples.extend(tumor_slices)
            if n_bg > 0:
                samples.extend(rng.sample(bg_slices, min(n_bg, len(bg_slices))))
        return samples

    print(f'Loading BraTS 2021 from: {data_root} | modality={modality} | segmentation-only')
    return build(train_ids), build(val_ids), build(test_ids)


# ================================================================
#  PYTORCH DATASET AND DATALOADERS
# ================================================================

class BrainTumorDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _rotate_tensor_2d(tensor, angle, is_mask=False):
        """Rotate a [1,H,W] tensor using PIL; avoids requiring torchvision at runtime."""
        arr = tensor.squeeze(0).detach().cpu().numpy()
        if is_mask:
            pil = Image.fromarray(((arr > 0.5).astype(np.uint8)) * 255)
            rotated = pil.rotate(angle, resample=Image.NEAREST)
            out = (np.asarray(rotated, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        else:
            pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
            rotated = pil.rotate(angle, resample=Image.BILINEAR)
            out = np.asarray(rotated, dtype=np.float32) / 255.0
        return torch.from_numpy(out).unsqueeze(0)

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if random.random() > 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[1])
        angle = random.uniform(-15, 15)
        img = self._rotate_tensor_2d(img, angle, is_mask=False)
        mask = self._rotate_tensor_2d(mask, angle, is_mask=True)
        if random.random() > 0.5:
            img = torch.clamp(img * random.uniform(0.85, 1.15), 0.0, 1.0)
        return img, mask

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_t = torch.from_numpy(s['image']).unsqueeze(0)
        mask_t = torch.from_numpy(s['mask']).unsqueeze(0)
        if self.augment:
            img_t, mask_t = self._augment(img_t, mask_t)
        return {
            'image': img_t,
            'mask': mask_t,
            'label': torch.tensor(s.get('label', 0), dtype=torch.long),
            'subject_id': str(s.get('subject_id', idx)),
        }


def get_dataloaders(train_s, val_s, test_s, batch_size=8, num_workers=2):
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    tr = DataLoader(BrainTumorDataset(train_s, augment=True), batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    va = DataLoader(BrainTumorDataset(val_s), batch_size=batch_size, shuffle=False, **kw)
    te = DataLoader(BrainTumorDataset(test_s), batch_size=batch_size, shuffle=False, **kw)
    print(f'  DataLoaders → train:{len(tr)} val:{len(va)} test:{len(te)} batches')
    return tr, va, te


# ================================================================
#  METRICS, TRAINING, EVALUATION, INFERENCE TIME
# ================================================================

def dice_coefficient(pred, target, threshold=0.5, smooth=1.0):
    p = (pred > threshold).float().view(-1)
    t = target.float().view(-1)
    return ((2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    p = (pred > threshold).float().view(-1)
    t = target.float().view(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def train_model(model, criterion, train_dl, val_dl, device, cfg, task='joint'):
    optimizer = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg['lr_factor'], patience=cfg['lr_patience'], min_lr=cfg['min_lr'])
    best_val_loss = float('inf')
    best_weights = None
    wait = 0
    history = defaultdict(list)
    t0 = time.time()

    print('=' * 72)
    print(f'  DenseUNet-SC Training | task={task}')
    print(f'  Device={device} | epochs={cfg["epochs"]} | batch={cfg["batch_size"]}')
    print('=' * 72)

    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        tr = defaultdict(float)
        n_tr = 0
        for batch in train_dl:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            seg_p, cls_p = model(imgs)
            loss, l_seg, l_cls = criterion(seg_p, masks, cls_p, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr['loss'] += loss.item()
            if task in {'joint', 'segmentation'}:
                tr['dice'] += dice_coefficient(seg_p.detach().cpu(), masks.cpu())
                tr['iou'] += iou_score(seg_p.detach().cpu(), masks.cpu())
            if task in {'joint', 'classification'}:
                tr['cls_acc'] += (cls_p.detach().cpu().argmax(1) == labels.cpu()).float().mean().item()
            n_tr += 1

        model.eval()
        va = defaultdict(float)
        n_va = 0
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                seg_p, cls_p = model(imgs)
                loss, l_seg, l_cls = criterion(seg_p, masks, cls_p, labels)
                va['loss'] += loss.item()
                if task in {'joint', 'segmentation'}:
                    va['dice'] += dice_coefficient(seg_p.cpu(), masks.cpu())
                    va['iou'] += iou_score(seg_p.cpu(), masks.cpu())
                if task in {'joint', 'classification'}:
                    va['cls_acc'] += (cls_p.cpu().argmax(1) == labels.cpu()).float().mean().item()
                n_va += 1

        if n_tr == 0 or n_va == 0:
            raise RuntimeError('Empty train or validation dataloader. Check dataset path and split sizes.')

        for k in tr:
            history[f'train_{k}'].append(tr[k] / n_tr)
        for k in va:
            history[f'val_{k}'].append(va[k] / n_va)
        val_loss = va['loss'] / n_va
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        ckpt = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save({'epoch': ep, 'model_state': best_weights, 'val_loss': best_val_loss, 'history': dict(history), 'task': task}, OUT_DIR / 'best_model.pth')
            ckpt = ' ✓'
        else:
            wait += 1

        msg = f'  Ep {ep:03d}/{cfg["epochs"]} | loss {tr["loss"]/n_tr:.4f}/{val_loss:.4f}'
        if task in {'joint', 'segmentation'}:
            msg += f' | Dice {va["dice"]/n_va:.4f} | IoU {va["iou"]/n_va:.4f}'
        if task in {'joint', 'classification'}:
            msg += f' | Acc {va["cls_acc"]/n_va:.4f}'
        msg += f' | LR {lr:.0e}{ckpt}'
        print(msg)

        if wait >= cfg['patience']:
            print(f'\n  Early stopping at epoch {ep}.')
            break

    if best_weights:
        model.load_state_dict(best_weights)
    print(f'\n  Training complete in {time.time() - t0:.0f}s | best val loss={best_val_loss:.6f}')
    return dict(history)


@torch.no_grad()
def evaluate_model(model, test_dl, device, task='joint', num_classes=3):
    model.eval()
    seg_dice, seg_iou = [], []
    subj_logits = defaultdict(list)
    subj_labels = {}

    for batch in test_dl:
        imgs = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label']
        sids = batch['subject_id']
        seg_p, cls_p = model(imgs)

        if task in {'joint', 'segmentation'}:
            for i in range(imgs.size(0)):
                seg_dice.append(dice_coefficient(seg_p[i].cpu(), masks[i].cpu()))
                seg_iou.append(iou_score(seg_p[i].cpu(), masks[i].cpu()))

        if task in {'joint', 'classification'}:
            probs = torch.softmax(cls_p.cpu(), dim=1)
            for i in range(imgs.size(0)):
                subj_logits[str(sids[i])].append(probs[i])
                subj_labels[str(sids[i])] = labels[i].item()

    results = {}
    if task in {'joint', 'segmentation'}:
        results.update({
            'dice': float(np.mean(seg_dice)) if seg_dice else None,
            'iou': float(np.mean(seg_iou)) if seg_iou else None,
            'segmentation_unit': 'slice',
        })

    if task in {'joint', 'classification'}:
        y_true, y_pred = [], []
        for sid, logit_list in subj_logits.items():
            avg_prob = torch.stack(logit_list).mean(0)
            y_pred.append(avg_prob.argmax().item())
            y_true.append(subj_labels[sid])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        labels_fixed = list(range(num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels_fixed)
        acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
        prec = precision_score(y_true, y_pred, labels=labels_fixed, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, labels=labels_fixed, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels_fixed, average='macro', zero_division=0)

        per_class = {}
        specs = []
        for c in labels_fixed:
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            specs.append(specificity)
            per_class[CLASS_NAMES.get(c, str(c))] = {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(tp / (tp + fp + 1e-8)),
                'f1': float(2 * tp / (2 * tp + fp + fn + 1e-8)),
            }

        results.update({
            'accuracy': acc,
            'precision_macro': float(prec),
            'recall_macro': float(rec),
            'f1_macro': float(f1),
            'specificity_macro': float(np.mean(specs)),
            'confusion_matrix': cm,
            'per_class': per_class,
            'n_subjects': int(len(y_true)),
            'classification_unit': 'subject; slice probabilities averaged per subject',
        })

    print('\n' + '=' * 60)
    print(f'  Test Results | task={task}')
    print('=' * 60)
    if 'dice' in results:
        print(f'  Segmentation  : Dice={results["dice"]:.4f} | IoU={results["iou"]:.4f} | unit=slice')
    if 'accuracy' in results:
        print(f'  Classification: Acc={results["accuracy"]:.4f} | F1={results["f1_macro"]:.4f} | Spec={results["specificity_macro"]:.4f} | n_subjects={results["n_subjects"]}')
        print(f'  Confusion matrix labels fixed to: {list(range(num_classes))}')
    print('=' * 60)
    return results


@torch.no_grad()
def measure_inference_time(model, data_loader, device, warmup_batches=3, max_batches=30):
    """Report practical inference time requested by reviewer."""
    model.eval()
    subject_counts = defaultdict(int)

    for b, batch in enumerate(data_loader):
        if b >= warmup_batches:
            break
        imgs = batch['image'].to(device)
        _ = model(imgs)

    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize()

    n_slices = 0
    t0 = time.perf_counter()
    for b, batch in enumerate(data_loader):
        if b >= max_batches:
            break
        imgs = batch['image'].to(device)
        _ = model(imgs)
        n_slices += imgs.size(0)
        for sid in batch['subject_id']:
            subject_counts[str(sid)] += 1

    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_slice = 1000.0 * elapsed / max(n_slices, 1)
    mean_slices_per_subject = float(np.mean(list(subject_counts.values()))) if subject_counts else 1.0
    ms_per_subject = ms_per_slice * mean_slices_per_subject
    return {
        'device': str(device),
        'timed_slices': int(n_slices),
        'ms_per_slice': float(ms_per_slice),
        'mean_slices_per_subject_in_timed_batches': mean_slices_per_subject,
        'estimated_ms_per_subject': float(ms_per_subject),
    }


def save_results(history, results, timing, task, dataset, out_dir=OUT_DIR):
    serializable = {
        'dataset': dataset,
        'task': task,
        'hyperparameters': CFG,
        'history': history,
        'test_results': {},
        'inference_time': timing,
    }
    for k, v in results.items():
        if k == 'confusion_matrix':
            serializable['test_results'][k] = v.tolist()
        else:
            serializable['test_results'][k] = v
    out_path = Path(out_dir) / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f'  Results saved → {out_path}')


def main():
    parser = argparse.ArgumentParser(description='DenseUNet-SC reviewer-safe training script')
    parser.add_argument('--dataset', required=True, choices=['figshare', 'brats'])
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--task', default='auto', choices=['auto', 'joint', 'segmentation', 'classification'],
                        help='auto: figshare→joint, brats→segmentation')
    parser.add_argument('--modality', default='flair', choices=['flair', 't1', 't1ce', 't2'])
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    parser.add_argument('--batch_size', type=int, default=CFG['batch_size'])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=CFG['seed'])
    args = parser.parse_args()

    CFG['epochs'] = args.epochs
    CFG['batch_size'] = args.batch_size
    CFG['seed'] = args.seed
    set_seed(args.seed)

    if args.task == 'auto':
        task = 'segmentation' if args.dataset == 'brats' else 'joint'
    else:
        task = args.task
    if args.dataset == 'brats' and task != 'segmentation':
        print('  WARNING: BraTS has no 3-class tumor-type labels in this pipeline. Switching to task=segmentation.')
        task = 'segmentation'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\n' + '=' * 72)
    print('  DenseUNet-SC: Reviewer-safe implementation')
    print(f'  Dataset={args.dataset} | task={task} | device={device}')
    print('=' * 72)

    if args.dataset == 'figshare':
        train_s, val_s, test_s = load_figshare(args.data_root, seed=args.seed)
    else:
        train_s, val_s, test_s = load_brats2021(args.data_root, modality=args.modality, seed=args.seed)

    train_dl, val_dl, test_dl = get_dataloaders(train_s, val_s, test_s, batch_size=CFG['batch_size'], num_workers=args.num_workers)
    print(f'  Samples → train:{len(train_s)} val:{len(val_s)} test:{len(test_s)}')

    model = DenseUNetSC(in_channels=CFG['in_channels'], num_classes=CFG['num_classes'], growth_rate=CFG['growth_rate'], dropout=CFG['dropout']).to(device)
    criterion = MultiTaskLoss(CFG['w_seg'], CFG['w_cls'], task=task)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable parameters: {params:,}')

    history = train_model(model, criterion, train_dl, val_dl, device, CFG, task=task)
    results = evaluate_model(model, test_dl, device, task=task, num_classes=CFG['num_classes'])
    timing = measure_inference_time(model, test_dl, device)
    print(f'  Inference time: {timing["ms_per_slice"]:.2f} ms/slice | approx {timing["estimated_ms_per_subject"]:.2f} ms/subject')
    save_results(history, results, timing, task, args.dataset)


if __name__ == '__main__':
    main()
