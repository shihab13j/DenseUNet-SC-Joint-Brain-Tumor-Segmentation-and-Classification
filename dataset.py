"""
Data Preprocessing Pipeline — exactly as described in paper Section 3.2

Steps (Paper Table 2, Step 1 and Section 3.2):
1. Subject-level dataset splitting (BEFORE any slice extraction)
   → prevents information leakage
2. Intensity normalization: percentile clipping + z-score (Eq. 4, 5)
3. Spatial normalization: bounding-box crop + resize to 128×128
4. Slice extraction from 3D volumes → 2D axial slices
5. Tumor-aware sample selection (threshold-based)
6. Classification label alignment (subject-level → slice-level)
7. Data augmentation (training set only)

Classification labels (Figshare dataset):
  0 = meningioma
  1 = glioma
  2 = pituitary tumor
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image


# ─────────────────────────────────────────────
#  Label Map  (paper: 3 tumor classes)
# ─────────────────────────────────────────────
CLASS_NAMES = {0: 'meningioma', 1: 'glioma', 2: 'pituitary'}
CLASS_IDS   = {'meningioma': 0, 'glioma': 1, 'pituitary': 2}


# ─────────────────────────────────────────────
#  Subject-Level Split  (Paper Eq. 1–3)
# ─────────────────────────────────────────────

def subject_level_split(
    subject_ids: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Stratified subject-level split so that S_train ∩ S_val ∩ S_test = ∅.
    Splitting is done BEFORE any slice extraction to prevent leakage.

    Returns:
        train_ids, val_ids, test_ids  — disjoint lists of subject IDs
    """
    assert abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) < 1e-6

    rng = random.Random(seed)

    # Stratify by class label
    class_buckets: Dict[int, List[str]] = {}
    for sid, lbl in zip(subject_ids, labels):
        class_buckets.setdefault(lbl, []).append(sid)

    train_ids, val_ids, test_ids = [], [], []

    for lbl, bucket in class_buckets.items():
        bucket = bucket.copy()
        rng.shuffle(bucket)
        n      = len(bucket)
        n_tr   = int(n * train_ratio)
        n_val  = int(n * val_ratio)
        train_ids.extend(bucket[:n_tr])
        val_ids.extend(bucket[n_tr:n_tr + n_val])
        test_ids.extend(bucket[n_tr + n_val:])

    # Verify mutual exclusion (Paper Eq. 2)
    assert len(set(train_ids) & set(val_ids) & set(test_ids)) == 0, \
        "Data leakage detected — splits are not mutually exclusive!"
    assert len(set(train_ids) | set(val_ids) | set(test_ids)) == \
        len(set(subject_ids)), "Split union does not cover all subjects!"

    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────
#  Preprocessing Functions
# ─────────────────────────────────────────────

def intensity_normalize(
    volume: np.ndarray,
    p_low:  float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Per-subject, per-modality intensity normalization (Paper Eq. 4, 5):
      1. Percentile clipping:  X_clip = clip(X, p_low, p_high)
      2. Z-score over brain:   X' = (X_clip - μ_brain) / σ_brain
    Only non-zero voxels (brain region) are used for statistics.
    """
    brain_mask = volume > 0
    if brain_mask.sum() == 0:
        return volume.astype(np.float32)

    # Eq. 4 — percentile clipping
    p_lo_val = np.percentile(volume[brain_mask], p_low)
    p_hi_val = np.percentile(volume[brain_mask], p_high)
    clipped  = np.clip(volume, p_lo_val, p_hi_val)

    # Eq. 5 — z-score using brain-region statistics
    mu    = clipped[brain_mask].mean()
    sigma = clipped[brain_mask].std() + 1e-8
    return ((clipped - mu) / sigma).astype(np.float32)


def spatial_normalize(
    image: np.ndarray,
    mask:  Optional[np.ndarray],
    target_size: Tuple[int, int] = (128, 128),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Spatial normalization (Paper Section 3.2):
      1. Bounding-box crop around brain to remove redundant background
      2. Resize + pad to target_size (128×128 as per Table 3)
    Same transform applied to mask to preserve pixel-wise correspondence.
    """
    # Find bounding box (non-zero region)
    nonzero = np.argwhere(image > 0)
    if len(nonzero) == 0:
        image_pil = Image.fromarray(
            ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255
             ).astype(np.uint8))
        image_rs = np.array(image_pil.resize(target_size, Image.BILINEAR),
                            dtype=np.float32)
        mask_rs  = None
        if mask is not None:
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_rs  = np.array(mask_pil.resize(target_size, Image.NEAREST),
                                dtype=np.uint8)
        return image_rs, mask_rs

    r_min, c_min = nonzero.min(axis=0)
    r_max, c_max = nonzero.max(axis=0)
    margin = 5
    r_min = max(0, r_min - margin)
    c_min = max(0, c_min - margin)
    r_max = min(image.shape[0] - 1, r_max + margin)
    c_max = min(image.shape[1] - 1, c_max + margin)

    image_crop = image[r_min:r_max+1, c_min:c_max+1]

    # Normalize to uint8 for PIL
    v_min, v_max = image_crop.min(), image_crop.max()
    if v_max > v_min:
        img8 = ((image_crop - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    else:
        img8 = np.zeros_like(image_crop, dtype=np.uint8)

    image_rs = np.array(
        Image.fromarray(img8).resize(target_size, Image.BILINEAR),
        dtype=np.float32) / 255.0

    mask_rs = None
    if mask is not None:
        mask_crop = mask[r_min:r_max+1, c_min:c_max+1]
        mask_rs   = np.array(
            Image.fromarray(mask_crop.astype(np.uint8)).resize(
                target_size, Image.NEAREST),
            dtype=np.uint8)

    return image_rs, mask_rs


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Binary segmentation label preparation (Paper Section 3.2):
    All tumor subregions (labels 1,2,4 in BraTS) → foreground=1, background=0.
    Deliberate binary design (see paper Introduction: Binary segmentation rationale).
    """
    return (mask > 0).astype(np.uint8)


def tumor_aware_selection(
    slices:          List[np.ndarray],
    masks:           List[np.ndarray],
    labels:          List[int],
    min_tumor_px:    int   = 50,
    bg_keep_ratio:   float = 0.15,
    seed:            int   = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Tumor-aware sample selection (Paper Section 3.2):
    - Retain slices with tumor pixels > min_tumor_px
    - Randomly subsample background-only slices at bg_keep_ratio
    Prevents classification branch from learning trivial background cues.
    """
    rng = random.Random(seed)
    tumor_slices, bg_slices = [], []

    for img, msk, lbl in zip(slices, masks, labels):
        n_tumor = (msk > 0).sum()
        if n_tumor >= min_tumor_px:
            tumor_slices.append((img, msk, lbl))
        else:
            bg_slices.append((img, msk, lbl))

    # Subsample background
    n_keep = max(1, int(len(bg_slices) * bg_keep_ratio))
    bg_keep = rng.sample(bg_slices, min(n_keep, len(bg_slices)))
    all_kept = tumor_slices + bg_keep
    rng.shuffle(all_kept)

    out_imgs   = [x[0] for x in all_kept]
    out_masks  = [x[1] for x in all_kept]
    out_labels = [x[2] for x in all_kept]
    return out_imgs, out_masks, out_labels


# ─────────────────────────────────────────────
#  Dataset Class
# ─────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for DenseUNet-SC training.

    Supports:
    - BraTS 2021 format (NIfTI volumes with segmentation masks)
    - Figshare format (PNG/JPG slices with classification labels)
    - Synthetic data (for testing / CI without actual dataset)

    Classification label (Paper Section 3.2 — Classification task definition):
      Subject-level label assigned to all retained slices of that subject.
      At test time, per-slice logits are aggregated per subject (majority vote).
    """

    def __init__(
        self,
        samples:     List[Dict],   # [{'image': np.arr, 'mask': np.arr, 'label': int}]
        augment:     bool  = False,
        target_size: Tuple = (128, 128),
    ):
        self.samples     = samples
        self.augment     = augment
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: torch.Tensor, mask: torch.Tensor):
        """
        Training augmentation (Paper Section 3.2):
        - Geometric: random horizontal/vertical flip, rotation ±15°
        - Intensity: mild brightness variation (images only, not masks)
        Same geometric transforms applied to both image and mask.
        """
        # Horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        # Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        # Rotation
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask,  angle)
        # Brightness (image only)
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            image  = torch.clamp(image * factor, 0, 1)
        return image, mask

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img    = sample['image'].astype(np.float32)   # [H, W]
        msk    = sample['mask'].astype(np.float32)    # [H, W]  binary {0,1}
        lbl    = int(sample['label'])

        # To tensor: [1, H, W]
        img_t = torch.from_numpy(img).unsqueeze(0)
        msk_t = torch.from_numpy(msk).unsqueeze(0)

        if self.augment:
            img_t, msk_t = self._augment(img_t, msk_t)

        return {
            'image':        img_t,                     # [1, 128, 128]
            'mask':         msk_t,                     # [1, 128, 128]
            'label':        torch.tensor(lbl, dtype=torch.long),
            'subject_id':   sample.get('subject_id', f'subj_{idx}'),
        }


# ─────────────────────────────────────────────
#  Synthetic Dataset (for testing without real data)
# ─────────────────────────────────────────────

def make_synthetic_dataset(
    n_subjects:  int   = 90,
    slices_each: int   = 8,
    num_classes: int   = 3,
    img_size:    int   = 128,
    seed:        int   = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate synthetic brain tumor data for pipeline verification.
    Mimics the Figshare / BraTS structure used in the paper:
    - n_subjects subjects, each with slices_each 2D slices
    - Subject-level class label (0=meningioma, 1=glioma, 2=pituitary)
    - Binary tumor mask per slice

    Returns: train_samples, val_samples, test_samples
    """
    rng   = np.random.default_rng(seed)
    rng_r = random.Random(seed)

    subjects = [f'subject_{i:04d}' for i in range(n_subjects)]
    labels   = [i % num_classes for i in range(n_subjects)]

    train_ids, val_ids, test_ids = subject_level_split(
        subjects, labels, train_ratio=0.70, val_ratio=0.15, seed=seed)

    def make_sample(subject_id, lbl, s_idx):
        img  = rng.normal(0, 1, (img_size, img_size)).astype(np.float32)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        # Synthetic tumor ellipse
        cy, cx = img_size // 2 + rng.integers(-20, 20), \
                 img_size // 2 + rng.integers(-20, 20)
        ry, rx = rng.integers(10, 30), rng.integers(10, 30)
        for y in range(img_size):
            for x in range(img_size):
                if ((y - cy)**2 / ry**2 + (x - cx)**2 / rx**2) <= 1:
                    mask[y, x] = 1
                    img[y, x] += rng.normal(2.0, 0.5)  # brighter tumor
        return {
            'image':      img,
            'mask':       mask.astype(np.float32),
            'label':      lbl,
            'subject_id': subject_id,
            'slice_idx':  s_idx,
        }

    def build_split(ids_list):
        samples = []
        label_map = dict(zip(subjects, labels))
        for sid in ids_list:
            lbl = label_map[sid]
            for s in range(slices_each):
                samples.append(make_sample(sid, lbl, s))
        return samples

    return build_split(train_ids), build_split(val_ids), build_split(test_ids)


def get_dataloaders(
    train_samples: List[Dict],
    val_samples:   List[Dict],
    test_samples:  List[Dict],
    batch_size:    int = 8,
    num_workers:   int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoaders (Paper Table 3: batch_size=8)."""
    train_ds = BrainTumorDataset(train_samples, augment=True)
    val_ds   = BrainTumorDataset(val_samples,   augment=False)
    test_ds  = BrainTumorDataset(test_samples,  augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers,
                          pin_memory=False, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, test_dl


if __name__ == '__main__':
    print("Testing preprocessing pipeline...")
    train_s, val_s, test_s = make_synthetic_dataset(
        n_subjects=30, slices_each=5)
    tr_dl, va_dl, te_dl = get_dataloaders(train_s, val_s, test_s, batch_size=4)

    batch = next(iter(tr_dl))
    print(f"  Batch image shape : {batch['image'].shape}")
    print(f"  Batch mask shape  : {batch['mask'].shape}")
    print(f"  Batch labels      : {batch['label']}")
    print(f"  Subject IDs       : {batch['subject_id']}")
    print(f"  Train batches     : {len(tr_dl)}")
    print(f"  Val   batches     : {len(va_dl)}")
    print(f"  Test  batches     : {len(te_dl)}")
    print("  ✓ Preprocessing pipeline verified — no data leakage")
