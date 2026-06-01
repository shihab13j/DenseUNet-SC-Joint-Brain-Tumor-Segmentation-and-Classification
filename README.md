# DenseUNet-SC: Densely Connected Multi-Task Learning for Brain Tumor Segmentation and Classification

This repository provides the official implementation of **DenseUNet-SC**, a densely connected shared-encoder deep learning framework for brain tumor segmentation and tumor-type classification from 2D MRI slices.

The model uses a **shared DenseNet-style encoder** with two parallel task-specific branches:

1. **Segmentation branch:** a U-Net-style decoder predicts a binary tumor/background mask.
2. **Classification branch:** a global-average-pooling classification head predicts tumor type.

The predicted segmentation mask is **not directly fed into the classification head**. Instead, segmentation and classification are coupled implicitly through the shared encoder and weighted multi-task optimization.

---

## Paper

**Title:** DenseUNet-SC: A Densely Connected Deep Learning Method for Joint Brain Tumor Segmentation and Classification
**Journal:** Healthcare Analytics
**Code availability:** https://github.com/your-username/DenseUNet-SC-Joint-Brain-Tumor-Segmentation-and-Classification

---

## Repository Structure

```text
.
├── train_denseunetsc.py
├── train_classification_baselines.py
├── train_segmentation_baselines.py
├── run_ablation.py
├── evaluate_subject_level.py
├── requirements.txt
├── README.md
```

---

## Main Features

* DenseUNet-SC shared-encoder multi-task architecture
* Binary brain tumor segmentation
* Three-class tumor-type classification
* Subject-level train/validation/test splitting
* Leakage-prevention assertions for train, validation, and test sets
* Subject-level classification evaluation using averaged slice probabilities
* Fixed 3-class confusion matrix evaluation
* Inference-time reporting
* Classification baselines:

  * ResNet-50
  * DenseNet-121
  * EfficientNet-B0
* Segmentation baselines:

  * U-Net
  * Attention U-Net
  * DenseUNet segmentation-only model
* Ablation experiments:

  * Full joint model
  * No dense encoder
  * Segmentation-only
  * Classification-only
  * Equal-loss weighting

---

## Datasets

This implementation supports two public brain tumor MRI datasets.

### 1. Figshare Brain Tumor Dataset

Used for three-class tumor-type classification and joint segmentation-classification experiments where both tumor masks and class labels are available.

Classes:

```text
0: Meningioma
1: Glioma
2: Pituitary tumor
```

Expected folder structure:

```text
data/figshare/
├── 1/
│   ├── *.mat
├── 2/
│   ├── *.mat
└── 3/
    ├── *.mat
```

The implementation uses the patient identifier field from the `.mat` file when available. This is used to perform subject-level splitting before training, validation, and testing.

### 2. BraTS 2021 Dataset

Used for binary tumor segmentation experiments only.

BraTS 2021 does not provide the three tumor-type labels required for the Figshare classification protocol. Therefore, this implementation does not compute three-class classification metrics on BraTS.

Expected folder structure:

```text
data/brats2021/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_flair.nii.gz
│   ├── BraTS2021_00000_seg.nii.gz
├── BraTS2021_00001/
│   ├── BraTS2021_00001_flair.nii.gz
│   ├── BraTS2021_00001_seg.nii.gz
└── ...
```

---

## Data Leakage Prevention

All splitting is performed at the subject level before slice extraction and augmentation.

The implementation enforces:

```text
train_subjects ∩ val_subjects = ∅
train_subjects ∩ test_subjects = ∅
val_subjects ∩ test_subjects = ∅
```

This prevents slices from the same subject appearing across training, validation, and test sets.

---

## Installation

Create a virtual environment:

```bash
python -m venv denseunetsc_env
source denseunetsc_env/bin/activate
```

For Windows:

```bash
python -m venv denseunetsc_env
denseunetsc_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train DenseUNet-SC on Figshare

For joint segmentation and three-class classification:

```bash
python train_denseunetsc.py --dataset figshare --data_root data/figshare --task joint --epochs 50 --batch_size 8
```

---

## Train DenseUNet-SC on BraTS

For segmentation-only training:

```bash
python train_denseunetsc.py --dataset brats --data_root data/brats2021 --task segmentation --modality flair --epochs 50 --batch_size 8
```

If `--dataset brats` is selected, the code automatically uses segmentation-only training because BraTS does not provide the three tumor-type classification labels required for this study.

---

## Classification Baselines

Run all classification baselines on Figshare:

```bash
python train_classification_baselines.py --data_root data/figshare --model all --epochs 50 --batch_size 8
```

Run a specific baseline:

```bash
python train_classification_baselines.py --data_root data/figshare --model resnet50 --epochs 50 --batch_size 8
python train_classification_baselines.py --data_root data/figshare --model densenet121 --epochs 50 --batch_size 8
python train_classification_baselines.py --data_root data/figshare --model efficientnet_b0 --epochs 50 --batch_size 8
```

---

## Segmentation Baselines

Run all segmentation baselines:

```bash
python train_segmentation_baselines.py --dataset brats --data_root data/brats2021 --model all --epochs 50 --batch_size 8
```

Run a specific segmentation baseline:

```bash
python train_segmentation_baselines.py --dataset brats --data_root data/brats2021 --model unet --epochs 50 --batch_size 8
python train_segmentation_baselines.py --dataset brats --data_root data/brats2021 --model attention_unet --epochs 50 --batch_size 8
python train_segmentation_baselines.py --dataset brats --data_root data/brats2021 --model denseunet --epochs 50 --batch_size 8
```

---

## Ablation Experiments

Run all ablation variants:

```bash
python run_ablation.py --data_root data/figshare --variant all --epochs 50 --batch_size 8
```

Run individual ablations:

```bash
python run_ablation.py --data_root data/figshare --variant full_joint --epochs 50 --batch_size 8
python run_ablation.py --data_root data/figshare --variant no_dense_encoder --epochs 50 --batch_size 8
python run_ablation.py --data_root data/figshare --variant segmentation_only --epochs 50 --batch_size 8
python run_ablation.py --data_root data/figshare --variant classification_only --epochs 50 --batch_size 8
python run_ablation.py --data_root data/figshare --variant equal_loss --epochs 50 --batch_size 8
```

---

## Evaluation Protocol

### Segmentation

Segmentation is evaluated using:

* Dice score
* Intersection over Union
* Inference time per slice

Segmentation metrics are computed on binary tumor/background masks.

### Classification

Classification is evaluated at the subject level.

For each subject, slice-level softmax probabilities are averaged, and the final class is selected using the highest averaged probability.

Classification metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* Confusion matrix
* Per-class sensitivity and specificity

The confusion matrix is computed using fixed class labels:

```python
labels = [0, 1, 2]
```

---

## Outputs

After training and evaluation, outputs are saved in the `outputs/` directory.
The exact output files may vary depending on the experiment.

---

## Reproducibility Notes

* Random seed is set to 42 by default.
* Subject-level splitting is used to reduce slice-level data leakage.
* Train, validation, and test subject identifiers should be saved in the `splits/` folder after running the experiments.
* The reported paper values should be updated using the outputs generated from this codebase.
* Dataset files are not included in this repository. Users must download the datasets from the official sources.

---

## Important Methodological Clarification

DenseUNet-SC is a shared-encoder parallel multi-task framework.

The segmentation branch and classification branch are trained jointly, but the predicted segmentation mask is not used as a direct input to the classifier. The task interaction occurs through the shared densely connected encoder and the combined loss function:

```text
L_total = 0.6 × L_segmentation + 0.4 × L_classification
```

For BraTS segmentation-only experiments, only the segmentation loss is used.

---

## Citation

If you use this code, please cite the associated manuscript:

```text
[Moinul Hossain], "DenseUNet-SC: A Densely Connected Deep Learning Method for Joint Brain Tumor Segmentation and Classification," Healthcare Analytics.
```

---

## Contact

For questions regarding the code or manuscript, please contact the corresponding author.

```
```
