"""
Standalone Classification Baselines for Comparison (Paper Table — standalone_classifiers)

Models:
  - ResNet-50   (standalone, classification only)
  - EfficientNet-B0 (standalone, classification only)
  - DenseNet-121    (standalone, classification only)

All trained on the same dataset split and evaluated at the subject level,
exactly as described in the paper's Discussion of High Classification Performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  ResNet-50 Baseline (standalone)
# ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """ResNet bottleneck block (He et al., 2016 — Paper Ref [15])."""
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(mid_ch)
        self.conv2  = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                                padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(mid_ch)
        self.conv3  = nn.Conv2d(mid_ch, mid_ch * self.expansion, 1, bias=False)
        self.bn3    = nn.BatchNorm2d(mid_ch * self.expansion)
        self.down   = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.down: identity = self.down(x)
        return F.relu(out + identity)


class ResNet50Classifier(nn.Module):
    """ResNet-50 adapted for single-channel 128×128 input, 3-class output."""
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Layers (3,4,6,3 blocks — ResNet-50)
        self.layer1 = self._make_layer(64,   64,  3, stride=1)
        self.layer2 = self._make_layer(256,  128, 4, stride=2)
        self.layer3 = self._make_layer(512,  256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(2048, num_classes)

    def _make_layer(self, in_ch, mid_ch, n_blocks, stride):
        down = None
        out_ch = mid_ch * ResidualBlock.expansion
        if stride != 1 or in_ch != out_ch:
            down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        layers = [ResidualBlock(in_ch, mid_ch, stride, down)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, mid_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────
#  EfficientNet-B0 Baseline (simplified)
# ─────────────────────────────────────────────

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv block (EfficientNet building block)."""
    def __init__(self, in_ch, out_ch, expand=6, stride=1, kernel=3):
        super().__init__()
        mid = in_ch * expand
        pad = kernel // 2
        self.use_skip = (stride == 1 and in_ch == out_ch)
        self.block = nn.Sequential(
            # Expand
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(),
            # Depthwise
            nn.Conv2d(mid, mid, kernel, stride=stride,
                      padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(),
            # Project
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_skip else out


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 adapted for 1-channel 128×128 input, 3-class output."""
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
        )
        # Simplified B0 stage configuration
        self.blocks = nn.Sequential(
            MBConv(32,  16,  expand=1, stride=1),
            MBConv(16,  24,  expand=6, stride=2),
            MBConv(24,  24,  expand=6, stride=1),
            MBConv(24,  40,  expand=6, stride=2),
            MBConv(40,  40,  expand=6, stride=1),
            MBConv(40,  80,  expand=6, stride=2),
            MBConv(80,  80,  expand=6, stride=1),
            MBConv(80,  112, expand=6, stride=1),
            MBConv(112, 112, expand=6, stride=1),
            MBConv(112, 192, expand=6, stride=2),
            MBConv(192, 192, expand=6, stride=1),
            MBConv(192, 320, expand=6, stride=1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────
#  DenseNet-121 Baseline (standalone)
# ─────────────────────────────────────────────

class _DenseLayer121(nn.Module):
    def __init__(self, in_ch, growth_rate=32):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 4*growth_rate, 1, bias=False),
            nn.BatchNorm2d(4*growth_rate), nn.ReLU(inplace=True),
            nn.Conv2d(4*growth_rate, growth_rate, 3, padding=1, bias=False),
        )
    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class _DenseBlock121(nn.Module):
    def __init__(self, in_ch, n_layers, g=32):
        super().__init__()
        self.layers = nn.ModuleList()
        c = in_ch
        for _ in range(n_layers):
            self.layers.append(_DenseLayer121(c, g))
            c += g
        self.out_channels = c
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x


class _Transition121(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2, stride=2),
        )
        self.out_channels = out_ch
    def forward(self, x): return self.block(x)


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 (Huang et al., 2017 — Paper Ref [21]) standalone classifier.
    Adapted for 1-channel 128×128 input, 3-class output.
    Uses same growth_rate=32 and block depths (6,12,24,16) as original.
    """
    def __init__(self, in_channels=1, num_classes=3, growth_rate=32):
        super().__init__()
        g = growth_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.db1 = _DenseBlock121(64,  6,  g)   # out=64+6*32=256
        self.tr1 = _Transition121(self.db1.out_channels, 128)
        self.db2 = _DenseBlock121(128, 12, g)   # out=128+12*32=512
        self.tr2 = _Transition121(self.db2.out_channels, 256)
        self.db3 = _DenseBlock121(256, 24, g)   # out=256+24*32=1024
        self.tr3 = _Transition121(self.db3.out_channels, 512)
        self.db4 = _DenseBlock121(512, 16, g)   # out=512+16*32=1024

        self.head = nn.Sequential(
            nn.BatchNorm2d(self.db4.out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(self.db4.out_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.tr1(self.db1(x))
        x = self.tr2(self.db2(x))
        x = self.tr3(self.db3(x))
        x = self.db4(x)
        x = self.head(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────
#  Quick verification
# ─────────────────────────────────────────────

if __name__ == '__main__':
    x = torch.randn(2, 1, 128, 128)
    models = {
        'ResNet-50':      ResNet50Classifier(),
        'EfficientNet-B0': EfficientNetB0Classifier(),
        'DenseNet-121':   DenseNet121Classifier(),
    }
    print("=" * 55)
    print("  Standalone Classifier Baselines — Verification")
    print("=" * 55)
    for name, model in models.items():
        out    = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:<20} out={list(out.shape)}  "
              f"params={params:,}")
    print("  ✓ All standalone baselines verified")
    print("=" * 55)
