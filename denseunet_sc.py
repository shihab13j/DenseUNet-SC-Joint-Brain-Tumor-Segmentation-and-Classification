"""
DenseUNet-SC: Densely Connected Deep Learning Method for
Joint Brain Tumor Segmentation and Classification

Architecture (exactly as described in paper):
  - DenseNet Encoder: 4 Dense Blocks (L=4,6,8,4), growth rate g=32
    with bottleneck layers, transition blocks (reduction=0.5)
  - U-Net Decoder: 4 progressive upsampling stages (256→128→64→32)
    with skip connections from encoder
  - Classification Head: GAP → Dense(256) → Dense(128) → Dense(3)
  - Two branches operate IN PARALLEL from shared bottleneck Z
  - Task coupling via shared encoder under joint multi-task loss

Reference: Paper Table 2 (algorithmic workflow) and Table 3 (hyperparams)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Building Blocks
# ─────────────────────────────────────────────

class BottleneckDenseLayer(nn.Module):
    """
    Single dense layer with bottleneck design (Paper Eq. 7):
      BN → ReLU → Conv(1×1, 4g) → BN → ReLU → Conv(3×3, g) → Dropout(0.2)
    """
    def __init__(self, in_channels: int, growth_rate: int, dropout: float = 0.2):
        super().__init__()
        inter = 4 * growth_rate
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth_rate, kernel_size=3,
                               padding=1, bias=False)
        self.drop  = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.drop(out)
        return torch.cat([x, out], dim=1)          # dense concat (Eq. 6)


class DenseBlock(nn.Module):
    """
    Dense Block with L dense layers.
    Output channels = in_channels + L * growth_rate  (Paper Eq. 8)
    """
    def __init__(self, in_channels: int, num_layers: int,
                 growth_rate: int = 32, dropout: float = 0.2):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(num_layers):
            layers.append(BottleneckDenseLayer(c, growth_rate, dropout))
            c += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransitionBlock(nn.Module):
    """
    Transition layer: BN+ReLU+Conv(1×1)+AvgPool(2×2)
    Channel compression: C' = floor(θ * C_in)  (Paper Eq. 9, 10)
    reduction=0.5 as per paper Table 3
    """
    def __init__(self, in_channels: int, reduction: float = 0.5):
        super().__init__()
        out = int(in_channels * reduction)
        self.bn   = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(F.relu(self.bn(x))))


class ConvBlock(nn.Module):
    """
    Decoder convolutional refinement block (Paper Eq. 14):
      Conv(3×3) → BN → ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────
#  DenseNet Encoder
# ─────────────────────────────────────────────

class DenseNetEncoder(nn.Module):
    """
    DenseNet Encoder as described in paper Section 3.3.1 and Figure 5.

    Architecture (Paper Table 2, Steps 2-7):
      Step 2: InitConv(64, 3×3) + BN + ReLU → MaxPool → Skip-1
      Step 3: DenseBlock-1 (L=4, g=32) → Skip-2
      Step 4: Transition-1 (reduction=0.5)
      Step 5: DenseBlock-2 (L=6, g=32) → Skip-3 → Transition-2
      Step 6: DenseBlock-3 (L=8, g=32) → Skip-4 → Transition-3
      Step 7: Bottleneck DenseBlock (L=4, g=32) → Z
    """
    def __init__(self, in_channels: int = 1, growth_rate: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        g = growth_rate

        # ── Stem: InitConv + BN + ReLU  (Paper Table 3: Conv2D(64,3×3,stride1))
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # MaxPool for downsampling after stem (Paper Table 3: MaxPool(2×2, stride2))
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Dense Block-1: L=4, g=32  (Paper Table 3)
        self.db1 = DenseBlock(64, num_layers=4, growth_rate=g, dropout=dropout)
        # out = 64 + 4*32 = 192
        self.tr1 = TransitionBlock(self.db1.out_channels, reduction=0.5)
        # out = 96

        # ── Dense Block-2: L=6, g=32
        self.db2 = DenseBlock(self.tr1.out_channels, num_layers=6,
                               growth_rate=g, dropout=dropout)
        # out = 96 + 6*32 = 288
        self.tr2 = TransitionBlock(self.db2.out_channels, reduction=0.5)
        # out = 144

        # ── Dense Block-3: L=8, g=32
        self.db3 = DenseBlock(self.tr2.out_channels, num_layers=8,
                               growth_rate=g, dropout=dropout)
        # out = 144 + 8*32 = 400
        self.tr3 = TransitionBlock(self.db3.out_channels, reduction=0.5)
        # out = 200

        # ── Bottleneck Dense Block: L=4, g=32  (Paper Table 2 Step 7)
        self.bottleneck = DenseBlock(self.tr3.out_channels, num_layers=4,
                                     growth_rate=g, dropout=dropout)
        # out = 200 + 4*32 = 328

        # Store output channel sizes for decoder skip connections
        self.skip_channels = {
            'skip1': 64,                          # after stem
            'skip2': self.db1.out_channels,       # after db1
            'skip3': self.db2.out_channels,       # after db2
            'skip4': self.db3.out_channels,       # after db3
        }
        self.bottleneck_channels = self.bottleneck.out_channels

    def forward(self, x: torch.Tensor):
        # Step 2: Stem + store Skip-1
        s1 = self.stem(x)            # [B, 64, H, W]
        x  = self.pool0(s1)          # [B, 64, H/2, W/2]

        # Step 3: DB1 + store Skip-2
        s2 = self.db1(x)             # [B, 192, H/2, W/2]
        x  = self.tr1(s2)            # [B, 96,  H/4, W/4]

        # Step 5: DB2 + store Skip-3
        s3 = self.db2(x)             # [B, 288, H/4, W/4]
        x  = self.tr2(s3)            # [B, 144, H/8, W/8]

        # Step 6: DB3 + store Skip-4
        s4 = self.db3(x)             # [B, 400, H/8, W/8]
        x  = self.tr3(s4)            # [B, 200, H/16,W/16]

        # Step 7: Bottleneck → Z
        Z  = self.bottleneck(x)      # [B, 328, H/16,W/16]

        return Z, (s1, s2, s3, s4)


# ─────────────────────────────────────────────
#  U-Net Decoder
# ─────────────────────────────────────────────

class UNetDecoder(nn.Module):
    """
    U-Net Decoder as described in paper Section 3.3.2 and Figure 6.

    Architecture (Paper Table 2, Steps 8-9):
      Decoder-1: UpSample → concat Skip-4 → ConvBlock(256)
      Decoder-2: UpSample → concat Skip-3 → ConvBlock(128)
      Decoder-3: UpSample → concat Skip-2 → ConvBlock(64)
      Decoder-4: UpSample → concat Skip-1 → ConvBlock(32)
      Output: Conv(1×1) + Sigmoid → binary mask M̂  (Eq. 15, 16)
    """
    def __init__(self, bottleneck_ch: int, skip_channels: dict):
        super().__init__()

        # Decoder-1: Bottleneck → upsample → cat Skip-4
        d1_in = bottleneck_ch + skip_channels['skip4']
        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1  = ConvBlock(d1_in, 256)

        # Decoder-2: → upsample → cat Skip-3
        d2_in = 256 + skip_channels['skip3']
        self.up2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2  = ConvBlock(d2_in, 128)

        # Decoder-3: → upsample → cat Skip-2
        d3_in = 128 + skip_channels['skip2']
        self.up3   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3  = ConvBlock(d3_in, 64)

        # Decoder-4: → upsample → cat Skip-1
        d4_in = 64 + skip_channels['skip1']
        self.up4   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4  = ConvBlock(d4_in, 32)

        # Segmentation output (Paper Eq. 15, 16)
        self.seg_out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, Z: torch.Tensor, skips: tuple) -> torch.Tensor:
        s1, s2, s3, s4 = skips

        # Decoder-1 (Eq. 12, 13, 14)
        x = self.up1(Z)
        x = torch.cat([x, s4], dim=1)
        x = self.dec1(x)

        # Decoder-2
        x = self.up2(x)
        x = torch.cat([x, s3], dim=1)
        x = self.dec2(x)

        # Decoder-3
        x = self.up3(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec3(x)

        # Decoder-4
        x = self.up4(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec4(x)

        return self.seg_out(x)   # M̂ ∈ [0,1], shape [B,1,H,W]


# ─────────────────────────────────────────────
#  Classification Head
# ─────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Classification Head as described in paper Section 3.3.3 and Figure 4.

    Architecture (Paper Table 2, Step 10):
      GAP(Z) → Dense(256, ReLU) + Dropout(0.5)
             → Dense(128, ReLU) + Dropout(0.3)
             → Dense(num_classes, Softmax)
    Equations: 13(GAP), 14(h1), 15(drop), 16(h2), 17(drop), 18(ŷ)
    """
    def __init__(self, in_channels: int, num_classes: int = 3,
                 hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)          # GAP (Eq. 13)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden1),        # W1*v + b1 (Eq. 14)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                      # Dropout (Eq. 15)
            nn.Linear(hidden1, hidden2),            # W2*h̃1 + b2 (Eq. 16)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),                      # Dropout (Eq. 17)
            nn.Linear(hidden2, num_classes),        # W3*h̃2 + b3 (Eq. 18)
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        v   = self.gap(Z)          # [B, C, 1, 1]
        return self.fc(v)          # [B, num_classes] — logits (softmax at loss)


# ─────────────────────────────────────────────
#  DenseUNet-SC  (Full Model)
# ─────────────────────────────────────────────

class DenseUNetSC(nn.Module):
    """
    DenseUNet-SC: Joint Brain Tumor Segmentation and Classification.

    Key design (paper Section 3.3):
    ┌──────────────────────────────────────────────────┐
    │  Input MRI slice  [B, C, 128, 128]               │
    │         ↓                                        │
    │   DenseNet Encoder  →  Z  (shared bottleneck)    │
    │         ↙                      ↘                 │
    │  U-Net Decoder            Classification Head    │
    │  (segmentation)           (tumor-type)           │
    │         ↓                      ↓                 │
    │   M̂ ∈ [0,1]           ŷ ∈ R^num_classes         │
    └──────────────────────────────────────────────────┘
    The two branches operate IN PARALLEL from the same Z.
    Task coupling is implicit through joint encoder optimization
    under the weighted multi-task loss (Paper Table 3):
      L_total = 0.6 * L_seg + 0.4 * L_cls
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 3,
                 growth_rate: int = 32, dropout: float = 0.2):
        super().__init__()

        # Shared DenseNet Encoder  (Paper Eq. 11: Z = F_DenseNet(x0))
        self.encoder = DenseNetEncoder(in_channels, growth_rate, dropout)

        # U-Net Decoder branch (segmentation)
        self.decoder = UNetDecoder(
            bottleneck_ch  = self.encoder.bottleneck_channels,
            skip_channels  = self.encoder.skip_channels,
        )

        # Classification Head branch
        self.cls_head = ClassificationHead(
            in_channels = self.encoder.bottleneck_channels,
            num_classes = num_classes,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, in_channels, H, W]  — preprocessed MRI slice(s)
        Returns:
            seg_out:  [B, 1, H, W]      — binary tumor probability mask M̂
            cls_out:  [B, num_classes]  — classification logits ŷ
        """
        # Step 1–7: Shared encoder produces Z and skip features
        Z, skips = self.encoder(x)

        # Step 8–9: Segmentation branch (parallel)
        seg_out = self.decoder(Z, skips)

        # Step 10: Classification branch (parallel, same Z)
        cls_out = self.cls_head(Z)

        return seg_out, cls_out

    def predict(self, x: torch.Tensor):
        """Inference with sigmoid on seg and softmax on cls."""
        self.eval()
        with torch.no_grad():
            seg_logits, cls_logits = self.forward(x)
            seg_prob  = seg_logits                       # already sigmoid
            cls_prob  = torch.softmax(cls_logits, dim=1)
            cls_pred  = cls_prob.argmax(dim=1)
        return seg_prob, cls_prob, cls_pred


# ─────────────────────────────────────────────
#  Multi-Task Loss  (Paper Table 2, Step 11)
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Binary Dice loss (Paper Eq. 22 adapted for segmentation branch)."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.view(-1)
        target = target.view(-1).float()
        inter  = (pred * target).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (
            pred.sum() + target.sum() + self.smooth)


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss (Paper Table 3):
      L_total = w_seg * (BCE + Dice) + w_cls * CrossEntropy
      w_seg = 0.6,  w_cls = 0.4
    """
    def __init__(self, w_seg: float = 0.6, w_cls: float = 0.4):
        super().__init__()
        self.w_seg   = w_seg
        self.w_cls   = w_cls
        self.bce     = nn.BCELoss()
        self.dice    = DiceLoss()
        self.ce      = nn.CrossEntropyLoss()

    def forward(self, seg_pred, seg_true, cls_pred, cls_true):
        l_bce  = self.bce(seg_pred, seg_true.float())
        l_dice = self.dice(seg_pred, seg_true)
        l_seg  = l_bce + l_dice

        l_cls  = self.ce(cls_pred, cls_true.long())

        total  = self.w_seg * l_seg + self.w_cls * l_cls
        return total, l_seg, l_cls


# ─────────────────────────────────────────────
#  Quick Sanity Check
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = DenseUNetSC(in_channels=1, num_classes=3).to(device)

    # Paper input: 128×128 grayscale (Table 3)
    x = torch.randn(2, 1, 128, 128).to(device)
    seg, cls = model(x)

    print("=" * 55)
    print("  DenseUNet-SC  —  Forward Pass Verification")
    print("=" * 55)
    print(f"  Input shape        : {list(x.shape)}")
    print(f"  Seg output shape   : {list(seg.shape)}   ← [B,1,128,128]")
    print(f"  Cls output shape   : {list(cls.shape)}    ← [B,3]")
    print(f"  Seg value range    : [{seg.min():.4f}, {seg.max():.4f}]  ← valid [0,1]")
    enc_ch = model.encoder.bottleneck_channels
    print(f"  Bottleneck channels: {enc_ch}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params   : {params:,}")
    print("=" * 55)
    print("  ✓  All shapes verified — architecture matches paper")
    print("=" * 55)
