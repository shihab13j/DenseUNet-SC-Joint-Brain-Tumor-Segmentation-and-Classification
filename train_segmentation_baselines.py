

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_denseunetsc import (
    CFG, OUT_DIR, ConvBlock, DenseUNetSC, load_figshare, load_brats2021,
    get_dataloaders, set_seed, MultiTaskLoss, train_model, evaluate_model,
    measure_inference_time
)


class UNetBaseline(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())
        self.num_classes = num_classes

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg = self.out(d1)
        dummy_logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
        return seg, dummy_logits


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=True), nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=True), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha


class AttentionUNetBaseline(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, max(base // 2, 1))
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())
        self.num_classes = num_classes

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        u4 = self.up4(b); a4 = self.att4(u4, e4)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))
        u3 = self.up3(d4); a3 = self.att3(u3, e3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))
        u2 = self.up2(d3); a2 = self.att2(u2, e2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))
        u1 = self.up1(d2); a1 = self.att1(u1, e1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))
        seg = self.out(d1)
        dummy_logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
        return seg, dummy_logits


def make_segmentation_model(name):
    name = name.lower()
    if name == 'unet':
        return UNetBaseline(CFG['in_channels'], CFG['num_classes'])
    if name == 'attention_unet':
        return AttentionUNetBaseline(CFG['in_channels'], CFG['num_classes'])
    if name == 'denseunet':
        # DenseUNet-SC architecture trained/evaluated in segmentation-only mode.
        return DenseUNetSC(CFG['in_channels'], CFG['num_classes'], CFG['growth_rate'], CFG['dropout'])
    raise ValueError(f'Unknown segmentation baseline: {name}')


def main():
    parser = argparse.ArgumentParser(description='Train segmentation baselines')
    parser.add_argument('--dataset', required=True, choices=['figshare', 'brats'])
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', default='all', choices=['all', 'unet', 'attention_unet', 'denseunet'])
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'figshare':
        train_s, val_s, test_s = load_figshare(args.data_root, seed=args.seed)
    else:
        train_s, val_s, test_s = load_brats2021(args.data_root, modality=args.modality, seed=args.seed)
    train_dl, val_dl, test_dl = get_dataloaders(train_s, val_s, test_s, args.batch_size, args.num_workers)

    names = ['unet', 'attention_unet', 'denseunet'] if args.model == 'all' else [args.model]
    summary = {}
    for name in names:
        print('\n' + '=' * 70)
        print(f'  Training segmentation baseline: {name}')
        print('=' * 70)
        model = make_segmentation_model(name).to(device)
        criterion = MultiTaskLoss(task='segmentation')
        history = train_model(model, criterion, train_dl, val_dl, device, CFG, task='segmentation')
        results = evaluate_model(model, test_dl, device, task='segmentation', num_classes=CFG['num_classes'])
        timing = measure_inference_time(model, test_dl, device)
        summary[name] = {
            'history': history,
            'results': results,
            'inference_time': timing,
        }

    out = OUT_DIR / 'segmentation_baselines_results.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nSaved segmentation baseline results → {out}')


if __name__ == '__main__':
    main()
