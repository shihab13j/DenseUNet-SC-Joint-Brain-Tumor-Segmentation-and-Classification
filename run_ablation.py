import argparse
import json

import torch
import torch.nn as nn

from train_denseunetsc import (
    CFG, OUT_DIR, ConvBlock, UNetDecoder, ClassificationHead, DenseUNetSC,
    load_figshare, get_dataloaders, set_seed, MultiTaskLoss, train_model,
    evaluate_model, measure_inference_time
)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class PlainConvEncoder(nn.Module):
    """
    Non-dense encoder ablation.
    Output channel sizes are matched to DenseNetEncoder's decoder interface so that
    the decoder and classifier heads remain comparable.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = ConvBlock(in_channels, 64)       # 128x128
        self.enc1 = DownBlock(64, 192)               # 64x64
        self.enc2 = DownBlock(192, 288)              # 32x32
        self.enc3 = DownBlock(288, 400)              # 16x16
        self.bottleneck = DownBlock(400, 328)        # 8x8
        self.skip_channels = {'skip1': 64, 'skip2': 192, 'skip3': 288, 'skip4': 400}
        self.bottleneck_channels = 328

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.enc1(s1)
        s3 = self.enc2(s2)
        s4 = self.enc3(s3)
        z = self.bottleneck(s4)
        return z, (s1, s2, s3, s4)


class PlainUNetSC(nn.Module):
    """DenseUNet-SC without dense connectivity; used for ablation only."""
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.encoder = PlainConvEncoder(in_channels)
        self.decoder = UNetDecoder(self.encoder.bottleneck_channels, self.encoder.skip_channels)
        self.cls_head = ClassificationHead(self.encoder.bottleneck_channels, num_classes)

    def forward(self, x):
        z, skips = self.encoder(x)
        seg = self.decoder(z, skips)
        cls = self.cls_head(z)
        return seg, cls


def make_variant(name):
    name = name.lower()
    if name in {'full_joint', 'segmentation_only', 'classification_only', 'equal_loss'}:
        return DenseUNetSC(CFG['in_channels'], CFG['num_classes'], CFG['growth_rate'], CFG['dropout'])
    if name == 'no_dense_encoder':
        return PlainUNetSC(CFG['in_channels'], CFG['num_classes'])
    raise ValueError(f'Unknown ablation variant: {name}')


def variant_task_and_loss(name):
    if name == 'segmentation_only':
        return 'segmentation', 1.0, 0.0
    if name == 'classification_only':
        return 'classification', 0.0, 1.0
    if name == 'equal_loss':
        return 'joint', 0.5, 0.5
    return 'joint', CFG['w_seg'], CFG['w_cls']


def main():
    parser = argparse.ArgumentParser(description='Run DenseUNet-SC ablation experiments')
    parser.add_argument('--data_root', required=True, help='Figshare data root')
    parser.add_argument('--variant', default='all', choices=['all', 'full_joint', 'no_dense_encoder', 'segmentation_only', 'classification_only', 'equal_loss'])
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

    train_s, val_s, test_s = load_figshare(args.data_root, seed=args.seed)
    train_dl, val_dl, test_dl = get_dataloaders(train_s, val_s, test_s, args.batch_size, args.num_workers)

    variants = ['full_joint', 'no_dense_encoder', 'segmentation_only', 'classification_only', 'equal_loss'] if args.variant == 'all' else [args.variant]
    summary = {}
    for variant in variants:
        task, w_seg, w_cls = variant_task_and_loss(variant)
        print('\n' + '=' * 70)
        print(f'  Ablation: {variant} | task={task} | w_seg={w_seg} | w_cls={w_cls}')
        print('=' * 70)
        model = make_variant(variant).to(device)
        criterion = MultiTaskLoss(w_seg=w_seg, w_cls=w_cls, task=task)
        history = train_model(model, criterion, train_dl, val_dl, device, CFG, task=task)
        results = evaluate_model(model, test_dl, device, task=task, num_classes=CFG['num_classes'])
        timing = measure_inference_time(model, test_dl, device)
        summary[variant] = {
            'task': task,
            'w_seg': w_seg,
            'w_cls': w_cls,
            'history': history,
            'results': {k: (v.tolist() if k == 'confusion_matrix' else v) for k, v in results.items()},
            'inference_time': timing,
        }

    out = OUT_DIR / 'ablation_results.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nSaved ablation results → {out}')


if __name__ == '__main__':
    main()
