
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from train_denseunetsc import (
    CFG, OUT_DIR, load_figshare, get_dataloaders, set_seed,
    MultiTaskLoss, train_model, evaluate_model, measure_inference_time
)


class ClassificationWrapper(nn.Module):
    """Wrap a classifier so it matches the DenseUNet-SC output signature."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        logits = self.backbone(x)
        dummy_seg = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return dummy_seg, logits


def _replace_first_conv_for_grayscale(model, name):
    if name == 'resnet50':
        old = model.conv1
        model.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                                stride=old.stride, padding=old.padding, bias=False)
    elif name == 'densenet121':
        old = model.features.conv0
        model.features.conv0 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                                         stride=old.stride, padding=old.padding, bias=False)
    elif name == 'efficientnet_b0':
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                                         stride=old.stride, padding=old.padding, bias=False)
    return model


def make_classifier(name, num_classes=3):
    name = name.lower()
    if name == 'resnet50':
        model = models.resnet50(weights=None)
        model = _replace_first_conv_for_grayscale(model, name)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'densenet121':
        model = models.densenet121(weights=None)
        model = _replace_first_conv_for_grayscale(model, name)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model = _replace_first_conv_for_grayscale(model, name)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f'Unknown baseline: {name}')
    return ClassificationWrapper(model)


def main():
    parser = argparse.ArgumentParser(description='Train classification baselines on Figshare')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', default='all', choices=['all', 'resnet50', 'densenet121', 'efficientnet_b0'])
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

    names = ['resnet50', 'densenet121', 'efficientnet_b0'] if args.model == 'all' else [args.model]
    summary = {}
    for name in names:
        print('\n' + '=' * 70)
        print(f'  Training classification baseline: {name}')
        print('=' * 70)
        model = make_classifier(name, CFG['num_classes']).to(device)
        criterion = MultiTaskLoss(task='classification')
        history = train_model(model, criterion, train_dl, val_dl, device, CFG, task='classification')
        results = evaluate_model(model, test_dl, device, task='classification', num_classes=CFG['num_classes'])
        timing = measure_inference_time(model, test_dl, device)
        summary[name] = {
            'history': history,
            'results': {k: (v.tolist() if k == 'confusion_matrix' else v) for k, v in results.items()},
            'inference_time': timing,
        }

    out = OUT_DIR / 'classification_baselines_results.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nSaved classification baseline results → {out}')


if __name__ == '__main__':
    main()
