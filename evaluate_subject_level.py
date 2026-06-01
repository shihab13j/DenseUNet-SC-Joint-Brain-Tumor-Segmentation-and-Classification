import argparse
import json
from pathlib import Path

import torch

from train_denseunetsc import (
    CFG, DenseUNetSC, load_figshare, load_brats2021, get_dataloaders,
    evaluate_model, measure_inference_time, set_seed
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate DenseUNet-SC checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to best_model.pth')
    parser.add_argument('--dataset', required=True, choices=['figshare', 'brats'])
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--task', default='auto', choices=['auto', 'joint', 'segmentation', 'classification'])
    parser.add_argument('--modality', default='flair', choices=['flair', 't1', 't1ce', 't2'])
    parser.add_argument('--batch_size', type=int, default=CFG['batch_size'])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=CFG['seed'])
    parser.add_argument('--out', default='outputs/evaluation_subject_level.json')
    args = parser.parse_args()

    set_seed(args.seed)
    task = 'segmentation' if (args.task == 'auto' and args.dataset == 'brats') else ('joint' if args.task == 'auto' else args.task)
    if args.dataset == 'brats' and task != 'segmentation':
        print('WARNING: BraTS classification is disabled; switching to segmentation.')
        task = 'segmentation'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'figshare':
        train_s, val_s, test_s = load_figshare(args.data_root, seed=args.seed)
    else:
        train_s, val_s, test_s = load_brats2021(args.data_root, modality=args.modality, seed=args.seed)
    _, _, test_dl = get_dataloaders(train_s, val_s, test_s, args.batch_size, args.num_workers)

    model = DenseUNetSC(CFG['in_channels'], CFG['num_classes'], CFG['growth_rate'], CFG['dropout']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    model.load_state_dict(state)

    results = evaluate_model(model, test_dl, device, task=task, num_classes=CFG['num_classes'])
    timing = measure_inference_time(model, test_dl, device)
    payload = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'task': task,
        'results': {k: (v.tolist() if k == 'confusion_matrix' else v) for k, v in results.items()},
        'inference_time': timing,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f'Saved evaluation → {out}')


if __name__ == '__main__':
    main()
