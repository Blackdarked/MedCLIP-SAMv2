import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import Paths
from datasets import load_breast, load_brain, load_xray, load_ct
from evaluate import dice, load_predictions_from_dir

# ─────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────

_DATASETS = {
    'breast': {
        'loader':   load_breast,
        'args':     lambda: {'breast_dir': Paths.BREAST_DIR},
        'pred_dir': lambda: os.path.join(Paths.PRED_DIR, 'breast'),
        'zs_pkl':   lambda: os.path.join(Paths.MASK_DIR, 'breast', 'zero_shot.pkl'),
    },
    'brain': {
        'loader':   load_brain,
        'args':     lambda: {'brain_dir': Paths.BRAIN_DIR},
        'pred_dir': lambda: os.path.join(Paths.PRED_DIR, 'brain'),
        'zs_pkl':   lambda: os.path.join(Paths.MASK_DIR, 'brain', 'zero_shot.pkl'),
    },
    'xray': {
        'loader':   load_xray,
        'args':     lambda: {'xray_dir': Paths.XRAY_DIR},
        'pred_dir': lambda: os.path.join(Paths.PRED_DIR, 'lungxray'),
        'zs_pkl':   lambda: os.path.join(Paths.MASK_DIR, 'xray', 'zero_shot.pkl'),
    },
    'ct': {
        'loader':   load_ct,
        'args':     lambda: {'ct_dir': Paths.CT_DIR},
        'pred_dir': lambda: os.path.join(Paths.PRED_DIR, 'lungct'),
        'zs_pkl':   lambda: os.path.join(Paths.MASK_DIR, 'ct', 'zero_shot.pkl'),
    },
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def overlay(image_np: np.ndarray, mask: np.ndarray,
            color: tuple, alpha: float = 0.45) -> np.ndarray:
    out = image_np.astype(np.float32).copy()
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask > 0,
                                out[:, :, c] * (1 - alpha) + v * alpha,
                                out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def _resize_mask(mask: np.ndarray, target_hw: tuple) -> np.ndarray:
    h, w = target_hw
    return np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    ) > 127


def save_sample(out_path: str, image_np: np.ndarray,
                gt_mask: np.ndarray, zs_mask, pred_mask: np.ndarray,
                dsc: float, idx: int):
    h, w = image_np.shape[:2]
    gt_r   = _resize_mask(gt_mask,   (h, w))
    pred_r = _resize_mask(pred_mask, (h, w))

    panels = [
        (image_np,                                          'Input'),
        (overlay(image_np, gt_r,   (0, 200, 0)),           'Ground Truth'),
        (None,                                              'Zero-shot'),
        (overlay(image_np, pred_r, (220, 50, 50)),          f'nnUNet  DSC={dsc*100:.1f}%'),
    ]

    if zs_mask is not None:
        zs_r = _resize_mask(zs_mask, (h, w))
        panels[2] = (overlay(image_np, zs_r, (255, 200, 0)), 'Zero-shot')

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (panel_img, title) in zip(axes, panels):
        if panel_img is None:
            ax.text(0.5, 0.5, 'not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')
            ax.set_facecolor('#f0f0f0')
        else:
            ax.imshow(panel_img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    fig.suptitle(f'Sample {idx:04d}', fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def save_summary_grid(samples: list, out_path: str, title: str):
    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    for row, (dsc, idx, image_np, gt_mask, zs_mask, pred_mask) in enumerate(samples):
        h, w = image_np.shape[:2]
        gt_r   = _resize_mask(gt_mask,   (h, w))
        pred_r = _resize_mask(pred_mask, (h, w))

        panels = [
            (image_np,                             f'#{idx:04d} Input'),
            (overlay(image_np, gt_r, (0, 200, 0)), 'Ground Truth'),
            (None,                                 'Zero-shot'),
            (overlay(image_np, pred_r, (220,50,50)), f'DSC={dsc*100:.1f}%'),
        ]
        if zs_mask is not None:
            zs_r = _resize_mask(zs_mask, (h, w))
            panels[2] = (overlay(image_np, zs_r, (255, 200, 0)), 'Zero-shot')

        for col, (panel_img, col_title) in enumerate(panels):
            ax = axes[row][col]
            if panel_img is None:
                ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='gray')
                ax.set_facecolor('#f0f0f0')
            else:
                ax.imshow(panel_img)
            ax.set_title(col_title, fontsize=8)
            ax.axis('off')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ─────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────

def visualize_dataset(ds_name: str, out_root: str, max_n: int):
    info     = _DATASETS[ds_name]
    pred_dir = info['pred_dir']()
    zs_path  = info['zs_pkl']()

    if not os.path.isdir(pred_dir) or not any(
        f.endswith('.png') for f in os.listdir(pred_dir)
    ):
        print(f'[{ds_name}] No predictions found in {pred_dir} — skipping.')
        return

    print(f'\n[{ds_name}] Loading dataset...')
    _, _, test_pairs = info['loader'](**info['args']())
    pred_masks, gt_masks = load_predictions_from_dir(pred_dir, test_pairs)

    if not pred_masks:
        print(f'[{ds_name}] No matched predictions — skipping.')
        return

    zs_dict = {}
    if os.path.exists(zs_path):
        with open(zs_path, 'rb') as f:
            zs_dict = pickle.load(f)
        print(f'[{ds_name}] Zero-shot pkl loaded ({len(zs_dict)} entries)')

    out_dir = os.path.join(out_root, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    n = min(len(pred_masks), max_n) if max_n else len(pred_masks)
    print(f'[{ds_name}] Rendering {n} samples...')

    summary_data = []
    pred_idx = 0
    for i, (img_path, mask_path, _) in enumerate(test_pairs):
        if pred_idx >= n:
            break
        pred_path = os.path.join(pred_dir, f'test_{i+1:04d}.png')
        if not os.path.exists(pred_path):
            continue
        if mask_path is None or not os.path.exists(mask_path):
            continue

        image_np  = np.array(Image.open(img_path).convert('RGB'))
        pred_mask = pred_masks[pred_idx]
        gt_mask   = gt_masks[pred_idx]
        zs_mask   = zs_dict.get(img_path)
        dsc       = dice(pred_mask, gt_mask)

        out_path = os.path.join(out_dir, f'sample_{i+1:04d}.png')
        save_sample(out_path, image_np, gt_mask, zs_mask, pred_mask, dsc, i + 1)
        summary_data.append((dsc, i + 1, image_np, gt_mask, zs_mask, pred_mask))
        pred_idx += 1

    print(f'[{ds_name}] {pred_idx} sample PNGs saved to {out_dir}')

    if summary_data:
        by_dsc = sorted(summary_data, key=lambda x: x[0])
        worst  = by_dsc[:min(10, len(by_dsc))]
        best   = by_dsc[-min(10, len(by_dsc)):][::-1]
        save_summary_grid(worst, os.path.join(out_dir, 'summary_worst.png'),
                          f'{ds_name} — 10 worst by DSC')
        save_summary_grid(best,  os.path.join(out_dir, 'summary_best.png'),
                          f'{ds_name} — 10 best by DSC')


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize MedCLIP-SAMv2 predictions')
    parser.add_argument('--dataset', choices=['breast', 'brain', 'xray', 'ct', 'all'],
                        default='all')
    parser.add_argument('--n',   type=int, default=0,
                        help='Max samples per dataset (0 = all)')
    parser.add_argument('--out', type=str,
                        default=os.path.join(Paths.WORK_DIR, 'visualizations'))
    args = parser.parse_args()

    datasets = list(_DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    max_n    = args.n if args.n > 0 else None

    os.makedirs(args.out, exist_ok=True)
    for ds in datasets:
        visualize_dataset(ds, args.out, max_n)

    print(f'\nDone. Results in: {args.out}')


if __name__ == '__main__':
    main()
