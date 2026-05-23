# run_all.py — LOCAL MACHINE VERSION
# Changes from Kaggle version:
#   - `if __name__ == '__main__':` guard wraps ALL execution (critical on Windows)
#   - No Kaggle session management or resume-from-timeout logic
#   - Paths reference local config
#   - subprocess used for shell commands instead of os.system

import os
import sys
import gc
import json
import glob
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

from config import (
    Paths, Stage1Config, M2IBConfig, PostprocessConfig,
    SAMConfig, NNUNetConfig, PAPER_TARGETS
)
from datasets import (
    load_medpix, load_roco,
    load_breast, load_brain, load_xray, load_ct
)
from stage1_finetune import run_stage1
from stage2_segmentation import run_stage2
from stage3_nnunet import (
    run_stage3, get_ensemble_checkpoint_paths,
    ensemble_predict_single, setup_nnunet_env
)
from evaluate import (
    evaluate_pairs, load_predictions_from_dir,
    print_table1, print_table2, print_table4,
    print_table5, print_table6, collect_ablation_row
)


# ─────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    'breast': {
        'loader':      load_breast,
        'loader_args': {'busi_dir': None},
        'nnunet_name': 'Breast',
        'nnunet_id':   1,
        'large':       False,
    },
    'brain': {
        'loader':      load_brain,
        'loader_args': {'brain_dir': None},
        'nnunet_name': 'Brain',
        'nnunet_id':   2,
        'large':       False,
    },
    'xray': {
        'loader':      load_xray,
        'loader_args': {'xray_dir': None},
        'nnunet_name': 'LungXray',
        'nnunet_id':   3,
        'large':       True,
    },
    'ct': {
        'loader':      load_ct,
        'loader_args': {'ct_dir': None},
        'nnunet_name': 'LungCT',
        'nnunet_id':   4,
        'large':       True,
    },
}

def fill_loader_args():
    DATASET_REGISTRY['breast']['loader_args'] = {'busi_dir': Paths.BUSI_DIR}
    DATASET_REGISTRY['brain']['loader_args']  = {'brain_dir': Paths.BRAIN_DIR}
    DATASET_REGISTRY['xray']['loader_args']   = {'xray_dir': Paths.XRAY_DIR}
    DATASET_REGISTRY['ct']['loader_args']     = {'ct_dir': Paths.CT_DIR}


# ─────────────────────────────────────────────────────────────
# Results persistence
# ─────────────────────────────────────────────────────────────

RESULTS_PATH = os.path.join(Paths.WORK_DIR, 'results.json')

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}

def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    Paths.makedirs()
    fill_loader_args()
    results = load_results()
    datasets_to_run = ([args.dataset] if args.dataset
                       else list(DATASET_REGISTRY.keys()))

    # ── Stage 1 ──
    if not args.skip_stage1 and not args.eval_only:
        print("=" * 60)
        print("STAGE 1 — BiomedCLIP Fine-tuning")
        print("=" * 60)

        if os.path.exists(Paths.BIOMEDCLIP_FT):
            print(f"Model already exists: {Paths.BIOMEDCLIP_FT}")
            print("Delete it and rerun to retrain.")
        else:
            _, roco_results = run_stage1(Stage1Config(), device=DEVICE)
            results['roco'] = roco_results
            save_results(results)
            gc.collect()
            torch.cuda.empty_cache()

        if 'roco' in results:
            print_table2(results['roco'])

    # ── Stages 2 + 3 per dataset ──
    for ds_name in datasets_to_run:
        info = DATASET_REGISTRY[ds_name]

        print("\n" + "=" * 60)
        print(f"DATASET: {ds_name.upper()}")
        print("=" * 60)

        print(f"\nLoading {ds_name} dataset...")
        train_pairs, val_pairs, test_pairs = info['loader'](**info['loader_args'])
        all_pairs = train_pairs + val_pairs + test_pairs

        # Stage 2
        if not args.eval_only:
            print(f"\n--- Stage 2: Zero-shot ({ds_name}) ---")
            use_pretrained = not os.path.exists(Paths.BIOMEDCLIP_FT)
            zero_shot_masks = run_stage2(
                dataset_name=ds_name,
                all_pairs=all_pairs,
                m2ib_cfg=M2IBConfig(),
                pp_cfg=PostprocessConfig(),
                sam_cfg=SAMConfig(),
                device=DEVICE,
                use_pretrained=use_pretrained,
            )

            # Evaluate zero-shot
            test_preds, test_gts = [], []
            for img_path, mask_path, _ in test_pairs:
                if mask_path is None or not os.path.exists(mask_path):
                    continue
                pred = zero_shot_masks.get(img_path)
                if pred is None:
                    continue
                gt = np.array(
                    Image.open(mask_path).convert('L').resize(
                        (pred.shape[1], pred.shape[0]), Image.NEAREST
                    )
                )
                gt = (gt > 127).astype(np.uint8)
                test_preds.append(pred)
                test_gts.append(gt)

            from PIL import Image
            if test_preds:
                zs = evaluate_pairs(test_preds, test_gts)
                results.setdefault(ds_name, {}).update({
                    'zs_dsc': zs['dsc_mean'], 'zs_dsc_std': zs['dsc_std'],
                    'zs_nsd': zs['nsd_mean'], 'zs_nsd_std': zs['nsd_std'],
                })
                save_results(results)
                tgt = PAPER_TARGETS[ds_name]
                print(f"\nZero-shot {ds_name}:")
                print(f"  DSC: {zs['dsc_mean']:.2f}% (target {tgt['zero_shot_dsc']:.2f}%)")
                print(f"  NSD: {zs['nsd_mean']:.2f}% (target {tgt['zero_shot_nsd']:.2f}%)")

        # Stage 3
        if not args.eval_only:
            print(f"\n--- Stage 3: nnUNet training ({ds_name}) ---")

            zshot_path = os.path.join(Paths.MASK_DIR, ds_name, 'zero_shot.pkl')
            if os.path.exists(zshot_path):
                with open(zshot_path, 'rb') as f:
                    zero_shot_masks = pickle.load(f)
            else:
                print(f"ERROR: zero_shot.pkl not found. Run Stage 2 first.")
                continue

            run_stage3(
                dataset_name=info['nnunet_name'],
                dataset_id=info['nnunet_id'],
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                zero_shot_masks=zero_shot_masks,
                cfg=NNUNetConfig(),
                resume=args.resume,
            )

        # Evaluation
        nn_name  = info['nnunet_name']
        pred_dir = os.path.join(Paths.PRED_DIR, nn_name.lower())

        if os.path.exists(pred_dir) and any(Path(pred_dir).glob('*.png')):
            ws_preds, ws_gts = load_predictions_from_dir(pred_dir, test_pairs)
            if ws_preds:
                ws = evaluate_pairs(ws_preds, ws_gts)
                results.setdefault(ds_name, {}).update({
                    'ws_dsc': ws['dsc_mean'], 'ws_dsc_std': ws['dsc_std'],
                    'ws_nsd': ws['nsd_mean'], 'ws_nsd_std': ws['nsd_std'],
                })
                save_results(results)
                tgt = PAPER_TARGETS[ds_name]
                print(f"\nWeakly Supervised {ds_name}:")
                print(f"  DSC: {ws['dsc_mean']:.2f}% (target {tgt['weakly_sup_dsc']:.2f}%)")
                print(f"  NSD: {ws['nsd_mean']:.2f}% (target {tgt['weakly_sup_nsd']:.2f}%)")

        del train_pairs, val_pairs, test_pairs, all_pairs
        gc.collect()

    # Final tables
    print("\n\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if 'roco' in results:
        print_table2(results['roco'])
    print_table1({ds: results.get(ds, {}) for ds in ['breast', 'brain', 'xray', 'ct']})
    if results.get('ablation'):
        print_table4(results['ablation'])

    print(f"\nAll results: {RESULTS_PATH}")


# ─────────────────────────────────────────────────────────────
# CRITICAL: Windows requires this guard.
# Without it, every import of this module spawns a new process,
# causing infinite recursion and crashing Python.
# ALL execution must be inside this block.
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MedCLIP-SAMv2 replication')
    parser.add_argument('--skip-stage1', action='store_true')
    parser.add_argument('--dataset', type=str,
                        choices=['breast', 'brain', 'xray', 'ct'], default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    print("MedCLIP-SAMv2 — Local Machine")
    print(f"  Dataset:     {args.dataset or 'all'}")
    print(f"  Skip Stage1: {args.skip_stage1}")
    print(f"  Resume:      {args.resume}")
    print(f"  Eval only:   {args.eval_only}")
    print()

    main(args)
