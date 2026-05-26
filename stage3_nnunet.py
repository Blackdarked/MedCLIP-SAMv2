# stage3_nnunet.py — LOCAL MACHINE VERSION
# Changes from Kaggle version:
#   - os.system() replaced with subprocess (more reliable on Windows)
#   - nnUNet env vars set via Python os.environ (same on Windows)
#   - install_cyclical_trainer() updated for Windows site-packages path
#   - DataLoader num_workers=0
#   - No session-resume logic (runs continuously on local machine)

import os
import gc
import glob
import json
import shutil
import subprocess
import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from config import Paths, NNUNetConfig, NUM_WORKERS


# ─────────────────────────────────────────────────────────────
# nnUNet environment setup
# ─────────────────────────────────────────────────────────────

def setup_nnunet_env():
    """Set nnUNet environment variables. Same logic, works on Windows."""
    os.environ['nnUNet_raw']          = Paths.NNUNET_RAW
    os.environ['nnUNet_preprocessed'] = Paths.NNUNET_PREP
    os.environ['nnUNet_results']      = Paths.NNUNET_RES

    for p in [Paths.NNUNET_RAW, Paths.NNUNET_PREP, Paths.NNUNET_RES]:
        os.makedirs(p, exist_ok=True)

    print("nnUNet env set:")
    print(f"  raw:          {Paths.NNUNET_RAW}")
    print(f"  preprocessed: {Paths.NNUNET_PREP}")
    print(f"  results:      {Paths.NNUNET_RES}")


def install_cyclical_trainer():
    """
    Copy cyclical_trainer.py into nnunetv2 package.

    Windows site-packages path differs from Linux:
    Linux: /opt/conda/lib/python3.10/site-packages/...
    Windows: C:\\Users\\...\\site-packages\\... or venv path

    This function finds the correct path automatically.
    """
    # Find cyclical_trainer.py
    src_candidates = [
        os.path.join(os.path.dirname(__file__), 'cyclical_trainer.py'),
        os.path.join(os.getcwd(), 'cyclical_trainer.py'),
    ]
    src = next((p for p in src_candidates if os.path.exists(p)), None)
    if src is None:
        raise FileNotFoundError(
            "cyclical_trainer.py not found. "
            "Place it in the same directory as stage3_nnunet.py."
        )

    # Find nnunetv2 trainer directory
    import site
    dst = None
    for site_pkg in site.getsitepackages():
        candidate = os.path.join(
            site_pkg, 'nnunetv2', 'training', 'nnUNetTrainer'
        )
        if os.path.isdir(candidate):
            dst = candidate
            break

    # Also check current venv if active
    if dst is None and hasattr(sys, 'real_prefix'):
        venv_candidate = os.path.join(
            sys.prefix, 'Lib', 'site-packages',
            'nnunetv2', 'training', 'nnUNetTrainer'
        )
        if os.path.isdir(venv_candidate):
            dst = venv_candidate

    if dst is None:
        raise RuntimeError(
            "Cannot find nnunetv2 trainer directory.\n"
            "Try: pip show nnunetv2 → find Location → manually copy cyclical_trainer.py\n"
            "to: <Location>/nnunetv2/training/nnUNetTrainer/"
        )

    shutil.copy(src, dst)
    print(f"Cyclical trainer installed to: {dst}")


# ─────────────────────────────────────────────────────────────
# Run shell commands — use subprocess on Windows
# ─────────────────────────────────────────────────────────────

def run_cmd(cmd: str) -> int:
    """
    Run a shell command via subprocess.
    More reliable than os.system() on Windows — captures errors properly.
    """
    import os as _os
    env = _os.environ.copy()
    # Ensure venv Scripts dir is on PATH so nnUNetv2_* CLI commands are found
    venv_scripts = _os.path.dirname(sys.executable)
    env['PATH'] = venv_scripts + _os.pathsep + env.get('PATH', '')
    print(f"\nRunning: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    return result.returncode


# ─────────────────────────────────────────────────────────────
# Dataset preparation (unchanged logic)
# ─────────────────────────────────────────────────────────────

def prepare_nnunet_dataset(dataset_name, dataset_id, train_pairs,
                            zero_shot_masks, test_pairs=None,
                            image_size=256):
    base = Path(Paths.NNUNET_RAW) / f'Dataset{dataset_id:03d}_{dataset_name}'
    (base / 'imagesTr').mkdir(parents=True, exist_ok=True)
    (base / 'labelsTr').mkdir(parents=True, exist_ok=True)
    if test_pairs:
        (base / 'imagesTs').mkdir(parents=True, exist_ok=True)

    size = (image_size, image_size)
    n_train = 0

    for i, (img_path, _, _) in enumerate(train_pairs):
        try:
            img = Image.open(img_path).convert('RGB').resize(size, Image.BILINEAR)
        except Exception as e:
            print(f"  Warning: {img_path}: {e}")
            img = Image.new('RGB', size, 0)
        img.save(str(base / 'imagesTr' / f'case_{i+1:04d}_0000.png'))

        mask = zero_shot_masks.get(img_path)
        if mask is None:
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
        else:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255)
                     .resize(size, Image.NEAREST)
            )
            mask = (mask > 127).astype(np.uint8)
        Image.fromarray(mask).save(
            str(base / 'labelsTr' / f'case_{i+1:04d}.png')
        )
        n_train += 1

    if test_pairs:
        for i, (img_path, _, _) in enumerate(test_pairs):
            try:
                img = Image.open(img_path).convert('RGB').resize(size, Image.BILINEAR)
            except Exception:
                img = Image.new('RGB', size, 0)
            img.save(str(base / 'imagesTs' / f'test_{i+1:04d}_0000.png'))

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": n_train,
        "file_ending": ".png",
        "dataset_name": dataset_name,
    }
    with open(str(base / 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Dataset {dataset_id:03d}_{dataset_name}: {n_train} training samples")
    return str(base)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def run_nnunet_train(dataset_id, cfg=None, resume=False):
    """
    Run nnUNet training.

    On Windows: nnUNet uses multiprocessing internally.
    The `if __name__ == '__main__':` guard in run_all.py handles this.
    Do not call this function from a script without that guard.
    """
    if cfg is None:
        cfg = NNUNetConfig()

    cmd = (
        f'nnUNetv2_train {dataset_id:03d} {cfg.config_2d} 0 '
        f'-tr {cfg.trainer_name} '
        f'-p {cfg.plans_name}'
    )
    if resume:
        cmd += ' --c'

    return run_cmd(cmd)


def run_nnunet_predict(dataset_id, dataset_name, cfg=None):
    if cfg is None:
        cfg = NNUNetConfig()

    input_dir  = os.path.join(
        Paths.NNUNET_RAW, f'Dataset{dataset_id:03d}_{dataset_name}', 'imagesTs'
    )
    output_dir = os.path.join(Paths.PRED_DIR, dataset_name.lower())
    os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f'nnUNetv2_predict '
        f'-i "{input_dir}" '
        f'-o "{output_dir}" '
        f'-d {dataset_id:03d} '
        f'-c {cfg.config_2d} '
        f'-p {cfg.plans_name} '
        f'-f 0 '
        f'-tr {cfg.trainer_name}'
    )
    # ── Windows: quote paths in case BASE_DIR contains spaces ──
    run_cmd(cmd)
    return output_dir


# ─────────────────────────────────────────────────────────────
# Checkpoint ensemble (unchanged logic)
# ─────────────────────────────────────────────────────────────

def get_ensemble_checkpoint_paths(dataset_id, dataset_name, cfg=None):
    if cfg is None:
        cfg = NNUNetConfig()

    ens_dir = os.path.join(
        Paths.NNUNET_RES,
        f'Dataset{dataset_id:03d}_{dataset_name}',
        f'{cfg.trainer_name}__{cfg.plans_name}__{cfg.config_2d}',
        'fold_0',
        'ensemble_checkpoints'
    )
    ckpts = sorted(glob.glob(os.path.join(ens_dir, '*.pth')))
    print(f"{dataset_name}: {len(ckpts)}/30 ensemble checkpoints")
    return ckpts


def _load_nnunet_network(ckpt_path, plans_path, device='cuda'):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    with open(plans_path) as f:
        plans = json.load(f)

    plans_manager = PlansManager(plans)
    network = get_network_from_plans(
        plans_manager.network_arch_class_name,
        plans_manager.network_arch_init_kwargs,
        plans_manager.network_arch_init_kwargs_req_import,
        num_input_channels=3,
        num_output_channels=2
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    network.load_state_dict(ckpt['network_weights'])
    network = network.half().to(device)
    network.eval()
    return network


def ensemble_predict_single(image_np, ckpt_paths, plans_path,
                              image_size=256, device='cuda'):
    all_probs = []
    for ckpt_path in ckpt_paths:
        try:
            net = _load_nnunet_network(ckpt_path, plans_path, device=device)
            img = Image.fromarray(image_np).resize(
                (image_size, image_size), Image.BILINEAR
            )
            img_t = torch.from_numpy(
                np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
            ).unsqueeze(0).half().to(device)

            with torch.no_grad():
                logits = net(img_t)
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().float().numpy()
            all_probs.append(prob)
            del net
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Checkpoint error: {e}")

    if not all_probs:
        z = np.zeros((image_size, image_size))
        return z, z.astype(np.uint8), z

    stacked = np.stack(all_probs, axis=0)
    mean_prob = stacked.mean(axis=0)
    binary_pred = (mean_prob > 0.5).astype(np.uint8)
    eps = 1e-8
    entropy = -(mean_prob * np.log(mean_prob + eps) +
                (1 - mean_prob) * np.log(1 - mean_prob + eps))
    return mean_prob, binary_pred, entropy


# ─────────────────────────────────────────────────────────────
# Full Stage 3 runner
# ─────────────────────────────────────────────────────────────

def run_stage3(dataset_name, dataset_id, train_pairs, test_pairs,
               zero_shot_masks, cfg=None, resume=False):
    if cfg is None:
        cfg = NNUNetConfig()

    setup_nnunet_env()
    install_cyclical_trainer()

    prepare_nnunet_dataset(
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        train_pairs=train_pairs,
        zero_shot_masks=zero_shot_masks,
        test_pairs=test_pairs,
        image_size=cfg.image_size,
    )

    # Remove stale preprocessed dir so plan_and_preprocess regenerates from
    # the new zero-shot labels rather than reusing old cached labels.
    stale_prep = os.path.join(
        Paths.NNUNET_PREP, f'Dataset{dataset_id:03d}_{dataset_name}'
    )
    if os.path.isdir(stale_prep):
        shutil.rmtree(stale_prep)
        print(f"Cleared stale preprocessed dir: {stale_prep}")

    ret = run_cmd(
        f'nnUNetv2_plan_and_preprocess '
        f'-d {dataset_id:03d} -c {cfg.config_2d} '
        f'-pl {cfg.planner_name} '
        f'--verify_dataset_integrity'
    )
    if ret != 0:
        raise RuntimeError(f"nnUNet preprocessing failed (code {ret})")

    run_nnunet_train(dataset_id, cfg, resume=resume)
    pred_dir = run_nnunet_predict(dataset_id, dataset_name, cfg)
    print(f"Predictions: {pred_dir}")
    return pred_dir
