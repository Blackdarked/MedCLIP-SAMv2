# verify.py
# Run before run_all.py to confirm every dataset is correctly placed.
# Usage: python verify.py
#
# Checks the actual pre-split folder structure used by datasets.py:
#   data/{breast_tumors,brain_tumors,lung_Xray,lung_CT}/
#       train_images/  train_masks/
#       val_images/    val_masks/
#       test_images/   test_masks/

import os
import sys
import json
import csv
from pathlib import Path

# Force UTF-8 output so Unicode tree/arrow characters don't crash on Windows cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from config import Paths
except ImportError:
    print("ERROR: Cannot import config.py. Run this from the project folder.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def print_tree(root, max_depth=3, max_files=6):
    root = Path(root)
    if not root.exists():
        print(f"  [NOT FOUND] {root}")
        return

    def _walk(path, depth, prefix=""):
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir())
        except PermissionError:
            return
        dirs  = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        for i, d in enumerate(dirs):
            connector = "└── " if (i == len(dirs) - 1 and not files) else "├── "
            try:
                n = sum(1 for _ in d.iterdir())
            except Exception:
                n = 0
            print(f"  {prefix}{connector}{d.name}/  [{n} items]")
            ext_prefix = "    " if connector.startswith("└") else "│   "
            _walk(d, depth + 1, prefix + ext_prefix)

        shown = files[:max_files]
        for i, f in enumerate(shown):
            connector = "└── " if i == len(shown) - 1 else "├── "
            size = f.stat().st_size
            size_str = f"{size/1e6:.1f}MB" if size > 1e6 else f"{size/1e3:.0f}KB"
            print(f"  {prefix}{connector}{f.name}  ({size_str})")
        if len(files) > max_files:
            print(f"  {prefix}    ... and {len(files) - max_files} more files")

    try:
        n_total = sum(1 for _ in root.rglob('*'))
    except Exception:
        n_total = 0
    print(f"  {root}  [{n_total} total items]")
    _walk(root, 1)


def check_presplit(base_path, min_test=10):
    """
    Verify a dataset that follows the pre-split structure:
        base_path/
            train_images/  train_masks/
            val_images/    val_masks/
            test_images/   test_masks/
    Returns (ok, message).
    """
    base = Path(base_path)
    print_tree(base)

    if not base.exists():
        return False, f"Directory not found: {base}"

    counts = {}
    for split in ('train', 'val', 'test'):
        img_dir  = base / f'{split}_images'
        mask_dir = base / f'{split}_masks'

        imgs  = (list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
                 if img_dir.exists() else [])
        masks = (list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.jpg'))
                 if mask_dir.exists() else [])
        counts[split] = (len(imgs), len(masks))
        print(f"  {split:5s}: {len(imgs):4d} images  {len(masks):4d} masks"
              f"  {'OK' if len(imgs) and len(masks) else '!!'}")

    total_imgs = sum(v[0] for v in counts.values())
    if total_imgs == 0:
        return False, (
            "No images found in any split folder.\n"
            f"  Expected subfolders: train_images/, val_images/, test_images/\n"
            f"  under: {base}"
        )

    test_imgs, test_masks = counts['test']
    if test_imgs == 0:
        return False, "test_images/ is empty or missing."
    if test_masks < min_test:
        return False, (
            f"test_masks/ has only {test_masks} files (expected ≥{min_test}).\n"
            f"  Check: {base / 'test_masks'}"
        )

    tr_i, tr_m = counts['train']
    v_i,  v_m  = counts['val']
    return True, (
        f"OK — train:{tr_i}/{tr_m}  val:{v_i}/{v_m}  test:{test_imgs}/{test_masks}"
        f"  (images/masks)"
    )


# ─────────────────────────────────────────────────────────────
# Check registry
# ─────────────────────────────────────────────────────────────

outcomes = {}

def check(name):
    def decorator(fn):
        outcomes[name] = {'fn': fn, 'status': None, 'message': ''}
        return fn
    return decorator


# ── Breast Tumors ─────────────────────────────────────────────
@check('Breast Tumors')
def check_breast():
    path = Paths.BREAST_DIR
    print(f"\n{'='*60}")
    print(f"Breast Tumors  →  {path}")
    print('='*60)
    return check_presplit(path, min_test=10)


# ── Brain Tumors ──────────────────────────────────────────────
@check('Brain Tumors')
def check_brain():
    path = Paths.BRAIN_DIR
    print(f"\n{'='*60}")
    print(f"Brain Tumors  →  {path}")
    print('='*60)
    return check_presplit(path, min_test=10)


# ── Lung X-ray ────────────────────────────────────────────────
@check('Lung X-ray')
def check_xray():
    path = Paths.XRAY_DIR
    print(f"\n{'='*60}")
    print(f"Lung X-ray  →  {path}")
    print('='*60)
    return check_presplit(path, min_test=10)


# ── Lung CT ───────────────────────────────────────────────────
@check('Lung CT')
def check_ct():
    path = Paths.CT_DIR
    print(f"\n{'='*60}")
    print(f"Lung CT  →  {path}")
    print('='*60)
    return check_presplit(path, min_test=10)


# ── MedPix 2.0 ────────────────────────────────────────────────
@check('MedPix 2.0 (Stage 1 fine-tuning)')
def check_medpix():
    path = Path(Paths.MEDPIX_DIR)
    print(f"\n{'='*60}")
    print(f"MedPix 2.0  →  {path}")
    print('='*60)
    print_tree(path)

    if not path.exists():
        return False, f"Directory not found: {path}"

    img_files  = list(path.rglob('*.png')) + list(path.rglob('*.jpg'))
    csv_files  = list(path.rglob('*.csv'))
    json_files = list(path.rglob('*.json'))
    print(f"\n  Found: {len(img_files)} images, "
          f"{len(csv_files)} CSV, {len(json_files)} JSON files")

    if len(img_files) == 0:
        return False, (
            "No images found.\n"
            f"  Expected: {path}/images/  containing synpic####.png files\n"
            "  Download from: https://zenodo.org/records/12624810"
        )

    # Verify at least one image-caption pair resolves
    n_valid = 0
    if csv_files:
        try:
            with open(csv_files[0], encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            for row in rows[:50]:
                caption = (row.get('caption') or row.get('Caption')
                           or row.get('Description') or row.get('description')
                           or row.get('text') or '')
                if len(str(caption).strip()) >= 20:
                    n_valid += 1
            print(f"  CSV columns: {list(rows[0].keys()) if rows else '?'}")
        except Exception as e:
            return False, f"Cannot read CSV: {e}"
    elif json_files:
        try:
            with open(json_files[0], encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data[:50]:
                    caption = (item.get('caption') or item.get('Description')
                               or item.get('description') or '')
                    if len(str(caption).strip()) >= 20:
                        n_valid += 1
        except Exception as e:
            return False, f"Cannot read JSON: {e}"
    else:
        # .txt sidecars
        txt_files = list(path.rglob('*.txt'))
        n_valid = len(txt_files)

    if n_valid == 0:
        return False, (
            "Metadata file found but no valid image-caption pairs resolved.\n"
            "  Check column names in the CSV/JSON match what load_medpix() expects."
        )

    img_dir = path / 'images'
    return True, (
        f"OK — {len(img_files)} images, {n_valid}/50 spot-checked captions valid.\n"
        f"  images dir exists: {img_dir.exists()}"
    )


# ── ROCO ─────────────────────────────────────────────────────
@check('ROCO (Stage 1 validation)')
def check_roco():
    path = Path(Paths.ROCO_DIR)
    print(f"\n{'='*60}")
    print(f"ROCO  →  {path}")
    print('='*60)
    print_tree(path)

    if not path.exists():
        return False, f"Directory not found: {path}"

    # Expected: path/train/radiology/images/ + captions.txt
    cap_files = list(path.rglob('captions.txt'))
    img_files = list(path.rglob('*.jpg')) + list(path.rglob('*.png'))
    print(f"\n  Found: {len(img_files)} images, {len(cap_files)} captions.txt files")

    if len(img_files) < 100:
        return False, (
            f"Only {len(img_files)} images. ROCO should have ~7,042.\n"
            "  Download from Kaggle: virajbagal/roco-dataset\n"
            "  Expected: path/train/radiology/images/ + captions.txt"
        )

    if not cap_files:
        return False, (
            "No captions.txt found.\n"
            "  Expected at: path/{train,validation,test}/radiology/captions.txt"
        )

    # Spot-check a caption file
    n_valid = 0
    try:
        with open(cap_files[0], encoding='utf-8') as f:
            for line in f.readlines()[:10]:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and len(parts[1]) >= 10:
                    n_valid += 1
    except Exception as e:
        return False, f"Cannot read captions.txt: {e}"

    if n_valid == 0:
        return False, (
            "captions.txt found but could not parse.\n"
            "  Expected format: image_name<TAB>caption"
        )

    return True, f"OK — {len(img_files)} images, {len(cap_files)} caption files."


# ── BiomedCLIP fine-tuned model ───────────────────────────────
@check('BiomedCLIP fine-tuned checkpoint')
def check_biomedclip():
    path = Path(Paths.BIOMEDCLIP_FT)
    print(f"\n{'='*60}")
    print(f"BiomedCLIP FT  →  {path}")
    print('='*60)

    if not path.exists():
        return False, (
            f"File not found: {path}\n"
            "  Option A — train from scratch (needs MedPix + ROCO):\n"
            "    python run_all.py\n"
            "  Option B — download authors' checkpoint from Google Drive:\n"
            "    https://drive.google.com/file/d/1jjnZabUlc9_gpcP0d2nz_GNS-EGX0lq5\n"
            f"  Then place at: {path}"
        )

    size_mb = path.stat().st_size / 1e6
    print(f"  File size: {size_mb:.1f} MB")

    if size_mb < 50:
        return False, (
            f"File too small ({size_mb:.1f} MB). Expected ~300–400 MB.\n"
            "  The download may be incomplete. Delete and re-download."
        )

    return True, f"OK — {size_mb:.1f} MB."


# ── SAM ViT-H weights ─────────────────────────────────────────
@check('SAM ViT-H weights')
def check_sam():
    path = Path(Paths.SAM_CKPT)
    print(f"\n{'='*60}")
    print(f"SAM checkpoint  →  {path}")
    print('='*60)

    if not path.exists():
        return False, (
            f"File not found: {path}\n"
            "  Download with:\n"
            "  curl.exe -L https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_h_4b8939.pth"
            f' -o "{path}"'
        )

    size_gb = path.stat().st_size / 1e9
    print(f"  File size: {size_gb:.2f} GB")

    if size_gb < 2.0:
        return False, (
            f"File too small ({size_gb:.2f} GB). Expected ~2.4 GB.\n"
            "  Delete and re-download."
        )

    return True, f"OK — {size_gb:.2f} GB."


# ─────────────────────────────────────────────────────────────
# Run all checks
# ─────────────────────────────────────────────────────────────

def run_all_checks():
    check_fns = {
        'Breast Tumors':                    check_breast,
        'Brain Tumors':                     check_brain,
        'Lung X-ray':                       check_xray,
        'Lung CT':                          check_ct,
        'MedPix 2.0 (Stage 1 fine-tuning)': check_medpix,
        'ROCO (Stage 1 validation)':         check_roco,
        'BiomedCLIP fine-tuned checkpoint':  check_biomedclip,
        'SAM ViT-H weights':                check_sam,
    }

    results = {}
    for name, fn in check_fns.items():
        try:
            ok, msg = fn()
            results[name] = (ok, msg)
        except Exception as e:
            results[name] = (False, f"Check crashed: {e}")

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    all_pass = True
    seg_pass = True
    for name, (ok, msg) in results.items():
        icon   = "OK" if ok else "!!"
        status = "PASS" if ok else "FAIL"
        print(f"\n  {icon} [{status}] {name}")
        for line in msg.split('\n'):
            print(f"         {line}")
        if not ok:
            all_pass = False
            if name in ('Breast Tumors', 'Brain Tumors', 'Lung X-ray',
                        'Lung CT', 'BiomedCLIP fine-tuned checkpoint',
                        'SAM ViT-H weights'):
                seg_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("All checks passed. Run:  python run_all.py --skip-stage1")
    elif seg_pass:
        print("Segmentation pipeline ready. Run:")
        print("  python run_all.py --skip-stage1")
        print("\nMedPix/ROCO only needed if you want to retrain Stage 1.")
    else:
        print("Fix the FAIL items above, then re-run: python verify.py")
    print('='*60)


if __name__ == '__main__':
    run_all_checks()
