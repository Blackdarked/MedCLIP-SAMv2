# datasets.py
# Rewritten to match the authors' pre-split folder structure:
#
#   data/{breast_tumors,brain_tumors,lung_Xray,lung_CT}/
#       train_images/   train_masks/
#       val_images/     val_masks/
#       test_images/    test_masks/
#
# All splitting is done by the authors — we just read what's there.
# Each loader returns (train_pairs, val_pairs, test_pairs)
# where each sample = (image_path, mask_path, prompt_key).

import os
import json
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
import torch


Sample = Tuple[str, Optional[str], str]


# ─────────────────────────────────────────────────────────────
# Utility: load a pre-split folder pair
# ─────────────────────────────────────────────────────────────

def _load_split(img_dir: Path,
                mask_dir: Path,
                prompt_fn) -> List[Sample]:
    """
    Load all image-mask pairs from one split folder.

    img_dir:   folder containing image PNGs
    mask_dir:  folder containing mask PNGs (same filenames)
    prompt_fn: callable(filename) -> prompt_key string
    """
    pairs = []
    if not img_dir.exists():
        return pairs

    for img_path in sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg')):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            # Try alternate extension
            for ext in ['.png', '.jpg']:
                alt = mask_dir / (img_path.stem + ext)
                if alt.exists():
                    mask_path = alt
                    break
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path),
                          prompt_fn(img_path.name)))
    return pairs


# ─────────────────────────────────────────────────────────────
# 1. MedPix — for BiomedCLIP fine-tuning
# ─────────────────────────────────────────────────────────────

class MedPixDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        try:
            img = self.transform(Image.open(img_path).convert('RGB'))
        except Exception:
            img = torch.zeros(3, 224, 224)
        return img, caption


def load_medpix(medpix_dir: str,
                min_caption_len: int = 20,
                train_split: float = 0.85,
                seed: int = 42):
    """
    Load MedPix image-caption pairs.

    Actual structure (from Google Drive download):
        medpix_dataset/
            images/              ← synpic####.png files
            medpix_dataset.csv   ← CSV with image name and caption columns
    """
    pairs = []
    base = Path(medpix_dir)

    # Locate images directory
    img_dir = base / 'images'
    if not img_dir.exists():
        for candidate in base.rglob('images'):
            if candidate.is_dir():
                img_dir = candidate
                break

    # Locate metadata file — CSV or JSON
    csv_files  = list(base.rglob('*.csv'))
    json_files = list(base.rglob('*.json'))

    # ── Try CSV first (matches medpix_dataset.csv) ──
    if csv_files:
        csv_path = csv_files[0]
        try:
            with open(csv_path, encoding='utf-8', errors='replace', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Find image name and caption columns
                    img_name = (row.get('image') or row.get('image_name')
                                or row.get('filename') or row.get('file')
                                or row.get('synpic') or '')
                    caption  = (row.get('caption') or row.get('Caption')
                                or row.get('Description') or row.get('description')
                                or row.get('text') or row.get('report') or '')

                    img_name = str(img_name).strip()
                    caption  = str(caption).strip()

                    if not img_name or len(caption) < min_caption_len:
                        continue

                    # Try exact name; if img_name is a path, use just the basename
                    img_path = img_dir / img_name
                    if not img_path.exists():
                        img_path = img_dir / Path(img_name).name
                    if not img_path.exists():
                        img_path = img_dir / (Path(img_name).name + '.png')
                    if not img_path.exists():
                        img_path = img_dir / (Path(img_name).name + '.jpg')
                    if img_path.exists():
                        pairs.append((str(img_path), caption))
        except Exception as e:
            print(f"  Warning: could not read {csv_path}: {e}")

    # ── Try JSON if CSV gave nothing ──
    if not pairs and json_files:
        json_path = json_files[0]
        try:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    img_name = (item.get('image') or item.get('image_name')
                                or item.get('filename') or '')
                    caption  = (item.get('Description') or item.get('caption')
                                or item.get('description') or item.get('text')
                                or '')
                    img_name = str(img_name).strip()
                    caption  = str(caption).strip()
                    if not img_name or len(caption) < min_caption_len:
                        continue
                    img_path = img_dir / (img_name + '.png')
                    if not img_path.exists():
                        img_path = img_dir / img_name
                    if img_path.exists():
                        pairs.append((str(img_path), caption))
        except Exception as e:
            print(f"  Warning: could not read {json_path}: {e}")

    # ── Fallback: .txt sidecars ──
    if not pairs and img_dir.exists():
        for img_path in sorted(img_dir.rglob('*.png')):
            txt = img_path.with_suffix('.txt')
            if txt.exists():
                cap = txt.read_text(encoding='utf-8').strip()
                if len(cap) >= min_caption_len:
                    pairs.append((str(img_path), cap))

    if not pairs:
        # Print diagnostic info to help debug
        print(f"\nDiagnostic — medpix directory: {base}")
        print(f"  images/ exists: {img_dir.exists()}")
        print(f"  CSV files: {[f.name for f in csv_files]}")
        print(f"  JSON files: {[f.name for f in json_files]}")
        if csv_files:
            with open(csv_files[0], encoding='utf-8') as f:
                header = f.readline()
            print(f"  CSV columns: {header.strip()}")
        raise FileNotFoundError(
            f"No image-caption pairs found in {medpix_dir}.\n"
            "Check the diagnostic output above to identify the correct "
            "column names and update load_medpix() accordingly."
        )

    print(f"MedPix: {len(pairs)} pairs loaded")
    np.random.seed(seed)
    idx   = np.random.permutation(len(pairs))
    split = int(train_split * len(pairs))
    train = [pairs[i] for i in idx[:split]]
    val   = [pairs[i] for i in idx[split:]]
    print(f"  Train: {len(train)} | Val: {len(val)}")
    return train, val


# ─────────────────────────────────────────────────────────────
# 2. ROCO — cross-modal retrieval validation
# ─────────────────────────────────────────────────────────────

def load_roco(roco_dir: str) -> List[Tuple[str, str]]:
    """Load ROCO radiology image-caption pairs."""
    pairs = []
    base  = Path(roco_dir)

    # CSV names per split — each has columns: id, name, caption
    # 'name' is the actual image filename (e.g. PMC####_....jpg)
    csv_names = {
        'train':      'traindata.csv',
        'validation': 'valdata.csv',
        'test':       'testdata.csv',
    }

    for split, csv_name in csv_names.items():
        csv_file = base / split / 'radiology' / csv_name
        img_dir  = base / split / 'radiology' / 'images'
        if not csv_file.exists():
            continue
        with open(csv_file, encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = (row.get('name') or '').strip()
                caption  = (row.get('caption') or '').strip()
                if not img_name or not caption:
                    continue
                img_path = img_dir / img_name
                if img_path.exists():
                    pairs.append((str(img_path), caption))

    # Deduplicate
    seen, unique = set(), []
    for p in pairs:
        if p[0] not in seen:
            seen.add(p[0])
            unique.append(p)

    print(f"ROCO: {len(unique)} pairs loaded")
    return unique


# ─────────────────────────────────────────────────────────────
# 3. Breast Ultrasound
# Pre-split by authors. Filenames contain 'benign'/'malignant'
# which we use to assign the correct text prompt.
# ─────────────────────────────────────────────────────────────

def _breast_prompt(filename: str) -> str:
    name = filename.lower()
    if 'malignant' in name:
        return 'breast_malignant'
    if 'benign' in name:
        return 'breast_benign'
    return 'breast_generic'


def load_breast(breast_dir: str) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load breast tumor dataset from pre-split folders.

    Structure:
        breast_dir/
            train_images/   train_masks/
            val_images/     val_masks/
            test_images/    test_masks/
    """
    base = Path(breast_dir)

    train = _load_split(base/'train_images', base/'train_masks', _breast_prompt)
    val   = _load_split(base/'val_images',   base/'val_masks',   _breast_prompt)
    test  = _load_split(base/'test_images',  base/'test_masks',  _breast_prompt)

    print(f"Breast: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# 4. Brain Tumor MRI
# ─────────────────────────────────────────────────────────────

def _brain_prompt(filename: str) -> str:
    name = filename.lower()
    if 'gl' in name or 'glioma' in name:
        return 'brain_glioma'
    if 'me' in name or 'meningioma' in name:
        return 'brain_meningioma'
    if 'pi' in name or 'pituitary' in name:
        return 'brain_pituitary'
    return 'brain_generic'


def load_brain(brain_dir: str) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load brain tumor dataset from pre-split folders.

    Structure:
        brain_dir/
            train_images/   train_masks/
            val_images/     val_masks/
            test_images/    test_masks/
    """
    base = Path(brain_dir)

    train = _load_split(base/'train_images', base/'train_masks', _brain_prompt)
    val   = _load_split(base/'val_images',   base/'val_masks',   _brain_prompt)
    test  = _load_split(base/'test_images',  base/'test_masks',  _brain_prompt)

    print(f"Brain: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# 5. Lung X-ray
# ─────────────────────────────────────────────────────────────

def load_xray(xray_dir: str) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load lung X-ray dataset from pre-split folders.

    Structure:
        xray_dir/
            train_images/   train_masks/
            val_images/     val_masks/
            test_images/    test_masks/
    """
    base  = Path(xray_dir)
    _xray = lambda f: 'xray'   # same prompt for all X-ray images

    train = _load_split(base/'train_images', base/'train_masks', _xray)
    val   = _load_split(base/'val_images',   base/'val_masks',   _xray)
    test  = _load_split(base/'test_images',  base/'test_masks',  _xray)

    print(f"X-ray: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# 6. Lung CT
# ─────────────────────────────────────────────────────────────

def load_ct(ct_dir: str) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load lung CT dataset from pre-split folders.

    Structure:
        ct_dir/
            train_images/   train_masks/
            val_images/     val_masks/
            test_images/    test_masks/

    Patient-ID split is already done by the authors in these folders.
    """
    base = Path(ct_dir)
    _ct  = lambda f: 'ct'

    train = _load_split(base/'train_images', base/'train_masks', _ct)
    val   = _load_split(base/'val_images',   base/'val_masks',   _ct)
    test  = _load_split(base/'test_images',  base/'test_masks',  _ct)

    print(f"CT: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def load_image_np(path: str,
                  size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def load_mask_np(path: Optional[str],
                 target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if path is None or not os.path.exists(path):
        return np.zeros(target_size or (224, 224), dtype=np.uint8)
    mask = Image.open(path).convert('L')
    if target_size:
        mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    return (np.array(mask) > 127).astype(np.uint8)