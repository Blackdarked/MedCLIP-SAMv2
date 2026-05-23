# datasets.py
# Dataset loaders for all 4 modalities.
# Each loader returns (train_pairs, val_pairs, test_pairs)
# where each pair is (image_path, mask_path, prompt_key).

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
import torch

# ─────────────────────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────────────────────

# Each sample: (image_path, mask_path_or_None, prompt_key)
Sample = Tuple[str, Optional[str], str]


# ─────────────────────────────────────────────────────────────
# 1. MedPix 2.0 — for BiomedCLIP fine-tuning
# ─────────────────────────────────────────────────────────────

class MedPixDataset(Dataset):
    """
    MedPix 2.0 image-caption pairs for contrastive fine-tuning.
    Returns (preprocessed_image_tensor, caption_string).
    """
    def __init__(self, pairs: List[Tuple[str, str]], transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return img, caption


def load_medpix(medpix_dir: str,
                min_caption_len: int = 20,
                train_split: float = 0.85,
                seed: int = 42) -> Tuple[List, List]:
    """
    Load image-caption pairs from MedPix 2.0.

    Tries these formats in order:
      1. JSON file with list of {image_path, caption} dicts
      2. CSV with image and caption columns
      3. Image files paired with .txt sidecars

    Returns:
        train_pairs, val_pairs — lists of (image_path, caption)
    """
    pairs = []
    base = Path(medpix_dir)

    # Format 1: JSON
    for jf in sorted(base.rglob('*.json'))[:3]:
        try:
            data = json.load(open(jf))
            if not isinstance(data, list):
                continue
            for item in data:
                img_p = item.get('image_path') or item.get('image') or item.get('file')
                cap   = item.get('caption') or item.get('text') or item.get('description') or ''
                if not img_p or len(cap.strip()) < min_caption_len:
                    continue
                full = str(base / img_p) if not os.path.isabs(img_p) else img_p
                if os.path.exists(full):
                    pairs.append((full, cap.strip()))
        except Exception:
            continue
        if pairs:
            break

    # Format 2: CSV
    if not pairs:
        import pandas as pd
        for cf in sorted(base.rglob('*.csv'))[:3]:
            try:
                df = pd.read_csv(cf)
                img_col = next((c for c in df.columns
                                if any(k in c.lower() for k in ['image','file','path'])), None)
                cap_col = next((c for c in df.columns
                                if any(k in c.lower() for k in ['caption','text','report'])), None)
                if img_col is None or cap_col is None:
                    continue
                for _, row in df.iterrows():
                    img_p = str(row[img_col]).strip()
                    cap   = str(row[cap_col]).strip()
                    if len(cap) < min_caption_len:
                        continue
                    full = str(base / img_p) if not os.path.isabs(img_p) else img_p
                    if os.path.exists(full):
                        pairs.append((full, cap))
            except Exception:
                continue
            if pairs:
                break

    # Format 3: image + .txt sidecar
    if not pairs:
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            for img_path in sorted(base.rglob(ext)):
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists():
                    cap = txt_path.read_text().strip()
                    if len(cap) >= min_caption_len:
                        pairs.append((str(img_path), cap))

    if not pairs:
        raise FileNotFoundError(
            f"No image-caption pairs found in {medpix_dir}. "
            "Check directory structure and update load_medpix()."
        )

    print(f"MedPix: {len(pairs)} pairs loaded")
    np.random.seed(seed)
    idx = np.random.permutation(len(pairs))
    split = int(train_split * len(pairs))
    train = [pairs[i] for i in idx[:split]]
    val   = [pairs[i] for i in idx[split:]]
    print(f"  Train: {len(train)} | Val: {len(val)}")
    return train, val


# ─────────────────────────────────────────────────────────────
# 2. ROCO — cross-modal retrieval validation
# ─────────────────────────────────────────────────────────────

def load_roco(roco_dir: str) -> List[Tuple[str, str]]:
    """
    Load ROCO image-caption pairs for retrieval evaluation.
    Returns list of (image_path, caption).
    """
    pairs = []
    base = Path(roco_dir)

    # Try multiple directory structures
    search_paths = [
        base / 'train' / 'radiology',
        base / 'validation' / 'radiology',
        base / 'test' / 'radiology',
        base / 'radiology',
        base,
    ]

    for sp in search_paths:
        cap_candidates = list(sp.glob('captions.txt')) + list(sp.glob('*.txt'))
        for cap_file in cap_candidates:
            img_dir = sp / 'images'
            if not img_dir.exists():
                img_dir = sp
            try:
                with open(cap_file) as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) < 2:
                            continue
                        img_name, caption = parts[0], '\t'.join(parts[1:])
                        for ext in ['.jpg', '.png', '.jpeg']:
                            img_path = img_dir / (img_name + ext)
                            if img_path.exists():
                                pairs.append((str(img_path), caption))
                                break
            except Exception:
                continue

    # Deduplicate
    seen = set()
    unique = []
    for p in pairs:
        if p[0] not in seen:
            seen.add(p[0])
            unique.append(p)

    print(f"ROCO: {len(unique)} pairs loaded")
    return unique


# ─────────────────────────────────────────────────────────────
# 3. BUSI — Breast Ultrasound
# ─────────────────────────────────────────────────────────────

def load_breast(busi_dir: str,
                seed: int = 42) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load BUSI breast ultrasound dataset.

    Structure:
        busi_dir/{benign,malignant}/
            *.png           (images)
            *_mask.png      (binary masks)

    Returns:
        train_pairs, val_pairs, test_pairs
        Each sample: (image_path, mask_path, prompt_key)
    """
    pairs = []
    base = Path(busi_dir)

    for class_name in ['benign', 'malignant']:
        class_dir = base / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue

        all_png = sorted(class_dir.glob('*.png'))
        img_files = [f for f in all_png if '_mask' not in f.name]

        for img_path in img_files:
            # Primary mask
            mask_path = class_dir / (img_path.stem + '_mask.png')
            if not mask_path.exists():
                candidates = sorted(class_dir.glob(f'{img_path.stem}_mask*.png'))
                mask_path = candidates[0] if candidates else None
            if mask_path and mask_path.exists():
                prompt_key = f'breast_{class_name}'
                pairs.append((str(img_path), str(mask_path), prompt_key))

    print(f"BUSI: {len(pairs)} pairs "
          f"({sum(1 for _,_,l in pairs if 'benign' in l)} benign, "
          f"{sum(1 for _,_,l in pairs if 'malignant' in l)} malignant)")

    np.random.seed(seed)
    idx = np.random.permutation(len(pairs))
    n = len(pairs)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)

    train = [pairs[i] for i in idx[:n_train]]
    val   = [pairs[i] for i in idx[n_train:n_train + n_val]]
    test  = [pairs[i] for i in idx[n_train + n_val:]]
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# 4. Brain Tumor MRI
# ─────────────────────────────────────────────────────────────

def load_brain(brain_dir: str,
               n_train: int = 1462,
               n_val:   int = 1002,
               seed: int = 42) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load brain tumor MRI dataset (Cheng 2017 format).

    Expected structure:
        brain_dir/{Training,Testing}/{glioma,meningioma,pituitary,notumor}/
            *.jpg or *.png

    Masks may be absent (classification-only dataset).
    If absent, mask_path = None — zero-shot masks from M2IB+SAM serve as labels.
    """
    CLASS_MAP = {
        'glioma':           'brain_glioma',
        'glioma_tumor':     'brain_glioma',
        'meningioma':       'brain_meningioma',
        'meningioma_tumor': 'brain_meningioma',
        'pituitary':        'brain_pituitary',
        'pituitary_tumor':  'brain_pituitary',
        'notumor':          None,
        'no_tumor':         None,
    }

    pairs = []
    base = Path(brain_dir)

    for split_dir in sorted(base.iterdir()):
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            prompt_key = CLASS_MAP.get(class_dir.name.lower())
            if prompt_key is None:
                continue
            for img_path in sorted(class_dir.glob('*.jpg')) + sorted(class_dir.glob('*.png')):
                if '_mask' in img_path.name:
                    continue
                mask_path = class_dir / (img_path.stem + '_mask.png')
                mask = str(mask_path) if mask_path.exists() else None
                pairs.append((str(img_path), mask, prompt_key))

    # Deduplicate
    seen = set()
    unique = []
    for p in pairs:
        if p[0] not in seen:
            seen.add(p[0])
            unique.append(p)
    pairs = unique

    print(f"Brain: {len(pairs)} pairs "
          f"({sum(1 for _,m,_ in pairs if m is not None)} with masks)")

    np.random.seed(seed)
    idx = np.random.permutation(len(pairs))
    n_test = len(pairs) - n_train - n_val
    train = [pairs[i] for i in idx[:n_train]]
    val   = [pairs[i] for i in idx[n_train:n_train + n_val]]
    test  = [pairs[i] for i in idx[n_train + n_val:]]
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# 5. COVID-QU-Ex — Lung Chest X-ray
# ─────────────────────────────────────────────────────────────

def load_xray(xray_dir: str,
              max_train: int = 16280,
              max_val:   int = 1372,
              max_test:  int = 957,
              seed: int = 42) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load COVID-QU-Ex chest X-ray dataset.

    Expected structure:
        xray_dir/COVID-19_Radiography_Dataset/{COVID,Normal,Lung_Opacity,Viral Pneumonia}/
            images/*.png
            masks/*.png
    """
    pairs = []
    base = Path(xray_dir)

    # Find dataset root
    for sub in ['COVID-19_Radiography_Dataset', 'COVID-QU-Ex', '']:
        candidate = base / sub if sub else base
        if candidate.exists() and any(candidate.iterdir()):
            base = candidate
            break

    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir():
            continue
        img_dir  = class_dir / 'images'
        mask_dir = class_dir / 'masks'
        if not img_dir.exists():
            img_dir = class_dir

        for img_path in sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg')):
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + '.png')
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path), 'xray'))

    print(f"X-ray: {len(pairs)} pairs")

    np.random.seed(seed)
    idx = np.random.permutation(len(pairs))
    train = [pairs[i] for i in idx[:max_train]]
    val   = [pairs[i] for i in idx[max_train:max_train + max_val]]
    test  = [pairs[i] for i in idx[max_train + max_val:max_train + max_val + max_test]]
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# 6. Konya — Lung CT (patient-ID split)
# ─────────────────────────────────────────────────────────────

def load_ct(ct_dir: str,
            train_frac: float = 0.70,
            val_frac:   float = 0.15,
            seed: int = 42) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Load Lung CT dataset with patient-ID-based split.

    CRITICAL: Split by patient, not by slice, to prevent data leakage.

    Expected structure:
        ct_dir/2d_images/*.png   (slices named PatientXX_SliceYY.png)
        ct_dir/2d_masks/*.png    (matching masks)

    If structure differs, adjust img_dir and mask_dir below.
    Run: !ls /kaggle/input/finding-lungs-in-ct-data/
    """
    base = Path(ct_dir)

    # Find image and mask directories
    img_dir = mask_dir = None
    for id_name in ['2d_images', 'images', 'Images', 'img']:
        if (base / id_name).exists():
            img_dir = base / id_name
            break
    for md_name in ['2d_masks', 'masks', 'Masks', 'mask']:
        if (base / md_name).exists():
            mask_dir = base / md_name
            break

    if img_dir is None:
        # Search recursively for PNGs
        all_imgs = list(base.rglob('*.png'))
        if not all_imgs:
            raise FileNotFoundError(
                f"No PNG files found in {ct_dir}. "
                "Check the dataset structure and update load_ct()."
            )
        print(f"Sample CT files: {[str(f) for f in all_imgs[:3]]}")
        raise FileNotFoundError(
            "Cannot determine img/mask directories. "
            "Update img_dir and mask_dir in load_ct() manually."
        )

    # Build patient → slices mapping
    patient_slices = {}
    for img_path in sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg')):
        if mask_dir is None:
            continue
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = mask_dir / (img_path.stem + '.png')
        if not mask_path.exists():
            continue

        # Extract patient ID: works for patterns like
        # "Patient_01_Slice_003.png", "pat01_sl003.png", "01_003.png"
        stem = img_path.stem
        parts = stem.replace('-', '_').split('_')
        # Heuristic: patient ID = first 1-2 parts
        patient_id = '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        if patient_id not in patient_slices:
            patient_slices[patient_id] = []
        patient_slices[patient_id].append((str(img_path), str(mask_path), 'ct'))

    print(f"CT: {len(patient_slices)} patients, "
          f"{sum(len(v) for v in patient_slices.values())} slices")

    # Patient-level split
    np.random.seed(seed)
    patient_ids = np.random.permutation(sorted(patient_slices.keys()))
    n = len(patient_ids)
    train_ids = patient_ids[:int(train_frac * n)]
    val_ids   = patient_ids[int(train_frac * n):int((train_frac + val_frac) * n)]
    test_ids  = patient_ids[int((train_frac + val_frac) * n):]

    train = [s for pid in train_ids for s in patient_slices[pid]]
    val   = [s for pid in val_ids   for s in patient_slices[pid]]
    test  = [s for pid in test_ids  for s in patient_slices[pid]]
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)} (by patient)")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# Utility: load a single image as numpy array
# ─────────────────────────────────────────────────────────────

def load_image_np(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load an image file as uint8 RGB numpy array (H, W, 3)."""
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def load_mask_np(path: str,
                 target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load a mask file as binary uint8 numpy array (H, W)."""
    if path is None or not os.path.exists(path):
        if target_size:
            return np.zeros(target_size, dtype=np.uint8)
        return None
    mask = Image.open(path).convert('L')
    if target_size:
        mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    return (np.array(mask) > 127).astype(np.uint8)
