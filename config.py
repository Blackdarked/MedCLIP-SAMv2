# config.py
# Central configuration for MedCLIP-SAMv2 replication.
# All hyperparameters, paths, and paper targets in one place.
#
# BEFORE RUNNING:
#   1. Set BASE_DIR to your project root
#   2. Set VRAM_GB to match your GPU (8 or 16)
#   3. Verify all dataset paths under Paths

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# HARDWARE
# ─────────────────────────────────────────────────────────────

# Set to your actual GPU VRAM in GB.
# RTX 4060 Ti 8GB  → VRAM_GB = 8
# RTX 4060 Ti 16GB → VRAM_GB = 16
# Kaggle T4        → VRAM_GB = 16
VRAM_GB = 16

# Windows: DataLoader workers must be 0 (multiprocessing limitation).
# Linux/Mac: can set to 2–4 for faster data loading.
NUM_WORKERS = 0  # keep 0 on Windows


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

# Project root — change this to wherever you store everything.
BASE_DIR = r'C:\medclip_samv2'


class Paths:
    # ── Input datasets ──────────────────────────────────────
    MEDPIX_DIR   = os.path.join(BASE_DIR, 'data', 'medpix')
    ROCO_DIR     = os.path.join(BASE_DIR, 'data', 'roco')
    BUSI_DIR     = os.path.join(BASE_DIR, 'data', 'BUSI', 'Dataset_BUSI_with_GT')
    BRAIN_DIR    = os.path.join(BASE_DIR, 'data', 'brain_tumor_mri')
    XRAY_DIR     = os.path.join(BASE_DIR, 'data', 'COVID_QU_Ex')
    CT_DIR       = os.path.join(BASE_DIR, 'data', 'lung_ct')
    SAM_CKPT     = os.path.join(BASE_DIR, 'models', 'sam_vit_h_4b8939.pth')

    # ── Working directories ──────────────────────────────────
    WORK_DIR     = os.path.join(BASE_DIR, 'working')
    CKPT_DIR     = os.path.join(BASE_DIR, 'working', 'checkpoints')
    MASK_DIR     = os.path.join(BASE_DIR, 'working', 'masks')
    PRED_DIR     = os.path.join(BASE_DIR, 'working', 'predictions')
    NNUNET_RAW   = os.path.join(BASE_DIR, 'working', 'nnunet_raw')
    NNUNET_PREP  = os.path.join(BASE_DIR, 'working', 'nnunet_preprocessed')
    NNUNET_RES   = os.path.join(BASE_DIR, 'working', 'nnunet_results')

    # ── Saved model paths ────────────────────────────────────
    BIOMEDCLIP_FT = os.path.join(BASE_DIR, 'working', 'checkpoints', 'biomedclip_dhn_nce.pt')

    @classmethod
    def makedirs(cls):
        for attr in ['CKPT_DIR', 'MASK_DIR', 'PRED_DIR',
                     'NNUNET_RAW', 'NNUNET_PREP', 'NNUNET_RES']:
            os.makedirs(getattr(cls, attr), exist_ok=True)
        for ds in ['breast', 'brain', 'xray', 'ct']:
            os.makedirs(os.path.join(cls.MASK_DIR, ds), exist_ok=True)
            os.makedirs(os.path.join(cls.PRED_DIR, ds), exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STAGE 1 — BiomedCLIP Fine-tuning
# Paper: LR=1e-6, batch=64, 50% decay rate, τ=0.6, β₁=β₂=0.15
# ─────────────────────────────────────────────────────────────

@dataclass
class Stage1Config:
    model_name: str      = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    lr: float            = 1e-6
    batch_size: int      = 64 if VRAM_GB >= 16 else 32   # paper uses 64
    lr_decay: float      = 0.5      # 50% decay rate (StepLR gamma)
    min_caption_len: int = 20       # paper: exclude captions < 20 chars
    train_split: float   = 0.85     # paper: 85/15 → 20,292 train / 3,515 val
    max_epochs: int      = 20       # not stated in paper; use early stopping
    patience: int        = 3        # early stopping patience
    # DHN-NCE hyperparameters (paper Section 3.4.1)
    tau: float           = 0.6
    beta1: float         = 0.15     # image→text hardness parameter
    beta2: float         = 0.15     # text→image hardness parameter
    # ROCO validation (paper: 5 runs, batch=50)
    roco_batch_size: int = 50
    roco_n_runs: int     = 5
    num_workers: int     = NUM_WORKERS


# ─────────────────────────────────────────────────────────────
# STAGE 2 — Zero-shot Segmentation
# ─────────────────────────────────────────────────────────────

@dataclass
class M2IBConfig:
    # Wang et al. (NeurIPS 2023) — M2IB hyperparameters
    layer_idx: int     = 8      # ViT-B transformer block for bottleneck
    beta: float        = 0.1    # compression weight (γ in paper eq. 16)
    sigma: float       = 1.0    # noise variance
    n_steps_small: int = 100    # breast, brain (small datasets)
    n_steps_large: int = 50     # xray, ct (large datasets, speed tradeoff)
    lr: float          = 0.01   # mask optimizer LR


@dataclass
class PostprocessConfig:
    confidence_threshold: float = 0.5   # connected component confidence cutoff
    min_area: int               = 50    # minimum component size (pixels)


@dataclass
class SAMConfig:
    model_type: str = 'vit_h'
    use_fp16: bool  = True   # fp16 reduces VRAM by ~50%; no accuracy loss
    # Per-dataset visual prompt strategy (paper Table 6)
    prompt_strategy: Dict = field(default_factory=lambda: {
        'breast': 'bbox',    # bounding boxes best for breast/brain/ct
        'brain':  'bbox',
        'xray':   'points',  # point prompts best for lung x-ray
        'ct':     'bbox',
    })
    n_points: int = 10       # number of point prompts when using points


# ─────────────────────────────────────────────────────────────
# STAGE 3 — Weakly Supervised nnUNet
# Paper: 600 epochs, 3 cycles, LR=0.01, cyclical schedule (Zhao 2022)
# 10 checkpoints per cycle → 30 total for ensemble
# ─────────────────────────────────────────────────────────────

@dataclass
class NNUNetConfig:
    config_2d: str             = '2d'
    trainer_name: str          = 'nnUNetTrainerCyclicalLR'
    total_epochs: int          = 600
    n_cycles: int              = 3
    epochs_per_cycle: int      = 200    # 600 / 3
    initial_lr: float          = 0.01
    restart_lr: float          = 0.1    # LR at start of each cycle
    gamma_fraction: float      = 0.8    # plateau after 80% of cycle
    checkpoints_per_cycle: int = 10     # 10 × 3 = 30 total
    image_size: int            = 256    # resize images for nnUNet input
    dataset_ids: Dict = field(default_factory=lambda: {
        'breast': 1, 'brain': 2, 'xray': 3, 'ct': 4,
    })


# ─────────────────────────────────────────────────────────────
# TEXT PROMPTS
# Best configuration per dataset (paper Table 3):
#   breast, brain → P3 (class-specific descriptive sentence)
#   xray          → P0 (short generic)
#   ct            → P2 (generic descriptive sentence)
# ─────────────────────────────────────────────────────────────

PROMPTS = {
    # Breast Ultrasound — P3
    'breast_benign': (
        "An ultrasound image of the breast showing a well-defined, oval or round "
        "hypoechoic mass with smooth margins and posterior acoustic enhancement "
        "suggestive of a benign breast tumor."
    ),
    'breast_malignant': (
        "An ultrasound image of the breast showing an irregularly shaped, spiculated "
        "hypoechoic mass with posterior acoustic shadowing and angular margins "
        "suggestive of a malignant breast tumor."
    ),
    'breast_generic': "breast tumor in ultrasound",

    # Brain Tumor MRI — P3
    'brain_glioma': (
        "A T1-weighted brain MRI showing a heterogeneous mass with irregular borders, "
        "surrounding edema, and ring enhancement in the cerebral hemisphere "
        "suggestive of a high-grade glioma tumor."
    ),
    'brain_meningioma': (
        "A T1-weighted brain MRI showing a well-circumscribed, extra-axial "
        "homogeneous mass with a broad dural base suggestive of a meningioma tumor."
    ),
    'brain_pituitary': (
        "A T1-weighted brain MRI showing a sellar or suprasellar mass arising "
        "from the pituitary gland suggestive of a pituitary tumor."
    ),
    'brain_generic': "brain tumor in MRI",

    # Lung X-ray — P0
    'xray': "lungs",

    # Lung CT — P2
    'ct': (
        "A CT scan showing bilateral lung lobes with parenchymal changes and "
        "altered lung tissue density consistent with fibrotic lung disease."
    ),
}


# ─────────────────────────────────────────────────────────────
# PAPER TARGET NUMBERS (Table 1)
# ─────────────────────────────────────────────────────────────

PAPER_TARGETS = {
    'breast': {
        'zero_shot_dsc': 77.76, 'zero_shot_nsd': 81.11,
        'weakly_sup_dsc': 78.87, 'weakly_sup_nsd': 84.58,
    },
    'brain': {
        'zero_shot_dsc': 76.52, 'zero_shot_nsd': 82.23,
        'weakly_sup_dsc': 80.03, 'weakly_sup_nsd': 88.25,
    },
    'xray': {
        'zero_shot_dsc': 75.79, 'zero_shot_nsd': 80.88,
        'weakly_sup_dsc': 80.77, 'weakly_sup_nsd': 84.53,
    },
    'ct': {
        'zero_shot_dsc': 80.38, 'zero_shot_nsd': 82.03,
        'weakly_sup_dsc': 88.78, 'weakly_sup_nsd': 91.95,
    },
    'average': {
        'zero_shot_dsc': 77.61, 'zero_shot_nsd': 81.56,
        'weakly_sup_dsc': 82.11, 'weakly_sup_nsd': 87.33,
    },
}
