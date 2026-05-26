# config.py
# Paths updated to match actual downloaded folder structure.

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

VRAM_GB     = 16
NUM_WORKERS = 0   # Windows: must be 0

# Project root — folder containing this file (portable across machines)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Paths:
    # ── Segmentation datasets (pre-split by authors) ─────────
    BREAST_DIR  = os.path.join(BASE_DIR, 'data', 'breast_tumors')
    BRAIN_DIR   = os.path.join(BASE_DIR, 'data', 'brain_tumors')
    XRAY_DIR    = os.path.join(BASE_DIR, 'data', 'lung_Xray')
    CT_DIR      = os.path.join(BASE_DIR, 'data', 'lung_CT')

    # ── BiomedCLIP fine-tuning datasets ──────────────────────
    MEDPIX_DIR  = os.path.join(BASE_DIR, 'data', 'medpix_dataset')
    ROCO_DIR    = os.path.join(BASE_DIR, 'data', 'roco-dataset-kaggle', 'all_data')

    # ── SAM checkpoint (authors' expected location) ───────────
    SAM_CKPT    = os.path.join(BASE_DIR, 'models', 'sam_vit_h_4b8939.pth')

    # ── Working directories ───────────────────────────────────
    WORK_DIR    = os.path.join(BASE_DIR, 'working')
    CKPT_DIR    = os.path.join(BASE_DIR, 'working', 'checkpoints')
    MASK_DIR    = os.path.join(BASE_DIR, 'working', 'masks')
    PRED_DIR    = os.path.join(BASE_DIR, 'working', 'predictions')
    NNUNET_RAW  = os.path.join(BASE_DIR, 'working', 'nnunet_raw')
    NNUNET_PREP = os.path.join(BASE_DIR, 'working', 'nnunet_preprocessed')
    NNUNET_RES  = os.path.join(BASE_DIR, 'working', 'nnunet_results')

    # ── Fine-tuned BiomedCLIP ─────────────────────────────────
    # Authors' checkpoint from Google Drive:
    # https://drive.google.com/file/d/1jjnZabUlc9_gpcP0d2nz_GNS-EGX0lq5
    # Place as: working/checkpoints/pytorch_model.bin
    BIOMEDCLIP_FT = os.path.join(BASE_DIR, 'working', 'checkpoints',
                                  'pytorch_model.bin')

    @classmethod
    def makedirs(cls):
        for attr in ['CKPT_DIR', 'MASK_DIR', 'PRED_DIR',
                     'NNUNET_RAW', 'NNUNET_PREP', 'NNUNET_RES']:
            os.makedirs(getattr(cls, attr), exist_ok=True)
        for ds in ['breast', 'brain', 'xray', 'ct']:
            os.makedirs(os.path.join(cls.MASK_DIR, ds), exist_ok=True)
            os.makedirs(os.path.join(cls.PRED_DIR, ds), exist_ok=True)


@dataclass
class Stage1Config:
    model_name: str      = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    lr: float            = 1e-6
    batch_size: int      = 64
    lr_decay: float      = 0.5
    min_caption_len: int = 20
    train_split: float   = 0.85
    max_epochs: int      = 20
    patience: int        = 3
    tau: float           = 0.6
    beta1: float         = 0.15
    beta2: float         = 0.15
    roco_batch_size: int = 50
    roco_n_runs: int     = 5
    num_workers: int     = NUM_WORKERS


@dataclass
class M2IBConfig:
    layer_idx: int     = 7
    beta: float        = 0.3   # stronger compression → sparser, more focused masks
    sigma: float       = 1.0
    n_steps_small: int = 300   # more steps → saliency converges further from 0.5
    n_steps_large: int = 150
    lr: float          = 0.01


@dataclass
class PostprocessConfig:
    confidence_threshold: float = 0.5
    min_area: int               = 50


@dataclass
class SAMConfig:
    model_type: str = 'vit_h'
    use_fp16: bool  = True
    prompt_strategy: Dict = field(default_factory=lambda: {
        'breast': 'bbox',
        'brain':  'bbox',
        'xray':   'points',
        'ct':     'bbox',
    })
    n_points: int = 10


@dataclass
class NNUNetConfig:
    config_2d: str             = '2d'
    trainer_name: str          = 'nnUNetTrainerCyclicalLR'
    planner_name: str          = 'nnUNetPlannerResEncM'   # ResEnc M: 8 GB VRAM target, safe on 17 GB
    plans_name: str            = 'nnUNetResEncUNetMPlans'  # produced by nnUNetPlannerResEncM
    total_epochs: int          = 600
    n_cycles: int              = 3
    epochs_per_cycle: int      = 200
    initial_lr: float          = 0.01
    restart_lr: float          = 0.1
    gamma_fraction: float      = 0.8
    checkpoints_per_cycle: int = 10
    image_size: int            = 256
    dataset_ids: Dict = field(default_factory=lambda: {
        'breast': 1, 'brain': 2, 'xray': 3, 'ct': 4,
    })


PROMPTS = {
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
    'breast_generic':    "breast tumor in ultrasound",
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
    'brain_generic':     "brain tumor in MRI",
    'xray':              "lungs",
    'ct': (
        "A CT scan showing bilateral lung lobes with parenchymal changes and "
        "altered lung tissue density consistent with fibrotic lung disease."
    ),
}

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