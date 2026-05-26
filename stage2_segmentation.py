# stage2_segmentation.py
# Zero-shot segmentation pipeline:
#   Fine-tuned BiomedCLIP → M2IB saliency maps
#   → KMeans (k=2) + top-K blob filter (matches paper's --postprocess kmeans --filter)
#   → SAM prompting (bounding box or point)

import os
import gc
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import open_clip
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans

from config import Paths, M2IBConfig, PostprocessConfig, SAMConfig, PROMPTS


# ─────────────────────────────────────────────────────────────
# M2IB — Multi-modal Information Bottleneck
# ─────────────────────────────────────────────────────────────

class M2IBExtractor:
    """
    Multi-modal Information Bottleneck saliency attribution.

    Ref: Wang, Rudner, Wilson — NeurIPS 2023
    "Visual Explanations of Image-Text Representations via
     Multi-Modal Information Bottleneck Attribution"

    Core idea:
        Given image I and text T with embeddings Z_img, Z_text,
        find a per-patch spatial mask λ_S ∈ [0,1]^{H×W} that:
          - maximizes MI(Z_img, Z_text)  [relevance: preserve text-related info]
          - minimizes MI(Z_img, I)       [compression: discard irrelevant content]

    Implementation:
        Insert a stochastic noise layer at ViT transformer block `layer_idx`.
        Patch tokens are multiplied by sigmoid(λ) and mixed with noise
        proportional to (1 - sigmoid(λ)).
        Optimize λ per image-text pair at inference time.

    Args:
        model     : fine-tuned BiomedCLIP model (open_clip) on CUDA
        cfg       : M2IBConfig
    """

    def __init__(self, model, cfg: M2IBConfig = None):
        self.model = model
        self.cfg = cfg or M2IBConfig()
        self.model.eval()

    def extract(self, image_tensor: torch.Tensor,
                text_tokens: torch.Tensor,
                n_steps: Optional[int] = None) -> np.ndarray:
        """
        Compute M2IB saliency map for one image-text pair.

        Args:
            image_tensor : (1, 3, 224, 224) preprocessed, on same device as model
            text_tokens  : (1, L) tokenized text, on same device as model
            n_steps      : override optimization steps (use cfg.n_steps_small by default)

        Returns:
            saliency_map : (H, W) float32 numpy array in [0, 1]
        """
        cfg = self.cfg
        n_steps = n_steps or cfg.n_steps_small
        H, W = image_tensor.shape[-2:]
        n_h, n_w = H // 16, W // 16   # patch grid for ViT-B/16
        n_patches = n_h * n_w          # 196 for 224×224
        device = image_tensor.device

        # Learnable mask logits — one per patch, initialized to 0 (sigmoid → 0.5)
        mask_logits = torch.zeros(1, n_patches, 1,
                                   requires_grad=True, device=device)
        optimizer = torch.optim.Adam([mask_logits], lr=cfg.lr)

        # Pre-compute fixed text embedding
        with torch.no_grad():
            txt_feat = F.normalize(self.model.encode_text(text_tokens), dim=-1)

        # Find the target ViT transformer block
        vit = self.model.visual
        try:
            # OpenCLIP standard path
            blocks = vit.transformer.resblocks
        except AttributeError:
            try:
                blocks = vit.trunk.blocks
            except AttributeError:
                raise RuntimeError(
                    "Cannot find ViT transformer blocks. "
                    "Check open_clip model architecture."
                )

        # Hook: inject bottleneck noise at the specified block
        def _bottleneck_hook(module, input, output):
            """
            output shape: (seq_len, batch, dim) or (batch, seq_len, dim)
            depending on open_clip version.
            We handle both.
            """
            mask = torch.sigmoid(mask_logits)  # (1, N, 1) ∈ [0,1]
            noise = torch.randn_like(output) * (cfg.sigma ** 0.5)

            # Determine layout
            if output.shape[0] == image_tensor.shape[0]:
                # (B, N+1, D) layout — batch first
                patch_out   = output[:, 1:, :]   # (B, N, D)
                cls_out     = output[:, :1, :]   # (B, 1, D)
                patch_noise = noise[:, 1:, :]
                masked_patches = patch_out * mask + patch_noise * (1 - mask)
                return torch.cat([cls_out, masked_patches], dim=1)
            else:
                # (N+1, B, D) layout — sequence first
                patch_out   = output[1:, :, :]   # (N, B, D)
                cls_out     = output[:1, :, :]   # (1, B, D)
                patch_noise = noise[1:, :, :]
                mask_t      = mask.squeeze(0).unsqueeze(1)  # (N, 1, 1)
                masked_patches = patch_out * mask_t + patch_noise * (1 - mask_t)
                return torch.cat([cls_out, masked_patches], dim=0)

        hook = blocks[cfg.layer_idx].register_forward_hook(_bottleneck_hook)

        try:
            for step in range(n_steps):
                optimizer.zero_grad()

                img_feat = F.normalize(
                    self.model.encode_image(image_tensor), dim=-1
                )

                # Relevance term: maximize image-text cosine similarity
                relevance = (img_feat * txt_feat).sum()

                # Compression term: minimize average mask activation
                compression = cfg.beta * torch.sigmoid(mask_logits).mean()

                loss = -relevance + compression
                loss.backward()
                optimizer.step()
        finally:
            hook.remove()

        # Final mask → spatial saliency
        with torch.no_grad():
            final_mask = torch.sigmoid(mask_logits).detach().cpu()  # (1, N, 1)
        flat = final_mask.squeeze().numpy()              # (N,)
        grid = flat.reshape(n_h, n_w)                   # (14, 14) for 224×224

        # Bilinear upsample to original image resolution
        saliency_pil = Image.fromarray((grid * 255).astype(np.uint8))
        saliency_up  = np.array(
            saliency_pil.resize((W, H), Image.BILINEAR)
        ) / 255.0

        return saliency_up.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Post-processing: KMeans clustering + top-K blob filter
# Matches original paper's postprocess_saliency_maps.py:
#   --postprocess kmeans --filter
# ─────────────────────────────────────────────────────────────

def postprocess_saliency(saliency_map: np.ndarray,
                          cfg: PostprocessConfig = None,
                          top_k: int = 3) -> np.ndarray:
    """
    Convert continuous saliency map to binary coarse segmentation mask.

    Step 1: KMeans (k=2) on pixel values → foreground = cluster with
            higher centroid (background cluster has lower mean saliency)
    Step 2: Keep the top-k largest connected components (--filter flag)

    Args:
        saliency_map : (H, W) float32 in [0, 1]
        cfg          : PostprocessConfig (min_area used for small-blob rejection)
        top_k        : maximum number of largest blobs to retain

    Returns:
        binary mask  : (H, W) uint8
    """
    if cfg is None:
        cfg = PostprocessConfig()

    if saliency_map.max() == saliency_map.min():
        return np.zeros_like(saliency_map, dtype=np.uint8)

    # KMeans 2-cluster on flattened saliency values
    flat = saliency_map.reshape(-1, 1).astype(np.float32)
    km = KMeans(n_clusters=2, n_init=3, random_state=0)
    labels = km.fit_predict(flat).reshape(saliency_map.shape)

    # Foreground = cluster with higher centroid
    fg_label = int(np.argmax(km.cluster_centers_[:, 0]))
    binary = (labels == fg_label).astype(np.uint8)

    # Top-K largest blobs filter
    labeled = label(binary)
    regions = regionprops(labeled)
    regions = [r for r in regions if r.area >= cfg.min_area]
    regions = sorted(regions, key=lambda r: r.area, reverse=True)[:top_k]

    final = np.zeros_like(binary)
    for region in regions:
        y, x = region.coords[:, 0], region.coords[:, 1]
        final[y, x] = 1

    return final


def get_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    """Extract bounding box [x1, y1, x2, y2] from binary mask. None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array([cmin, rmin, cmax, rmax])


def get_points(mask: np.ndarray, n: int = 10) -> np.ndarray:
    """Sample n random points inside binary mask. Returns (N, 2) in [x, y]."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return np.array([[w // 2, h // 2]])
    idx = np.random.choice(len(ys), min(n, len(ys)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1)


# ─────────────────────────────────────────────────────────────
# SAM inference
# ─────────────────────────────────────────────────────────────

def load_sam(sam_cfg: SAMConfig = None, device: str = 'cuda'):
    """Load SAM ViT-H in fp16 to save VRAM."""
    from segment_anything import sam_model_registry, SamPredictor
    if sam_cfg is None:
        sam_cfg = SAMConfig()
    sam = sam_model_registry[sam_cfg.model_type](checkpoint=Paths.SAM_CKPT)
    sam = sam.to(device)
    return SamPredictor(sam), sam


def refine_with_sam(predictor,
                     image_np: np.ndarray,
                     coarse_mask: np.ndarray,
                     use_bbox: bool = True,
                     n_points: int = 10) -> np.ndarray:
    """
    Refine a coarse segmentation mask using SAM.

    Args:
        predictor   : SamPredictor (with image already set via predictor.set_image)
        image_np    : (H, W, 3) uint8 RGB
        coarse_mask : (H, W) binary uint8
        use_bbox    : if True, use bounding box prompt; else use point prompts
        n_points    : number of point prompts when use_bbox=False

    Returns:
        refined_mask : (H, W) uint8 binary
    """
    if coarse_mask.sum() == 0:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    if use_bbox:
        bbox = get_bbox(coarse_mask)
        if bbox is None:
            return np.zeros(image_np.shape[:2], dtype=np.uint8)
        masks, scores, _ = predictor.predict(
            box=bbox[None, :],
            multimask_output=False
        )
    else:
        points = get_points(coarse_mask, n=n_points)
        labels = np.ones(len(points), dtype=int)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )

    return masks[0].astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Progress checkpointing
# ─────────────────────────────────────────────────────────────

def save_masks(masks: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(masks, f)


def load_masks(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


# ─────────────────────────────────────────────────────────────
def _remap_hf_visual_to_openclip(hf_sd: dict) -> dict:
    """Convert HuggingFace CLIP visual encoder keys to open_clip CustomTextCLIP keys.

    The authors' pytorch_model.bin was saved from a HF CLIPModel, whose visual
    encoder has different key names and stores q/k/v as separate projections.
    open_clip's timm ViT uses merged qkv and a different naming convention.
    Only visual keys are converted; text keys are intentionally omitted so the
    caller can load with strict=False and keep the pretrained text encoder.
    """
    import re
    new_sd = {}

    # cls token: HF shape [dim] → timm shape [1, 1, dim]
    if 'vision_model.embeddings.class_embedding' in hf_sd:
        new_sd['visual.trunk.cls_token'] = (
            hf_sd['vision_model.embeddings.class_embedding'].unsqueeze(0).unsqueeze(0)
        )

    # position embedding: HF shape [seq, dim] → timm shape [1, seq, dim]
    if 'vision_model.embeddings.position_embedding.weight' in hf_sd:
        new_sd['visual.trunk.pos_embed'] = (
            hf_sd['vision_model.embeddings.position_embedding.weight'].unsqueeze(0)
        )

    # patch projection
    for sfx in ('weight', 'bias'):
        k = f'vision_model.embeddings.patch_embedding.{sfx}'
        if k in hf_sd:
            new_sd[f'visual.trunk.patch_embed.proj.{sfx}'] = hf_sd[k]

    # transformer blocks
    block_ids = sorted({
        int(m.group(1))
        for k in hf_sd
        if (m := re.match(r'vision_model\.encoder\.layers\.(\d+)\.', k))
    })
    for i in block_ids:
        src = f'vision_model.encoder.layers.{i}'
        dst = f'visual.trunk.blocks.{i}'

        # layer norms
        for s_norm, d_norm in [('layer_norm1', 'norm1'), ('layer_norm2', 'norm2')]:
            for sfx in ('weight', 'bias'):
                k = f'{src}.{s_norm}.{sfx}'
                if k in hf_sd:
                    new_sd[f'{dst}.{d_norm}.{sfx}'] = hf_sd[k]

        # merge separate q, k, v → single qkv
        for sfx in ('weight', 'bias'):
            parts = [hf_sd.get(f'{src}.self_attn.{p}_proj.{sfx}') for p in ('q', 'k', 'v')]
            if all(p is not None for p in parts):
                new_sd[f'{dst}.attn.qkv.{sfx}'] = torch.cat(parts, dim=0)

        # output projection
        for sfx in ('weight', 'bias'):
            k = f'{src}.self_attn.out_proj.{sfx}'
            if k in hf_sd:
                new_sd[f'{dst}.attn.proj.{sfx}'] = hf_sd[k]

        # MLP
        for fc in ('fc1', 'fc2'):
            for sfx in ('weight', 'bias'):
                k = f'{src}.mlp.{fc}.{sfx}'
                if k in hf_sd:
                    new_sd[f'{dst}.mlp.{fc}.{sfx}'] = hf_sd[k]

    # final layer norm
    for sfx in ('weight', 'bias'):
        k = f'vision_model.post_layernorm.{sfx}'
        if k in hf_sd:
            new_sd[f'visual.trunk.norm.{sfx}'] = hf_sd[k]

    # visual projection head
    if 'visual_projection.weight' in hf_sd:
        new_sd['visual.head.proj.weight'] = hf_sd['visual_projection.weight']

    return new_sd


# ─────────────────────────────────────────────────────────────
# Full Stage 2 runner — one dataset at a time
# ─────────────────────────────────────────────────────────────

def run_stage2(dataset_name: str,
               all_pairs: List[Tuple],
               m2ib_cfg: M2IBConfig = None,
               pp_cfg: PostprocessConfig = None,
               sam_cfg: SAMConfig = None,
               device: str = 'cuda',
               use_pretrained: bool = False) -> dict:
    """
    Run Stage 2 on one dataset: M2IB → postprocess → SAM.

    Args:
        dataset_name  : one of 'breast', 'brain', 'xray', 'ct'
        all_pairs     : list of (image_path, mask_path_or_None, prompt_key)
        m2ib_cfg      : M2IBConfig (defaults applied if None)
        pp_cfg        : PostprocessConfig
        sam_cfg       : SAMConfig
        device        : 'cuda' or 'cpu'
        use_pretrained: if True, use pre-trained BiomedCLIP (no fine-tuning)

    Returns:
        dict mapping image_path → zero-shot binary mask (H, W) uint8
    """
    if m2ib_cfg is None: m2ib_cfg = M2IBConfig()
    if pp_cfg   is None: pp_cfg   = PostprocessConfig()
    if sam_cfg  is None: sam_cfg  = SAMConfig()

    # Decide n_steps based on dataset size
    n_steps = (m2ib_cfg.n_steps_large
               if dataset_name in ('xray', 'ct')
               else m2ib_cfg.n_steps_small)

    # Paths for progress saving
    coarse_path = os.path.join(Paths.MASK_DIR, dataset_name, 'coarse.pkl')
    rawsal_path = os.path.join(Paths.MASK_DIR, dataset_name, 'raw_saliency.pkl')
    zshot_path  = os.path.join(Paths.MASK_DIR, dataset_name, 'zero_shot.pkl')

    # ── Step 1: Load BiomedCLIP ──
    print(f"\n[Stage 2 / {dataset_name}] Loading BiomedCLIP...")
    model_name = m2ib_cfg.__class__.__name__   # just a label
    biomedclip, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = open_clip.get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )

    if not use_pretrained and os.path.exists(Paths.BIOMEDCLIP_FT):
        ckpt = torch.load(Paths.BIOMEDCLIP_FT, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            ckpt = (ckpt.get('state_dict')
                    or ckpt.get('model_state_dict')
                    or ckpt.get('model')
                    or ckpt)
        if isinstance(ckpt, dict) and 'vision_model.embeddings.class_embedding' in ckpt:
            # Checkpoint is in HuggingFace CLIP format; remap visual keys to open_clip.
            # Text encoder is architecturally incompatible (HF pre-norm vs BERT post-norm)
            # so pretrained text weights are kept.
            visual_sd = _remap_hf_visual_to_openclip(ckpt)
            missing, unexpected = biomedclip.load_state_dict(visual_sd, strict=False)
            visual_loaded = sum(1 for k in visual_sd if k not in missing)
            print(f"  Visual encoder loaded from HF checkpoint ({visual_loaded} keys).")
            print(f"  Text encoder uses pretrained weights (HF/open_clip architecture mismatch).")
        else:
            biomedclip.load_state_dict(ckpt)
            print("  Fine-tuned weights loaded")
    else:
        print("  Using pre-trained weights (Stage 1 not run)")

    biomedclip = biomedclip.to(device)
    biomedclip.eval()

    # ── Step 2: M2IB saliency maps ──
    print(f"[Stage 2 / {dataset_name}] Running M2IB "
          f"({n_steps} steps, {len(all_pairs)} images)...")

    extractor    = M2IBExtractor(biomedclip, m2ib_cfg)
    coarse_masks = load_masks(coarse_path)
    raw_saliency = load_masks(rawsal_path)
    processed    = set(coarse_masks.keys())
    remaining    = len(all_pairs) - len(processed)
    print(f"  Resuming: {len(processed)} done, {remaining} remaining")

    for i, (img_path, _, prompt_key) in enumerate(all_pairs):
        if img_path in processed:
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img).unsqueeze(0).to(device)
            prompt = PROMPTS.get(prompt_key, PROMPTS.get(dataset_name, ''))
            tokens = tokenizer([prompt]).to(device)

            saliency = extractor.extract(img_t, tokens, n_steps=n_steps)

            coarse = postprocess_saliency(saliency, pp_cfg)
            coarse_masks[img_path] = coarse
            raw_saliency[img_path] = saliency

        except Exception as e:
            print(f"  M2IB error [{img_path}]: {e}")
            h, w = 224, 224
            coarse_masks[img_path] = np.zeros((h, w), dtype=np.uint8)
            raw_saliency[img_path] = np.zeros((h, w), dtype=np.float32)

        if (i + 1) % 200 == 0:
            save_masks(coarse_masks, coarse_path)
            save_masks(raw_saliency, rawsal_path)
            pct = len(coarse_masks) / len(all_pairs) * 100
            print(f"  [{dataset_name}] {len(coarse_masks)}/{len(all_pairs)} "
                  f"({pct:.1f}%)", flush=True)

    save_masks(coarse_masks, coarse_path)
    save_masks(raw_saliency, rawsal_path)
    print(f"  M2IB done: {len(coarse_masks)} masks")

    # ── Clear BiomedCLIP from GPU before loading SAM ──
    del biomedclip, extractor, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  GPU freed. Available: {free_gb:.1f} GB")

    # ── Step 3: SAM refinement ──
    print(f"[Stage 2 / {dataset_name}] Running SAM...")
    use_bbox = (sam_cfg.prompt_strategy.get(dataset_name, 'bbox') == 'bbox')
    print(f"  Prompt type: {'bounding box' if use_bbox else 'points'}")

    predictor, sam_model = load_sam(sam_cfg, device)
    zero_shot = load_masks(zshot_path)
    processed_sam = set(zero_shot.keys())

    for img_path, _, _ in all_pairs:
        if img_path in processed_sam:
            continue
        coarse = coarse_masks.get(img_path, np.zeros((224, 224), dtype=np.uint8))
        try:
            img_np = np.array(Image.open(img_path).convert('RGB'))
            h_orig, w_orig = img_np.shape[:2]
            h_c, w_c = coarse.shape
            if (h_c, w_c) != (h_orig, w_orig):
                # M2IB runs on 224×224 preprocessed images; SAM needs coordinates
                # in original image space — scale the coarse mask up first.
                coarse = (np.array(
                    Image.fromarray((coarse * 255).astype(np.uint8))
                    .resize((w_orig, h_orig), Image.NEAREST)
                ) > 127).astype(np.uint8)
            predictor.set_image(img_np)
            refined = refine_with_sam(
                predictor, img_np, coarse,
                use_bbox=use_bbox, n_points=sam_cfg.n_points
            )
            zero_shot[img_path] = refined
        except Exception as e:
            print(f"  SAM error [{img_path}]: {e}")
            zero_shot[img_path] = coarse

        if len(zero_shot) % 500 == 0:
            save_masks(zero_shot, zshot_path)
            print(f"  SAM: {len(zero_shot)}/{len(all_pairs)}", flush=True)

    save_masks(zero_shot, zshot_path)
    print(f"  SAM done: {len(zero_shot)} zero-shot masks")

    del sam_model, predictor
    gc.collect()
    torch.cuda.empty_cache()

    return zero_shot
