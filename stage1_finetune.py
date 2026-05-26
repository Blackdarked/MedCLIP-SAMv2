# stage1_finetune.py — CORRECTED
#
# Bug fixed: DHN-NCE hardness weight exponent was wrong.
#
# Paper eq. 12:
#   W^{v→t}_{ij} = (B-1) × exp(β₁ · I_i·T_j / τ) / Σ_{k≠i} exp(β₁ · I_i·T_k / τ)
#
# The exponent is β₁ multiplied by the TEMPERATURE-SCALED similarity (dot/τ),
# NOT β₁ multiplied by the raw dot product.
#
# Previous code:
#   raw = sim_nodiv * tau * beta   ← wrong: = dot * beta (missing /τ)
#
# Corrected code:
#   raw = sim_scaled * beta        ← correct: = (dot/τ) * β = dot * β / τ
#
# At β=0.15, τ=0.6:
#   Wrong:   exp(0.15 · dot)
#   Correct: exp(0.15 · dot / 0.6) = exp(0.25 · dot)
#
# The hardness contrast is 40% sharper in the paper's formulation.
# This materially affects which negatives get up-weighted during training.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import open_clip

from config import Paths, Stage1Config, NUM_WORKERS
from datasets import MedPixDataset, load_medpix, load_roco


# ─────────────────────────────────────────────────────────────
# DHN-NCE Loss — corrected
# ─────────────────────────────────────────────────────────────

def dhn_nce_loss(img_emb: torch.Tensor,
                 txt_emb: torch.Tensor,
                 tau:   float = 0.6,
                 beta1: float = 0.15,
                 beta2: float = 0.15) -> torch.Tensor:
    """
    Decoupled Hard Negative Noise Contrastive Estimation loss.

    Implements paper equations (9)–(13):

      L^{v→t} = -Σ_i (I_i·T_i/τ)
                + Σ_i log( Σ_{j≠i} exp(I_i·T_j/τ) · W^{v→t}_{ij} )

      W^{v→t}_{ij} = (B-1) · exp(β₁·I_i·T_j/τ)
                             / Σ_{k≠i} exp(β₁·I_i·T_k/τ)       [eq.12]

    Key: the hardness weight exponent uses the TEMPERATURE-SCALED similarity
    (dot product divided by τ), matching eq.12 exactly.

    Args:
        img_emb : (B, D) L2-normalized image embeddings
        txt_emb : (B, D) L2-normalized text embeddings
        tau     : temperature (paper: 0.6)
        beta1   : image→text hardness parameter (paper: 0.15)
        beta2   : text→image hardness parameter (paper: 0.15)
    """
    B = img_emb.shape[0]
    device = img_emb.device

    # Temperature-scaled similarity matrices (B, B)
    # sim_i2t[i,j] = I_i · T_j / τ  (eq. 2 notation)
    sim_i2t = img_emb @ txt_emb.T / tau
    sim_t2i = txt_emb @ img_emb.T / tau

    # Negative mask: True at all off-diagonal positions
    neg_mask  = ~torch.eye(B, dtype=torch.bool, device=device)
    neg_float = neg_mask.float()

    def _hardness_weights(sim_scaled: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute per-negative hardness weights.

        sim_scaled: (B, B) already divided by τ  → this IS the I·T/τ term
        beta:       β₁ or β₂

        Exponent = β · sim_scaled = β · (I·T/τ)   ← matches paper eq.12
        """
        # Zero out diagonal before softmax-style normalization
        logits = beta * sim_scaled * neg_float   # (B, B)
        # Numerically stable: subtract row max before exp
        logits = logits - logits.max(dim=1, keepdim=True).values * neg_float
        hard   = torch.exp(logits) * neg_float
        # Normalize so weights sum to (B-1) per row [eq.12]
        hard   = hard / (hard.sum(dim=1, keepdim=True) + 1e-8) * (B - 1)
        return hard

    W_i2t = _hardness_weights(sim_i2t, beta1)
    W_t2i = _hardness_weights(sim_t2i, beta2)

    # Positive terms: diagonal of temperature-scaled similarity [eq.9 first term]
    pos_i2t = torch.diagonal(sim_i2t)   # (B,)
    pos_t2i = torch.diagonal(sim_t2i)   # (B,)

    # Weighted negative log-sum: Σ_{j≠i} exp(I_i·T_j/τ) · W_{ij}  [eq.9 second term]
    neg_sum_i2t = (torch.exp(sim_i2t) * neg_float * W_i2t).sum(dim=1)  # (B,)
    neg_sum_t2i = (torch.exp(sim_t2i) * neg_float * W_t2i).sum(dim=1)  # (B,)

    # Final DHN-NCE: -positive + log(weighted_negative_sum)  [eq.9, 10, 11]
    loss_i2t = (-pos_i2t + torch.log(neg_sum_i2t + 1e-8)).mean()
    loss_t2i = (-pos_t2i + torch.log(neg_sum_t2i + 1e-8)).mean()

    return loss_i2t + loss_t2i   # L_DHN-NCE = L^{v→t} + L^{t→v}  [eq.11]


# ─────────────────────────────────────────────────────────────
# ROCO Evaluation (unchanged logic, confirmed against paper)
# ─────────────────────────────────────────────────────────────

def evaluate_roco(model, preprocess, tokenizer,
                  pairs, cfg: Stage1Config, device: str) -> dict:
    """
    Top-1 and Top-2 cross-modal retrieval on ROCO.

    Paper: "We ran the experiments five times with a batch size of 50,
    using shuffling to randomize image-text pairs
    (resulting in 70,420 shuffled examples)."

    70,420 = 7,042 images × 5 runs × 2 directions (img→txt and txt→img)
    """
    model.eval()
    top1_i2t, top2_i2t = [], []
    top1_t2i, top2_t2i = [], []

    for run in range(cfg.roco_n_runs):   # 5 runs
        np.random.seed(run)
        shuffled = [pairs[i] for i in np.random.permutation(len(pairs))]

        for start in range(0, len(shuffled) - cfg.roco_batch_size + 1,
                           cfg.roco_batch_size):   # batch_size=50
            batch = shuffled[start:start + cfg.roco_batch_size]

            imgs = []
            for img_path, _ in batch:
                try:
                    imgs.append(preprocess(Image.open(img_path).convert('RGB')))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            imgs = torch.stack(imgs).to(device)
            caps = tokenizer([c for _, c in batch]).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                img_emb = F.normalize(model.encode_image(imgs), dim=-1)
                txt_emb = F.normalize(model.encode_text(caps),  dim=-1)
                sim = img_emb @ txt_emb.T   # (B, B) cosine similarities

            B = len(batch)
            for i in range(B):
                # Image → Text
                i2t = sim[i].argsort(descending=True).tolist()
                top1_i2t.append(float(i2t[0] == i))
                top2_i2t.append(float(i in i2t[:2]))
                # Text → Image
                t2i = sim[:, i].argsort(descending=True).tolist()
                top1_t2i.append(float(t2i[0] == i))
                top2_t2i.append(float(i in t2i[:2]))

    return {
        'img2txt_top1': np.mean(top1_i2t) * 100,
        'img2txt_top2': np.mean(top2_i2t) * 100,
        'txt2img_top1': np.mean(top1_t2i) * 100,
        'txt2img_top2': np.mean(top2_t2i) * 100,
    }


# ─────────────────────────────────────────────────────────────
# Fine-tuning runner
# ─────────────────────────────────────────────────────────────

def run_stage1(cfg=None, device='cuda'):
    """
    Fine-tune BiomedCLIP with DHN-NCE on MedPix 2.0,
    then validate on ROCO.

    Paper training details:
      - LR: 1e-6
      - Batch size: 64
      - 50% decay rate (StepLR, gamma=0.5, applied each epoch)
      - Images: 224×224, normalized with CLIP RGB stats
      - MedPix split: 85/15 → 20,292 train / 3,515 val

    Number of epochs: NOT stated in the paper.
    We use early stopping on validation loss with patience=3.
    Training stops when val loss has not improved for 3 consecutive epochs,
    or after 20 epochs maximum.
    """
    if cfg is None:
        cfg = Stage1Config()

    Paths.makedirs()

    print("Loading BiomedCLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(cfg.model_name)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")

    train_pairs, val_pairs = load_medpix(
        Paths.MEDPIX_DIR,
        min_caption_len=cfg.min_caption_len,
        train_split=cfg.train_split,
    )

    train_ds = MedPixDataset(train_pairs, preprocess)
    val_ds   = MedPixDataset(val_pairs,   preprocess)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # Paper: "50% decay rate" → StepLR gamma=0.5 applied each epoch
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=cfg.lr_decay   # lr_decay=0.5
    )
    scaler = torch.cuda.amp.GradScaler()

    save_path = Paths.BIOMEDCLIP_FT
    best_val_loss  = float('inf')
    patience_count = 0
    MAX_EPOCHS     = 20   # hard ceiling; paper doesn't specify
    PATIENCE       = 3    # early stopping patience

    print(f"\nFine-tuning for up to {MAX_EPOCHS} epochs "
          f"(early stop patience={PATIENCE})...")

    for epoch in range(MAX_EPOCHS):
        # ── Train ──
        model.train()
        train_loss_accum = torch.zeros(1, device=device)
        for batch_idx, (images, captions) in enumerate(train_loader):
            # Start H2D image transfer immediately (non-blocking),
            # then tokenize on CPU while the transfer is in flight.
            images = images.to(device, non_blocking=True)
            tokens = tokenizer(list(captions)).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                img_emb = F.normalize(model.encode_image(images), dim=-1)
                txt_emb = F.normalize(model.encode_text(tokens),  dim=-1)
                loss = dhn_nce_loss(img_emb, txt_emb,
                                     cfg.tau, cfg.beta1, cfg.beta2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_accum += loss.detach()

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {batch_idx}/{len(train_loader)} "
                      f"| Loss {loss.item():.4f}", flush=True)

        # ── Validate ──
        model.eval()
        val_loss_accum = torch.zeros(1, device=device)
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device, non_blocking=True)
                tokens = tokenizer(list(captions)).to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    img_emb = F.normalize(model.encode_image(images), dim=-1)
                    txt_emb = F.normalize(model.encode_text(tokens),  dim=-1)
                    loss = dhn_nce_loss(img_emb, txt_emb,
                                         cfg.tau, cfg.beta1, cfg.beta2)
                val_loss_accum += loss.detach()

        train_loss = (train_loss_accum / len(train_loader)).item()
        val_loss   = (val_loss_accum   / len(val_loader)).item()
        lr_now     = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Train {train_loss:.4f} "
              f"| Val {val_loss:.4f} | LR {lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"  → Best saved (val={val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print("Early stopping triggered.")
                break

        scheduler.step()

    # ── ROCO Validation ──
    print("\nValidating on ROCO (Table 2)...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    roco_pairs   = load_roco(Paths.ROCO_DIR)
    roco_results = evaluate_roco(model, preprocess, tokenizer,
                                  roco_pairs, cfg, device)

    print("\n── Table 2: ROCO Cross-Modal Retrieval ──")
    print(f"  Img→Txt Top-1: {roco_results['img2txt_top1']:.2f}%  (target 84.70%)")
    print(f"  Img→Txt Top-2: {roco_results['img2txt_top2']:.2f}%  (target 94.73%)")
    print(f"  Txt→Img Top-1: {roco_results['txt2img_top1']:.2f}%  (target 85.99%)")
    print(f"  Txt→Img Top-2: {roco_results['txt2img_top2']:.2f}%  (target 95.17%)")

    if roco_results['img2txt_top1'] < 82.0:
        print("\n⚠ Top-1 below 82%. Do NOT proceed to Stage 2.")
        print("  Possible fixes:")
        print("  1. Re-clean MedPix data (check caption min_len, special chars)")
        print("  2. Reduce LR to 5e-7 and retrain from scratch")
        print("  3. Confirm batch_size=64 (needed to match paper)")
    else:
        print(f"\n✓ Stage 1 complete. Model: {save_path}")

    return save_path, roco_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_stage1(device=device)
