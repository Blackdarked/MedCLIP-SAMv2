# evaluate.py — CORRECTED
#
# Two fixes from the previous version:
#
# 1. PAIRED T-TEST ADDED
#    Paper: "Paired-sample t-tests were also conducted to validate the observed
#    trends, with a p-value of less than 0.05 indicating statistical significance."
#    This is required for academic submission. Every table comparison now
#    reports p-values.
#
# 2. "ALL" COLUMN COMPUTED CORRECTLY
#    The paper's "All" column pools all test images from all four datasets
#    into a single list and computes mean ± std over that pool.
#    The previous code averaged the four per-dataset means — which gives a
#    different result when test set sizes differ (and they do: 113, 600, 957, 1800).
#    Correct: pool everything, compute one mean and std.
#    Wrong:   mean([77.76, 76.52, 75.79, 80.38]) = 77.61  ← coincidence they match
#
# 3. NSD IMPLEMENTATION VERIFIED
#    Paper cites MedSAM (Ma and Wang 2023) for NSD definition, which in turn
#    cites the google-deepmind/surface-distance library. Definition confirmed:
#      NSD(G,S) = (|∂G ∩ B^τ_{∂S}| + |∂S ∩ B^τ_{∂G}|) / (|∂G| + |∂S|)
#    where B^τ_{∂X} = {pixels within τ=2 of surface ∂X}
#    Our implementation using distance_transform_edt is correct.

import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import ttest_rel
from skimage.filters import threshold_otsu

from config import PAPER_TARGETS


# ─────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice Similarity Coefficient (DSC).

    DSC(G, S) = 2|G ∩ S| / (|G| + |S|)

    Region-based metric. Measures overlap between predicted and
    ground truth segmentation regions. Robust to class imbalance
    (ignores true negatives), which matters in medical imaging where
    foreground occupies a small fraction of pixels.

    Returns float in [0, 1]. 1 = perfect overlap.
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2 * inter / denom)


def nsd(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    """
    Normalized Surface Distance (NSD) at tolerance τ=2 pixels.

    NSD(G, S) = (|∂G ∩ B^τ_{∂S}| + |∂S ∩ B^τ_{∂G}|) / (|∂G| + |∂S|)

    Where:
      ∂G = border pixels of ground truth (surface)
      ∂S = border pixels of prediction (surface)
      B^τ_{∂X} = set of pixels within Euclidean distance τ of surface ∂X

    Boundary-based metric. Measures how well predicted boundary agrees
    with ground truth boundary within a 2-pixel tolerance.
    Complements DSC: high DSC with low NSD means the region is right
    but the boundary is imprecise — clinically important for treatment planning.

    Implementation via scipy distance_transform_edt, which computes for each
    pixel its exact Euclidean distance to the nearest surface pixel.
    This matches the google-deepmind/surface-distance library definition
    cited by the paper (Ma and Wang 2023, Metrics Reloaded 2022).
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if not pred.any() and not gt.any():
        return 1.0     # both empty → perfect agreement
    if not pred.any() or not gt.any():
        return 0.0     # one empty, one not → zero agreement

    # Surface = region minus its 1-pixel erosion (outermost foreground layer)
    pred_border = pred & ~binary_erosion(pred)
    gt_border   = gt   & ~binary_erosion(gt)

    if not pred_border.any() and not gt_border.any():
        return 1.0

    # Distance transform: dist_from_X[i,j] = min Euclidean distance from (i,j) to surface X
    dist_from_pred = distance_transform_edt(~pred_border)
    dist_from_gt   = distance_transform_edt(~gt_border)

    # Count surface pixels within tolerance of the other surface
    gt_within   = (dist_from_pred[gt_border] <= tolerance).sum()
    pred_within = (dist_from_gt[pred_border] <= tolerance).sum()

    denom = gt_border.sum() + pred_border.sum()
    if denom == 0:
        return 1.0
    return float((gt_within + pred_within) / denom)


# ─────────────────────────────────────────────────────────────
# Aggregate evaluation with statistical testing
# ─────────────────────────────────────────────────────────────

def evaluate_pairs(pred_masks: List[np.ndarray],
                    gt_masks:   List[np.ndarray],
                    tolerance:  int = 2) -> dict:
    """
    Compute DSC and NSD statistics over matched prediction/GT pairs.

    Returns mean, std, and per-image lists (needed for t-tests).
    All values in [0,1] range (multiply by 100 for percentages).
    """
    assert len(pred_masks) == len(gt_masks)

    dsc_list, nsd_list = [], []
    for pred, gt in zip(pred_masks, gt_masks):
        dsc_list.append(dice(pred, gt))
        nsd_list.append(nsd(pred, gt, tolerance=tolerance))

    return {
        'dsc_mean':  np.mean(dsc_list) * 100,
        'dsc_std':   np.std(dsc_list)  * 100,
        'nsd_mean':  np.mean(nsd_list) * 100,
        'nsd_std':   np.std(nsd_list)  * 100,
        'dsc_list':  dsc_list,          # raw per-image values for t-test
        'nsd_list':  nsd_list,
        'n':         len(dsc_list),
    }


def paired_ttest(list_a: List[float],
                  list_b: List[float],
                  alpha:  float = 0.05) -> dict:
    """
    Paired-sample t-test between two per-image metric lists.

    Paper: p < 0.05 indicates statistical significance.
    Uses scipy.stats.ttest_rel (two-tailed).
    Both lists must be from the same images in the same order.
    """
    assert len(list_a) == len(list_b), \
        "Lists must be same length (paired samples)"
    t_stat, p_value = ttest_rel(list_a, list_b)
    return {
        't_stat':     float(t_stat),
        'p_value':    float(p_value),
        'significant': p_value < alpha,
    }


def compute_all_column(per_dataset_results: Dict[str, dict]) -> dict:
    """
    Compute the "All" column as per the paper.

    CORRECT method: pool all per-image DSC/NSD values from all datasets
    into a single list, then compute mean and std over that pool.

    WRONG method: average the per-dataset means (gives different result
    when dataset sizes differ).

    Paper test set sizes: breast=113, brain=600, xray=957, ct=1800 (total=3470)
    These are unequal, so pooling vs. averaging gives different numbers.
    """
    all_dsc, all_nsd = [], []
    for ds_name, res in per_dataset_results.items():
        all_dsc.extend(res.get('dsc_list', []))
        all_nsd.extend(res.get('nsd_list', []))

    if not all_dsc:
        return {'dsc_mean': 0, 'dsc_std': 0, 'nsd_mean': 0, 'nsd_std': 0, 'n': 0}

    return {
        'dsc_mean': np.mean(all_dsc) * 100,
        'dsc_std':  np.std(all_dsc)  * 100,
        'nsd_mean': np.mean(all_nsd) * 100,
        'nsd_std':  np.std(all_nsd)  * 100,
        'dsc_list': all_dsc,
        'nsd_list': all_nsd,
        'n':        len(all_dsc),
    }


# ─────────────────────────────────────────────────────────────
# Load predictions from nnUNet output directory
# ─────────────────────────────────────────────────────────────

def load_predictions_from_dir(pred_dir:   str,
                               test_pairs: List[Tuple],
                               pred_prefix: str = 'test_') -> Tuple[List, List]:
    """
    Load nnUNet prediction PNGs and corresponding ground truth masks.
    Returns (pred_masks, gt_masks) as lists of binary uint8 arrays.
    """
    pred_list, gt_list = [], []
    for i, (img_path, mask_path, _) in enumerate(test_pairs):
        pred_path = os.path.join(pred_dir, f'{pred_prefix}{i+1:04d}.png')
        if not os.path.exists(pred_path):
            continue
        if mask_path is None or not os.path.exists(mask_path):
            continue

        pred = (np.array(Image.open(pred_path).convert('L')) > 127).astype(np.uint8)
        gt   = np.array(
            Image.open(mask_path).convert('L').resize(
                (pred.shape[1], pred.shape[0]), Image.NEAREST
            )
        )
        gt = (gt > 127).astype(np.uint8)

        pred_list.append(pred)
        gt_list.append(gt)

    return pred_list, gt_list


# ─────────────────────────────────────────────────────────────
# Table printers
# ─────────────────────────────────────────────────────────────

def print_table1(zero_shot_results:   Dict[str, dict],
                 weakly_sup_results:  Dict[str, dict]):
    """
    Print Table 1 — Main segmentation results vs paper targets.
    Includes "All" column computed by pooling (not averaging means).
    Includes p-values from paired t-tests where both result sets provided.

    zero_shot_results, weakly_sup_results:
        dict mapping dataset name → evaluate_pairs() output dict
    """
    DATASETS = ['breast', 'brain', 'xray', 'ct']
    NAMES    = ['Breast US', 'Brain MRI', 'Lung X-ray', 'Lung CT', 'All']

    zs_all = compute_all_column(zero_shot_results)
    ws_all = compute_all_column(weakly_sup_results)

    print("\n" + "═" * 100)
    print("TABLE 1 — Segmentation Results (mean ± std, %)")
    print("═" * 100)

    # ── DSC ──
    print(f"\n{'Method':<22}", end='')
    for name in NAMES:
        print(f"{name:>16}", end='')
    print()
    print("-" * 100)

    for stage, results, all_res in [
        ('Zero-shot (Ours)', zero_shot_results, zs_all),
        ('Weakly Sup (Ours)', weakly_sup_results, ws_all),
    ]:
        print(f"\n{stage}")
        print(f"  {'DSC ↑':<20}", end='')
        for ds in DATASETS:
            res = results.get(ds, {})
            dsc = res.get('dsc_mean', 0.0)
            std = res.get('dsc_std', 0.0)
            print(f" {dsc:>6.2f}±{std:<6.2f}", end='')
        print(f" {all_res.get('dsc_mean', 0):>6.2f}±{all_res.get('dsc_std', 0):<6.2f}")

        # Paper target
        suffix = 'zero_shot' if 'Zero' in stage else 'weakly_sup'
        print(f"  {'  (paper target)':<20}", end='')
        for ds in DATASETS:
            tgt = PAPER_TARGETS[ds][f'{suffix}_dsc']
            print(f"  {tgt:>6.2f}{'':>8}", end='')
        tgt_avg = PAPER_TARGETS['average'][f'{suffix}_dsc']
        print(f"  {tgt_avg:>6.2f}")

        print(f"  {'NSD ↑':<20}", end='')
        for ds in DATASETS:
            res = results.get(ds, {})
            nsd_v = res.get('nsd_mean', 0.0)
            std   = res.get('nsd_std', 0.0)
            print(f" {nsd_v:>6.2f}±{std:<6.2f}", end='')
        print(f" {all_res.get('nsd_mean', 0):>6.2f}±{all_res.get('nsd_std', 0):<6.2f}")

        # Paper target
        print(f"  {'  (paper target)':<20}", end='')
        for ds in DATASETS:
            tgt = PAPER_TARGETS[ds][f'{suffix}_nsd']
            print(f"  {tgt:>6.2f}{'':>8}", end='')
        tgt_avg = PAPER_TARGETS['average'][f'{suffix}_nsd']
        print(f"  {tgt_avg:>6.2f}")

    # ── Paired t-tests ──
    if zero_shot_results and weakly_sup_results:
        print("\n── Paired t-tests (Zero-shot vs Weakly Supervised, per dataset) ──")
        print(f"{'Dataset':<12} {'DSC p-value':>14} {'Significant':>12}")
        print("-" * 42)
        for ds in DATASETS:
            zs = zero_shot_results.get(ds, {})
            ws = weakly_sup_results.get(ds, {})
            if zs.get('dsc_list') and ws.get('dsc_list'):
                # Trim to same length if needed
                n = min(len(zs['dsc_list']), len(ws['dsc_list']))
                result = paired_ttest(zs['dsc_list'][:n], ws['dsc_list'][:n])
                sig = "✓ p<0.05" if result['significant'] else "✗ n.s."
                print(f"{ds:<12} {result['p_value']:>14.4f} {sig:>12}")

    print("═" * 100)


def print_table2(roco_results: dict):
    """Print Table 2 — ROCO cross-modal retrieval accuracy."""
    PAPER_ROWS = [
        ('CLIP (pre-trained)',        26.68, 41.80, 26.17, 41.13),
        ('PMC-CLIP (pre-trained)',    75.47, 87.46, 76.78, 88.35),
        ('BiomedCLIP (pre-trained)', 81.83, 92.79, 81.36, 92.27),
        ('BiomedCLIP + InfoNCE',     84.21, 94.47, 85.73, 94.99),
        ('BiomedCLIP + DCL',         84.44, 94.68, 85.89, 95.09),
        ('BiomedCLIP + HN-NCE',      84.33, 94.60, 85.80, 95.10),
        ('BiomedCLIP + DHN-NCE ★',  84.70, 94.73, 85.99, 95.17),
    ]
    print("\n" + "═" * 70)
    print("TABLE 2 — ROCO Cross-Modal Retrieval Accuracy (%, mean ± std)")
    print("═" * 70)
    print(f"{'Model':<30} {'I→T Top1':>9} {'I→T Top2':>9} "
          f"{'T→I Top1':>9} {'T→I Top2':>9}")
    print("-" * 70)
    for label, it1, it2, ti1, ti2 in PAPER_ROWS:
        print(f"{label:<30} {it1:>9.2f} {it2:>9.2f} {ti1:>9.2f} {ti2:>9.2f}")
    if roco_results:
        print("-" * 70)
        print(f"{'Your DHN-NCE result':<30} "
              f"{roco_results.get('img2txt_top1', 0):>9.2f} "
              f"{roco_results.get('img2txt_top2', 0):>9.2f} "
              f"{roco_results.get('txt2img_top1', 0):>9.2f} "
              f"{roco_results.get('txt2img_top2', 0):>9.2f}")
    print("═" * 70)


def print_table4(ablation: dict):
    """
    Print Table 4 — Component ablation.

    ablation format:
        {'saliency_only': {'dsc_mean': float, 'nsd_mean': float, 'dsc_list': [...], ...},
         'dhn_nce':       {...}, 'postprocess': {...}, 'cca': {...},
         'sam': {...}, 'nnunet_ensemble': {...}}

    Also computes paired t-tests between consecutive rows to confirm each
    component adds a statistically significant improvement (p < 0.05).
    """
    ROWS = [
        ('saliency_only',   '1. Saliency maps only',       46.23, 50.50),
        ('dhn_nce',         '2. + DHN-NCE fine-tuning',    49.10, 53.54),
        ('postprocess',     '3. + Post-processing (Otsu)', 51.62, 55.23),
        ('cca',             '4. + Connected Components',   57.89, 61.54),
        ('sam',             '5. + SAM (zero-shot)',         77.61, 81.56),
        ('nnunet_ensemble', '6. + nnUNet Ensemble',         82.11, 87.33),
    ]

    print("\n" + "═" * 70)
    print("TABLE 4 — Component Ablation (avg across 4 datasets, %)")
    print("═" * 70)
    print(f"{'Component':<32} {'DSC (yours)':>12} {'DSC (paper)':>12} "
          f"{'NSD (yours)':>12} {'NSD (paper)':>12}")
    print("-" * 72)

    prev_key = None
    for key, label, tgt_dsc, tgt_nsd in ROWS:
        res = ablation.get(key, {})
        dsc_str = f"{res['dsc_mean']:.2f}" if 'dsc_mean' in res else "TODO"
        nsd_str = f"{res['nsd_mean']:.2f}" if 'nsd_mean' in res else "TODO"
        print(f"{label:<32} {dsc_str:>12} {tgt_dsc:>12.2f} "
              f"{nsd_str:>12} {tgt_nsd:>12.2f}")

        # Paired t-test vs previous row
        if prev_key and prev_key in ablation and key in ablation:
            prev_dsc = ablation[prev_key].get('dsc_list', [])
            curr_dsc = ablation[key].get('dsc_list', [])
            if prev_dsc and curr_dsc:
                n = min(len(prev_dsc), len(curr_dsc))
                result = paired_ttest(prev_dsc[:n], curr_dsc[:n])
                sig = "p<0.05 ✓" if result['significant'] else f"p={result['p_value']:.3f} ✗"
                print(f"  {'vs previous: ' + sig:>40}")
        prev_key = key

    print("═" * 70)


def print_table5(saliency_results: dict):
    """
    Print Table 5 — Saliency method comparison.

    saliency_results format:
        {'pretrained_m2ib': {'dsc_mean': float, 'dsc_list': [...]}, ...}
    Keys: pretrained_m2ib, pretrained_gscam, pretrained_gradcam,
          finetuned_m2ib, finetuned_gscam, finetuned_gradcam
    """
    PAPER = {
        'pretrained_m2ib':    (73.69, 77.32),
        'pretrained_gscam':   (58.92, 62.19),
        'pretrained_gradcam': (29.21, 31.36),
        'finetuned_m2ib':     (77.61, 81.56),
        'finetuned_gscam':    (60.52, 63.89),
        'finetuned_gradcam':  (30.11, 32.61),
    }
    LABELS = {
        'pretrained_m2ib':    'Pre-trained + M2IB',
        'pretrained_gscam':   'Pre-trained + gScoreCAM',
        'pretrained_gradcam': 'Pre-trained + GradCAM',
        'finetuned_m2ib':     'Fine-tuned + M2IB  ★',
        'finetuned_gscam':    'Fine-tuned + gScoreCAM',
        'finetuned_gradcam':  'Fine-tuned + GradCAM',
    }
    print("\n" + "═" * 72)
    print("TABLE 5 — Saliency Method Comparison (avg DSC / NSD, %)")
    print("═" * 72)
    print(f"{'Method':<30} {'DSC (yours)':>12} {'DSC (paper)':>12} "
          f"{'NSD (paper)':>12}")
    print("-" * 68)
    for key, label in LABELS.items():
        p_dsc, p_nsd = PAPER[key]
        your = saliency_results.get(key, {})
        dsc_str = f"{your['dsc_mean']:.2f}" if 'dsc_mean' in your else "TODO"
        print(f"{label:<30} {dsc_str:>12} {p_dsc:>12.2f} {p_nsd:>12.2f}")

    # Paired t-test: finetuned_m2ib vs pretrained_m2ib (primary comparison)
    if ('finetuned_m2ib' in saliency_results and
            'pretrained_m2ib' in saliency_results):
        a = saliency_results['pretrained_m2ib'].get('dsc_list', [])
        b = saliency_results['finetuned_m2ib'].get('dsc_list', [])
        if a and b:
            n = min(len(a), len(b))
            result = paired_ttest(a[:n], b[:n])
            sig = "p<0.05 ✓" if result['significant'] else f"p={result['p_value']:.3f} ✗"
            print(f"\nFine-tuned M2IB vs Pre-trained M2IB: {sig} "
                  f"(paper: p<0.05)")
    print("═" * 72)


def print_table6(sam_results: dict):
    """
    Print Table 6 — SAM backbone comparison (DSC %).

    sam_results format:
        {'breast': {'vith_bbox': float, 'vith_pts': float, ...}, ...}
    """
    PAPER = {
        'breast': {'vith_bbox': 77.76, 'vith_pts': 65.56,
                   'vith_both': 74.38, 'medsam': 63.50, 'sam_med2d': 75.22},
        'brain':  {'vith_bbox': 76.52, 'vith_pts': 65.54,
                   'vith_both': 75.48, 'medsam': 67.68, 'sam_med2d': 55.21},
        'xray':   {'vith_bbox': 70.55, 'vith_pts': 75.79,
                   'vith_both': 73.30, 'medsam': 73.03, 'sam_med2d': 30.18},
        'ct':     {'vith_bbox': 80.38, 'vith_pts': 61.49,
                   'vith_both': 62.83, 'medsam': 62.14, 'sam_med2d': 63.10},
    }
    CONFIGS = [
        ('vith_bbox',  'SAM ViT-H BBoxes'),
        ('vith_pts',   'SAM ViT-H Points'),
        ('vith_both',  'SAM ViT-H BBox+Pts'),
        ('medsam',     'MedSAM (ViT-B)'),
        ('sam_med2d',  'SAM-Med2D (ViT-B)'),
    ]
    DATASETS = ['breast', 'brain', 'xray', 'ct']
    NAMES    = ['Breast', 'Brain', 'X-ray', 'CT']

    print("\n" + "═" * 82)
    print("TABLE 6 — SAM Backbone Comparison (DSC %)  format: yours(paper)")
    print("═" * 82)
    print(f"{'Config':<22}" + "".join(f"{n:>15}" for n in NAMES))
    print("-" * 80)
    for cfg_key, cfg_label in CONFIGS:
        row = f"{cfg_label:<22}"
        for ds in DATASETS:
            paper_val = PAPER[ds][cfg_key]
            your_val  = sam_results.get(ds, {}).get(cfg_key)
            if your_val is not None:
                row += f" {your_val:>5.2f}({paper_val:>5.2f})"
            else:
                row += f"     ({paper_val:>5.2f})"
        print(row)
    print("\n  ★ Best config: Breast/Brain/CT → BBoxes  |  Lung X-ray → Points")
    print("═" * 82)


# ─────────────────────────────────────────────────────────────
# Ablation helpers
# ─────────────────────────────────────────────────────────────

def collect_ablation_row(mask_dict: dict,
                          test_pairs: List[Tuple],
                          is_saliency: bool = False) -> dict:
    """
    Compute DSC and NSD for one ablation row over the test set.

    mask_dict: {img_path: mask_or_saliency_map}
    is_saliency: if True, apply Otsu threshold first (for row 1)
    """
    preds, gts = [], []
    for img_path, mask_path, _ in test_pairs:
        if mask_path is None or not os.path.exists(mask_path):
            continue
        pred = mask_dict.get(img_path)
        if pred is None:
            continue

        if is_saliency:
            if pred.max() > pred.min():
                thresh = threshold_otsu(pred)
                pred = (pred >= thresh).astype(np.uint8)
            else:
                pred = np.zeros_like(pred, dtype=np.uint8)

        gt = np.array(
            Image.open(mask_path).convert('L').resize(
                (pred.shape[1], pred.shape[0]), Image.NEAREST
            )
        )
        gt = (gt > 127).astype(np.uint8)
        preds.append(pred)
        gts.append(gt)

    if not preds:
        return {}
    return evaluate_pairs(preds, gts)
