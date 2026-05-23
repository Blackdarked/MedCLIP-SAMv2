<<<<<<< HEAD
# MedCLIP-SAMv2 Replication

Replication of **MedCLIP-SAMv2: Towards Universal Text-Driven Medical Image Segmentation** (Koleilat et al., 2025).

> Koleilat T., Asgariandehkordi H., Rivaz H., Xiao Y. (2025). MedCLIP-SAMv2: Towards Universal Text-Driven Medical Image Segmentation. *Medical Image Analysis*. arXiv:2409.19483v4.

Original paper code: https://github.com/HealthX-Lab/MedCLIP-SAMv2

---

## What this replication covers

| Component | Status |
|-----------|--------|
| DHN-NCE fine-tuning of BiomedCLIP | ✓ Full implementation |
| M2IB zero-shot saliency maps | ✓ Full implementation |
| Otsu + Connected Component post-processing | ✓ Full implementation |
| SAM ViT-H visual prompting | ✓ Full implementation |
| nnUNet weakly supervised training | ✓ Full implementation |
| Cyclical LR + checkpoint ensemble (Zhao 2022) | ✓ Full implementation |
| DSC + NSD evaluation with paired t-tests | ✓ Full implementation |
| All 4 datasets (Breast US, Brain MRI, Lung X-ray, Lung CT) | ✓ |

---

## Target results (Table 1)

| Dataset | Zero-shot DSC | Zero-shot NSD | Weakly Sup DSC | Weakly Sup NSD |
|---------|:------------:|:-------------:|:--------------:|:--------------:|
| Breast Ultrasound | 77.76% | 81.11% | 78.87% | 84.58% |
| Brain MRI | 76.52% | 82.23% | 80.03% | 88.25% |
| Lung X-ray | 75.79% | 80.88% | 80.77% | 84.53% |
| Lung CT | 80.38% | 82.03% | 88.78% | 91.95% |
| **Average** | **77.61%** | **81.56%** | **82.11%** | **87.33%** |

---

## File structure

```
config.py               Central config: all hyperparameters, paths, paper targets
datasets.py             Dataset loaders: MedPix, ROCO, BUSI, Brain, X-ray, CT
stage1_finetune.py      BiomedCLIP fine-tuning with DHN-NCE loss
stage2_segmentation.py  M2IB saliency → Otsu post-processing → SAM refinement
stage3_nnunet.py        nnUNet dataset prep, training, ensemble inference
cyclical_trainer.py     nnUNet custom trainer (cyclical LR + checkpoint saving)
evaluate.py             DSC, NSD, paired t-tests, all result tables
run_all.py              Master pipeline runner
requirements.txt        Python dependencies
```

---

## Setup

### 1. Hardware requirements

- GPU with ≥ 8GB VRAM (tested on RTX 4060 Ti 16GB)
- Python 3.10+
- Windows 10/11 or Linux

### 2. Install PyTorch

Install first, separately from the rest, to get the right CUDA build:

```bash
# CUDA 12.x (RTX 40-series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download SAM ViT-H weights

```bash
# ~2.4 GB
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -o sam_vit_h_4b8939.pth
```

Move to: `C:\medclip_samv2\models\sam_vit_h_4b8939.pth`

### 5. Configure paths

Edit `config.py`:

```python
BASE_DIR = r'C:\medclip_samv2'   # your project root
VRAM_GB  = 16                     # 8 or 16
```

### 6. Download datasets

| Dataset | Source | Path |
|---------|--------|------|
| MedPix 2.0 | https://medpix.nlm.nih.gov | `data/medpix/` |
| ROCO | https://github.com/razorx89/roco-dataset | `data/roco/` |
| BUSI | Kaggle: `aryashah2k/breast-ultrasound-images-dataset` | `data/BUSI/` |
| Brain Tumor MRI | Kaggle: `masoudnickparvar/brain-tumor-mri-dataset` | `data/brain_tumor_mri/` |
| COVID-QU-Ex | Kaggle: `tawsifurrahman/covid19-radiography-database` | `data/COVID_QU_Ex/` |
| Lung CT | Kaggle: `kmader/finding-lungs-in-ct-data` | `data/lung_ct/` |

### 7. Create directories

```bash
python -c "from config import Paths; Paths.makedirs()"
```

---

## Running

### Full pipeline (all 4 datasets)

```bash
python run_all.py
```

### One dataset at a time

```bash
python run_all.py --dataset breast
python run_all.py --dataset brain
python run_all.py --dataset xray
python run_all.py --dataset ct
```

### Skip Stage 1 (use pre-trained BiomedCLIP)

```bash
python run_all.py --skip-stage1
```

### Resume after interruption

```bash
python run_all.py --resume
```

M2IB progress auto-saves every 200 images. nnUNet resumes via `--c` flag.

### Evaluation only

```bash
python run_all.py --eval-only
```

---

## Implementation notes

### DHN-NCE loss (paper equations 9–13)

The hardness weight exponent uses the **temperature-scaled** similarity `β · (I·T)/τ`, not the raw dot product. This is the correct reading of equation 12. At β=0.15 and τ=0.6, the exponent is `exp(0.25 · dot_product)`, giving noticeably sharper hardness contrast than `exp(0.15 · dot_product)`.

### Fine-tuning epochs

The paper states LR=1e-6, 50% decay rate, batch=64 but does not specify epoch count. This replication uses early stopping (patience=3, max 20 epochs) as the stopping criterion, which is the standard approach when epoch count is unspecified.

### "All" column computation

The paper's "All" column pools all test-set predictions from all four datasets into a single list before computing mean ± std. This differs from averaging per-dataset means when dataset sizes are unequal (113, 600, 957, 1800 images).

### Paired t-tests

Paper: "Paired-sample t-tests were also conducted to validate the observed trends, with a p-value of less than 0.05 indicating statistical significance." All table comparisons include p-values via `scipy.stats.ttest_rel`.

### Windows-specific

- `num_workers=0` in all DataLoaders (Windows multiprocessing)
- `subprocess.run()` instead of `os.system()` for nnUNet CLI calls
- `if __name__ == '__main__':` guard in `run_all.py` (required for nnUNet on Windows)

---

## Known limitations

- **UDIAT dataset** (breast validation/test): requires institutional access from Byra et al. (2020). If unavailable, use an 80/10/10 split of BUSI — breast numbers will differ slightly from the paper.
- **M2IB implementation**: our implementation approximates the official Wang et al. (2024) code. For exact reproduction, replace `M2IBExtractor` in `stage2_segmentation.py` with the official repo at https://github.com/YingWang-NYU/M2IB.
- **Number of fine-tuning epochs**: not stated in paper; results may vary slightly depending on when early stopping triggers.

---

## Citation

```bibtex
@article{koleilat2025medclipsamv2,
  title={MedCLIP-SAMv2: Towards Universal Text-Driven Medical Image Segmentation},
  author={Koleilat, Taha and Asgariandehkordi, Hojat and Rivaz, Hassan and Xiao, Yiming},
  journal={Medical Image Analysis},
  year={2025},
  note={arXiv:2409.19483}
}
```
=======
# MedCLIP-SAMv2
>>>>>>> 68d31be0adf1ae258ddff95b2b565674d30a9c06
