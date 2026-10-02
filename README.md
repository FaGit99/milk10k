# MILK10k skin-lesion classification — PanDerm-B late fusion

Code and result summaries for an entry to the **ISIC MILK10k Benchmark**: 11-class skin-lesion diagnosis from a
clinical close-up and a dermoscopic image per lesion.

| | Calibrated macro F1 |
|---|---|
| 5-fold cross-validation (5,240 training lesions) | 0.584 (95% CI 0.561–0.606) |
| MILK10k Benchmark leaderboard (479 test lesions) | 0.546 (rank 16, end of September 2026) |

Model: PanDerm-Base (ViT-B/16) fine-tuned separately on the dermoscopic and the clinical image, predictions combined
by geometric mean (late fusion), 3 seeds × 5 folds, per-class thresholds fitted on out-of-fold predictions.
Only MILK10k training data was used (no external images).

The full write-up — what was done and why, what worked and what did not — is in
[`paper/MILK10k_Skin-Lesion_Classification_Fabian_Wolz.pdf`](paper/MILK10k_Skin-Lesion_Classification_Fabian_Wolz.pdf).
A shorter running log is in [`LEARNINGS.md`](LEARNINGS.md).

*Authorship note: the work was largely done by Claude Opus 5.5 (Anthropic), with guidance and experience contributed
by the author, Fabian Wolz.*

## Repository layout

| Path | Content |
|---|---|
| `src/common.py` | data loading, challenge metric, nested threshold calibration, bootstrap CIs |
| `src/make_folds.py`, `data/folds.csv` | canonical stratified 5-fold split (seed 42) used by every step |
| `src/step1_baseline.py` | Step 1: metadata + MONET baselines (LogReg, LightGBM) |
| `src/step2_*.py`, `run_step2.bat` | Step 2: frozen embeddings + linear probe (PanDerm-B, DINOv2-B, EVA-02-B/S) |
| `src/step3_finetune.py`, `run_step3.bat` | Step 3: full fine-tuning on dermoscopy |
| `src/step4_*.py` | Step 4: clinical image — late fusion, shared and separate joint models; `--img 448` = Step 7 |
| `src/step5_precheck.py` | Step 5: metadata fused with image models |
| `src/step8_ensemble.py` | Step 8: EVA-02-B + PanDerm-B ensemble |
| `src/step9_crt.py` | Step 9: classifier re-training / class balancing |
| `src/step10_*.py`, `run_step10.bat` | Step 10: epoch budget sweep |
| `src/priorshift.py` | label-shift correction (BCTS + EM), validated by simulation |
| `src/aug_hiv15.py` | alternative augmentation pipeline (A/B test) |
| `results/*/` | per-step README, `summary.csv`, `metrics.json` |
| `submissions/LOG.md`, `submissions/leaderboard/` | leaderboard submissions and per-class details |

Step 6 has no recorded experiment; the numbering follows the code.

## Setup

1. **Python environment** (versions in `requirements.txt`; torch from the CUDA 12.8 index):
   ```
   pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
2. **MILK10k data** from the ISIC Challenge site (CC-BY-NC). Place in the repository root:
   `MILK10k_Training_Input/`, `MILK10k_Training_GroundTruth.csv`, `MILK10k_Training_Metadata.csv`,
   `MILK10k_Training_Supplement.csv`, `MILK10k_Test_Input/`, `MILK10k_Test_Metadata.csv`.
3. **PanDerm code** (CC-BY-NC-ND 4.0, not redistributed here). Download the unmodified file at the pinned commit:
   ```
   curl -L -o third_party/panderm/modeling_finetune.py https://raw.githubusercontent.com/SiyuanYan1/PanDerm/fd7a80748ba7fc3e203fed88f909f4689d0d6f24/classification/models/modeling_finetune.py
   ```
4. **PanDerm-Base weights**: download `panderm_bb_data6_checkpoint-499.pth` (link in the
   [PanDerm README](https://github.com/SiyuanYan1/PanDerm)) to `weights/panderm_bb_data6_checkpoint-499.pth`.
   EVA-02 and DINOv2 weights are loaded through `timm`.

## Running

Steps build on each other's outputs (Step 3+ compare against Steps 1–2). Typical order:
```
python src/make_folds.py            # or use the provided data/folds.csv
python src/step1_baseline.py
run_step2.bat
run_step3.bat --backbones panderm_b eva02_b --seeds 0 1 2
python src/step4_twoview.py --variants der cli --backbones panderm_b --seeds 0 1 2
```
Each script documents its arguments in the header docstring. A Step 4 run (PanDerm-B, one view, 5 folds, 25 epochs)
took about 17 min on an RTX PRO 6000 Blackwell (96 GB).

## Evaluation protocol

All CV numbers are out-of-fold on the canonical 5 folds. Metric = challenge rule (per-class F1 at ≥ 0.5, macro mean).
Per-class thresholds are fitted on the other 4 folds only ("calib"). Comparisons use a paired lesion bootstrap.
Test-set class counts derived from leaderboard feedback were never used for tuning.

## Licence

Code in this repository: MIT (see `LICENSE`). MILK10k data, PanDerm code and weights, and other third-party
components keep their own licences and are not included.

## References

- MILK10k Benchmark, ISIC Challenge (organisers H. Kittler, P. Tschandl, Medical University of Vienna).
- Yan S. et al., PanDerm — a multimodal vision foundation model for clinical dermatology. https://github.com/SiyuanYan1/PanDerm
- Kim C. et al., MONET. Nat Med 2024;30(4):1154–1165. doi:10.1038/s41591-024-02887-x
- Menon A.K. et al., Long-tail learning via logit adjustment. ICLR 2021.
