# MILK10k – what we learned so far (status 2026-09-28)

Repo: Documents\GitHub\260925MILK10k. Per-step details: results/step*/README.md, submissions/LOG.md, submissions/leaderboard/.
Metric: 11 per-class F1 scores (each class binarised at >= 0.5), macro mean ("Dice" on ISIC). Upload CSV first column must be `lesion_id`.

## Data (MILK10k training only — no external data used so far)
- 5,240 lesions, each 1 clinical close-up + 1 dermoscopic image, all 600x450 px. No patient ID.
- Extreme imbalance: BCC 48%, NV 14%, BKL 10%, SCCKA 9%, MEL 9%, AKIEC 6%, DF 52, INF 50, VASC 47, BEN_OTH 44, MAL_OTH 9 lesions.
- Evaluation: canonical stratified 5-fold (data/folds.csv), nested per-class threshold calibration, lesion bootstrap CIs, paired bootstrap for comparisons.
- Indirect data via pretrained weights: PanDerm-B (>2M derm images; overlap with ISIC/MILK10k sources NOT verified), EVA-02 (ImageNet-22k), DINOv2 (LVD-142M).

## Results ladder (CV, calibrated macro F1)
| Step | Setup | CV |
|---|---|---|
| 1 | LightGBM on metadata + MONET | 0.363 |
| 2 | frozen linear probe, best (PanDerm-B, both images + meta) | 0.453 |
| 3 | fine-tune dermoscopy only, 3-seed ens: EVA-02-B / PanDerm-B | 0.560 / 0.548 |
| 4 | + clinical image, late fusion (separate der + cli models, geo 50/50), 3-seed ens: EVA / PanDerm | 0.576 / **0.584** |
| 4 | joint models (shared / separate backbones), seed 0 | not better than late fusion |
| 5 | + metadata (late fused) | no robust gain |
| 8 | EVA + PanDerm late-fusion ensemble (chosen post hoc) | 0.596 |

## Leaderboard (479 test lesions)
- PanDerm-B late fusion: **0.546**, rank 16. EVA+PanDerm ensemble: 0.505 (worse; rare-class decisions changed).
- Test class mix differs strongly from training (derived from ISIC per-class details; must NOT be used for tuning):
  SCCKA 117, BKL 100, AKIEC 71, BCC 61 (13% vs 48% train), NV 52, MEL 39, INF 11, MAL_OTH ~10, BEN_OTH 7, DF 6, VASC 5.
- Our test AUC is high (mean 0.929, > leader's 0.905): the model ranks well; losses come from thresholds under prior shift (BCC F1 0.91 -> 0.61) and rare classes (MAL_OTH 0.00, BEN_OTH 0.22, INF 0.37).
- Rank 1 (Allegoria, external data: yes, 0.705): gap 0.159, of which 79% from MAL_OTH, BEN_OTH, INF, VASC (~33 test lesions). Leader has much higher average precision on rare classes (few, confident, correct rare predictions). Frequent classes: we are on par.

## Things tested that did NOT help
- Metadata/MONET fused with the image models (nested weight choice: -0.001..+0.011, n.s.).
- Label-shift correction (BCTS + EM prior estimate + plug-in F1): EM prior estimate too noisy at n=479 in simulation; true prior would give +0.03..+0.08.
- HiV15 augmentation pipeline (ported, bit-identical): with grayscale -0.018 late (clinical -0.046); without grayscale -0.005. Kept original augmentation.
- Post-hoc ensembles on the leaderboard: noisy (+-~0.06), dominated by a few rare-class lesions.

## Open / pending
- 448 px resolution run (script ready: src/step4_twoview.py --img 448 --tag _448). Earlier EVA study (other project) found no AUROC gain 224->448 on binary melanoma; MILK10k differs (native 600x450, thresholded macro F1).
- Hierarchical auxiliary head (48 granular diagnoses) – not started.

## Hand-off for the external-data exploration
- Biggest remaining lever by evidence: rare-class data (MAL_OTH, BEN_OTH, INF, VASC). Checked in MILK10k_Training_Supplement.csv:
  - MAL_OTH training = only 9 collision lesions. Allowed test diagnoses NOT in training: atypical fibroxanthoma, Kaposi sarcoma,
    T-cell/NK lymphocytic proliferations, malignant peripheral nerve sheath tumor, Merkel cell carcinoma.
  - BEN_OTH not in training: cylindroma, fibroepithelial polyp, scar, spiradenoma, trichilemmal/pilar cyst (training has 12 other subtypes, 1-8 each).
  - INF not in training: verruca (training: 46 generic inflammatory/infectious, 3 porokeratosis, 1 molluscum).
  - VASC not in training: arterio-venous malformation, lymphangioma (training: hemangioma 33, angiokeratoma 7, pyogenic granuloma 6, hobnail 1).
  - Which of these actually occur in the test set is undisclosed (paper: benchmark includes new granular diagnoses, mainly in BEN_OTH/MAL_OTH).
- Must check before use: (1) overlap/duplicates between external images and MILK10k test (MILK10k sources: Vienna, MSKCC, Ankara, Skopje, Queensland; paper matched 985 train lesions to HAM10000 via DINOv2 similarity); (2) licensing — per image/dataset (MILK10k itself is CC-BY-NC; check each external source); (3) modality — clinical close-ups needed as well as dermoscopy (check availability per source); late fusion needs a clinical model too; (4) label mapping to the 11 categories via ISIC-DX.
- Keep evaluation on MILK10k folds only (external data in training folds only); compare against late PanDerm-B 0.584 CV / 0.546 LB.
