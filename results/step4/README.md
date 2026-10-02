# Step 4 - add the clinical close-up

## Pre-check (CPU, no training) - src/step4_precheck.py
Fusion of existing OOF predictions: Step 3 fine-tuned dermoscopy model (3-seed ensemble) + Step 2 frozen clinical probe (crop).
Calibrated macro F1; delta = paired lesion bootstrap vs dermoscopy-only reference.

| backbone | der (ref) | geo50 | geo75 | stack |
|---|---|---|---|---|
| eva02_b | 0.560 | 0.572, +0.012 (-0.013..+0.034) | 0.566, +0.007 (-0.006..+0.019) | 0.513, -0.047 (-0.072..-0.022) |
| panderm_b | 0.548 | 0.564, +0.016 (-0.003..+0.036) | 0.559, +0.011 (-0.001..+0.025) | 0.513, -0.034 (-0.054..-0.014) |

Geometric-mean fusion: positive point estimates for both backbones, CIs include 0. Stacker: worse.
Per class: precheck_per_class.csv; all numbers: precheck_metrics.json.

## Training - src/step4_twoview.py
Variants cli (-> late fusion), shared, sep; Step 3 settings unchanged. Smoke-tested (training path on CPU with random init,
evaluation/fusion/submission path with synthetic predictions) before the GPU run.

## Stage A results (seed 0, 2026-09-26) - calibrated macro F1, delta vs Step 3 dermoscopy-only seed 0 (paired bootstrap)

| variant | eva02_b | panderm_b |
|---|---|---|
| der only (Step 3 s0) | 0.551 | 0.545 |
| cli only | 0.462, -0.089 (-0.123..-0.057) | 0.482, -0.063 (-0.089..-0.037) |
| late (der s0 + cli s0, geo 50/50) | 0.573, +0.022 (-0.001..+0.048) | 0.578, +0.033 (+0.014..+0.053) |
| shared | 0.562, +0.011 (-0.015..+0.037) | 0.565, +0.020 (+0.001..+0.040) |
| sep | 0.555, +0.004 (-0.020..+0.030) | 0.572, +0.027 (+0.005..+0.050) |

Macro F1 excluding MAL_OTH (9 lesions): eva02_b der 0.589 / late 0.625 / shared 0.612 / sep 0.604;
panderm_b der 0.598 / late 0.633 / shared 0.618 / sep 0.614.
Variant vs variant (paired): no difference resolved except eva02_b late - sep +0.018 (+0.000..+0.036, borderline).
Ensembling control: two dermoscopy models (s0+s1 or s0+s2, same fusion) = 0.551/0.559 (eva02_b), 0.543/0.542 (panderm_b),
i.e. the late-fusion gain is not explained by having two networks.
Time per 5 folds: cli ~23 min (eva) / ~17 min (panderm); shared ~41 / ~30; sep ~40 / ~29.

## Stage B results (late fusion = cli variant, seeds 0,1,2; 2026-09-28)

| | eva02_b | panderm_b |
|---|---|---|
| cli only, 3-seed ens | 0.476 | 0.504 |
| late, per seed (delta vs same-seed der) | 0.573 (+0.022) / 0.565 (+0.023) / 0.574 (+0.034) | 0.578 (+0.033) / 0.603 (+0.079) / 0.578 (+0.036) |
| late, 3-seed ens vs der 3-seed ens | 0.576, +0.016 (-0.007..+0.039) | 0.584, +0.036 (+0.018..+0.054) |
| excl. MAL_OTH: der ens -> late ens | 0.600 -> 0.629 | 0.601 -> 0.639 |

Late eva - late panderm (ens): -0.008 (-0.029..+0.014). SD across seeds of cli-only: 0.013 / 0.002.
Largest per-class gains (both backbones): INF, BEN_OTH, BKL, AKIEC, DF, SCCKA; MAL_OTH lower for eva (0.160 -> 0.049).
Info only (chosen after seeing results): late eva + late panderm (geo 50/50) = 0.596; vs late panderm +0.012 (-0.007..+0.031).
Submissions: submission_late_eva02_b.csv, submission_late_panderm_b.csv (3-seed ensembles).

## Augmentation A/B (PanDerm-B, seed 0, 224; 2026-09-28) - ported HiV15 GPU pipeline (src/aug_hiv15.py, verified identical to the original module)
Paired delta vs current 'milk' augmentation (Step 3 der s0 / Step 4 cli s0):
| | der | cli | late |
|---|---|---|---|
| milk (current) | 0.545 | 0.482 | 0.578 |
| hiv15 (grayscale p=.5) | 0.530, -0.015 (-0.036..+0.005) | 0.436, -0.046 (-0.064..-0.028) | 0.560, -0.018 (-0.035..+0.000) |
| hiv15_nogray | 0.548, +0.003 (-0.014..+0.019) | 0.464, -0.018 (-0.036..+0.000) | 0.573, -0.005 (-0.020..+0.011) |
Pre-specified rule (late >= +0.01) not met by either variant -> keep 'milk'. Grayscale hurts mainly the clinical model (VASC 0.59 -> 0.41, INF, AKIEC).
