# Step 1 – Non-image baselines (metadata + MONET)

Setup: 5,240 training lesions, canonical stratified 5-fold split (`data/folds.csv`, seed 42, lesion-level; no patient ID available).
Metric = challenge rule: macro F1, each class binarised at >= 0.5. "calib" = per-class thresholds fit on the other 4 folds' OOF, then rescaled so threshold -> 0.5 (nested, no self-tuning).

| Model / features | raw | argmax | calib | calib 95% CI (bootstrap) | balanced recall (argmax) |
|---|---|---|---|---|---|
| class prior | 0.000 | 0.059 | 0.144 | – | 0.091 |
| LogReg demo | 0.020 | 0.105 | 0.208 | 0.201–0.215 | 0.245 |
| LogReg demo+MONET | 0.235 | 0.285 | 0.331 | 0.313–0.348 | 0.420 |
| LogReg demo+MONET+manip | 0.231 | 0.285 | 0.324 | 0.305–0.341 | 0.419 |
| LightGBM demo | 0.089 | 0.156 | 0.214 | 0.203–0.226 | 0.213 |
| LightGBM demo+MONET | 0.303 | 0.340 | 0.360 | 0.342–0.377 | 0.340 |
| LightGBM demo+MONET+manip | 0.304 | 0.346 | 0.363 | 0.344–0.382 | 0.346 |

LightGBM: 3 model seeds, seed SD of calib macro F1 <= 0.003.
Floor for image models to beat: **macro F1 (calib) ~0.36**.

Files: `src/common.py` (loading, metric, calibration), `src/make_folds.py`, `src/step1_baseline.py`, `oof_*.csv`, `metrics.json`.
