# Step 2 - frozen-feature linear probe (run 2026-09-25, RTX PRO 6000, torch 2.9.1+cu128)

Setup: frozen 224px embeddings; class-balanced softmax regression, C chosen by inner CV;
same folds / nested calibration / metric as Step 1. Reference: Step 1 lgbm__demo_monet_manip calib 0.363.

Top configs (calibrated macro F1, 95% bootstrap CI):
- panderm_b crop both_meta 0.453 (0.428-0.475), vs Step 1 +0.090 (+0.062..+0.114)
- dinov2_b  crop both_meta 0.448 (0.424-0.470)
- dinov2_b  crop both      0.446 (0.422-0.467)
- panderm_b crop both      0.443 (0.418-0.464)
- eva02_b   crop both_meta 0.439 (0.416-0.459)
- eva02_s best: pad both_meta 0.372 ; eva02_s der only 0.309-0.313 (below Step 1)

Paired bootstrap differences (95% CI):
- panderm - dinov2 (crop both_meta): +0.005 (-0.017..+0.026)  -> not resolved
- panderm - eva02_b (crop both_meta): +0.014 (-0.007..+0.034) -> not resolved
- dinov2 - eva02_b (crop both_meta): +0.009 (-0.012..+0.030)  -> not resolved
- panderm both - der only: +0.005 (-0.015..+0.025)            -> clinical adds nothing resolvable
- panderm both_meta - both: +0.010 (+0.002..+0.018)
- crop - pad, dermoscopy: panderm +0.024 (+0.008..+0.039), dinov2 +0.018 (-0.002..+0.037)

Full table: summary.csv / metrics.json; log: probe_log.txt.
