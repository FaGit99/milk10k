# Label-shift (prior) correction - src/priorshift.py (PanDerm-B late fusion)
Calibration BCTS on OOF -> EM prior estimation on unlabeled target predictions -> decision rule.
Validation: simulated shifted target sets (n=479) resampled from each held-out fold, prior (1-lam)*pi_train + lam*Dir(1),
50 draws x 5 folds per lam. Pre-specified; not derived from leaderboard feedback.

Mean macro F1 (delta vs base; share of draws better than base):
| method | lam 0 | lam 0.5 | lam 1 |
|---|---|---|---|
| base (current) | 0.584 | 0.593 | 0.524 |
| em_plugin | 0.547 (-0.037; 21%) | 0.576 (-0.016; 34%) | 0.529 (+0.005; 53%) |
| emhalf_plugin | 0.558 (-0.026; 32%) | 0.582 (-0.011; 37%) | 0.529 (+0.004; 57%) |
| cal_plugin (no EM) | 0.589 (+0.006; 58%) | 0.564 (-0.029) | 0.472 (-0.053) |
| em_argmax / cal_argmax / base_em | all worse (-0.03 .. -0.14) | | |
| oracle_plugin (TRUE prior) | 0.592 (+0.008; 63%) | 0.623 (+0.030; 79%) | 0.604 (+0.079; 93%) |
EM prior error (L1): 0.17 / 0.31 / 0.39.

Conclusion: knowing the target prior would help (+0.03..+0.08), but EM estimates it too imprecisely at n=479 with 11 classes;
no label-free method beats the current thresholds reliably. NOT validated -> not recommended for submission.
EM test estimate (label-free) and candidate files kept for reference: summary.json, submission_*_panderm_b.csv.
After-the-fact check only (uses leaderboard-derived counts, must not drive decisions): EM test L1 ~0.14 vs training prior ~0.79.
