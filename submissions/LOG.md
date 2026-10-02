# Leaderboard submissions (ISIC MILK10k Benchmark, task 57)
Note: the upload checker requires the first column 'lesion_id' (the challenge text says 'lesion').

| date | file | model | CV calib macro F1 (95% CI) | leaderboard | rank |
|---|---|---|---|---|---|
| 2026-09-28 | 2026-09-28_late_panderm_b_v2.csv | PanDerm-B, der + cli fine-tuned separately, late fusion (geo 50/50), 3 seeds, per-class thresholds from OOF | 0.584 (0.561-0.606) | 0.546 | 16 |
| 2026-09-28 | 2026-09-28_ens_late_eva_panderm.csv | geo50(late EVA-02-B, late PanDerm-B); combination chosen after seeing OOF | 0.596 (0.569-0.623), optimistic | 0.505 | |
| 2026-09-29 | 2026-09-29_late_panderm_b_448.csv | PanDerm-B, der + cli fine-tuned separately at 448 px, late fusion (geo 50/50), seed 0 only, per-class thresholds from OOF | 0.592 (0.563-0.622); vs 224 s0 +0.014 (-0.008..+0.036), n.s. | 0.545 | |
