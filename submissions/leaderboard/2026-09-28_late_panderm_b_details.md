# ISIC details for submission 2026-09-28_late_panderm_b_v2.csv (Dice mean 0.546 = leaderboard score; balanced multiclass accuracy 0.554)
Per class (from the ISIC details page, screenshot provided by user):
| | AKIEC | BCC | BEN_OTH | BKL | DF | INF | MAL_OTH | MEL | NV | SCCKA | VASC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | .920 | .966 | .892 | .881 | .989 | .941 | .782 | .952 | .966 | .940 | .992 |
| Sensitivity | .535 | .951 | .143 | .610 | .833 | .455 | .000 | .538 | .769 | .701 | .600 |
| PPV | .776 | .453 | .500 | .709 | .556 | .313 | .000 | .808 | .800 | .796 | .750 |
| Dice (F1) | .633 | .614 | .222 | .656 | .667 | .370 | .000 | .646 | .784 | .745 | .667 |

Derived test-set class counts (TP = PPV x predicted positives of our file; n = TP / sensitivity; MAL_OTH from NPV). Rounded, +-1:
AKIEC 71, BCC 61, BEN_OTH 7, BKL 100, DF 6, INF 11, MAL_OTH ~10, MEL 39, NV 52, SCCKA 117, VASC 5  (sum 479)
WARNING: these counts come from leaderboard feedback. Do not use them to tune priors/thresholds (that would fit the hidden test labels).
