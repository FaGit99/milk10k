# Rank 1: Allegoria "3D-ViT v3", external data: Yes, Dice mean 0.705, balanced acc 0.607 (screenshot from user, 2026-09-28)
| | AKIEC | BCC | BEN_OTH | BKL | DF | INF | MAL_OTH | MEL | NV | SCCKA | VASC | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | .918 | .975 | .746 | .879 | .991 | .958 | .670 | .930 | .962 | .932 | .992 | .905 |
| AP | .697 | .867 | .517 | .751 | .646 | .611 | .289 | .749 | .830 | .825 | .758 | .685 |
| Sens | .634 | .820 | .571 | .680 | .833 | .727 | .400 | .795 | .827 | .889 | .800 | .725 |
| PPV | .608 | .758 | .800 | .701 | .625 | .571 | .800 | .660 | .827 | .650 | .800 | .709 |
| Dice | .621 | .787 | .667 | .690 | .714 | .640 | .533 | .721 | .827 | .751 | .800 | .705 |
| ours (late_panderm_b) Dice | .633 | .614 | .222 | .656 | .667 | .370 | .000 | .646 | .784 | .745 | .667 | .546 |
| ours AUC | .920 | .966 | .892 | .881 | .989 | .941 | .782 | .952 | .966 | .940 | .992 | .929 |
Gap 0.159 = sum of per-class Dice differences / 11; MAL_OTH+BEN_OTH+INF+VASC contribute 0.126 (79%).
Their sens/PPV are consistent with our derived test counts (MAL_OTH 10, BEN_OTH 7, INF 11, VASC 5).
