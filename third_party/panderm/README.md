`modeling_finetune.py` is copied unmodified from https://github.com/SiyuanYan1/PanDerm
(classification/models/modeling_finetune.py, commit fd7a80748ba7fc3e203fed88f909f4689d0d6f24, 2026-02-17).

License: CC-BY-NC-ND 4.0 (non-commercial academic research, with attribution, no modified redistribution).
Authors: Yan et al., "A multimodal vision foundation model for clinical dermatology" (PanDerm).

Used by src/step2_extract.py via `panderm_base_patch16_224()` (ViT-B/16, CLS token after the final
LayerNorm, as in the repo's `PanDerm_Base_LP` path). Weights are not included: download
`panderm_bb_data6_checkpoint-499.pth` (PanDerm_Base, Google Drive link in the PanDerm README) to
`weights/panderm_bb_data6_checkpoint-499.pth`.
