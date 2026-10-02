# Step 5 pre-check - metadata on top of the image models (CPU, no training) - src/step5_precheck.py
img = Step 4 late fusion 3-seed ens (eva02_b 0.576, panderm_b 0.584). meta = Step 1 LightGBM OOF. Geometric fusion.

Pre-specified weights (delta vs img, paired bootstrap 95% CI):
| | eva02_b | panderm_b |
|---|---|---|
| + demo, w_img 0.75 | 0.593, +0.017 (+0.004..+0.032) | 0.577, -0.007 (-0.021..+0.008) |
| + demo, w_img 0.50 | 0.552, -0.024 (-0.042..-0.003) | 0.521, -0.063 (-0.084..-0.042) |
| + demo_monet_manip, w_img 0.75 | 0.586, +0.010 (-0.003..+0.023) | 0.592, +0.008 (-0.011..+0.026) |
| + demo_monet_manip, w_img 0.50 | 0.563, -0.013 (-0.030..+0.004) | 0.548, -0.036 (-0.059..-0.012) |

Nested weight choice (w chosen on the other 4 folds, grid 0.6..1.0):
| | eva02_b | panderm_b |
|---|---|---|
| + demo | 0.586, +0.011 (-0.004..+0.026) | 0.583, -0.001 (-0.010..+0.008) |
| + demo_monet_manip | 0.575, -0.001 (-0.018..+0.016) | 0.588, +0.004 (-0.010..+0.019) |

Conclusion: no robust gain from late-fused metadata; equal weighting hurts. The single resolved +0.017 (eva02_b + demo, w 0.75)
does not replicate on panderm_b and disappears under honest (nested) weight selection.
Not tested: a metadata branch trained jointly with the image model.
