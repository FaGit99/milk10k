"""Step 5 pre-check (CPU, no training): does metadata add information on top of the image models?
Fuses existing OOF predictions only:
  img  = Step 4 late fusion (Step 3 der 3-seed ens  x  Step 4 cli 3-seed ens, geometric 50/50)   -> reference
  meta = Step 1 LightGBM OOF (seed 0): 'demo' (age, sex, skin tone, site) or 'demo_monet_manip' (+ MONET + altered)
Fusions (pre-specified): geometric mean with image weight 0.75 and 0.5.
Evaluation: Step 1-4 metric, nested calibration, paired lesion bootstrap vs img.
Caveat: MONET scores are derived from the same images, so part of their information may already be in the image models.
"""
import json
import numpy as np, pandas as pd
from common import ROOT, CLASSES, load_train, macro_f1, per_class_f1, nested_calibrate, bootstrap_ci

OUT = ROOT / 'results' / 'step5'; OUT.mkdir(parents=True, exist_ok=True)
df = load_train(); Y = df[CLASSES].values; folds = df['fold'].values


def rd(p):
    return pd.read_csv(p, index_col=0).loc[df.index, CLASSES].values


def geo(a, b, w):
    z = w * np.log(a + 1e-7) + (1 - w) * np.log(b + 1e-7); z -= z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def paired(A, B, n=1000):
    r = np.random.default_rng(0); i = np.arange(len(Y)); d = []
    for _ in range(n):
        s = r.choice(i, len(i)); d.append(macro_f1(Y[s], A[s]) - macro_f1(Y[s], B[s]))
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


res, pcs = {}, {}
for bb in ('eva02_b', 'panderm_b'):
    der = np.mean([rd(ROOT / 'results' / 'step3' / f'oof_ft_{bb}__s{s}.csv') for s in range(3)], 0)
    cli = np.mean([rd(ROOT / 'results' / 'step4' / f'oof_cli_{bb}__s{s}.csv') for s in range(3)], 0)
    img = geo(der, cli, 0.5); Ci = nested_calibrate(Y, img, folds)
    print(f'\n== {bb}: img (late ens) calib={macro_f1(Y, Ci):.3f}')
    pcs[f'{bb}__img'] = per_class_f1(Y, Ci)
    for fs in ('demo', 'demo_monet_manip'):
        meta = rd(ROOT / 'results' / 'step1' / f'oof_lgbm__{fs}.csv')
        for w in (0.75, 0.5):
            C = nested_calibrate(Y, geo(img, meta, w), folds)
            k = f'{bb}__img+{fs}__w{w}'
            r = {'calib': macro_f1(Y, C), 'ci95': bootstrap_ci(Y, C), 'delta_vs_img': macro_f1(Y, C) - macro_f1(Y, Ci),
                 'delta_ci95': paired(C, Ci)}
            res[k] = r; pcs[k] = per_class_f1(Y, C)
            print(f"{fs:17s} w_img={w}: calib={r['calib']:.3f} delta={r['delta_vs_img']:+.3f} "
                  f"({r['delta_ci95'][0]:+.3f}..{r['delta_ci95'][1]:+.3f})")
json.dump(res, open(OUT / 'precheck_metrics.json', 'w'), indent=1)
t = pd.DataFrame(pcs, index=CLASSES); t.to_csv(OUT / 'precheck_per_class.csv')
print(t.round(3).to_string())


# ---- nested weight selection: w chosen on the other 4 folds (with their own nested calibration), applied to the 5th
print('\nNested weight choice (grid 0.60..1.00; w=1.00 means image only):')
GRID = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
for bb in ('eva02_b', 'panderm_b'):
    der = np.mean([rd(ROOT / 'results' / 'step3' / f'oof_ft_{bb}__s{s}.csv') for s in range(3)], 0)
    cli = np.mean([rd(ROOT / 'results' / 'step4' / f'oof_cli_{bb}__s{s}.csv') for s in range(3)], 0)
    img = geo(der, cli, 0.5); Ci = nested_calibrate(Y, img, folds)
    for fs in ('demo', 'demo_monet_manip'):
        meta = rd(ROOT / 'results' / 'step1' / f'oof_lgbm__{fs}.csv')
        P = np.zeros_like(img); chosen = []
        for f in range(5):
            tr = folds != f
            sc = [macro_f1(Y[tr], nested_calibrate(Y[tr], geo(img, meta, w)[tr], folds[tr])) for w in GRID]
            w = GRID[int(np.argmax(sc))]; chosen.append(w)
            P[folds == f] = geo(img, meta, w)[folds == f]
        C = nested_calibrate(Y, P, folds)
        d = macro_f1(Y, C) - macro_f1(Y, Ci); lo, hi = paired(C, Ci)
        res[f'{bb}__img+{fs}__nestedw'] = {'calib': macro_f1(Y, C), 'chosen_w': chosen, 'delta_vs_img': d, 'delta_ci95': [lo, hi]}
        print(f'{bb} {fs:17s} chosen w per fold={chosen} calib={macro_f1(Y, C):.3f} delta={d:+.3f} ({lo:+.3f}..{hi:+.3f})')
json.dump(res, open(OUT / 'precheck_metrics.json', 'w'), indent=1)
