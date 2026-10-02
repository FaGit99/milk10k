"""Step 4 pre-check (CPU, no training): does the clinical close-up add information on top of the
fine-tuned dermoscopy model?  Fuses existing OOF predictions only:
  der  = Step 3 fine-tuned dermoscopy model, 3-seed ensemble (seeds 0,1,2)   -> reference
  cli  = Step 2 frozen linear probe on the clinical embedding (crop), same backbone
Fusions (all pre-specified, not tuned on the evaluation folds):
  geo50   : softmax(0.5*log p_der + 0.5*log p_cli)
  geo75   : softmax(0.75*log p_der + 0.25*log p_cli)
  stack   : multinomial LogReg on [log p_der | log p_cli], trained per outer fold on the OTHER 4 folds' OOF
Evaluation: same metric and nested calibration as Steps 1-3; paired lesion bootstrap vs der.
Caveat: the frozen clinical probe is weaker than a fine-tuned clinical model would be, so a null result here
does not rule out a gain from Step 4 proper; a positive result indicates complementary information.
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from common import ROOT, CLASSES, load_train, macro_f1, per_class_f1, nested_calibrate, bootstrap_ci

OUT = ROOT / 'results' / 'step4'; OUT.mkdir(parents=True, exist_ok=True)
df = load_train(); Y = df[CLASSES].values; y = df['label'].values; folds = df['fold'].values
EPS = 1e-7


def rd(p):
    return pd.read_csv(p, index_col=0).loc[df.index, CLASSES].values


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def paired(Pa, Pb, n=1000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(Y)); d = []
    for _ in range(n):
        s = rng.choice(idx, len(idx)); d.append(macro_f1(Y[s], Pa[s]) - macro_f1(Y[s], Pb[s]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


res = {}
for bb in ('eva02_b', 'panderm_b'):
    der = np.mean([rd(ROOT / 'results' / 'step3' / f'oof_ft_{bb}__s{s}.csv') for s in (0, 1, 2)], 0)
    cli = rd(ROOT / 'results' / 'step2' / f'oof_{bb}__crop__cli.csv')
    Ld, Lc = np.log(der + EPS), np.log(cli + EPS)
    fused = {'der (ref)': der, 'cli frozen': cli,
             'geo50': softmax(0.5 * Ld + 0.5 * Lc), 'geo75': softmax(0.75 * Ld + 0.25 * Lc)}
    St = np.zeros_like(der); X = np.hstack([Ld, Lc])
    for f in range(5):
        tr, te = folds != f, folds == f
        m = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(X[tr], y[tr])
        St[te] = m.predict_proba(X[te])
    fused['stack'] = St
    Cref = nested_calibrate(Y, der, folds)
    print(f'\n== {bb}')
    for name, P in fused.items():
        Pc = nested_calibrate(Y, P, folds)
        r = {'calib': macro_f1(Y, Pc), 'ci95': bootstrap_ci(Y, Pc), 'per_class': dict(zip(CLASSES, np.round(per_class_f1(Y, Pc), 3).tolist()))}
        if name != 'der (ref)':
            r['delta_vs_der'] = r['calib'] - macro_f1(Y, Cref); r['delta_ci95'] = paired(Pc, Cref)
        res[f'{bb}__{name}'] = r
        dl = f"  delta {r['delta_vs_der']:+.3f} ({r['delta_ci95'][0]:+.3f}..{r['delta_ci95'][1]:+.3f})" if 'delta_vs_der' in r else ''
        print(f"{name:11s} calib={r['calib']:.3f} CI95=({r['ci95'][0]:.3f},{r['ci95'][1]:.3f}){dl}")
json.dump(res, open(OUT / 'precheck_metrics.json', 'w'), indent=1)
pc = pd.DataFrame({k: v['per_class'] for k, v in res.items()}); pc.to_csv(OUT / 'precheck_per_class.csv')
print(pc.round(3).to_string())
