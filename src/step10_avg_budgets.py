"""Step 10b: average the final-epoch late-fusion predictions of several epoch-budget runs (PanDerm-B, 224, seed 0).
Candidates fixed a priori from the step-10 table (budgets 20/25/30/35 = the flat optimum); no selection among ensembles is claimed.
Reports nested-calibrated macro F1, mean AUC/AP, paired bootstrap vs each member and vs the mean of the members' single scores."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, load_train, macro_f1, fit_thresholds, rescale
from step10_analyze import load_budget, geo, score, OUT, paired
df = load_train(); Y, folds = df[CLASSES].values, df['fold'].values
B = [20, 25, 30, 35]
oof, test, tid = {}, {}, None
for E in B:
    R = load_budget(E, 'panderm_b', 0, df)
    oof[E] = geo(R['der'][0][-1], R['cli'][0][-1]); test[E] = geo(R['der'][1][-1], R['cli'][1][-1]); tid = R['der'][2]
combos = [(20,), (25,), (30,), (35,), (20,25), (20,30), (25,30), (20,25,30), (20,25,30,35)]
rng = np.random.default_rng(0); boot = [rng.choice(len(Y), len(Y)) for _ in range(1000)]
res, cal = {}, {}
for c in combos:
    P = np.mean([oof[e] for e in c], 0); s, f1, Pc = score(Y, folds, P); res[c] = s; cal[c] = Pc
    if len(c) > 1:
        Pt = np.mean([test[e] for e in c], 0)
        pd.DataFrame(rescale(Pt, fit_thresholds(Y, P)), index=pd.Index(tid, name='lesion_id'), columns=CLASSES).to_csv(
            OUT / ('submission_avg_' + '_'.join(map(str, c)) + '.csv'), float_format='%.6f')
rows = []
for c in combos:
    b = [macro_f1(Y[s], cal[c][s]) for s in boot]
    r = dict(members=str(c), n=len(c), **res[c], ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5))
    if len(c) > 1:
        r['mean_of_members'] = np.mean([res[(e,)]['calib'] for e in c])
        r['d_vs_best_member'] = res[c]['calib'] - max(res[(e,)]['calib'] for e in c)
        for ref in (25, 20):
            m, lo, hi, p = paired(Y, cal[c], cal[(ref,)], boot); r[f'd_vs_e{ref}'] = m; r[f'lo{ref}'] = lo; r[f'hi{ref}'] = hi
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(OUT / 'avg_budgets.csv', index=False)
pd.set_option('display.width', 250); print(S.round(4).to_string(index=False))
