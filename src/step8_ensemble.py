"""Step 8 (first version): ensemble of the two Step 4 late-fusion models.
  late_{bb} = geo50( Step 3 der 3-seed ens , Step 4 cli 3-seed ens )      bb in eva02_b, panderm_b
  ens       = geo50( late_eva02_b , late_panderm_b )
CV: nested calibration + paired bootstrap vs each component (same as Steps 1-5).
Test: thresholds fit on the full OOF of the ensemble, applied to the test ensemble (common.rescale).
Caveat: this combination was chosen after its OOF score had been seen (0.596), so its CV estimate is optimistic.
"""
import json
import numpy as np, pandas as pd
from common import ROOT, CLASSES, load_train, macro_f1, per_class_f1, nested_calibrate, bootstrap_ci, fit_thresholds, rescale

OUT = ROOT / 'results' / 'step8'; OUT.mkdir(parents=True, exist_ok=True)
df = load_train(); Y = df[CLASSES].values; folds = df['fold'].values
BBS = ('eva02_b', 'panderm_b')


def rd(p, idx):
    return pd.read_csv(p, index_col=0).loc[idx, CLASSES].values


def geo(a, b):
    z = 0.5 * np.log(a + 1e-7) + 0.5 * np.log(b + 1e-7); z -= z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def paired(A, B, n=1000):
    r = np.random.default_rng(0); i = np.arange(len(Y)); d = []
    for _ in range(n):
        s = r.choice(i, len(i)); d.append(macro_f1(Y[s], A[s]) - macro_f1(Y[s], B[s]))
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


test_ids = pd.read_csv(ROOT / 'results' / 'step4' / 'test_late_panderm_b.csv', index_col=0).index
L, Lt = {}, {}
for bb in BBS:
    der = np.mean([rd(ROOT / 'results' / 'step3' / f'oof_ft_{bb}__s{s}.csv', df.index) for s in range(3)], 0)
    cli = np.mean([rd(ROOT / 'results' / 'step4' / f'oof_cli_{bb}__s{s}.csv', df.index) for s in range(3)], 0)
    L[bb] = geo(der, cli)
    Lt[bb] = rd(ROOT / 'results' / 'step4' / f'test_late_{bb}.csv', test_ids)
E, Et = geo(L['eva02_b'], L['panderm_b']), geo(Lt['eva02_b'], Lt['panderm_b'])

res = {}
Ce = nested_calibrate(Y, E, folds)
res['ens'] = {'calib': macro_f1(Y, Ce), 'ci95': bootstrap_ci(Y, Ce),
              'calib_fold_sd': float(np.std([macro_f1(Y[folds == f], Ce[folds == f]) for f in range(5)])),
              'per_class': dict(zip(CLASSES, np.round(per_class_f1(Y, Ce), 3).tolist()))}
for bb in BBS:
    Cb = nested_calibrate(Y, L[bb], folds)
    res[f'late_{bb}'] = {'calib': macro_f1(Y, Cb), 'per_class': dict(zip(CLASSES, np.round(per_class_f1(Y, Cb), 3).tolist()))}
    res['ens'][f'delta_vs_late_{bb}'] = [macro_f1(Y, Ce) - macro_f1(Y, Cb)] + paired(Ce, Cb)
print(json.dumps(res, indent=1))

th = fit_thresholds(Y, E)
sub = pd.DataFrame(rescale(Et, th), index=pd.Index(test_ids, name='lesion_id'), columns=CLASSES)
pd.DataFrame(Et, index=sub.index, columns=CLASSES).to_csv(OUT / 'test_ens_late_eva_panderm.csv')
sub.to_csv(OUT / 'submission_ens_late_eva_panderm.csv', float_format='%.6f')
res['_thresholds'] = dict(zip(CLASSES, np.round(th, 4).tolist()))
res['test_pred_pos_frac'] = dict(zip(CLASSES, np.round((sub.values >= 0.5).mean(0), 3).tolist()))
res['test_pos_per_lesion'] = {int(k): int(v) for k, v in pd.Series((sub.values >= 0.5).sum(1)).value_counts().sort_index().items()}
json.dump(res, open(OUT / 'metrics.json', 'w'), indent=1)
print('test predicted-positive fraction:', res['test_pred_pos_frac'])
print('positives per lesion:', res['test_pos_per_lesion'])
