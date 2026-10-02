"""Step 1: non-image baselines on metadata + MONET concept scores.
Models: class-prior, multinomial logistic regression, LightGBM. Canonical folds from data/folds.csv.
Reports macro F1 under the challenge rule (>=0.5 per class) for:
  raw      - model probabilities as-is
  argmax   - one-hot of the top class
  calib    - per-class thresholds fit on the OTHER folds' OOF, rescaled so threshold -> 0.5 (nested, honest)
"""
import json, sys, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from common import (ROOT, CLASSES, FEATURES, load_train, macro_f1, per_class_f1,
                    nested_calibrate, bootstrap_ci)

OUT = ROOT / 'results' / 'step1'; OUT.mkdir(parents=True, exist_ok=True)
df = load_train()
Y = df[CLASSES].values; y = df['label'].values; folds = df['fold'].values
K = len(CLASSES)


def run(model, feats, seed=0):
    P = np.zeros((len(df), K))
    for f in range(5):
        tr, te = folds != f, folds == f
        X = df[feats].values
        if model == 'prior':
            P[te] = np.bincount(y[tr], minlength=K) / tr.sum()
        elif model == 'logreg':
            clf = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced'))
            clf.fit(X[tr], y[tr]); P[te] = clf.predict_proba(X[te])
        elif model == 'lgbm':
            clf = lgb.LGBMClassifier(objective='multiclass', n_estimators=300, learning_rate=0.03,
                                     num_leaves=15, min_child_samples=5, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                                     n_jobs=2, verbose=-1)
            clf.fit(X[tr], y[tr], sample_weight=compute_sample_weight('balanced', y[tr]))
            P[te] = clf.predict_proba(X[te])
    return P


def evaluate(P):
    Pa = np.eye(K)[P.argmax(1)]
    Pc = nested_calibrate(Y, P, folds)
    r = {'raw': macro_f1(Y, P), 'argmax': macro_f1(Y, Pa), 'calib': macro_f1(Y, Pc)}
    r['calib_ci95'] = bootstrap_ci(Y, Pc, n=1000)
    r['calib_fold_sd'] = float(np.std([macro_f1(Y[folds == f], Pc[folds == f]) for f in range(5)]))
    r['calib_per_class'] = dict(zip(CLASSES, np.round(per_class_f1(Y, Pc), 3).tolist()))
    r['argmax_balanced_recall'] = float(np.mean([(P.argmax(1)[y == k] == k).mean() for k in range(K)]))
    return r, Pc


configs = [('prior', 'demo', [0])] + [(m, fs, [0] if m == 'logreg' else [0, 1, 2])
                                      for m in ('logreg', 'lgbm') for fs in FEATURES]
results = {}
for model, fs, seeds in configs:
    t0 = time.time()
    P = np.mean([run(model, FEATURES[fs], s) for s in seeds], axis=0) if len(seeds) == 1 else None
    if len(seeds) > 1:  # evaluate each seed separately, report mean/sd, keep seed-0 OOF
        runs = [run(model, FEATURES[fs], s) for s in seeds]
        evs = [evaluate(p) for p in runs]
        r = evs[0][0]; r['calib_seed_mean'] = float(np.mean([e[0]['calib'] for e in evs]))
        r['calib_seed_sd'] = float(np.std([e[0]['calib'] for e in evs])); P = runs[0]; Pc = evs[0][1]
    else:
        r, Pc = evaluate(P)
    name = f'{model}__{fs}'; results[name] = r
    pd.DataFrame(P, index=df.index, columns=CLASSES).assign(fold=folds).to_csv(OUT / f'oof_{name}.csv')
    print(f"{name:28s} raw={r['raw']:.3f} argmax={r['argmax']:.3f} calib={r['calib']:.3f} "
          f"CI95=({r['calib_ci95'][0]:.3f},{r['calib_ci95'][1]:.3f}) foldSD={r['calib_fold_sd']:.3f} "
          f"balRec={r['argmax_balanced_recall']:.3f} [{time.time()-t0:.0f}s]", flush=True)

json.dump(results, open(OUT / 'metrics.json', 'w'), indent=1)
