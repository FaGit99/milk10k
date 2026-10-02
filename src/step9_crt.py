"""Step 9: classifier-only retraining (decoupled cRT, Kang et al. ICLR 2020) on frozen fine-tuned features.

Input: results/step4/feats/{der,cli}_{bb}{tag}__s{seed}__f{fold}.npz written by
       step4_twoview.py --variants der cli --tag _feat --save_feats
Per fold the backbone stays frozen; only the linear head (768 -> 11) is re-fit on that fold's TRAINING lesions
(4 flip views per lesion = 4 samples). Held-out predictions = mean softmax over the 4 flips (same as the TTA).

Heads (all hyper-parameters fixed in advance, no tuning on CV):
  orig      the head trained end-to-end in Step 4 (logit-adjusted CE)   -> baseline, reproduces the run
  refit_ib  new head, plain CE, instance-balanced                      -> control: effect of re-fitting alone
  crt_sqrt  new head, CE with class weights 1/sqrt(n_k)                 (= square-root sampling in expectation)
  crt_cb    new head, CE with class weights 1/n_k                       (= class-balanced sampling in expectation)
  taunorm   orig head with rows scaled by 1/||w_k|| (tau=1), bias 0     (no training)
New heads: zero init, full-batch Adam lr 1e-2, 300 steps, L2 1e-4 on W. Class weights are normalised to mean 1 per sample.
Weighted full-batch loss is used instead of a sampler: same expected gradient, deterministic.

Evaluation: der, cli and late (geo 50/50) per head; nested per-class threshold calibration; paired bootstrap vs orig;
per-class F1, AP, AUC. Also: training-fold accuracy per class of the orig head (memorisation check).
Outputs: results/step9/summary.csv, per_class.csv, oof_*.csv, submission_late_*.csv, log.txt

Usage (repo root, CPU is enough):  python src/step9_crt.py --tag _feat --seed 0
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CLASSES, load_train, macro_f1, per_class_f1, nested_calibrate, fit_thresholds, rescale

K = len(CLASSES); EPS = 1e-7
FEATS = ROOT / 'results' / 'step4' / 'feats'; RUNS = ROOT / 'results' / 'step4' / 'runs'
OUT = ROOT / 'results' / 'step9'


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def geo(Pa, Pb, w=0.5):
    return softmax(w * np.log(Pa + EPS) + (1 - w) * np.log(Pb + EPS))


def predict(F, W, b):  # F: (N, 4, D) -> mean softmax over flips
    return softmax(F.astype(np.float32) @ W.T + b).mean(1).astype(np.float64)


def fit_head(X, y, sw, steps=300, lr=1e-2, l2=1e-4):
    """Weighted multinomial logistic regression, full batch Adam. X (n, D) float32, y (n,), sw (n,) mean 1."""
    n, D = X.shape; W = np.zeros((K, D), np.float32); b = np.zeros(K, np.float32)
    Y = np.eye(K, dtype=np.float32)[y]; m = [np.zeros_like(W), np.zeros_like(b)]; v = [np.zeros_like(W), np.zeros_like(b)]
    for t in range(1, steps + 1):
        G = (softmax(X @ W.T + b) - Y) * sw[:, None] / n
        gW = G.T @ X + l2 * W; gb = G.sum(0)
        for i, (p, g) in enumerate(((W, gW), (b, gb))):
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def heads_for_fold(z, y_tr):
    F = z['tr'].astype(np.float32); n, V, D = F.shape
    X = F.reshape(n * V, D); yy = np.repeat(y_tr, V); cnt = np.bincount(y_tr, minlength=K).astype(float)
    H = {'orig': (z['W'], z['b'])}
    for name, wk in (('refit_ib', np.ones(K)), ('crt_sqrt', 1 / np.sqrt(cnt)), ('crt_cb', 1 / cnt)):
        sw = wk[yy]; sw = (sw / sw.mean()).astype(np.float32)
        H[name] = fit_head(X, yy, sw)
    W0 = z['W']; H['taunorm'] = (W0 / np.linalg.norm(W0, axis=1, keepdims=True), np.zeros(K, np.float32))
    return H


def ap_score(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def auc_score(y, s):
    r = pd.Series(s).rank().values; p = y.sum(); q = len(y) - p
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bb', default='panderm_b'); ap.add_argument('--tag', default='_feat')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / 'log.txt', 'a', encoding='utf-8')
    def pr(*s):
        t = ' '.join(str(x) for x in s); print(t, flush=True); log.write(t + '\n')

    df = load_train(); Y = df[CLASSES].values; y = df['label'].values; folds = df['fold'].values
    P, PT, memo = {}, {}, {}
    for view in ('der', 'cli'):
        for f in range(5):
            fp = FEATS / f'{view}_{a.bb}{a.tag}__s{a.seed}__f{f}.npz'
            z = np.load(fp, allow_pickle=True)
            tr_idx = df.index.get_indexer(z['tr_ids']); te_idx = df.index.get_indexer(z['te_ids'])
            assert (tr_idx >= 0).all() and (te_idx >= 0).all() and (folds[te_idx] == f).all()
            H = heads_for_fold(z, y[tr_idx])
            # reproduction check of the orig head against the saved run probabilities
            rz = np.load(RUNS / f'{view}_{a.bb}{a.tag}__s{a.seed}__f{f}.npz', allow_pickle=True)
            rep = predict(z['te'], *H['orig'])
            ro = pd.DataFrame(rz['oof'], index=rz['oof_ids']).loc[z['te_ids']].values
            pr(f'{view} f{f}: orig head reproduces run OOF, max|dp|={np.abs(rep - ro).max():.4f}')
            # memorisation check: orig head on the training-fold lesions
            ptr = predict(z['tr'], *H['orig']).argmax(1); ytr = y[tr_idx]
            memo.setdefault(view, []).append([(ptr[ytr == k] == k).mean() for k in range(K)])
            for h, (W, b) in H.items():
                P.setdefault((view, h), np.full((len(df), K), np.nan))[te_idx] = predict(z['te'], W, b)
                PT.setdefault((view, h), []).append(predict(z['test'], W, b))
    pr('training-fold accuracy per class, orig head (mean over folds):')
    for view in ('der', 'cli'):
        pr(f'  {view}: ' + ', '.join(f'{c} {v:.2f}' for c, v in zip(CLASSES, np.mean(memo[view], 0))))
    heads = ['orig', 'refit_ib', 'crt_sqrt', 'crt_cb', 'taunorm']
    for h in heads:
        P[('late', h)] = geo(P[('der', h)], P[('cli', h)])
        PT[('late', h)] = [geo(np.mean(PT[('der', h)], 0), np.mean(PT[('cli', h)], 0))]
    rng = np.random.default_rng(0); B = [rng.choice(len(Y), len(Y)) for _ in range(a.boot)]
    rows, pcs = [], []
    for view in ('der', 'cli', 'late'):
        C0 = nested_calibrate(Y, P[(view, 'orig')], folds)
        for h in heads:
            Pv = P[(view, h)]; assert np.isfinite(Pv).all()
            Cv = nested_calibrate(Y, Pv, folds)
            d = np.array([macro_f1(Y[s], Cv[s]) - macro_f1(Y[s], C0[s]) for s in B])
            f1 = per_class_f1(Y, Cv); aps = [ap_score(Y[:, k], Pv[:, k]) for k in range(K)]
            aucs = [auc_score(Y[:, k], Pv[:, k]) for k in range(K)]
            rows.append(dict(view=view, head=h, calib=macro_f1(Y, Cv), d_vs_orig=d.mean() if h != 'orig' else 0.0,
                             d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5), p_gt0=(d > 0).mean(),
                             mAP=np.mean(aps), mAUC=np.mean(aucs),
                             rareF1=np.mean([f1[CLASSES.index(c)] for c in ('BEN_OTH', 'DF', 'INF', 'MAL_OTH', 'VASC')])))
            for k, c in enumerate(CLASSES):
                pcs.append(dict(view=view, head=h, cls=c, n=int(Y[:, k].sum()), F1=f1[k], AP=aps[k], AUC=aucs[k]))
            pd.DataFrame(Pv, index=df.index, columns=CLASSES).assign(fold=folds).to_csv(OUT / f'oof_{view}_{h}.csv')
            if view == 'late':
                th = fit_thresholds(Y, Pv); Pt = PT[(view, h)][0]
                pd.DataFrame(rescale(Pt, th), index=pd.Index(np.load(FEATS / f'der_{a.bb}{a.tag}__s{a.seed}__f0.npz', allow_pickle=True)['test_ids'],
                             name='lesion_id'), columns=CLASSES).to_csv(OUT / f'submission_late_{h}.csv', float_format='%.6f')
    S = pd.DataFrame(rows); S.to_csv(OUT / 'summary.csv', index=False)
    PC = pd.DataFrame(pcs); PC.to_csv(OUT / 'per_class.csv', index=False)
    pr('\n' + S.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    late = PC[PC.view == 'late'].pivot(index='cls', columns='head', values=['F1', 'AP'])[['F1', 'AP']]
    pr('\nlate, per class:\n' + late.round(3).to_string())


if __name__ == '__main__':
    main()
