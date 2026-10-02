"""Label (prior) shift correction for the late-fusion models, validated by simulation in CV. CPU only.

Why: the test set's class mix differs strongly from training; per-class thresholds tuned on the training OOF
(training prevalence) transfer badly (e.g. BCC). Correction uses only UNLABELED test predictions.

Steps
 1. Calibration (BCTS, Alexandari et al. 2020): q = softmax(T*log p + b), T and b fit by NLL on training OOF.
 2. Prior estimation (EM, Saerens et al. 2002) on the unlabeled target predictions -> pi_hat, adapted posteriors r.
 3. Decision rule per class (the challenge binarises every class at >= 0.5 independently):
      argmax  : one-hot of the top class
      plugin  : label-free plug-in F1 maximisation: choose the top-m lesions per class that maximise
                2*sum(q_top_m) / (m + sum(q)), i.e. expected F1 under the (adapted) posteriors
Methods compared (all label-free on the target set):
  base         current method: raw late-fusion probs, thresholds fit on training OOF (other folds)
  cal_argmax / cal_plugin   calibrated, no prior adaptation
  em_argmax  / em_plugin    calibrated + EM prior adaptation
  oracle_plugin             adaptation with the TRUE target prior (simulation only, upper reference)

Simulation (pre-specified, NOT derived from leaderboard feedback): for each outer fold f, calibration and base
thresholds are fit on the other 4 folds; target sets of n=479 lesions are resampled from fold f with prior
  pi_t = (1-lam)*pi_train + lam*d,  d ~ Dirichlet(1,...,1),  lam in {0, 0.5, 1};  floor 0.004, >=1 lesion per class.
50 draws per fold and lam. Metric: challenge macro F1.
"""
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from common import ROOT, CLASSES, load_train, macro_f1, per_class_f1, fit_thresholds, rescale

OUT = ROOT / 'results' / 'priorshift'; OUT.mkdir(parents=True, exist_ok=True)
K = len(CLASSES); N_TEST = 479; EPS = 1e-7
df = load_train(); Y = df[CLASSES].values; y = df['label'].values; folds = df['fold'].values


def rd(p, idx):
    return pd.read_csv(p, index_col=0).loc[idx, CLASSES].values


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def geo(a, b):
    return softmax(0.5 * np.log(a + EPS) + 0.5 * np.log(b + EPS))


def late(bb):
    der = np.mean([rd(ROOT / 'results' / 'step3' / f'oof_ft_{bb}__s{s}.csv', df.index) for s in range(3)], 0)
    cli = np.mean([rd(ROOT / 'results' / 'step4' / f'oof_cli_{bb}__s{s}.csv', df.index) for s in range(3)], 0)
    t = pd.read_csv(ROOT / 'results' / 'step4' / f'test_late_{bb}.csv', index_col=0)
    return geo(der, cli), t[CLASSES].values, t.index


def fit_bcts(P, yy):
    L = np.log(P + EPS); n = len(yy); ar = np.arange(n)
    def f(th):
        Z = th[0] * L + th[1:]; Z = Z - Z.max(1, keepdims=True); lse = np.log(np.exp(Z).sum(1))
        S = np.exp(Z - lse[:, None]); G = S; G[ar, yy] -= 1; G /= n
        return -(Z[ar, yy] - lse).mean(), np.concatenate([[(G * L).sum()], G.sum(0)])
    r = minimize(f, np.r_[1.0, np.zeros(K)], jac=True, method='L-BFGS-B')
    return r.x[0], r.x[1:]


def bcts(P, T, b):
    return softmax(T * np.log(P + EPS) + b)


def em(Q, pi_s, iters=2000, tol=1e-9):
    pi = pi_s.copy()
    for _ in range(iters):
        R = Q * (pi / pi_s); R /= R.sum(1, keepdims=True); new = R.mean(0)
        done = np.abs(new - pi).max() < tol; pi = new
        if done:
            break
    R = Q * (pi / pi_s); R /= R.sum(1, keepdims=True)
    return pi, R


def argmax_dec(Q):
    return np.eye(K)[Q.argmax(1)]


def plugin(Q, return_th=False):
    D = np.zeros_like(Q); th = np.zeros(K)
    for k in range(K):
        o = np.argsort(-Q[:, k]); cs = np.cumsum(Q[o, k]); m = np.arange(1, len(Q) + 1)
        best = int(np.argmax(2 * cs / (m + Q[:, k].sum())))
        D[o[:best + 1], k] = 1; th[k] = Q[o[best], k]
    return (D, th) if return_th else D


def simulate(P, lams=(0.0, 0.5, 1.0), draws=50, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for f in range(5):
        tr, te = folds != f, folds == f
        T, b = fit_bcts(P[tr], y[tr]); pi_s = np.bincount(y[tr], minlength=K) / tr.sum()
        th_base = fit_thresholds(Y[tr], P[tr])
        te_idx = np.where(te)[0]; by_c = [te_idx[y[te_idx] == k] for k in range(K)]
        for lam in lams:
            for _ in range(draws):
                pi_t = (1 - lam) * pi_s + lam * rng.dirichlet(np.ones(K))
                pi_t = np.maximum(pi_t, 0.004); pi_t /= pi_t.sum()
                cnt = np.maximum(1, np.round(pi_t * N_TEST).astype(int))
                idx = np.concatenate([rng.choice(by_c[k], cnt[k]) for k in range(K)])
                Ys, Ps = Y[idx], P[idx]; Q = bcts(Ps, T, b)
                pi_hat, R = em(Q, pi_s)
                true_pi = cnt / cnt.sum(); Ro = Q * (true_pi / pi_s); Ro /= Ro.sum(1, keepdims=True)
                pi_h = np.sqrt(pi_hat * pi_s); pi_h /= pi_h.sum()
                Rh = Q * (pi_h / pi_s); Rh /= Rh.sum(1, keepdims=True)
                Pb = Ps * (pi_hat / pi_s); Pb /= Pb.sum(1, keepdims=True)
                dec = {'base': (rescale(Ps, th_base) >= 0.5).astype(float), 'cal_argmax': argmax_dec(Q),
                       'cal_plugin': plugin(Q), 'em_argmax': argmax_dec(R), 'em_plugin': plugin(R),
                       'emhalf_plugin': plugin(Rh), 'base_em': (rescale(Pb, th_base) >= 0.5).astype(float),
                       'oracle_plugin': plugin(Ro)}
                r = {'fold': f, 'lam': lam, 'em_L1': float(np.abs(pi_hat - true_pi).sum()),
                     'train_L1': float(np.abs(pi_s - true_pi).sum())}
                r.update({m: macro_f1(Ys, D) for m, D in dec.items()}); rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    bbs = sys.argv[1:] or ['panderm_b', 'eva02_b']
    summary = {}
    for bb in bbs:
        P, Pt, tid = late(bb)
        sim = simulate(P); sim.to_csv(OUT / f'sim_{bb}.csv', index=False)
        meth = ['base', 'cal_argmax', 'cal_plugin', 'em_argmax', 'em_plugin', 'emhalf_plugin', 'base_em', 'oracle_plugin']
        for m in meth[1:]:
            sim[f'd_{m}'] = sim[m] - sim['base']; sim[f'win_{m}'] = (sim[m] > sim['base']).astype(float)
        g = sim.groupby('lam')
        tab = g[meth + [c for c in sim.columns if c.startswith(('d_', 'win_'))]].mean().round(3)
        tab['em_L1'] = g['em_L1'].mean().round(3); tab['train_L1'] = g['train_L1'].mean().round(3)
        print(f'\n===== {bb}: simulation (mean over 5 folds x 50 draws per lam)'); print(tab.T.to_string())
        summary[bb] = {'sim': tab.to_dict()}
        # ---- apply to the real test set (label-free)
        T, b = fit_bcts(P, y); pi_s = np.bincount(y, minlength=K) / len(y)
        Q = bcts(Pt, T, b); pi_hat, R = em(Q, pi_s)
        Pb = Pt * (pi_hat / pi_s); Pb /= Pb.sum(1, keepdims=True); th_b = fit_thresholds(Y, P)
        pd.DataFrame(rescale(Pb, th_b), index=pd.Index(tid, name='lesion_id'), columns=CLASSES).to_csv(OUT / f'submission_base_em_{bb}.csv', float_format='%.6f')
        summary.setdefault(bb, {})['base_em_test_pos'] = dict(zip(CLASSES, (rescale(Pb, th_b) >= 0.5).sum(0).astype(int).tolist()))
        est = pd.Series(pi_hat * len(Pt), index=CLASSES).round(1)
        print('EM-estimated test class counts:', est.to_dict())
        summary[bb]['em_test_counts'] = est.to_dict(); summary[bb]['bcts'] = {'T': float(T), 'b': b.round(3).tolist()}
        for name, Qx in (('em_plugin', R), ('cal_plugin', Q)):
            D, th = plugin(Qx, return_th=True)
            sub = pd.DataFrame(np.where(D > 0, 0.5 + 0.5 * Qx, 0.4999 * Qx), index=pd.Index(tid, name='lesion_id'), columns=CLASSES)
            assert ((sub.values >= 0.5) == (D > 0)).all()
            sub.to_csv(OUT / f'submission_{name}_{bb}.csv', float_format='%.6f')
            summary[bb][f'{name}_test_pos'] = dict(zip(CLASSES, D.sum(0).astype(int).tolist()))
        Da = argmax_dec(R)
        pd.DataFrame(np.where(Da > 0, 0.75, 0.25) * 1.0, index=pd.Index(tid, name='lesion_id'), columns=CLASSES) \
            .to_csv(OUT / f'submission_em_argmax_{bb}.csv', float_format='%.6f')
        print('em_plugin test positives:', summary[bb]['em_plugin_test_pos'])
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1, default=str)
