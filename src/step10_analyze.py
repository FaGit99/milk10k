"""Step 10 analysis: epoch-budget study (numpy/pandas only, no GPU needed).

Reads results/step10/preds/{view}_{bb}_e{E}__s{seed}__f{fold}.npz written by step10_epochs.py.
Each file holds, for EVERY epoch of that run, the held-out (OOF) and test probabilities (flip TTA) + the training loss.

analyze_budget(E):  per epoch of the E-epoch run: der, cli and late (geo 50/50, same epoch) -> nested-calibrated
                    macro F1, mean AUC, mean AP, rare-class F1 -> results/step10/curve_e{E}.csv
                    returns the epochs ranked by late calibrated macro F1 (used to keep the 3 best checkpoints).
compare_all():      one row per budget at its FINAL epoch (the unbiased comparison: no epoch chosen on the CV folds),
                    paired bootstrap vs the 25-epoch run, per-class F1; plus the best epoch of each run
                    (optimistic: chosen on the same OOF it is scored on) -> summary.csv, per_class_final.csv,
                    submission_late_e{E}.csv, curves.png (if matplotlib is available).
Per class:          curve_perclass_e{E}.csv (epoch x view x class: F1, AP, AUC) and
                    curves_perclass_{F1,AP,AUC}_e{E}.png (11 class panels; der, cli, late per epoch).

Usage (repo root):  python src/step10_analyze.py            # re-run the comparison from the saved predictions
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CLASSES, load_train, macro_f1, per_class_f1, nested_calibrate, fit_thresholds, rescale

K = len(CLASSES); EPS = 1e-7
OUT = ROOT / 'results' / 'step10'
RARE = ['BEN_OTH', 'DF', 'INF', 'MAL_OTH', 'VASC']
REF_BUDGET = 25


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def geo(Pa, Pb, w=0.5):
    return softmax(w * np.log(Pa + EPS) + (1 - w) * np.log(Pb + EPS))


def ap_score(y, s):
    o = np.argsort(-s, kind='stable'); y = y[o]; tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1))


def auc_score(y, s):
    r = pd.Series(s).rank().values; p = y.sum(); q = len(y) - p
    if p == 0 or q == 0:
        return float('nan')
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * q))


def pred_path(view, bb, E, seed, f):
    return OUT / 'preds' / f'{view}_{bb}_e{E}__s{seed}__f{f}.npz'


def load_budget(E, bb, seed, df, views=('der', 'cli')):
    """{view: (P_oof (E, n, K), P_test (E, n_test, K), test_ids, loss (5, E))} or None if a fold file is missing."""
    folds = df['fold'].values; res = {}
    for v in views:
        P = np.full((E, len(df), K), np.nan); Pt = []; L = []; tids = None
        for f in range(5):
            p = pred_path(v, bb, E, seed, f)
            if not p.exists():
                return None
            z = np.load(p, allow_pickle=True)
            idx = df.index.get_indexer(z['oof_ids']); assert (idx >= 0).all() and (folds[idx] == f).all()
            assert z['oof'].shape[0] == E, (p, z['oof'].shape)
            P[:, idx] = z['oof']; Pt.append(z['test']); L.append(z['loss']); tids = z['test_ids']
        assert np.isfinite(P).all()
        res[v] = (P, np.mean(Pt, 0), tids, np.array(L))
    return res


def score(Y, folds, P, per_class=False):
    Pc = nested_calibrate(Y, P, folds); f1 = per_class_f1(Y, Pc)
    aucs = np.array([auc_score(Y[:, k], P[:, k]) for k in range(K)]); aps = np.array([ap_score(Y[:, k], P[:, k]) for k in range(K)])
    s = dict(calib=float(f1.mean()), mAUC=float(np.nanmean(aucs)), mAP=float(aps.mean()),
             rareF1=float(np.mean([f1[CLASSES.index(c)] for c in RARE])))
    return (s, f1, Pc, aucs, aps) if per_class else (s, f1, Pc)


def analyze_budget(E, bb='panderm_b', seed=0, df=None, log=print):
    df = load_train() if df is None else df
    Y, folds = df[CLASSES].values, df['fold'].values
    R = load_budget(E, bb, seed, df)
    if R is None:
        raise FileNotFoundError(f'budget {E}: prediction files incomplete')
    rows, pcs = [], []
    for ep in range(E):
        Pl = geo(R['der'][0][ep], R['cli'][0][ep])
        for view, P in (('der', R['der'][0][ep]), ('cli', R['cli'][0][ep]), ('late', Pl)):
            s, f1, _, aucs, aps = score(Y, folds, P, per_class=True)
            rows.append(dict(budget=E, epoch=ep + 1, view=view, **s,
                             train_loss=float(R[view][3][:, ep].mean()) if view != 'late' else np.nan))
            pcs += [dict(budget=E, epoch=ep + 1, view=view, cls=c, n=int(Y[:, k].sum()), F1=float(f1[k]),
                         AP=float(aps[k]), AUC=float(aucs[k])) for k, c in enumerate(CLASSES)]
    C = pd.DataFrame(rows); C.to_csv(OUT / f'curve_e{E}.csv', index=False)
    PC = pd.DataFrame(pcs); PC.to_csv(OUT / f'curve_perclass_e{E}.csv', index=False)
    plot_perclass(PC, E, log)
    late = C[C.view == 'late'].sort_values('calib', ascending=False)
    log(f'budget {E}: late calib at final epoch {C[(C.view == "late") & (C.epoch == E)].calib.iloc[0]:.3f}; '
        f'top epochs (late calib, optimistic): ' +
        ', '.join(f'ep{int(r.epoch)}={r.calib:.3f}' for r in late.head(5).itertuples()))
    return [int(e) for e in late.epoch]


def plot_perclass(PC, E, log=print):
    """curves_perclass_{metric}_e{E}.png: 11 class panels, F1 / AP / AUC per epoch for der, cli, late."""
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        for metric in ('F1', 'AP', 'AUC'):
            fig, axs = plt.subplots(3, 4, figsize=(17, 10), sharex=True); axs = axs.ravel()
            for k, c in enumerate(CLASSES):
                for view, st in (('der', ':'), ('cli', '--'), ('late', '-')):
                    d = PC[(PC.cls == c) & (PC.view == view)]
                    axs[k].plot(d.epoch, d[metric], st, lw=1.3, label=view)
                axs[k].set_title(f'{c} (n={int(d.n.iloc[0])})'); axs[k].grid(alpha=.3)
            axs[0].legend(fontsize=8); axs[-1].axis('off')
            fig.suptitle(f'{E}-epoch run: per-class {metric} per epoch (OOF' + (', nested-calibrated)' if metric == 'F1' else ')'))
            fig.tight_layout(); fig.savefig(OUT / f'curves_perclass_{metric}_e{E}.png', dpi=110); plt.close(fig)
    except Exception as e:
        log(f'(no per-class plot: {e})')


def paired(Y, A, B, boot):
    d = np.array([macro_f1(Y[s], A[s]) - macro_f1(Y[s], B[s]) for s in boot])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def compare_all(budgets, bb='panderm_b', seed=0, nboot=1000, log=print, df=None):
    df = load_train() if df is None else df; Y, folds = df[CLASSES].values, df['fold'].values
    rng = np.random.default_rng(0); boot = [rng.choice(len(Y), len(Y)) for _ in range(nboot)]
    have = {E: load_budget(E, bb, seed, df) for E in budgets}; have = {E: r for E, r in have.items() if r is not None}
    if not have:
        log('no complete budget'); return
    fin = {}  # final-epoch calibrated OOF per budget and view
    rows, pcs = [], []
    for E, R in have.items():
        Pd, Pc_ = R['der'][0][-1], R['cli'][0][-1]; Pl = geo(Pd, Pc_)
        Ptl = geo(R['der'][1][-1], R['cli'][1][-1])
        for view, P in (('der', Pd), ('cli', Pc_), ('late', Pl)):
            s, f1, Pcal, aucs, aps = score(Y, folds, P, per_class=True); fin[(E, view)] = Pcal
            rows.append(dict(budget=E, view=view, **s, ci_lo=np.nan, ci_hi=np.nan))
            pcs += [dict(budget=E, view=view, cls=c, n=int(Y[:, k].sum()), F1=float(f1[k]), AP=float(aps[k]),
                         AUC=float(aucs[k])) for k, c in enumerate(CLASSES)]
        th = fit_thresholds(Y, Pl)
        pd.DataFrame(rescale(Ptl, th), index=pd.Index(R['der'][2], name='lesion_id'), columns=CLASSES).to_csv(
            OUT / f'submission_late_e{E}.csv', float_format='%.6f')
    S = pd.DataFrame(rows)
    for i, r in S.iterrows():
        E, view = int(r['budget']), r['view']; Pcal = fin[(E, view)]
        b = [macro_f1(Y[s], Pcal[s]) for s in boot]
        S.loc[i, 'ci_lo'], S.loc[i, 'ci_hi'] = np.percentile(b, 2.5), np.percentile(b, 97.5)
        if (REF_BUDGET, view) in fin and E != REF_BUDGET:
            m, lo, hi, p = paired(Y, Pcal, fin[(REF_BUDGET, view)], boot)
            S.loc[i, 'd_vs_e25'], S.loc[i, 'd_lo'], S.loc[i, 'd_hi'], S.loc[i, 'p_gt0'] = m, lo, hi, p
    # best epoch inside each run (selection on the same OOF -> optimistic, information only)
    for E in have:
        C = pd.read_csv(OUT / f'curve_e{E}.csv') if (OUT / f'curve_e{E}.csv').exists() else None
        if C is None:
            analyze_budget(E, bb, seed, df, log); C = pd.read_csv(OUT / f'curve_e{E}.csv')
        for view in ('der', 'cli', 'late'):
            c = C[C.view == view].sort_values('calib', ascending=False).iloc[0]
            m = (S.budget == E) & (S.view == view)
            S.loc[m, 'best_epoch_optimistic'] = int(c.epoch); S.loc[m, 'best_calib_optimistic'] = c.calib
    S = S.sort_values(['view', 'budget']); S.to_csv(OUT / 'summary.csv', index=False)
    PC = pd.DataFrame(pcs); PC.to_csv(OUT / 'per_class_final.csv', index=False)
    pd.set_option('display.width', 250)
    log('\n=== final-epoch comparison (unbiased; paired vs 25 epochs) ===\n' +
        S.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    for m in ('F1', 'AP', 'AUC'):
        log(f'\nlate, per-class {m} at the final epoch:\n' +
            PC[PC.view == 'late'].pivot(index='cls', columns='budget', values=m).round(3).to_string())
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        for j, view in enumerate(('der', 'cli', 'late')):
            for E in sorted(have):
                C = pd.read_csv(OUT / f'curve_e{E}.csv'); c = C[C.view == view]
                ax[j].plot(c.epoch, c.calib, label=f'{E} ep', lw=1.2)
            ax[j].set_title(f'{view}: calibrated macro F1 per epoch'); ax[j].set_xlabel('epoch'); ax[j].grid(alpha=.3)
        ax[0].set_ylabel('calibrated macro F1 (OOF)'); ax[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / 'curves.png', dpi=130); plt.close(fig)
        log(f'plot: {OUT / "curves.png"}')
    except Exception as e:
        log(f'(no plot: {e})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budgets', nargs='+', type=int, default=[15, 20, 25, 30, 35, 40, 50])
    ap.add_argument('--bb', default='panderm_b'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--redo_curves', action='store_true')
    a = ap.parse_args()
    if a.redo_curves:
        df = load_train()
        for E in a.budgets:
            if load_budget(E, a.bb, a.seed, df) is not None:
                analyze_budget(E, a.bb, a.seed, df)
    compare_all(a.budgets, a.bb, a.seed)


if __name__ == '__main__':
    main()
