"""Step 2b: linear probe (multinomial logistic regression) on frozen embeddings from step2_extract.py.

Same canonical folds (data/folds.csv), same challenge metric and nested threshold calibration as Step 1,
so 'calib' is directly comparable to the Step 1 baseline.

Model: softmax regression, L2 penalty, class-balanced sample weights (as sklearn class_weight='balanced'),
       objective 0.5*||W||^2 + C * sum_i w_i * CE_i, full-batch L-BFGS in torch (GPU if available).
C:     chosen per outer fold by inner CV on the 4 training folds (weighted log-loss), grid C_GRID.
       The outer test fold is never used for C selection or threshold fitting.
Setups per backbone x prep:
  der        dermoscopic embedding
  cli        clinical embedding
  both       [der | cli]
  both_meta  [der | cli | demo + MONET + manipulation]   (Step 1 feature set 'demo_monet_manip')
Each block is standardised on the training fold only (metadata: median imputation first).

Output: results/step2/metrics.json, summary.csv, oof_<config>.csv, probe_log.txt
Usage:  python src/step2_probe.py [--backbones ...] [--preps ...] [--setups ...]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, CLASSES, FEATURES, load_train, macro_f1, per_class_f1, nested_calibrate, bootstrap_ci)

EMB = ROOT / 'data' / 'emb'
OUT = ROOT / 'results' / 'step2'
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
K = len(CLASSES)
MAX_ITER = 500
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tee:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8'); self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()


# ---------------------------------------------------------------- model
def fit_predict(Xtr, ytr, Xte, C, init=None):
    """Weighted multinomial logistic regression; returns (P(Xte), (W, b)) - (W, b) can warm-start the next C."""
    X = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV)
    y = torch.as_tensor(ytr, dtype=torch.long, device=DEV)
    cnt = torch.bincount(y, minlength=K).float()
    wc = torch.where(cnt > 0, len(y) / (K * cnt.clamp(min=1)), torch.zeros_like(cnt))
    w = wc[y]
    if init is None:
        W = torch.zeros(X.shape[1], K, device=DEV); b = torch.zeros(K, device=DEV)
    else:
        W, b = init[0].clone(), init[1].clone()
    W.requires_grad_(True); b.requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=MAX_ITER, tolerance_grad=1e-6, tolerance_change=1e-9,
                            history_size=20, line_search_fn='strong_wolfe')
    n = len(y)

    def closure():
        opt.zero_grad()
        loss = (C * (w * F.cross_entropy(X @ W + b, y, reduction='none')).sum() + 0.5 * (W * W).sum()) / n
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        Xt = torch.as_tensor(Xte, dtype=torch.float32, device=DEV)
        return torch.softmax(Xt @ W + b, 1).double().cpu().numpy(), (W.detach(), b.detach())


def weighted_logloss(y, P):
    cnt = np.bincount(y, minlength=K)
    wc = np.where(cnt > 0, len(y) / (K * np.maximum(cnt, 1)), 0)
    return float(np.sum(wc[y] * -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))) / np.sum(wc[y]))


def standardise(blocks, tr):
    """blocks: list of (n, d) arrays (NaN allowed in metadata). Fit median/mean/std on rows tr only."""
    out = []
    for B in blocks:
        B = B.astype(np.float64).copy()
        if np.isnan(B).any():
            med = np.nanmedian(B[tr], 0); med = np.where(np.isnan(med), 0, med)
            B = np.where(np.isnan(B), med, B)
        mu, sd = B[tr].mean(0), B[tr].std(0)
        out.append((B - mu) / np.where(sd > 1e-8, sd, 1))
    return np.hstack(out)


def run_config(blocks, y, folds):
    P = np.zeros((len(y), K)); chosen = []
    for f in range(5):
        tr, te = folds != f, folds == f
        # inner CV on the 4 training folds selects C (C ascending, warm-started within each inner fold)
        Pi = {C: np.zeros((len(y), K)) for C in C_GRID}
        for g in np.unique(folds[tr]):
            itr, ite = tr & (folds != g), folds == g
            Xg = standardise(blocks, itr); init = None
            for C in C_GRID:
                Pi[C][ite], init = fit_predict(Xg[itr], y[itr], Xg[ite], C, init)
        inner_loss = [weighted_logloss(y[tr], Pi[C][tr]) for C in C_GRID]
        C = C_GRID[int(np.argmin(inner_loss))]
        chosen.append(C)
        X = standardise(blocks, tr)
        P[te], _ = fit_predict(X[tr], y[tr], X[te], C)
    return P, chosen


# ---------------------------------------------------------------- evaluation (identical to Step 1)
def evaluate(Y, y, P, folds):
    Pa = np.eye(K)[P.argmax(1)]
    Pc = nested_calibrate(Y, P, folds)
    r = {'raw': macro_f1(Y, P), 'argmax': macro_f1(Y, Pa), 'calib': macro_f1(Y, Pc)}
    r['calib_ci95'] = bootstrap_ci(Y, Pc, n=1000)
    r['calib_fold_sd'] = float(np.std([macro_f1(Y[folds == f], Pc[folds == f]) for f in range(5)]))
    r['calib_per_class'] = dict(zip(CLASSES, np.round(per_class_f1(Y, Pc), 3).tolist()))
    r['argmax_balanced_recall'] = float(np.mean([(P.argmax(1)[y == k] == k).mean() for k in range(K)]))
    return r, Pc


def paired_diff_ci(Y, Pa, Pb, n=1000, seed=0):
    """Bootstrap CI of macro_f1(Pa) - macro_f1(Pb) on the same resampled lesions."""
    rng = np.random.default_rng(seed); idx = np.arange(len(Y)); d = []
    for _ in range(n):
        s = rng.choice(idx, len(idx)); d.append(macro_f1(Y[s], Pa[s]) - macro_f1(Y[s], Pb[s]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', nargs='+', default=None, help='default: all found in data/emb')
    ap.add_argument('--preps', nargs='+', default=None)
    ap.add_argument('--setups', nargs='+', default=['der', 'cli', 'both', 'both_meta'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUT / 'probe_log.txt')
    print(f'\n===== step2_probe {time.strftime("%Y-%m-%d %H:%M:%S")} device={DEV} args={vars(a)}')

    df = load_train()
    lids = pd.read_csv(EMB / 'training_lesions.csv')['lesion_id']
    missing = set(df.index) - set(lids)
    if missing:
        print(f'!! {len(missing)} lesions of folds.csv have no embeddings (smoke-test extract?) -> restricting')
    df = df.loc[[l for l in lids if l in df.index]]
    order = pd.Index(lids).get_indexer(df.index)
    Y = df[CLASSES].values; y = df['label'].values; folds = df['fold'].values
    meta = df[FEATURES['demo_monet_manip']].values.astype(float)

    avail = sorted({p.name.split('__')[1] for p in EMB.glob('training__*__*__der.npy')})
    backbones = a.backbones or avail
    preps_all = sorted({p.name.split('__')[2] for p in EMB.glob('training__*__*__der.npy')})
    preps = a.preps or preps_all

    # Step 1 reference (best calibrated config), recomputed from its OOF so the paired bootstrap is exact
    s1 = json.load(open(ROOT / 'results' / 'step1' / 'metrics.json'))
    ref_name = max(s1, key=lambda k: s1[k]['calib'])
    ref_oof = pd.read_csv(ROOT / 'results' / 'step1' / f'oof_{ref_name}.csv', index_col=0).loc[df.index, CLASSES].values
    ref_c = nested_calibrate(Y, ref_oof, folds)
    print(f'Step 1 reference: {ref_name} calib={macro_f1(Y, ref_c):.3f} (on {len(df)} lesions)')

    results = {}
    mpath = OUT / 'metrics.json'
    if mpath.exists():
        results = json.load(open(mpath))
    for b in backbones:
        for p in preps:
            fd = EMB / f'training__{b}__{p}__der.npy'
            if not fd.exists():
                print(f'-- skip {b}/{p}: no embeddings'); continue
            E = {m: np.load(EMB / f'training__{b}__{p}__{m}.npy')[order] for m in ('der', 'cli')}
            for s in a.setups:
                blocks = {'der': [E['der']], 'cli': [E['cli']], 'both': [E['der'], E['cli']],
                          'both_meta': [E['der'], E['cli'], meta]}[s]
                t0 = time.time()
                P, chosen = run_config(blocks, y, folds)
                r, Pc = evaluate(Y, y, P, folds)
                r['C_per_fold'] = chosen
                r['dim'] = int(sum(B.shape[1] for B in blocks))
                r['delta_vs_step1'] = float(r['calib'] - macro_f1(Y, ref_c))
                r['delta_vs_step1_ci95'] = paired_diff_ci(Y, Pc, ref_c)
                r['step1_ref'] = ref_name
                if b == 'own':
                    r['note'] = 'biased upward: backbone fine-tuned on data overlapping MILK10k; not ranked'
                name = f'{b}__{p}__{s}'; results[name] = r
                pd.DataFrame(P, index=df.index, columns=CLASSES).assign(fold=folds).to_csv(OUT / f'oof_{name}.csv')
                print(f"{name:32s} dim={r['dim']:5d} raw={r['raw']:.3f} argmax={r['argmax']:.3f} "
                      f"calib={r['calib']:.3f} CI95=({r['calib_ci95'][0]:.3f},{r['calib_ci95'][1]:.3f}) "
                      f"foldSD={r['calib_fold_sd']:.3f} d_vs_S1={r['delta_vs_step1']:+.3f} "
                      f"({r['delta_vs_step1_ci95'][0]:+.3f},{r['delta_vs_step1_ci95'][1]:+.3f}) C={chosen} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                json.dump(results, open(mpath, 'w'), indent=1)

    rows = [{'config': k, 'backbone': k.split('__')[0], 'prep': k.split('__')[1], 'setup': k.split('__')[2],
             'calib': v['calib'], 'ci_lo': v['calib_ci95'][0], 'ci_hi': v['calib_ci95'][1],
             'fold_sd': v['calib_fold_sd'], 'delta_vs_step1': v['delta_vs_step1'],
             'delta_ci_lo': v['delta_vs_step1_ci95'][0], 'delta_ci_hi': v['delta_vs_step1_ci95'][1],
             'argmax_bal_recall': v['argmax_balanced_recall'], 'ranked': not k.startswith('own__')}
            for k, v in results.items()]
    summ = pd.DataFrame(rows).sort_values(['ranked', 'calib'], ascending=[False, False])
    summ.to_csv(OUT / 'summary.csv', index=False)
    print('\n' + summ.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
