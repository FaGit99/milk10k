"""Step 10: epoch sweep for PanDerm-B (dermoscopic + clinical single-view models -> late fusion).

Question: is the fixed 25-epoch budget of Steps 3/4 a good choice? One 40-epoch run per view and fold is trained;
after EVERY epoch the model is (a) saved and (b) used to predict the held-out fold and the test set. At the end all
epochs are evaluated with the full Step 3/4 indicator set, for der, cli and late fusion (geo 50/50 of der and cli of
the SAME epoch).

Everything else is Step 4 unchanged (variants 'der' / 'cli' of step4_twoview): canonical folds, per-fold seed
1000*seed+fold, logit-adjusted CE (tau=1), AdamW + LLRD 0.75, peak LR 1e-4, warm-up 2 epochs, cosine to 1e-6,
drop_path 0.1, bf16, 'milk' augmentation, eval 'crop' 224, flip TTA, nested per-class threshold calibration.

IMPORTANT - what an epoch-e snapshot is: the cosine schedule runs over the full --epochs (40). The epoch-25 snapshot of
this run is therefore NOT the same model as the existing 25-epoch runs (there the LR had already decayed to 1e-6 at
epoch 25; here it is still ~3.7e-5). The existing 25-epoch seed-0 runs are reported as the reference.

Model selection and bias:
  best epoch      = argmax over epochs of the pooled calibrated macro F1 (challenge metric after nested thresholds).
                    This number is optimistic, because the same OOF predictions choose and score the epoch.
  nested estimate = honest estimate of the whole procedure 'train 40 epochs, pick the epoch': for each outer fold f,
                    the epoch is chosen on the other 4 folds only (their own nested calibration), then fold f is
                    scored at that epoch with thresholds from the other 4 folds.
  paired bootstrap of the best epoch vs the last epoch of this run, vs epoch 25 of this run and vs the existing
  25-epoch reference runs.

Outputs (results/step10/):
  runs/{view}_{bb}{tag}__s{seed}__f{fold}.npz   per-epoch OOF (E x n x K) + test (E x m x K) probabilities, train loss, LR
  ckpt/{view}_{bb}{tag}__s{seed}__f{fold}/ep{EEE}.pth   state_dict after each epoch (--ckpt all; ~350 MB each, fp32)
  per_epoch_{bb}{tag}__s{seed}.csv   all indicators per view and epoch
  metrics.json                       per-epoch results, best epochs, nested estimates, paired comparisons
  curves_{bb}{tag}__s{seed}.png      calibrated macro F1 (95% CI), train loss, held-out NLL vs epoch
  submission_late_{bb}{tag}__s{seed}__ep{E}.csv   late fusion at the best epoch (thresholds from the full OOF)
  best_model_{bb}{tag}__s{seed}.json  best epoch + the 10 checkpoint paths (der + cli, 5 folds)
  train_log.txt

Usage (repo root):
  python src/step10_epoch_sweep.py                                  # der + cli, PanDerm-B, seed 0, 40 epochs, 5 folds
  python src/step10_epoch_sweep.py --eval_only                      # re-run the evaluation from saved runs
  python src/step10_epoch_sweep.py --limit 60 --epochs 2 --img 64 --tag _smoke --no_pretrained --workers 0   # smoke test
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CLASSES, load_train, macro_f1, per_class_f1, nested_calibrate, fit_thresholds, rescale, bootstrap_ci
from step2_extract import image_table, Tee
from step2_probe import evaluate, paired_diff_ci
from step3_finetune import ImbalanceLoss, seed_worker
from step4_twoview import Pair, TwoView, param_groups, predict, geo

K = len(CLASSES)
OUT = ROOT / 'results' / 'step10'
S3, S4 = ROOT / 'results' / 'step3', ROOT / 'results' / 'step4'
VIEWS = ('der', 'cli')
EPS = 1e-7


# ---------------------------------------------------------------- training
def run_fold(a, view, f, df, test, dev, name):
    """Train one view on the 4 other folds for a.epochs; predict held-out + test after every epoch."""
    s = 1000 * a.seed + f  # same per-fold seed scheme as Steps 3/4
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    tr, te = df['fold'].values != f, df['fold'].values == f
    y = df['label'].values
    net = TwoView(view, a.backbone, a.drop_path, 0.0, 0.0, not a.no_pretrained, a.img).to(dev)
    mean = torch.tensor(net.mean, device=dev).view(1, 1, 3, 1, 1); std = torch.tensor(net.std, device=dev).view(1, 1, 3, 1, 1)
    views = (view,)

    g = torch.Generator(); g.manual_seed(s)
    kw = dict(num_workers=a.workers, pin_memory=dev.type == 'cuda', persistent_workers=a.workers > 0,
              worker_init_fn=seed_worker)
    D, C = df['der_isic_id'].values, df['cli_isic_id'].values
    dl_tr = DataLoader(Pair('training', df.index[tr], D[tr], C[tr], y[tr], True, views, a.img, 'milk'),
                       batch_size=a.batch, shuffle=True, drop_last=True, generator=g, **kw)
    dl_te = DataLoader(Pair('training', df.index[te], D[te], C[te], y[te], False, views, a.img), batch_size=a.batch * 2, **kw)
    dl_test = DataLoader(Pair('test', test.index, test['der'].values, test['cli'].values, np.zeros(len(test)), False, views, a.img),
                         batch_size=a.batch * 2, **kw)

    counts = np.bincount(y[tr], minlength=K)
    crit = ImbalanceLoss('la', counts, dev, a.tau)
    opt = torch.optim.AdamW(param_groups(net, a.wd, a.layer_decay), lr=a.lr, betas=(0.9, 0.999))
    steps = a.epochs * len(dl_tr); warm = min(a.warmup * len(dl_tr), steps - 1)

    def lr_at(i):
        if i < warm:
            return a.lr * (i + 1) / warm
        t = (i - warm) / max(1, steps - warm)
        return a.min_lr + (a.lr - a.min_lr) * 0.5 * (1 + math.cos(math.pi * t))

    ckdir = OUT / 'ckpt' / f'{name}__f{f}'
    if a.ckpt == 'all':
        ckdir.mkdir(parents=True, exist_ok=True)
    print(f'--- {name} fold={f}: train={tr.sum()} heldout={te.sum()} test={len(test)} steps={steps} '
          f'ckpt={a.ckpt}', flush=True)
    E = a.epochs
    oof = np.zeros((E, te.sum(), K), np.float32); tst = np.zeros((E, len(test), K), np.float32)
    losses, lrs = np.zeros(E), np.zeros(E)
    Yte = np.eye(K)[y[te]]
    it, t0 = 0, time.time()
    for ep in range(E):
        net.train(); tot, nb = 0.0, 0
        for x, yb in dl_tr:
            lr = lr_at(it)
            for gr in opt.param_groups:
                gr['lr'] = lr * gr['lr_scale']
            x = (x.to(dev, non_blocking=True).float().div_(255) - mean) / std
            yb = yb.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                logits = net(x)
            loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            opt.step()
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {ep} step {it}')
            tot += loss.item(); nb += 1; it += 1
        losses[ep], lrs[ep] = tot / max(nb, 1), lr
        # prediction must not change the training trajectory: DataLoader iteration draws from the global RNG,
        # so all RNG states are saved and restored (the run then equals a plain a.epochs run of step4_twoview)
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if dev.type == 'cuda' else None,
               np.random.get_state(), random.getstate())
        oof[ep] = predict(net, dl_te, mean, std, dev, a.tta)
        tst[ep] = predict(net, dl_test, mean, std, dev, a.tta)
        torch.set_rng_state(rng[0]); np.random.set_state(rng[2]); random.setstate(rng[3])
        if rng[1] is not None:
            torch.cuda.set_rng_state_all(rng[1])
        if a.ckpt == 'all':
            torch.save(net.state_dict(), ckdir / f'ep{ep + 1:03d}.pth')
        # held-out score per epoch: information only (selection happens in evaluate_sweep, nested)
        print(f'  ep {ep + 1:3d}/{E} loss={losses[ep]:.4f} lr={lr:.2e} held-out raw F1={macro_f1(Yte, oof[ep]):.3f} '
              f'[{time.time() - t0:.0f}s]', flush=True)
    del net, opt
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return oof, tst, losses, lrs


# ---------------------------------------------------------------- extra indicators
def ranking_metrics(Y, P):
    """Macro one-vs-rest AUROC and average precision over classes with >= 1 positive (sklearn if available)."""
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        roc_auc_score = average_precision_score = None
    aucs, aps = [], []
    for k in range(K):
        yk = Y[:, k].astype(bool)
        if yk.sum() == 0 or yk.all():
            continue
        if roc_auc_score is not None:
            aucs.append(roc_auc_score(yk, P[:, k])); aps.append(average_precision_score(yk, P[:, k]))
        else:  # Mann-Whitney AUROC with average ranks; AP as mean precision at each positive (descending score)
            r = pd.Series(P[:, k]).rank().values
            n1, n0 = yk.sum(), (~yk).sum()
            aucs.append((r[yk].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            o = np.argsort(-P[:, k], kind='mergesort'); hit = yk[o]
            aps.append(float((np.cumsum(hit)[hit] / (np.flatnonzero(hit) + 1)).mean()))
    return float(np.mean(aucs)), float(np.mean(aps))


def indicators(Y, y, P, folds):
    r, Pc = evaluate(Y, y, P, folds)  # raw, argmax, calib, calib_ci95, calib_fold_sd, calib_per_class, bal. recall
    r['auc_macro'], r['ap_macro'] = ranking_metrics(Y, P)
    r['nll'] = float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)).mean())
    pc = per_class_f1(Y, Pc)
    r['calib_excl_MAL_OTH'] = float(np.delete(pc, CLASSES.index('MAL_OTH')).mean())
    return r, Pc


def nested_epoch_selection(Y, Ps, folds):
    """Ps: (E, n, K). For each outer fold: choose the epoch on the other folds (scored with their own nested
    calibration), then score the outer fold at that epoch with thresholds fit on the other folds."""
    out = np.zeros_like(Ps[0]); chosen = {}
    for f in np.unique(folds):
        te = folds == f; tr = ~te
        sc = [macro_f1(Y[tr], nested_calibrate(Y[tr], P[tr], folds[tr])) for P in Ps]
        e = int(np.argmax(sc))  # first maximum = earliest epoch on ties
        out[te] = rescale(Ps[e][te], fit_thresholds(Y[tr], Ps[e][tr]))
        chosen[int(f)] = e + 1
    return out, chosen


# ---------------------------------------------------------------- reference (existing 25-epoch seed runs)
def load_runs(paths, df, test):
    if not all(p.exists() for p in paths):
        return None
    P = np.full((len(df), K), np.nan); Pt = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        o = pd.DataFrame(z['oof'], index=z['oof_ids']); ids = [i for i in o.index if i in df.index]
        P[df.index.get_indexer(ids)] = o.loc[ids].values
        Pt.append(pd.DataFrame(z['test'], index=z['test_ids']).reindex(test.index).values)
    return (P, np.mean(Pt, 0)) if np.isfinite(P).all() else None


def reference(a, df, test):
    der = load_runs([S3 / 'runs' / f'{a.backbone}__s{a.seed}__f{f}.npz' for f in range(5)], df, test)
    cli = load_runs([S4 / 'runs' / f'cli_{a.backbone}__s{a.seed}__f{f}.npz' for f in range(5)], df, test)
    ref = {}
    if der is not None:
        ref['der'] = der[0]
    if cli is not None:
        ref['cli'] = cli[0]
    if der is not None and cli is not None:
        ref['late'] = geo(der[0], cli[0])
    return ref


# ---------------------------------------------------------------- evaluation of all epochs
def evaluate_sweep(a, df, test, base):
    Y, y, folds = df[CLASSES].values, df['label'].values, df['fold'].values
    P, Pt, L = {}, {}, {}
    for v in VIEWS:
        fs = [OUT / 'runs' / f'{v}_{base}__f{f}.npz' for f in range(5)]
        if not all(p.exists() for p in fs):
            print(f'{v}: not all 5 folds present -> no evaluation'); return
        z = [np.load(p, allow_pickle=True) for p in fs]
        E = min(len(zz['loss']) for zz in z)
        if any(len(zz['loss']) != E for zz in z):
            print(f'!! {v}: folds have different epoch counts, using the first {E}')
        Pv = np.full((E, len(df), K), np.nan)
        for zz in z:
            idx = df.index.get_indexer(zz['oof_ids']); assert (idx >= 0).all()
            Pv[:, idx] = zz['oof'][:E]
        assert np.isfinite(Pv).all()
        P[v] = Pv.astype(np.float64)
        Pt[v] = np.mean([zz['test'][:E][:, pd.Index(zz['test_ids']).get_indexer(test.index)] for zz in z], 0).astype(np.float64)
        L[v] = np.mean([zz['loss'][:E] for zz in z], 0)
    E = min(P['der'].shape[0], P['cli'].shape[0])
    P['late'] = np.stack([geo(P['der'][e], P['cli'][e]) for e in range(E)])
    Pt['late'] = np.stack([geo(Pt['der'][e], Pt['cli'][e]) for e in range(E)])
    ref = reference(a, df, test)
    refc = {v: nested_calibrate(Y, R, folds) for v, R in ref.items()}

    res = {'args': vars(a), 'epochs': E, 'per_epoch': {}, 'best': {}, 'nested': {}}
    rows = []
    t0 = time.time()
    for v in ('der', 'cli', 'late'):
        Pcs = []
        for e in range(E):
            r, Pc = indicators(Y, y, P[v][e], folds); Pcs.append(Pc)
            res['per_epoch'][f'{v}_ep{e + 1:03d}'] = r
            rows.append({'view': v, 'epoch': e + 1, 'train_loss': (L[v][e] if v in L else np.nan),
                         'calib': r['calib'], 'ci_lo': r['calib_ci95'][0], 'ci_hi': r['calib_ci95'][1],
                         'fold_sd': r['calib_fold_sd'], 'calib_excl_MAL_OTH': r['calib_excl_MAL_OTH'],
                         'raw': r['raw'], 'argmax': r['argmax'], 'argmax_bal_recall': r['argmax_balanced_recall'],
                         'auc_macro': r['auc_macro'], 'ap_macro': r['ap_macro'], 'nll': r['nll'],
                         **{f'f1_{c}': val for c, val in r['calib_per_class'].items()}})
        cal = np.array([res['per_epoch'][f'{v}_ep{e + 1:03d}']['calib'] for e in range(E)])
        eb = int(np.argmax(cal))
        best = {'epoch': eb + 1, 'calib': float(cal[eb]), 'note': 'optimistic: chosen and scored on the same OOF',
                'top5': [[int(i + 1), float(cal[i])] for i in np.argsort(-cal)[:5]]}
        for lab, e2 in (('last', E - 1), ('ep025', 24)):
            if e2 < E and e2 != eb:
                best[f'delta_vs_{lab}'] = float(cal[eb] - cal[e2])
                best[f'delta_vs_{lab}_ci95'] = paired_diff_ci(Y, Pcs[eb], Pcs[e2])
        if v in refc:
            best['ref_25ep_calib'] = macro_f1(Y, refc[v])
            best['delta_vs_ref_25ep'] = float(cal[eb] - best['ref_25ep_calib'])
            best['delta_vs_ref_25ep_ci95'] = paired_diff_ci(Y, Pcs[eb], refc[v])
        res['best'][v] = best
        Pn, chosen = nested_epoch_selection(Y, P[v], folds)
        n = {'calib': macro_f1(Y, Pn), 'calib_ci95': bootstrap_ci(Y, Pn), 'chosen_epoch_per_fold': chosen,
             'calib_per_class': dict(zip(CLASSES, np.round(per_class_f1(Y, Pn), 3).tolist()))}
        if v in refc:
            n['delta_vs_ref_25ep'] = float(n['calib'] - macro_f1(Y, refc[v]))
            n['delta_vs_ref_25ep_ci95'] = paired_diff_ci(Y, Pn, refc[v])
        res['nested'][v] = n
        print(f'[{v}] evaluated {E} epochs [{time.time() - t0:.0f}s]', flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT / f'per_epoch_{base}.csv', index=False, float_format='%.4f')
    if a.limit:
        res['smoke_test'] = True

    # ---- report
    print('\n===== epoch sweep summary (calibrated macro F1 = challenge metric after nested per-class thresholds)')
    for v in ('der', 'cli', 'late'):
        b, n = res['best'][v], res['nested'][v]
        line = f"{v:5s} best epoch {b['epoch']:3d}: {b['calib']:.3f} (optimistic)"
        for lab in ('last', 'ep025'):
            if f'delta_vs_{lab}' in b:
                lo, hi = b[f'delta_vs_{lab}_ci95']; line += f" | vs {lab}: {b[f'delta_vs_{lab}']:+.3f} ({lo:+.3f}..{hi:+.3f})"
        if 'delta_vs_ref_25ep' in b:
            lo, hi = b['delta_vs_ref_25ep_ci95']
            line += f" | vs existing 25-ep run ({b['ref_25ep_calib']:.3f}): {b['delta_vs_ref_25ep']:+.3f} ({lo:+.3f}..{hi:+.3f})"
        print(line)
        line = (f"      nested epoch selection: {n['calib']:.3f} CI95=({n['calib_ci95'][0]:.3f},{n['calib_ci95'][1]:.3f}) "
                f"epochs per fold {n['chosen_epoch_per_fold']}")
        if 'delta_vs_ref_25ep' in n:
            lo, hi = n['delta_vs_ref_25ep_ci95']; line += f" | vs existing 25-ep run: {n['delta_vs_ref_25ep']:+.3f} ({lo:+.3f}..{hi:+.3f})"
        print(line)
        print(f"      top 5 epochs: {b['top5']}")
    cols = ['view', 'epoch', 'train_loss', 'calib', 'ci_lo', 'ci_hi', 'calib_excl_MAL_OTH', 'auc_macro', 'ap_macro', 'nll']
    print('\n' + per[per['view'] == 'late'][cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # ---- best late-fusion model: submission + checkpoint list
    eb = res['best']['late']['epoch']
    th = fit_thresholds(Y, P['late'][eb - 1])
    idx = pd.Index(test.index, name=a.id_col)
    sub = OUT / f'submission_late_{base}__ep{eb:03d}.csv'
    pd.DataFrame(rescale(Pt['late'][eb - 1], th), index=idx, columns=CLASSES).to_csv(sub, float_format='%.6f')
    res['_thresholds_late_best'] = dict(zip(CLASSES, np.round(th, 4).tolist()))
    ck = {v: [str((OUT / 'ckpt' / f'{v}_{base}__f{f}' / f'ep{eb:03d}.pth').relative_to(ROOT)) for f in range(5)] for v in VIEWS}
    bm = {'selection': 'late fusion, argmax of pooled calibrated macro F1 over epochs', 'epoch': eb,
          'cv_calib_optimistic': res['best']['late']['calib'], 'cv_calib_nested': res['nested']['late']['calib'],
          'submission': str(sub.relative_to(ROOT)), 'checkpoints': ck,
          'checkpoints_exist': all((ROOT / p).exists() for v in VIEWS for p in ck[v]),
          'thresholds': res['_thresholds_late_best']}
    json.dump(bm, open(OUT / f'best_model_{base}.json', 'w'), indent=1)
    print(f'\nbest late-fusion epoch {eb}: submission {sub.name}; checkpoints listed in best_model_{base}.json')

    p = OUT / 'metrics.json'
    cur = json.load(open(p)) if p.exists() else {}
    cur[base] = res
    json.dump(cur, open(p, 'w'), indent=1)
    plot_curves(per, res, ref and {v: macro_f1(Y, refc[v]) for v in refc}, OUT / f'curves_{base}.png', base)


def plot_curves(per, res, refs, path, base):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed -> no plot'); return
    col = {'der': '#2a78d6', 'cli': '#eb6834', 'late': '#1baf7a'}  # categorical slots 1-3 (validated default palette)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for v in ('der', 'cli', 'late'):
        d = per[per['view'] == v]
        ax[0].fill_between(d['epoch'], d['ci_lo'], d['ci_hi'], color=col[v], alpha=0.12, lw=0)
        ax[0].plot(d['epoch'], d['calib'], color=col[v], lw=2, label=f'{v} (best ep {res["best"][v]["epoch"]})')
        ax[0].plot(res['best'][v]['epoch'], res['best'][v]['calib'], 'o', color=col[v], ms=8, mec='white', mew=2)
        if refs and v in refs:
            ax[0].axhline(refs[v], color=col[v], lw=1, ls='--')
        if v != 'late':
            ax[1].plot(d['epoch'], d['train_loss'], color=col[v], lw=2, label=v)
        ax[2].plot(d['epoch'], d['nll'], color=col[v], lw=2, label=v)
    ax[0].set_title('Calibrated macro F1 (OOF, 95% CI)\ndashed = existing 25-epoch seed run', fontsize=10)
    ax[1].set_title('Training loss (mean over folds)', fontsize=10)
    ax[2].set_title('Held-out NLL (uncalibrated probabilities)', fontsize=10)
    for x in ax:
        x.set_xlabel('epoch'); x.grid(alpha=0.25, lw=0.6); x.spines[['top', 'right']].set_visible(False)
        x.axvline(25, color='0.6', lw=0.8, ls=':'); x.legend(frameon=False, fontsize=9)
    fig.suptitle(f'Epoch sweep {base}', fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f'plot: {path.name}')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='panderm_b', choices=['panderm_b', 'eva02_b', 'dinov2_b'])
    ap.add_argument('--views', nargs='+', default=list(VIEWS), choices=list(VIEWS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--ckpt', default='all', choices=['all', 'none'], help="'all': save the model after every epoch")
    ap.add_argument('--img', type=int, default=224)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--min_lr', type=float, default=1e-6)
    ap.add_argument('--layer_decay', type=float, default=0.75)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--drop_path', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--no_tta', dest='tta', action='store_false')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--id_col', default='lesion_id')
    ap.add_argument('--limit', type=int, default=0, help='first N lesions (smoke test; needs a --tag)')
    ap.add_argument('--no_pretrained', action='store_true', help='random init; pipeline test only')
    a = ap.parse_args()
    if (a.limit or a.img != 224 or a.no_pretrained) and not a.tag:
        ap.error('--limit / --img != 224 / --no_pretrained need a --tag (e.g. _smoke) so the runs do not mix')

    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'runs').mkdir(exist_ok=True)
    sys.stdout = Tee(OUT / 'train_log.txt')
    print(f'\n===== step10_epoch_sweep {time.strftime("%Y-%m-%d %H:%M:%S")} args={vars(a)}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__}, device {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev.type == 'cuda' else ''))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    df = load_train(); test = image_table('test')
    if a.limit:
        df, test = df.iloc[:a.limit], test.iloc[:min(a.limit, len(test))]
        print(f'!! --limit {a.limit}: smoke test, results are not meaningful')
    base = f'{a.backbone}{a.tag}__s{a.seed}'

    if not a.eval_only:
        for v in a.views:
            for f in a.folds:
                rf = OUT / 'runs' / f'{v}_{base}__f{f}.npz'
                if rf.exists() and not a.overwrite:
                    print(f'--- {v}_{base} fold={f}: exists, skipped ({rf.name})'); continue
                oof, tst, losses, lrs = run_fold(a, v, f, df, test, dev, f'{v}_{base}')
                te = df['fold'].values == f
                np.savez(rf, oof=oof, oof_ids=np.array(df.index[te]), test=tst, test_ids=np.array(test.index),
                         loss=losses, lr=lrs, args=json.dumps(vars(a)))
    evaluate_sweep(a, df, test, base)


if __name__ == '__main__':
    main()
