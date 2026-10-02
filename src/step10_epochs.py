"""Step 10: epoch-budget study for the PanDerm-B late-fusion setup (one seed).

For every budget E in --budgets (default 15 20 25 30 35 40 50) an INDEPENDENT run is trained, dermoscopic and clinical
model separately (Step 4 'der' / 'cli' variants, late fusion geo 50/50), 5 canonical folds. Everything else is the
Step 4 224 px setup unchanged: logit-adjusted CE (tau=1), AdamW + LLRD 0.75, lr 1e-4 -> 1e-6 cosine over the E epochs
with 2 warm-up epochs, batch 64, drop_path 0.1, 'milk' augmentation, bf16, flip TTA, per-fold seed 1000*seed+fold.

After EVERY epoch: held-out fold + test are predicted (images cached once per fold, flip TTA unless --no_epoch_tta)
and the weights are saved (fp16) to results/step10/ckpt/. When both views of a budget have all 5 folds,
step10_analyze.analyze_budget scores every epoch (der, cli, late; nested-calibrated macro F1, AUC, AP) and ONLY the
checkpoints of the --keep (default 3) best epochs by late calibrated macro F1 are kept; the others are deleted.
NOTE: those 'best' epochs are chosen on the CV folds themselves (optimistic). The unbiased comparison is between
budgets at their final epoch (step10_analyze.compare_all, run at the end).

Outputs (results/step10/): preds/*.npz (per-epoch OOF + test probabilities, loss), ckpt/*.pt (kept epochs),
  curve_e{E}.csv, kept_e{E}.json, summary.csv, per_class_final.csv, submission_late_e{E}.csv, curves.png, train_log.txt
Resume: finished folds (preds file present) are skipped; a finished budget (kept_e{E}.json present) is skipped.

Usage (repo root):
  python src/step10_epochs.py                                   # full study (~6 h on the RTX PRO 6000, rough estimate)
  python src/step10_epochs.py --budgets 2 3 --limit 250 --keep 1       # smoke test (writes to results/step10_smoke)
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step10_analyze as A
from common import ROOT, CLASSES, load_train
from step2_extract import image_table, Tee
from step3_finetune import ImbalanceLoss, seed_worker
from step4_twoview import Pair, TwoView, param_groups

K = len(CLASSES)


def cache(ds, a):
    """Load a (non-augmented) dataset once into a pinned uint8 tensor (N, V, 3, H, W). Own generator: does not
    consume the global torch RNG, so training randomness is the same as in Step 4."""
    dl = DataLoader(ds, batch_size=128, num_workers=a.workers, generator=torch.Generator().manual_seed(0))
    X = torch.cat([x for x, _ in dl])
    return X.pin_memory() if torch.cuda.is_available() else X


def predict_cached(net, X, mean, std, dev, tta, bs=128):
    """Same computation as step4_twoview.predict, on cached images."""
    net.eval(); out = []
    with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        for i in range(0, len(X), bs):
            x = (X[i:i + bs].to(dev, non_blocking=True).float().div_(255) - mean) / std
            views = [x, x.flip(4), x.flip(3), x.flip(3).flip(4)] if tta else [x]
            out.append(torch.stack([F.softmax(net(v).float(), 1) for v in views]).mean(0).cpu())
    return torch.cat(out).double().numpy()


def ckpt_path(out, view, bb, E, seed, f, ep):
    return out / 'ckpt' / f'{view}_{bb}_e{E}__s{seed}__f{f}__ep{ep:03d}.pt'


def run_fold(a, out, view, E, f, df, test, dev):
    seed, bb = a.seed, a.bb
    s = 1000 * seed + f
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    tr, te = df['fold'].values != f, df['fold'].values == f
    y = df['label'].values
    net = TwoView(view, bb, a.drop_path, 0.0, 0.0, True, a.img).to(dev)
    mean = torch.tensor(net.mean, device=dev).view(1, 1, 3, 1, 1); std = torch.tensor(net.std, device=dev).view(1, 1, 3, 1, 1)
    views = (view,)
    g = torch.Generator(); g.manual_seed(s)
    kw = dict(num_workers=a.workers, pin_memory=dev.type == 'cuda', persistent_workers=a.workers > 0,
              worker_init_fn=seed_worker)
    D, C = df['der_isic_id'].values, df['cli_isic_id'].values
    dl_tr = DataLoader(Pair('training', df.index[tr], D[tr], C[tr], y[tr], True, views, a.img, 'milk'),
                       batch_size=a.batch, shuffle=True, drop_last=True, generator=g, **kw)
    X_te = cache(Pair('training', df.index[te], D[te], C[te], y[te], False, views, a.img), a)
    X_test = cache(Pair('test', test.index, test['der'].values, test['cli'].values, np.zeros(len(test)), False, views, a.img), a)

    counts = np.bincount(y[tr], minlength=K)
    crit = ImbalanceLoss('la', counts, dev, 1.0, 0.5, 2.0, 0.0)
    opt = torch.optim.AdamW(param_groups(net, a.wd, a.layer_decay), lr=a.lr, betas=(0.9, 0.999))
    steps = E * len(dl_tr); warm = min(a.warmup * len(dl_tr), steps - 1)

    def lr_at(i):
        if i < warm:
            return a.lr * (i + 1) / warm
        t = (i - warm) / max(1, steps - warm)
        return a.min_lr + (a.lr - a.min_lr) * 0.5 * (1 + math.cos(math.pi * t))

    print(f'--- {view}/{bb} budget={E} seed={seed} fold={f}: train={tr.sum()} heldout={te.sum()} steps={steps}', flush=True)
    P_oof = np.zeros((E, te.sum(), K)); P_test = np.zeros((E, len(test), K)); losses = []
    it, t_start = 0, time.time()
    for ep in range(E):
        t0 = time.time(); net.train(); tot, nb = 0.0, 0
        for x, yb in dl_tr:
            lr = lr_at(it)
            for gr in opt.param_groups:
                gr['lr'] = lr * gr['lr_scale']
            x = (x.to(dev, non_blocking=True).float().div_(255) - mean) / std
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                loss = crit(net(x), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            opt.step()
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {ep} step {it}')
            tot += loss.item(); nb += 1; it += 1
        t1 = time.time()
        P_oof[ep] = predict_cached(net, X_te, mean, std, dev, a.epoch_tta)
        P_test[ep] = predict_cached(net, X_test, mean, std, dev, a.epoch_tta)
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                   ckpt_path(out, view, bb, E, seed, f, ep + 1))
        losses.append(tot / max(nb, 1))
        print(f'  ep {ep + 1:3d}/{E} loss={losses[-1]:.4f} lr={lr:.2e} train {t1 - t0:.0f}s, '
              f'eval+save {time.time() - t1:.0f}s [{time.time() - t_start:.0f}s total]', flush=True)
    del net, opt
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return P_oof, P_test, np.array(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budgets', nargs='+', type=int, default=[15, 20, 25, 30, 35, 40, 50])
    ap.add_argument('--views', nargs='+', default=['der', 'cli'], choices=['der', 'cli'])
    ap.add_argument('--bb', default='panderm_b', choices=['panderm_b', 'eva02_b', 'dinov2_b'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--img', type=int, default=224)
    ap.add_argument('--keep', type=int, default=3, help='checkpoints of this many best epochs are kept per budget')
    ap.add_argument('--no_epoch_tta', dest='epoch_tta', action='store_false',
                    help='per-epoch evaluation without flip TTA (faster; then no epoch reproduces the Step 4 numbers)')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--min_lr', type=float, default=1e-6)
    ap.add_argument('--layer_decay', type=float, default=0.75)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--drop_path', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N lesions, output to results/step10_smoke')
    a = ap.parse_args()
    if a.img != 224:
        ap.error('this study is defined at 224 px')

    out = ROOT / 'results' / ('step10_smoke' if a.limit else 'step10')
    A.OUT = out
    for d in (out, out / 'preds', out / 'ckpt'):
        d.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / 'train_log.txt')
    print(f'\n===== step10_epochs {time.strftime("%Y-%m-%d %H:%M:%S")} args={vars(a)}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__}, device {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev.type == 'cuda' else ''))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    df = load_train(); test = image_table('test')
    if a.limit:  # keep all 5 folds present
        df = df.groupby('fold', group_keys=False).head(a.limit // 5); test = test.iloc[:32]
        print(f'!! smoke test: {len(df)} lesions, {len(test)} test lesions; results are not meaningful')

    for E in a.budgets:
        kept = out / f'kept_e{E}.json'
        if kept.exists():
            print(f'=== budget {E}: already finished ({kept.name}), skipped'); continue
        for view in a.views:
            for f in range(5):
                p = A.pred_path(view, a.bb, E, a.seed, f)
                if p.exists():
                    print(f'--- {view} budget={E} fold={f}: loaded {p.name}'); continue
                P_oof, P_test, L = run_fold(a, out, view, E, f, df, test, dev)
                np.savez(p, oof=P_oof, oof_ids=np.array(df.index[df['fold'].values == f]), test=P_test,
                         test_ids=np.array(test.index), loss=L, args=json.dumps(vars(a)))
        if set(a.views) != {'der', 'cli'}:
            print(f'budget {E}: only {a.views} -> no late-fusion analysis, all checkpoints kept'); continue
        ranked = A.analyze_budget(E, a.bb, a.seed, df, log=print)
        keep = sorted(ranked[:a.keep]); removed = 0
        for view in a.views:
            for f in range(5):
                for ep in range(1, E + 1):
                    c = ckpt_path(out, view, a.bb, E, a.seed, f, ep)
                    if ep not in keep and c.exists():
                        c.unlink(); removed += 1
        json.dump({'budget': E, 'kept_epochs': keep, 'ranked_by': 'late nested-calibrated macro F1 (OOF, optimistic)'},
                  open(kept, 'w'), indent=1)
        print(f'=== budget {E}: kept checkpoints of epochs {keep}, deleted {removed} others', flush=True)

    A.compare_all(a.budgets, a.bb, a.seed, log=print, df=df)


if __name__ == '__main__':
    main()
