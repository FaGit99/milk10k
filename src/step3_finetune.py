"""Step 3: full fine-tune on the DERMOSCOPIC image only, imbalance-aware loss, nested threshold calibration.

Backbones: panderm_b (PanDerm-Base, local checkpoint), dinov2_b (timm vit_base_patch14_dinov2.lvd142m),
           eva02_b (timm eva02_base_patch14_224.mim_in22k; not run by default: --backbones eva02_b).
Everything that decides comparability is taken from Steps 1/2 unchanged:
  folds        data/folds.csv (canonical 5-fold, seed 42)
  metric       challenge rule: per-class F1, each class binarised at >= 0.5, macro mean (common.macro_f1)
  calibration  common.nested_calibrate: per-class thresholds fit on the OTHER 4 folds' OOF only
  CIs          lesion bootstrap (common.bootstrap_ci) and paired bootstrap (step2_probe.paired_diff_ci)
  eval input   Step 2 'crop': short side -> 224 (bicubic), centre crop 224x224, model's own mean/std

Per outer fold: train on the 4 other folds for a FIXED number of epochs (no early stopping, no model selection),
then predict the held-out fold (OOF) and the test set. The held-out fold is never used for any training decision.

Imbalance handling (--loss):
  la     logit-adjusted CE (Menon et al., ICLR 2021): CE(logits + tau*log prior). Inference uses the raw logits.
  wce    class-weighted CE, weight_k = (n / (K n_k))^beta   (beta=1: inverse frequency, as in the MILK10k baseline)
  focal  focal loss (gamma) with the same class weights as wce
  ce     plain CE (reference)
Class priors / weights are computed on the training folds of each outer fold only.

Threshold tuning: OOF probabilities -> common.nested_calibrate (reported 'calib'). For the test submission the
per-class thresholds are fit on the full OOF of that backbone and applied via common.rescale (threshold -> 0.5).

Outputs (results/step3/):
  runs/{bb}__s{seed}__f{fold}.npz   OOF + test probabilities per fold (resume: existing files are skipped)
  oof_ft_{bb}__s{seed}.csv          OOF probabilities (uncalibrated), same format as Step 2
  test_ft_{bb}.csv                  test probabilities, mean over folds (and seeds), uncalibrated
  submission_ft_{bb}.csv            test probabilities after threshold rescaling (challenge format)
  metrics.json, summary.csv, train_log.txt

Usage (repo root):
  python src/step3_finetune.py                                   # both backbones, seed 0, 5 folds
  python src/step3_finetune.py --backbones dinov2_b --seeds 0 1 2
  python src/step3_finetune.py --backbones eva02_b                # third candidate, same settings
  python src/step3_finetune.py --limit 300 --epochs 1 --folds 0  # smoke test (no evaluation without all folds)
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CLASSES, load_train, macro_f1, nested_calibrate, fit_thresholds, rescale
from step2_extract import to_square, strip_state_dict, image_table, PANDERM_CKPT, Tee
from step2_probe import evaluate, paired_diff_ci

K = len(CLASSES)
OUT = ROOT / 'results' / 'step3'
S1, S2 = ROOT / 'results' / 'step1', ROOT / 'results' / 'step2'
IMG = 224


# ---------------------------------------------------------------- data
class RandomRot90:
    """Dermoscopy has no canonical orientation: rotate by 0/90/180/270 degrees."""
    def __call__(self, img):
        k = random.randint(0, 3)
        return img if k == 0 else img.transpose([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][k - 1])


def train_transform(size):
    # scale lower bound 0.35: eval centre crop of a 600x450 image covers 75% of the area
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), RandomRot90(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    ])


class Derm(Dataset):
    def __init__(self, split, lids, iids, labels, train, size=IMG):
        self.dir = ROOT / f'MILK10k_{split.capitalize()}_Input'
        self.items = list(zip(lids, iids)); self.labels = np.asarray(labels, np.int64)
        self.tf = train_transform(size) if train else None; self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        lid, iid = self.items[i]
        img = Image.open(self.dir / lid / f'{iid}.jpg').convert('RGB')
        img = self.tf(img) if self.tf is not None else to_square(img, 'crop', self.size)
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        return x, self.labels[i]


def seed_worker(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


# ---------------------------------------------------------------- models
def build_backbone(name, drop_path, pretrained):
    """Returns (module mapping normalised images -> (B, dim) features, dim, mean, std)."""
    if name == 'panderm_b':
        tp = str(ROOT / 'third_party' / 'panderm')
        if tp not in sys.path:
            sys.path.insert(0, tp)
        from modeling_finetune import panderm_base_patch16_224
        m = panderm_base_patch16_224(drop_path_rate=drop_path)  # lin_probe/standard: CLS token after final norm
        if pretrained:
            sd = strip_state_dict(torch.load(PANDERM_CKPT, map_location='cpu', weights_only=False))
            missing, unexpected = m.load_state_dict(sd, strict=False)
            missing = [k for k in missing if not k.startswith('head.')]
            print(f'  PanDerm load: missing={missing[:10]} ({len(missing)}), unexpected={unexpected[:10]} ({len(unexpected)})')
            if any(k.startswith(('blocks.', 'patch_embed.', 'norm.', 'cls_token')) for k in missing):
                raise RuntimeError('PanDerm checkpoint does not cover the encoder')
        m.head = nn.Identity()  # (num_classes=0 is not supported by the vendored constructor)
        return m, m.embed_dim, (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)  # same constants as Step 2
    if name == 'dinov2_b':
        import timm
        m = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=pretrained, num_classes=0,
                              img_size=IMG, drop_path_rate=drop_path)
        cfg = timm.data.resolve_model_data_config(m)
        return m, m.num_features, tuple(cfg['mean']), tuple(cfg['std'])
    if name == 'eva02_b':
        import timm  # same weights/pooling as Step 2: MIM in22k pretrain, global_pool='avg' + fc_norm, RoPE
        m = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=pretrained, num_classes=0,
                              drop_path_rate=drop_path)
        cfg = timm.data.resolve_model_data_config(m)
        return m, m.num_features, tuple(cfg['mean']), tuple(cfg['std'])
    raise ValueError(name)


class Net(nn.Module):
    def __init__(self, backbone, dim, head_drop):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(head_drop)
        self.head = nn.Linear(dim, K)
        nn.init.trunc_normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.drop(self.backbone(x)))


def param_groups(net, wd, layer_decay):
    """AdamW groups with layer-wise LR decay (BEiT-style). Embeddings = layer 0, block i = i+1, norm/head = L+1.
    No weight decay on 1-D params (norms, biases, LayerScale) and on cls/pos/reg tokens."""
    L = len(net.backbone.blocks)
    groups = {}
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue  # PanDerm's fixed sin-cos pos_embed
        if n.startswith('backbone.'):
            s = n[len('backbone.'):]
            if s.startswith(('cls_token', 'pos_embed', 'patch_embed', 'reg_token', 'mask_token')):
                lid = 0
            elif s.startswith('blocks.'):
                lid = int(s.split('.')[1]) + 1
            else:
                lid = L + 1
        else:
            lid = L + 1
        no_wd = p.ndim == 1 or n.endswith(('cls_token', 'pos_embed', 'reg_token'))
        g = groups.setdefault((lid, no_wd), {'params': [], 'lr_scale': layer_decay ** (L + 1 - lid),
                                             'weight_decay': 0.0 if no_wd else wd})
        g['params'].append(p)
    return list(groups.values())


# ---------------------------------------------------------------- loss
class ImbalanceLoss:
    def __init__(self, kind, counts, dev, tau=1.0, beta=0.5, gamma=2.0, smoothing=0.0):
        c = torch.as_tensor(counts, dtype=torch.float32, device=dev).clamp(min=1)
        prior = c / c.sum()
        w = (c.sum() / (K * c)) ** beta
        self.w = w / (w * prior).sum()  # expected sample weight = 1
        self.log_prior = prior.log()
        self.kind, self.tau, self.gamma, self.s = kind, tau, gamma, smoothing

    def __call__(self, logits, y):
        logits = logits.float()
        if self.kind == 'la':
            return F.cross_entropy(logits + self.tau * self.log_prior, y, label_smoothing=self.s)
        if self.kind == 'wce':
            return F.cross_entropy(logits, y, weight=self.w, label_smoothing=self.s)
        if self.kind == 'ce':
            return F.cross_entropy(logits, y, label_smoothing=self.s)
        if self.kind == 'focal':
            lp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
            wy = self.w[y]
            return (-wy * (1 - lp.exp()) ** self.gamma * lp).sum() / wy.sum()
        raise ValueError(self.kind)


# ---------------------------------------------------------------- train / predict
def predict(net, dl, mean, std, dev, tta):
    net.eval(); out = []
    with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        for x, _ in dl:
            x = (x.to(dev, non_blocking=True).float().div_(255) - mean) / std
            views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
            out.append(torch.stack([F.softmax(net(v).float(), 1) for v in views]).mean(0).cpu())
    return torch.cat(out).double().numpy()


def run_fold(a, bb, seed, f, df, test, dev):
    s = 1000 * seed + f
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    tr, te = df['fold'].values != f, df['fold'].values == f
    y = df['label'].values
    backbone, dim, mean, std = build_backbone(bb, a.drop_path, not a.no_pretrained)
    net = Net(backbone, dim, a.head_drop).to(dev)
    mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1); std = torch.tensor(std, device=dev).view(1, 3, 1, 1)

    g = torch.Generator(); g.manual_seed(s)
    kw = dict(num_workers=a.workers, pin_memory=dev.type == 'cuda', persistent_workers=a.workers > 0,
              worker_init_fn=seed_worker)
    ds_tr = Derm('training', df.index[tr], df['der_isic_id'].values[tr], y[tr], train=True)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, drop_last=True, generator=g, **kw)
    dl_te = DataLoader(Derm('training', df.index[te], df['der_isic_id'].values[te], y[te], train=False),
                       batch_size=a.batch * 2, **kw)
    dl_test = DataLoader(Derm('test', test.index, test['der'].values, np.zeros(len(test)), train=False),
                         batch_size=a.batch * 2, **kw)

    counts = np.bincount(y[tr], minlength=K)
    crit = ImbalanceLoss(a.loss, counts, dev, a.tau, a.beta, a.gamma, a.smoothing)
    opt = torch.optim.AdamW(param_groups(net, a.wd, a.layer_decay), lr=a.lr, betas=(0.9, 0.999))
    steps = a.epochs * len(dl_tr); warm = min(a.warmup * len(dl_tr), steps - 1)

    def lr_at(i):
        if i < warm:
            return a.lr * (i + 1) / warm
        t = (i - warm) / max(1, steps - warm)
        return a.min_lr + (a.lr - a.min_lr) * 0.5 * (1 + math.cos(math.pi * t))

    print(f'--- {bb} seed={seed} fold={f}: train={tr.sum()} heldout={te.sum()} test={len(test)} '
          f'steps={steps} counts={counts.tolist()}', flush=True)
    it, t0 = 0, time.time()
    for ep in range(a.epochs):
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
        print(f'  ep {ep + 1:3d}/{a.epochs} loss={tot / max(nb, 1):.4f} lr={lr:.2e} [{time.time() - t0:.0f}s]', flush=True)

    P_oof = predict(net, dl_te, mean, std, dev, a.tta)
    P_test = predict(net, dl_test, mean, std, dev, a.tta)
    Yte = np.eye(K)[y[te]]
    # information only - not used for any decision
    print(f'  held-out fold {f}: raw={macro_f1(Yte, P_oof):.3f} argmax={macro_f1(Yte, np.eye(K)[P_oof.argmax(1)]):.3f} '
          f'[{time.time() - t0:.0f}s total]', flush=True)
    ck = None
    if a.save_ckpt:
        ck = OUT / 'ckpt' / f'{bb}__s{seed}__f{f}.pth'; ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), ck)
    del net, opt
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return P_oof, P_test


# ---------------------------------------------------------------- main
def save_results(results):
    """Merge into metrics.json instead of overwriting it, so runs started in parallel
    (e.g. eva02_b next to panderm_b/dinov2_b) do not drop each other's entries."""
    p = OUT / 'metrics.json'
    cur = json.load(open(p)) if p.exists() else {}
    for k, v in results.items():
        if k == '_thresholds':
            cur.setdefault(k, {}).update(v)
        else:
            cur[k] = v
    json.dump(cur, open(p, 'w'), indent=1)
    return cur


def calib_of(csv, df, Y, folds):
    P = pd.read_csv(csv, index_col=0).loc[df.index, CLASSES].values
    return nested_calibrate(Y, P, folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', nargs='+', default=['panderm_b', 'dinov2_b'],
                    choices=['panderm_b', 'dinov2_b', 'eva02_b'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--warmup', type=int, default=2, help='epochs of linear warm-up')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4, help='peak LR of the head / top layer')
    ap.add_argument('--min_lr', type=float, default=1e-6)
    ap.add_argument('--layer_decay', type=float, default=0.75)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--drop_path', type=float, default=0.1)
    ap.add_argument('--head_drop', type=float, default=0.0)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--loss', default='la', choices=['la', 'wce', 'focal', 'ce'])
    ap.add_argument('--tau', type=float, default=1.0, help='logit adjustment strength (la)')
    ap.add_argument('--beta', type=float, default=0.5, help='class-weight exponent (wce/focal); 1 = inverse frequency')
    ap.add_argument('--gamma', type=float, default=2.0, help='focal gamma')
    ap.add_argument('--smoothing', type=float, default=0.0)
    ap.add_argument('--no_tta', dest='tta', action='store_false', help='disable flip TTA (4 views) at prediction')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--save_ckpt', action='store_true', help='keep fold weights (~350 MB each)')
    ap.add_argument('--overwrite', action='store_true', help='retrain folds that already have run files')
    ap.add_argument('--tag', default='', help='suffix for run names, e.g. _wce, to keep variants apart')
    ap.add_argument('--id_col', default='lesion_id', help='first column name of the submission CSV')
    ap.add_argument('--limit', type=int, default=0, help='first N lesions (smoke test)')
    ap.add_argument('--no_pretrained', action='store_true', help='random init; pipeline test only')
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'runs').mkdir(exist_ok=True)
    sys.stdout = Tee(OUT / 'train_log.txt')
    print(f'\n===== step3_finetune {time.strftime("%Y-%m-%d %H:%M:%S")} args={vars(a)}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__}, device {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev.type == 'cuda' else ''))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    df = load_train()
    test = image_table('test')
    if a.limit:
        df, test = df.iloc[:a.limit], test.iloc[:min(a.limit, len(test))]
        print(f'!! --limit {a.limit}: smoke test, results are not meaningful')
    Y, y, folds = df[CLASSES].values, df['label'].values, df['fold'].values

    results = json.load(open(OUT / 'metrics.json')) if (OUT / 'metrics.json').exists() else {}
    s1 = json.load(open(S1 / 'metrics.json')); s1_name = max(s1, key=lambda k: s1[k]['calib'])
    ref = {'step1_' + s1_name: calib_of(S1 / f'oof_{s1_name}.csv', df, Y, folds)}
    final = {}  # bb -> (OOF used for comparison, test probs)

    for bb in a.backbones:
        name = f'ft_{bb}{a.tag}'
        bref = dict(ref)  # Step 1 best, frozen probe of this backbone (der only), Step 2 best overall
        for probe in (f'{bb}__crop__der', 'panderm_b__crop__both_meta'):
            if (S2 / f'oof_{probe}.csv').exists():
                bref['frozen_' + probe] = calib_of(S2 / f'oof_{probe}.csv', df, Y, folds)
        oofs, tests = [], []
        for seed in a.seeds:
            P = np.full((len(df), K), np.nan); Pt = []
            for f in a.folds:
                rf = OUT / 'runs' / f'{bb}{a.tag}__s{seed}__f{f}.npz'
                if rf.exists() and not a.overwrite and not a.limit:
                    z = np.load(rf, allow_pickle=True)
                    P[folds == f] = pd.DataFrame(z['oof'], index=z['oof_ids']).loc[df.index[folds == f]].values
                    Pt.append(z['test']); print(f'--- {bb} seed={seed} fold={f}: loaded {rf.name}')
                    continue
                P_oof, P_test = run_fold(a, bb, seed, f, df, test, dev)
                P[folds == f] = P_oof; Pt.append(P_test)
                if not a.limit:
                    np.savez(rf, oof=P_oof, oof_ids=np.array(df.index[folds == f]), test=P_test,
                             test_ids=np.array(test.index), args=json.dumps(vars(a)))
            if a.limit or sorted(a.folds) != [0, 1, 2, 3, 4]:
                print(f'{name} seed {seed}: not all 5 folds (or smoke test) -> no evaluation')
                continue
            assert np.isfinite(P).all()
            pd.DataFrame(P, index=df.index, columns=CLASSES).assign(fold=folds).to_csv(OUT / f'oof_{name}__s{seed}.csv')
            oofs.append(P); tests.append(np.mean(Pt, 0))
            run_name = f'{name}__s{seed}'
            r, Pc = evaluate(Y, y, P, folds)
            for rn, Rc in bref.items():
                r[f'delta_vs_{rn}'] = float(r['calib'] - macro_f1(Y, Rc))
                r[f'delta_vs_{rn}_ci95'] = paired_diff_ci(Y, Pc, Rc)
            r['args'] = vars(a); results[run_name] = r
            print(f"{run_name}: raw={r['raw']:.3f} argmax={r['argmax']:.3f} calib={r['calib']:.3f} "
                  f"CI95=({r['calib_ci95'][0]:.3f},{r['calib_ci95'][1]:.3f}) foldSD={r['calib_fold_sd']:.3f}")
            print('  per class (calib):', r['calib_per_class'])
            for rn in bref:
                lo, hi = r[f'delta_vs_{rn}_ci95']
                print(f"  vs {rn:40s} {r[f'delta_vs_{rn}']:+.3f} ({lo:+.3f}..{hi:+.3f})")
            results = save_results(results)
        if not oofs:
            continue
        P = np.mean(oofs, 0); Pt = np.mean(tests, 0)
        if len(oofs) > 1:  # seed ensemble
            r, _ = evaluate(Y, y, P, folds)
            r['seeds'] = a.seeds; results[f'{name}__seedens'] = r
            sd = np.std([results[f'{name}__s{s}']['calib'] for s in a.seeds])
            print(f"{name}__seedens ({len(oofs)} seeds): calib={r['calib']:.3f} "
                  f"CI95=({r['calib_ci95'][0]:.3f},{r['calib_ci95'][1]:.3f}); SD of calib across seeds={sd:.3f}")
        final[bb] = (P, Pt)
        # test submission: thresholds from the full OOF, applied to the fold-averaged test probabilities
        th = fit_thresholds(Y, P)
        pd.DataFrame(Pt, index=pd.Index(test.index, name=a.id_col), columns=CLASSES).to_csv(OUT / f'test_{name}.csv')
        pd.DataFrame(rescale(Pt, th), index=pd.Index(test.index, name=a.id_col), columns=CLASSES) \
            .to_csv(OUT / f'submission_{name}.csv', float_format='%.6f')
        results.setdefault('_thresholds', {})[name] = dict(zip(CLASSES, np.round(th, 4).tolist()))
        results = save_results(results)

    # pairwise paired-bootstrap differences between the fine-tuned backbones. Includes backbones finished in
    # earlier invocations with the same tag and seeds (read from their OOF csv), so eva02_b can be run separately.
    for bb in ['panderm_b', 'dinov2_b', 'eva02_b']:
        if bb in final:
            continue
        fs = [OUT / f'oof_ft_{bb}{a.tag}__s{s}.csv' for s in a.seeds]
        if all(f.exists() for f in fs) and not a.limit:
            final[bb] = (np.mean([pd.read_csv(f, index_col=0).loc[df.index, CLASSES].values for f in fs], 0), None)
    names = list(final)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            b1, b2 = names[i], names[j]
            C1, C2 = nested_calibrate(Y, final[b1][0], folds), nested_calibrate(Y, final[b2][0], folds)
            d = macro_f1(Y, C1) - macro_f1(Y, C2); lo, hi = paired_diff_ci(Y, C1, C2)
            results[f'_diff_ft_{b1}{a.tag}_minus_ft_{b2}{a.tag}'] = {'delta': d, 'ci95': [lo, hi], 'seeds': a.seeds}
            print(f'fine-tuned {b1} - {b2}: {d:+.3f} ({lo:+.3f}..{hi:+.3f})')
    if len(names) > 1:
        results = save_results(results)

    rows = [{'run': k, 'calib': v['calib'], 'ci_lo': v['calib_ci95'][0], 'ci_hi': v['calib_ci95'][1],
             'fold_sd': v['calib_fold_sd'], 'raw': v['raw'], 'argmax': v['argmax'],
             'argmax_bal_recall': v['argmax_balanced_recall'],
             **{kk: vv for kk, vv in v.items() if kk.startswith('delta_vs_') and not kk.endswith('_ci95')}}
            for k, v in results.items() if not k.startswith('_')]
    if rows:
        summ = pd.DataFrame(rows).sort_values('calib', ascending=False)
        summ.to_csv(OUT / 'summary.csv', index=False)
        print('\n' + summ.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
