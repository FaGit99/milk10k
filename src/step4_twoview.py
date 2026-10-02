"""Step 4: add the clinical close-up image to the fine-tuned model.

Everything that decides comparability is taken over from Step 3 unchanged: canonical folds, fixed 25 epochs
(no early stopping / model selection), logit-adjusted CE (tau=1), AdamW + LLRD 0.75, cosine LR 1e-4, warm-up 2,
drop_path 0.1, bf16, same augmentation (applied independently to each view), eval 'crop' at 224, flip TTA,
nested per-class threshold calibration, lesion bootstrap and paired bootstrap CIs.
Reference = Step 3 dermoscopy-only runs (results/step3/runs/{bb}__s{seed}__f{fold}.npz) of the SAME seeds.

Variants (--variants):
  cli     single-view fine-tune on the CLINICAL image only (Step 3 architecture). Used for
          late fusion 'late' = softmax(0.5*log p_der + 0.5*log p_cli)  (weight pre-specified, as in the pre-check)
  shared  one backbone applied to both images, pooled features concatenated [f_der | f_cli] -> linear head.
          (The position in the concatenated vector identifies the modality, so no modality embedding is needed.)
  sep     two backbones (dermoscopic / clinical, both from the same pretrained weights), features concatenated.
--mod_drop p (shared/sep): during training the clinical feature vector is zeroed with probability p per lesion.
          Default 0 (off). Use a --tag (e.g. _md25) to keep such runs apart.

Outputs (results/step4/):
  runs/{variant}_{bb}{tag}__s{seed}__f{fold}.npz   OOF + test probabilities per fold (resume: existing files skipped)
  feats/{variant}_{bb}{tag}__s{seed}__f{fold}.npz  with --save_feats (der/cli only): pooled backbone features of the
          fold's training, held-out and test lesions for the 4 flip views (fp16, shape N x 4 x D) + the trained head
          (W, b). Used by src/step9_crt.py for classifier-only retraining (decoupled cRT, Kang et al. 2020).
  oof_{variant}_{bb}{tag}__s{seed}.csv, test_*.csv, submission_*.csv, metrics.json, summary.csv, train_log.txt

Usage (repo root):
  python src/step4_twoview.py --variants cli shared sep --backbones eva02_b panderm_b --seeds 0      # stage A
  python src/step4_twoview.py --variants shared --backbones eva02_b --seeds 0 1 2                   # stage B
  python src/step4_twoview.py --variants shared sep --backbones eva02_b --limit 64 --epochs 1 --folds 0 --batch 8
  python src/step4_twoview.py --variants der cli --backbones panderm_b --img 448 --tag _448 --seeds 0   # Step 7 (resolution)
  python src/step4_twoview.py --variants der cli --backbones panderm_b --tag _feat --seeds 0 --save_feats  # Step 9 (cRT)
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CLASSES, load_train, macro_f1, nested_calibrate, fit_thresholds, rescale
from step2_extract import to_square, image_table, Tee, strip_state_dict, PANDERM_CKPT
from step2_probe import evaluate, paired_diff_ci
from step3_finetune import build_backbone, ImbalanceLoss, train_transform, seed_worker, IMG
from aug_hiv15 import hiv15_augment
AUG_GRAY = {'hiv15': 0.5, 'hiv15_nogray': 0.0}  # --aug variants using the ported GPU pipeline

K = len(CLASSES)
OUT = ROOT / 'results' / 'step4'
S3 = ROOT / 'results' / 'step3'
EPS = 1e-7


# ---------------------------------------------------------------- data
class Pair(Dataset):
    """Returns (V, 3, H, W) uint8 for the requested views ('der',), ('cli',) or ('der', 'cli'), and the label."""
    def __init__(self, split, lids, der, cli, labels, train, views, size=IMG, aug='milk'):
        # aug 'milk': Step 3 CPU augmentation (RandomResizedCrop, flips, rot90, colour jitter).
        # aug 'hiv15*': CPU only does the eval crop; all augmentation runs on the GPU (hiv15_augment).
        self.dir = ROOT / f'MILK10k_{split.capitalize()}_Input'
        self.items = list(zip(lids, der, cli)); self.labels = np.asarray(labels, np.int64)
        self.tf = train_transform(size) if (train and aug == 'milk') else None; self.views = views; self.size = size

    def __len__(self):
        return len(self.items)

    def _load(self, lid, iid):
        img = Image.open(self.dir / lid / f'{iid}.jpg').convert('RGB')
        img = self.tf(img) if self.tf is not None else to_square(img, 'crop', self.size)
        return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()

    def __getitem__(self, i):
        lid, d, c = self.items[i]; ids = {'der': d, 'cli': c}
        return torch.stack([self._load(lid, ids[v]) for v in self.views]), self.labels[i]


# ---------------------------------------------------------------- model
def build_backbone_sized(bb, drop_path, pretrained, img):
    """img == 224: exactly Step 3's build_backbone. Otherwise same weights at a larger input size:
      eva02_b   timm img_size=img (timm resamples the abs. pos. embedding; RoPE is rebuilt for the new grid)
      panderm_b vendored ViT with img_size=img; the checkpoint's 14x14 pos_embed (fixed sin-cos, not trained)
                is bicubically interpolated to the new grid (equivalent to fractional sin-cos positions)
      dinov2_b  timm img_size=img"""
    if img == IMG:
        return build_backbone(bb, drop_path, pretrained)
    if bb == 'panderm_b':
        tp = str(ROOT / 'third_party' / 'panderm')
        if tp not in sys.path:
            sys.path.insert(0, tp)
        from modeling_finetune import panderm_base_patch16_224
        m = panderm_base_patch16_224(drop_path_rate=drop_path, img_size=img)
        if pretrained:
            sd = strip_state_dict(torch.load(PANDERM_CKPT, map_location='cpu', weights_only=False))
            pe = sd.pop('pos_embed', None)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            missing = [k for k in missing if not k.startswith('head.') and k != 'pos_embed']
            print(f'  PanDerm load @{img}: missing={missing[:10]} ({len(missing)}), unexpected={unexpected[:10]} ({len(unexpected)})')
            if any(k.startswith(('blocks.', 'patch_embed.', 'norm.', 'cls_token')) for k in missing):
                raise RuntimeError('PanDerm checkpoint does not cover the encoder')
            if pe is not None:
                g0 = int(round((pe.shape[1] - 1) ** 0.5)); g1 = m.patch_embed.patch_shape
                grid = pe[:, 1:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2).float()
                grid = F.interpolate(grid, size=g1, mode='bicubic', align_corners=False)
                new = torch.cat([pe[:, :1].float(), grid.permute(0, 2, 3, 1).reshape(1, g1[0] * g1[1], -1)], 1)
                assert new.shape == m.pos_embed.shape, (new.shape, m.pos_embed.shape)
                m.pos_embed.data.copy_(new)
                print(f'  PanDerm pos_embed interpolated {g0}x{g0} -> {g1[0]}x{g1[1]}')
        m.head = nn.Identity()
        return m, m.embed_dim, (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)
    import timm
    name = {'eva02_b': 'eva02_base_patch14_224.mim_in22k', 'dinov2_b': 'vit_base_patch14_dinov2.lvd142m'}[bb]
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img, drop_path_rate=drop_path)
    cfg = timm.data.resolve_model_data_config(m)
    return m, m.num_features, tuple(cfg['mean']), tuple(cfg['std'])


class TwoView(nn.Module):
    def __init__(self, variant, bb, drop_path, head_drop, mod_drop, pretrained, img=IMG):
        super().__init__()
        self.variant, self.mod_drop = variant, mod_drop
        self.backbone, dim, self.mean, self.std = build_backbone_sized(bb, drop_path, pretrained, img)
        self.backbone_c = build_backbone_sized(bb, drop_path, pretrained, img)[0] if variant == 'sep' else None
        self.drop = nn.Dropout(head_drop)
        self.head = nn.Linear(dim * (1 if variant in ('cli', 'der') else 2), K)
        nn.init.trunc_normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)

    def forward(self, x):  # x: (B, V, 3, H, W), normalised
        if self.variant in ('cli', 'der'):
            return self.head(self.drop(self.backbone(x[:, 0])))
        if self.variant == 'shared':
            B = x.shape[0]
            z = self.backbone(x.flatten(0, 1)).view(B, 2, -1)
            fd, fc = z[:, 0], z[:, 1]
        else:
            fd, fc = self.backbone(x[:, 0]), self.backbone_c(x[:, 1])
        if self.training and self.mod_drop > 0:
            keep = (torch.rand(fc.shape[0], 1, device=fc.device) >= self.mod_drop).to(fc.dtype)
            fc = fc * keep
        return self.head(self.drop(torch.cat([fd, fc], 1)))


def param_groups(net, wd, layer_decay):
    """As Step 3 (BEiT-style LLRD), extended to a second backbone 'backbone_c' with the same layer schedule."""
    L = len(net.backbone.blocks)
    groups = {}
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        pre = next((q for q in ('backbone.', 'backbone_c.') if n.startswith(q)), None)
        if pre:
            s = n[len(pre):]
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


def predict(net, dl, mean, std, dev, tta):
    net.eval(); out = []
    with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        for x, _ in dl:
            x = (x.to(dev, non_blocking=True).float().div_(255) - mean) / std
            views = [x, x.flip(4), x.flip(3), x.flip(3).flip(4)] if tta else [x]  # same flip on both images
            out.append(torch.stack([F.softmax(net(v).float(), 1) for v in views]).mean(0).cpu())
    return torch.cat(out).double().numpy()


def extract_feats(net, dl, mean, std, dev):
    """Pooled backbone features (before the head) for the 4 flip views used by the TTA in predict(): (N, 4, D) fp16."""
    net.eval(); out = []
    with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        for x, _ in dl:
            x = (x.to(dev, non_blocking=True).float().div_(255) - mean) / std
            views = [x, x.flip(4), x.flip(3), x.flip(3).flip(4)]
            out.append(torch.stack([net.backbone(v[:, 0]).float() for v in views], 1).cpu())
    return torch.cat(out).half().numpy()


def run_fold(a, variant, bb, seed, f, df, test, dev):
    s = 1000 * seed + f  # same per-fold seed scheme as Step 3
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    tr, te = df['fold'].values != f, df['fold'].values == f
    y = df['label'].values
    net = TwoView(variant, bb, a.drop_path, a.head_drop, a.mod_drop, not a.no_pretrained, a.img).to(dev)
    mean = torch.tensor(net.mean, device=dev).view(1, 1, 3, 1, 1); std = torch.tensor(net.std, device=dev).view(1, 1, 3, 1, 1)
    views = {'cli': ('cli',), 'der': ('der',)}.get(variant, ('der', 'cli'))

    g = torch.Generator(); g.manual_seed(s)
    kw = dict(num_workers=a.workers, pin_memory=dev.type == 'cuda', persistent_workers=a.workers > 0,
              worker_init_fn=seed_worker)
    D, C = df['der_isic_id'].values, df['cli_isic_id'].values
    dl_tr = DataLoader(Pair('training', df.index[tr], D[tr], C[tr], y[tr], True, views, a.img, a.aug),
                       batch_size=a.batch, shuffle=True, drop_last=True, generator=g, **kw)
    dl_te = DataLoader(Pair('training', df.index[te], D[te], C[te], y[te], False, views, a.img), batch_size=a.batch * 2, **kw)
    dl_test = DataLoader(Pair('test', test.index, test['der'].values, test['cli'].values, np.zeros(len(test)), False, views, a.img),
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

    nparam = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'--- {variant}/{bb} seed={seed} fold={f}: train={tr.sum()} heldout={te.sum()} test={len(test)} '
          f'steps={steps} params={nparam:.1f}M', flush=True)
    it, t0 = 0, time.time()
    for ep in range(a.epochs):
        net.train(); tot, nb = 0.0, 0
        for x, yb in dl_tr:
            lr = lr_at(it)
            for gr in opt.param_groups:
                gr['lr'] = lr * gr['lr_scale']
            x = x.to(dev, non_blocking=True).float().div_(255)
            if a.aug in AUG_GRAY:  # per image, independently for each view
                x = hiv15_augment(x.flatten(0, 1), AUG_GRAY[a.aug]).view_as(x)
            x = (x - mean) / std
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True); loss = 0.0
            for xc, yc in zip(x.chunk(a.accum), yb.chunk(a.accum)):  # gradient accumulation, same effective batch
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                    lc = crit(net(xc), yc) / a.accum
                lc.backward(); loss = loss + lc.detach()
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
    print(f'  held-out fold {f}: raw={macro_f1(Yte, P_oof):.3f} argmax={macro_f1(Yte, np.eye(K)[P_oof.argmax(1)]):.3f} '
          f'[{time.time() - t0:.0f}s total]  (information only)', flush=True)
    if a.save_feats and not a.limit:
        # training-fold features without random augmentation (eval crop + flips), same loader order as df.index[tr]
        dl_trf = DataLoader(Pair('training', df.index[tr], D[tr], C[tr], y[tr], False, views, a.img),
                            batch_size=a.batch * 2, **kw)
        fd = OUT / 'feats'; fd.mkdir(exist_ok=True)
        np.savez(fd / f'{variant}_{bb}{a.tag}__s{seed}__f{f}.npz',
                 tr=extract_feats(net, dl_trf, mean, std, dev), tr_ids=np.array(df.index[tr]),
                 te=extract_feats(net, dl_te, mean, std, dev), te_ids=np.array(df.index[te]),
                 test=extract_feats(net, dl_test, mean, std, dev), test_ids=np.array(test.index),
                 W=net.head.weight.detach().float().cpu().numpy(), b=net.head.bias.detach().float().cpu().numpy(),
                 counts=counts, args=json.dumps(vars(a)))
        print(f'  features saved: feats/{variant}_{bb}{a.tag}__s{seed}__f{f}.npz [{time.time() - t0:.0f}s total]', flush=True)
    del net, opt
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return P_oof, P_test


# ---------------------------------------------------------------- reference / fusion helpers
def geo(Pa, Pb, w=0.5):
    z = w * np.log(Pa + EPS) + (1 - w) * np.log(Pb + EPS)
    z -= z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def step3_ref(bb, seed, df, test, tag=''):
    """Dermoscopy-only OOF + fold-averaged test probabilities of one seed, or None if incomplete.
    With a tag (e.g. _448) the Step 4 'der' runs of that tag are used; without, the Step 3 (224) runs."""
    fs = ([OUT / 'runs' / f'der_{bb}{tag}__s{seed}__f{f}.npz' for f in range(5)] if tag else
          [S3 / 'runs' / f'{bb}__s{seed}__f{f}.npz' for f in range(5)])
    if not all(p.exists() for p in fs):
        return None
    P = np.full((len(df), K), np.nan); Pt = []
    for p in fs:
        z = np.load(p, allow_pickle=True)
        o = pd.DataFrame(z['oof'], index=z['oof_ids']); ids = [i for i in o.index if i in df.index]
        P[df.index.get_indexer(ids)] = o.loc[ids].values
        Pt.append(pd.DataFrame(z['test'], index=z['test_ids']).loc[test.index].values)
    return (P, np.mean(Pt, 0)) if np.isfinite(P).all() else None


def compare(results, key, Y, y, folds, P, R, label):
    """Evaluate P, store under key, add paired delta vs reference R (uncalibrated probabilities)."""
    r, Pc = evaluate(Y, y, P, folds)
    if R is not None:
        Rc = nested_calibrate(Y, R, folds)
        r['ref'] = label; r['delta_vs_ref'] = float(r['calib'] - macro_f1(Y, Rc))
        r['delta_vs_ref_ci95'] = paired_diff_ci(Y, Pc, Rc)
    results[key] = r
    d = (f"  vs {label}: {r['delta_vs_ref']:+.3f} ({r['delta_vs_ref_ci95'][0]:+.3f}..{r['delta_vs_ref_ci95'][1]:+.3f})"
         if R is not None else '')
    print(f"{key}: calib={r['calib']:.3f} CI95=({r['calib_ci95'][0]:.3f},{r['calib_ci95'][1]:.3f}) "
          f"foldSD={r['calib_fold_sd']:.3f}{d}\n  per class (calib): {r['calib_per_class']}", flush=True)


def save_results(results):
    p = OUT / 'metrics.json'
    cur = json.load(open(p)) if p.exists() else {}
    for k, v in results.items():
        if k == '_thresholds':
            cur.setdefault(k, {}).update(v)
        else:
            cur[k] = v
    json.dump(cur, open(p, 'w'), indent=1)
    return cur


def write_submission(results, name, Pt, P_oof, Y, test, id_col):
    th = fit_thresholds(Y, P_oof)
    idx = pd.Index(test.index, name=id_col)
    pd.DataFrame(Pt, index=idx, columns=CLASSES).to_csv(OUT / f'test_{name}.csv')
    pd.DataFrame(rescale(Pt, th), index=idx, columns=CLASSES).to_csv(OUT / f'submission_{name}.csv', float_format='%.6f')
    results.setdefault('_thresholds', {})[name] = dict(zip(CLASSES, np.round(th, 4).tolist()))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['cli', 'shared', 'sep'], choices=['der', 'cli', 'shared', 'sep'])
    ap.add_argument('--img', type=int, default=224, help='input size (224 = Steps 3/4); e.g. 448 for Step 7')
    ap.add_argument('--aug', default='milk', choices=['milk', 'hiv15', 'hiv15_nogray'],
                    help="training augmentation: 'milk' (Steps 3/4) or the ported HiV15 GPU pipeline")
    ap.add_argument('--accum', type=int, default=1, help='micro-batches per step (memory only; effective batch unchanged)')
    ap.add_argument('--backbones', nargs='+', default=['eva02_b', 'panderm_b'], choices=['panderm_b', 'dinov2_b', 'eva02_b'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64, help='lesions per batch (shared/sep: 2 images per lesion)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--min_lr', type=float, default=1e-6)
    ap.add_argument('--layer_decay', type=float, default=0.75)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--drop_path', type=float, default=0.1)
    ap.add_argument('--head_drop', type=float, default=0.0)
    ap.add_argument('--mod_drop', type=float, default=0.0, help='shared/sep: P(zero clinical features) in training')
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--loss', default='la', choices=['la', 'wce', 'focal', 'ce'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--gamma', type=float, default=2.0)
    ap.add_argument('--smoothing', type=float, default=0.0)
    ap.add_argument('--no_tta', dest='tta', action='store_false')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--tag', default='', help='suffix for run names, e.g. _md25')
    ap.add_argument('--id_col', default='lesion_id')
    ap.add_argument('--limit', type=int, default=0, help='first N lesions (smoke test, nothing is saved)')
    ap.add_argument('--save_feats', action='store_true',
                    help='der/cli only: also save pooled features + head per fold (results/step4/feats/) for step9_crt.py')
    ap.add_argument('--no_pretrained', action='store_true', help='random init; pipeline test only')
    a = ap.parse_args()
    if a.mod_drop > 0 and not a.tag:
        ap.error('--mod_drop > 0 needs a --tag so the runs do not mix with the mod_drop=0 runs')
    if a.aug != 'milk' and not a.tag:
        ap.error('--aug other than milk needs a --tag (e.g. _augH) so the runs do not mix')
    if a.img != 224 and not a.tag:
        ap.error('--img other than 224 needs a --tag (e.g. _448) so the runs do not mix with the 224 runs')
    if a.save_feats and (not a.tag or set(a.variants) - {'der', 'cli'}):
        ap.error('--save_feats needs a --tag (e.g. _feat) and only supports --variants der cli')
    if a.batch % a.accum:
        ap.error('--batch must be divisible by --accum')

    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'runs').mkdir(exist_ok=True)
    sys.stdout = Tee(OUT / 'train_log.txt')
    print(f'\n===== step4_twoview {time.strftime("%Y-%m-%d %H:%M:%S")} args={vars(a)}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__}, device {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev.type == 'cuda' else ''))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    df = load_train()
    test = image_table('test')
    if a.limit:
        df, test = df.iloc[:a.limit], test.iloc[:min(a.limit, len(test))]
        print(f'!! --limit {a.limit}: smoke test, nothing is saved, results are not meaningful')
    Y, y, folds = df[CLASSES].values, df['label'].values, df['fold'].values
    complete = not a.limit and sorted(a.folds) == [0, 1, 2, 3, 4]
    results = json.load(open(OUT / 'metrics.json')) if (OUT / 'metrics.json').exists() else {}

    for variant in a.variants:
        for bb in a.backbones:
            name = f'{variant}_{bb}{a.tag}'
            oofs, tests, refs = [], [], []
            for seed in a.seeds:
                P = np.full((len(df), K), np.nan); Pt = []
                for f in a.folds:
                    rf = OUT / 'runs' / f'{name}__s{seed}__f{f}.npz'
                    if rf.exists() and not a.overwrite and not a.limit:
                        z = np.load(rf, allow_pickle=True)
                        P[folds == f] = pd.DataFrame(z['oof'], index=z['oof_ids']).loc[df.index[folds == f]].values
                        Pt.append(z['test']); print(f'--- {name} seed={seed} fold={f}: loaded {rf.name}')
                        continue
                    P_oof, P_test = run_fold(a, variant, bb, seed, f, df, test, dev)
                    P[folds == f] = P_oof; Pt.append(P_test)
                    if not a.limit:
                        np.savez(rf, oof=P_oof, oof_ids=np.array(df.index[folds == f]), test=P_test,
                                 test_ids=np.array(test.index), args=json.dumps(vars(a)))
                if not complete:
                    print(f'{name} seed {seed}: not all 5 folds (or smoke test) -> no evaluation')
                    continue
                assert np.isfinite(P).all()
                pd.DataFrame(P, index=df.index, columns=CLASSES).assign(fold=folds).to_csv(OUT / f'oof_{name}__s{seed}.csv')
                oofs.append(P); tests.append(np.mean(Pt, 0))
                # 'der' variant: compared with Step 3 der (224); other variants: with der of the same tag (if run)
                rtag = '' if variant == 'der' or not (OUT / 'runs' / f'der_{bb}{a.tag}__s{seed}__f0.npz').exists() else a.tag
                ref = step3_ref(bb, seed, df, test, rtag); refs.append(ref)
                R = ref[0] if ref else None; rl = f"der {bb}{rtag or ' (step3)'} s{seed}"
                compare(results, f'{name}__s{seed}', Y, y, folds, P, R, rl)
                if variant == 'cli' and R is not None:
                    compare(results, f'late_{bb}{a.tag}__s{seed}', Y, y, folds, geo(R, P), R, rl)
                results = save_results(results)
            if not oofs:
                continue
            P, Pt = np.mean(oofs, 0), np.mean(tests, 0)
            have_ref = all(r is not None for r in refs)
            R = np.mean([r[0] for r in refs], 0) if have_ref else None
            Rt = np.mean([r[1] for r in refs], 0) if have_ref else None
            tagS = f'seeds {a.seeds}'
            if len(oofs) > 1:
                compare(results, f'{name}__seedens', Y, y, folds, P, R, f"der {bb}{rtag or ' (step3)'} ens {tagS}")
                results[f'{name}__seedens']['seeds'] = a.seeds
                sd = np.std([results[f'{name}__s{s}']['calib'] for s in a.seeds])
                print(f'  SD of calib across seeds = {sd:.3f}')
            write_submission(results, name, Pt, P, Y, test, a.id_col)
            if variant == 'cli' and have_ref:
                Pl, Plt = geo(R, P), geo(Rt, Pt)
                if len(oofs) > 1:
                    compare(results, f'late_{bb}{a.tag}__seedens', Y, y, folds, Pl, R, f"der {bb}{rtag or ' (step3)'} ens {tagS}")
                write_submission(results, f'late_{bb}{a.tag}', Plt, Pl, Y, test, a.id_col)
            results = save_results(results)

    rows = [{'run': k, 'calib': v['calib'], 'ci_lo': v['calib_ci95'][0], 'ci_hi': v['calib_ci95'][1],
             'fold_sd': v['calib_fold_sd'], 'raw': v['raw'], 'argmax': v['argmax'],
             'ref': v.get('ref'), 'delta_vs_ref': v.get('delta_vs_ref'),
             'delta_lo': (v.get('delta_vs_ref_ci95') or [None, None])[0],
             'delta_hi': (v.get('delta_vs_ref_ci95') or [None, None])[1]}
            for k, v in results.items() if not k.startswith('_') and 'calib_ci95' in v]
    if rows:
        summ = pd.DataFrame(rows).sort_values('calib', ascending=False)
        summ.to_csv(OUT / 'summary.csv', index=False)
        print('\n' + summ.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
