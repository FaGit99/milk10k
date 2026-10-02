"""Step 2a: extract frozen embeddings (no training) per backbone x preprocessing x modality.

One pass over the images; every batch is fed to all loaded backbones (96 GB VRAM holds them all).
Output (data/emb/):
  {split}_lesions.csv                          lesion_id order shared by all arrays of that split
  {split}__{backbone}__{prep}__{mod}.npy       float32 (n_lesions, dim); mod in {der, cli}
  manifest.json                                model names, dims, normalisation, pooling, timings

prep:
  crop - resize short side to 224 (bicubic), center-crop 224x224 (drops the long-side edges)
  pad  - resize long side to 224 (bicubic), pad to 224x224 with RGB (124,116,104) ~ ImageNet/CLIP mean
Normalisation: each model's own mean/std (timm data config; PanDerm: values from its repo).

Usage (from repo root):
  python src/step2_extract.py                      # all backbones whose weights are available
  python src/step2_extract.py --backbones eva02_s dinov2_b --limit 64   # quick smoke test
  python src/step2_extract.py --own_ckpt D:/x/best.pth --own_arch eva02_small_patch14_224
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'data' / 'emb'
PAD_RGB = (124, 116, 104)
NO_PRETRAINED = False  # set by --no_pretrained (smoke tests without network only)
PANDERM_CKPT = ROOT / 'weights' / 'panderm_bb_data6_checkpoint-499.pth'


class Tee:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8'); self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()


# ---------------------------------------------------------------- data
def image_table(split):
    """One row per lesion with the isic_id of its clinical and dermoscopic image."""
    meta = ROOT / f'MILK10k_{split.capitalize()}_Metadata.csv'
    m = pd.read_csv(meta)
    m['mod'] = m['image_type'].map({'clinical: close-up': 'cli', 'dermoscopic': 'der'})
    assert m['mod'].notna().all(), 'unexpected image_type'
    t = m.pivot(index='lesion_id', columns='mod', values='isic_id')
    assert t.notna().all().all(), 'lesion without both images'
    return t.sort_index()


def to_square(img, prep, size=224):
    w, h = img.size
    if prep == 'crop':
        s = size / min(w, h)
        nw, nh = max(size, round(w * s)), max(size, round(h * s))
        img = img.resize((nw, nh), Image.BICUBIC)
        l, t = (nw - size) // 2, (nh - size) // 2
        return img.crop((l, t, l + size, t + size))
    if prep == 'pad':
        s = size / max(w, h)
        nw, nh = min(size, round(w * s)), min(size, round(h * s))
        img = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new('RGB', (size, size), PAD_RGB)
        canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
        return canvas
    raise ValueError(prep)


class Images(Dataset):
    def __init__(self, split, table, preps):
        self.items = [(lid, mod, table.loc[lid, mod]) for lid in table.index for mod in ('der', 'cli')]
        self.dir = ROOT / f'MILK10k_{split.capitalize()}_Input'
        self.preps = preps
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        lid, mod, iid = self.items[i]
        img = Image.open(self.dir / lid / f'{iid}.jpg').convert('RGB')
        out = {p: torch.from_numpy(np.array(to_square(img, p))).permute(2, 0, 1).contiguous() for p in self.preps}
        return i, out


# ---------------------------------------------------------------- models
def strip_state_dict(sd):
    for k in ('model', 'state_dict', 'model_ema', 'module'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]; break
    out = {}
    for k, v in sd.items():
        for pre in ('module.', 'encoder.', 'backbone.', 'model.'):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def load_panderm(path):
    tp = str(ROOT / 'third_party' / 'panderm')
    if tp not in sys.path:
        sys.path.insert(0, tp)
    from modeling_finetune import panderm_base_patch16_224  # vendored, unmodified (CC-BY-NC-ND 4.0)
    model = panderm_base_patch16_224()  # lin_probe=True, linear_type='standard' -> CLS token after final norm
    sd = strip_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith('head.')]
    print(f'  PanDerm load: {len(sd)} tensors, missing={missing[:10]} ({len(missing)}), '
          f'unexpected={unexpected[:10]} ({len(unexpected)})')
    if any(k.startswith(('blocks.', 'patch_embed.', 'norm.', 'cls_token')) for k in missing):
        raise RuntimeError('PanDerm checkpoint does not cover the encoder - check key names above')
    model.head = torch.nn.Identity()
    # normalisation used in PanDerm's own eval code (get_norm_constants('imagenet'), std[0]=0.228 as in repo)
    return model, (0.485, 0.456, 0.406), (0.228, 0.224, 0.225), 'CLS token after final LayerNorm'


def load_timm(name, **kw):
    import timm
    model = timm.create_model(name, pretrained=not NO_PRETRAINED, num_classes=0, **kw)
    cfg = timm.data.resolve_model_data_config(model)
    pool = getattr(model, 'global_pool', '?')
    return model, tuple(cfg['mean']), tuple(cfg['std']), f'timm pre-logits, global_pool={pool!r}'


def load_own(path, arch):
    import timm
    model = timm.create_model(arch, pretrained=False, num_classes=0)
    sd = strip_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    sd = {k: v for k, v in sd.items() if not k.startswith(('head.', 'fc.', 'classifier.'))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  own ckpt load: missing={missing[:10]} ({len(missing)}), unexpected={unexpected[:10]} ({len(unexpected)})')
    if len(missing) > 0:
        raise RuntimeError('own checkpoint does not match --own_arch (see missing keys)')
    cfg = timm.data.resolve_model_data_config(model)
    return model, tuple(cfg['mean']), tuple(cfg['std']), f'timm pre-logits, global_pool={getattr(model, "global_pool", "?")!r}'


BACKBONES = {
    'eva02_s': lambda a: load_timm('eva02_small_patch14_224.mim_in22k'),
    'eva02_b': lambda a: load_timm('eva02_base_patch14_224.mim_in22k'),
    'dinov2_b': lambda a: load_timm('vit_base_patch14_dinov2.lvd142m', img_size=224),
    'panderm_b': lambda a: load_panderm(a.panderm_ckpt),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', nargs='+', default=list(BACKBONES))
    ap.add_argument('--preps', nargs='+', default=['crop', 'pad'])
    ap.add_argument('--splits', nargs='+', default=['training', 'test'])
    ap.add_argument('--panderm_ckpt', default=str(PANDERM_CKPT))
    ap.add_argument('--own_ckpt', default=None)
    ap.add_argument('--own_arch', default='eva02_small_patch14_224')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0, help='first N lesions per split (smoke test)')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--no_pretrained', action='store_true', help='random weights; pipeline test only')
    a = ap.parse_args()
    global NO_PRETRAINED; NO_PRETRAINED = a.no_pretrained
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / 'extract_log.txt')
    print(f'\n===== step2_extract {time.strftime("%Y-%m-%d %H:%M:%S")} args={vars(a)}')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__}, cuda build {torch.version.cuda}, device {dev}'
          + (f' ({torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability(0)})' if dev.type == 'cuda' else ''))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    names = [b for b in a.backbones if b in BACKBONES]
    if 'panderm_b' in names and not Path(a.panderm_ckpt).is_file():
        print(f'!! PanDerm weights not found at {a.panderm_ckpt} -> skipping panderm_b'); names.remove('panderm_b')
    models, manifest = {}, {}
    for b in names:
        t0 = time.time(); print(f'loading {b} ...', flush=True)
        m, mean, std, pool = BACKBONES[b](a)
        models[b] = (m.eval().to(dev), torch.tensor(mean, device=dev).view(1, 3, 1, 1),
                     torch.tensor(std, device=dev).view(1, 3, 1, 1))
        manifest[b] = {'mean': mean, 'std': std, 'pooling': pool,
                       'params_M': round(sum(p.numel() for p in m.parameters()) / 1e6, 1)}
        print(f'  {b}: {manifest[b]} [{time.time()-t0:.0f}s]')
    if a.own_ckpt:
        m, mean, std, pool = load_own(a.own_ckpt, a.own_arch)
        models['own'] = (m.eval().to(dev), torch.tensor(mean, device=dev).view(1, 3, 1, 1),
                         torch.tensor(std, device=dev).view(1, 3, 1, 1))
        manifest['own'] = {'mean': mean, 'std': std, 'pooling': pool, 'ckpt': a.own_ckpt, 'arch': a.own_arch,
                           'note': 'CV biased upward (HAM10000/ISIC overlap with MILK10k not removable) - report separately'}
    if not models:
        sys.exit('no backbone loaded')

    for split in a.splits:
        table = image_table(split)
        if a.limit:
            table = table.iloc[:a.limit]
        pd.Series(table.index, name='lesion_id').to_csv(out / f'{split}_lesions.csv', index=False)
        ds = Images(split, table, a.preps)
        dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=dev.type == 'cuda',
                        persistent_workers=False)
        feats = {}  # (b, prep) -> array over ds.items
        t0 = time.time()
        with torch.inference_mode():
            for bi, (idx, imgs) in enumerate(dl):
                for p in a.preps:
                    x = imgs[p].to(dev, non_blocking=True).float().div_(255)
                    for b, (m, mu, sd) in models.items():
                        f = m((x - mu) / sd).float().cpu().numpy()
                        arr = feats.setdefault((b, p), np.full((len(ds), f.shape[1]), np.nan, np.float32))
                        arr[idx.numpy()] = f
                if bi % 10 == 0:
                    print(f'  {split}: {min((bi+1)*a.batch, len(ds))}/{len(ds)} images [{time.time()-t0:.0f}s]', flush=True)
        el = time.time() - t0
        print(f'  {split}: {len(ds)} images x {len(a.preps)} preps x {len(models)} models in {el:.0f}s')
        mods = np.array([mod for _, mod, _ in ds.items])
        for (b, p), arr in feats.items():
            assert np.isfinite(arr).all(), f'non-finite features {b} {p}'
            for mod in ('der', 'cli'):
                e = arr[mods == mod]
                np.save(out / f'{split}__{b}__{p}__{mod}.npy', e)
                print(f'    saved {split}__{b}__{p}__{mod}.npy {e.shape} mean|x|={np.abs(e).mean():.3f}')
        manifest.setdefault('_timing_s', {})[split] = round(el, 1)
    manifest['_preps'] = a.preps
    json.dump(manifest, open(out / 'manifest.json', 'w'), indent=1)
    print('done.')


if __name__ == '__main__':
    main()
