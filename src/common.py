"""Shared data loading, folds, challenge metric and threshold calibration for MILK10k."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
MONET = ['MONET_ulceration_crust', 'MONET_hair', 'MONET_vasculature_vessels', 'MONET_erythema',
         'MONET_pigmented', 'MONET_gel_water_drop_fluid_dermoscopy_liquid',
         'MONET_skin_markings_pen_ink_purple_pen']
SITES = ['trunk', 'head_neck_face', 'lower_extremity', 'upper_extremity', 'hand', 'foot', 'genital']


def lesion_table(meta_csv):
    """One row per lesion: demographics + MONET/manipulation for clinical (cli_) and dermoscopic (der_) image."""
    m = pd.read_csv(meta_csv)
    m['itype'] = m['image_type'].map({'clinical: close-up': 'cli', 'dermoscopic': 'der'})
    assert m['itype'].notna().all()
    demo = m.groupby('lesion_id')[['age_approx', 'sex', 'skin_tone_class', 'site']].first()
    per = m.pivot(index='lesion_id', columns='itype', values=MONET + ['image_manipulation', 'isic_id'])
    per.columns = [f'{t}_{c}' for c, t in per.columns]
    df = demo.join(per)
    df['sex_male'] = (df['sex'] == 'male').astype(float)
    for s in SITES:
        df[f'site_{s}'] = (df['site'] == s).astype(float)
    df['site_missing'] = df['site'].isna().astype(float)
    for t in ('cli', 'der'):
        df[f'{t}_altered'] = (df[f'{t}_image_manipulation'] == 'altered').astype(float)
        for c in MONET:
            df[f'{t}_{c}'] = df[f'{t}_{c}'].astype(float)
    return df


FEATURES = {
    'demo': ['age_approx', 'sex_male', 'skin_tone_class'] + [f'site_{s}' for s in SITES] + ['site_missing'],
}
FEATURES['demo_monet'] = FEATURES['demo'] + [f'{t}_{c}' for t in ('cli', 'der') for c in MONET]
FEATURES['demo_monet_manip'] = FEATURES['demo_monet'] + ['cli_altered', 'der_altered']


def load_train():
    df = lesion_table(ROOT / 'MILK10k_Training_Metadata.csv')
    gt = pd.read_csv(ROOT / 'MILK10k_Training_GroundTruth.csv').set_index('lesion_id')[CLASSES]
    df = df.join(gt, how='inner')
    df['label'] = gt.loc[df.index].values.argmax(1)
    folds = pd.read_csv(ROOT / 'data' / 'folds.csv').set_index('lesion_id')['fold']
    df['fold'] = folds.loc[df.index].values
    return df


# ---------------- challenge metric ----------------
def per_class_f1(Y, P, thr=0.5):
    """Y: (n,K) one-hot, P: (n,K) probabilities. Each class binarised independently at >= thr (challenge rule)."""
    Yb = Y.astype(bool)
    Pb = P >= thr
    tp = (Yb & Pb).sum(0); fp = (~Yb & Pb).sum(0); fn = (Yb & ~Pb).sum(0)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def macro_f1(Y, P, thr=0.5):
    return float(per_class_f1(Y, P, thr).mean())


# ---------------- threshold calibration ----------------
def fit_thresholds(Y, P):
    """Per-class threshold maximising that class's binary F1 (classes are scored independently)."""
    th = np.full(P.shape[1], 0.5)
    for k in range(P.shape[1]):
        cand = np.unique(np.concatenate([np.quantile(P[:, k], np.linspace(0, 1, 401)), [0.5]]))
        best, bt = -1, 0.5
        y = Y[:, k].astype(bool)
        for t in cand:
            p = P[:, k] >= t
            tp = (y & p).sum(); den = 2 * tp + (~y & p).sum() + (y & ~p).sum()
            f = 2 * tp / den if den else 0
            if f > best:
                best, bt = f, t
        th[k] = min(max(bt, 1e-6), 1 - 1e-6)
    return th


def rescale(P, th):
    """Monotone piecewise-linear map so that threshold th -> 0.5 (submission is binarised at 0.5)."""
    P = np.clip(P, 0, 1)
    return np.where(P < th, 0.5 * P / th, 0.5 + 0.5 * (P - th) / (1 - th))


def nested_calibrate(Y, P, folds):
    """For each fold, thresholds are fit on the OOF predictions of the other folds only (no self-tuning)."""
    out = np.zeros_like(P)
    for f in np.unique(folds):
        te = folds == f
        th = fit_thresholds(Y[~te], P[~te])
        out[te] = rescale(P[te], th)
    return out


def bootstrap_ci(Y, P, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Y))
    s = [macro_f1(Y[b], P[b]) for b in (rng.choice(idx, len(idx)) for _ in range(n))]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
