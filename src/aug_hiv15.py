"""GPU augmentation ported from '260715 Training with AI lesions/augmentations_gpu_balanced.py'
(FusedAugmentationPipeline._augment, HiV15), documented in AUGMENTATION_PIPELINE.md.

Ops and parameters are copied unchanged (order, probabilities, ranges, kornia-matched semantics):
  grayscale p=.5 | contrast 0.5-1.0 p=.2 | ONE fused warp: rotation full circle p=.7, hflip .5, vflip .5,
  affine p=.5 (scale 0.9-1.1, shear +-10 deg, translate +-0.1), zero padding | brightness 0.8-1.2 additive p=.5 |
  contrast 0.8-1.2 multiply p=.5 | saturation 0.8-1.2 p=.5 | hue +-0.1 (YIQ) p=.3 | gamma 0.8-1.2 p=.3 |
  gaussian noise std .01 p=.2 | p_original = 0.02 (whole sample left unaugmented).
Only change: grayscale probability is a parameter (0.5 = original; 0.0 for the 'hiv15_nogray' variant).
Input/output: float (B, 3, H, W) in [0, 1], BEFORE normalisation (normalisation stays in the training loop).
"""
import math
import torch
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)


def _u(b, dev, lo, hi):
    return torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, hi)


def _gate(b, dev, p):
    return (torch.rand(b, 1, 1, 1, device=dev) < p).float()


def _luma(x):
    return x[:, 0:1] * LUMA[0] + x[:, 1:2] * LUMA[1] + x[:, 2:3] * LUMA[2]


def _augment(x, gray_p):
    b, dev = x.size(0), x.device
    ones = torch.ones(b, 1, 1, 1, device=dev)
    g = _gate(b, dev, gray_p)
    x = x * (1 - g) + _luma(x).expand_as(x) * g
    c = torch.lerp(ones, _u(b, dev, 0.5, 1.0), _gate(b, dev, 0.2))
    x = (x * c).clamp_(0, 1)
    ang = _u(b, dev, -math.pi, math.pi).view(b) * _gate(b, dev, 0.7).view(b)
    ag = _gate(b, dev, 0.5).view(b)
    sc = 1.0 + _u(b, dev, -0.1, 0.1).view(b) * ag
    sh = torch.deg2rad(_u(b, dev, -10.0, 10.0).view(b) * ag)
    tx = _u(b, dev, -0.1, 0.1).view(b) * ag
    ty = _u(b, dev, -0.1, 0.1).view(b) * ag
    fx = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    cos, sin, tsh = torch.cos(ang), torch.sin(ang), torch.tan(sh)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = (cos + sin * tsh) * sc * fx
    theta[:, 0, 1] = -sin * sc
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = (sin - cos * tsh) * sc
    theta[:, 1, 1] = cos * sc * fy
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    br = torch.lerp(ones, _u(b, dev, 0.8, 1.2), _gate(b, dev, 0.5))
    x = (x + (br - 1.0)).clamp_(0, 1)
    c2 = torch.lerp(ones, _u(b, dev, 0.8, 1.2), _gate(b, dev, 0.5))
    x = (x * c2).clamp_(0, 1)
    s = torch.lerp(ones, _u(b, dev, 0.8, 1.2), _gate(b, dev, 0.5))
    lum = _luma(x)
    x = (lum + (x - lum) * s).clamp_(0, 1)
    th = (_u(b, dev, -0.1, 0.1) * _gate(b, dev, 0.3)) * (2 * math.pi)
    cs, sn = torch.cos(th), torch.sin(th)
    r, gc, bl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    yy = 0.299 * r + 0.587 * gc + 0.114 * bl
    ii = 0.596 * r - 0.274 * gc - 0.322 * bl
    qq = 0.211 * r - 0.523 * gc + 0.312 * bl
    i2 = ii * cs - qq * sn
    q2 = ii * sn + qq * cs
    x = torch.cat([yy + 0.956 * i2 + 0.621 * q2, yy - 0.272 * i2 - 0.647 * q2,
                   yy - 1.106 * i2 + 1.703 * q2], dim=1).clamp_(0, 1)
    gm = torch.lerp(ones, _u(b, dev, 0.8, 1.2), _gate(b, dev, 0.3))
    x = x.pow(gm)
    x = (x + torch.randn_like(x) * 0.01 * _gate(b, dev, 0.2)).clamp_(0, 1)
    return x


def hiv15_augment(x, gray_p=0.5, p_original=0.02):
    """x: float (B, 3, H, W) in [0, 1]. Returns augmented x (not normalised)."""
    xa = _augment(x, gray_p)
    keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) < p_original
    return torch.where(keep, x, xa)
