"""Step 2 environment check: prints versions and GPU facts, exits 1 if the GPU cannot be used."""
import importlib, platform, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'results' / 'step2'; out.mkdir(parents=True, exist_ok=True)
lines, ok = [f'python {sys.version.split()[0]} ({sys.executable}) on {platform.platform()}'], True
for mod in ('torch', 'torchvision', 'timm', 'huggingface_hub', 'numpy', 'pandas', 'PIL', 'sklearn'):
    try:
        m = importlib.import_module(mod); lines.append(f'{mod} {getattr(m, "__version__", "?")}')
    except Exception as e:
        lines.append(f'{mod} MISSING ({e.__class__.__name__})')
        if mod in ('torch', 'timm', 'numpy', 'pandas', 'PIL'):
            ok = False
try:
    import torch
    lines.append(f'torch CUDA build: {torch.version.cuda}; cuda available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        arch = torch.cuda.get_arch_list()
        lines.append(f'GPU: {torch.cuda.get_device_name(0)}, compute capability {cc}, '
                     f'{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB')
        lines.append(f'torch compiled archs: {arch}')
        need = f'sm_{cc[0]}{cc[1]}'
        if need not in arch:
            lines.append(f'!! this torch build has no kernels for {need} -> install a build that includes it '
                         f'(for Blackwell sm_120: a CUDA 12.8+ wheel)'); ok = False
        x = torch.randn(256, 256, device='cuda'); (x @ x).sum().item()
        lines.append('GPU matmul test: OK')
    else:
        lines.append('!! CUDA not available - extraction would run on CPU'); ok = False
except Exception as e:
    lines.append(f'!! torch/GPU test failed: {e!r}'); ok = False
lines.append('RESULT: ' + ('OK' if ok else 'NOT OK'))
txt = '\n'.join(lines)
print(txt)
(out / 'env.txt').write_text(txt + '\n', encoding='utf-8')
sys.exit(0 if ok else 1)
