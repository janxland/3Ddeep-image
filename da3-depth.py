# da3-depth.py — DA3-BASE 单图深度 → 管线 npy（适配 CVPR20 管线 read_MiDaS_depth 约定）
# 用法: python da3-depth.py <image.jpg> <out.npy>
# 输出: 归一化 0..1 float32 npy（1=近），与 MiDaS npy 同约定
import sys
import os
import numpy as np

IMG, OUT = sys.argv[1], sys.argv[2]

import torch
from depth_anything_3.api import DepthAnything3

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('加载 DA3-BASE (MPS)…', flush=True)
model = DepthAnything3.from_pretrained('depth-anything/DA3-BASE')
model = model.to(dev).eval()
print('模型就绪', flush=True)

from PIL import Image
img = Image.open(IMG).convert('RGB')
pred = model.inference([img], process_res=960)
# Prediction: 单图 → depth 在 pred.depth 或 pred.scaled_depth
d = getattr(pred, 'depth', None)
if d is None and hasattr(pred, 'scaled_depth'):
    d = pred.scaled_depth
if d is None:
    print('Prediction 字段:', [a for a in dir(pred) if not a.startswith('_')])
    raise SystemExit(1)
import numpy as _np
arr = _np.asarray(d)
if arr.ndim == 3:
    arr = arr[0]
print('raw depth:', arr.shape, arr.dtype, 'min/max', float(arr.min()), float(arr.max()), flush=True)
# DA3: metric/relative — 大值=远(深度)。管线约定: 1=近 → 取反归一
arr = arr.astype(np.float32)
arr = (arr - arr.min()) / max(1e-6, arr.max() - arr.min())
arr = 1.0 - arr  # 归一后: 1=近
np.save(OUT, arr.astype(np.float32))
print('✔', OUT, arr.shape, '1=近')
