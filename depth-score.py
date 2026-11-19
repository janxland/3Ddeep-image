# depth-score.py — 深度图客观评分器（无需真值, 几何/梯度/SAM2 语义指标）
# 四维指标:
#   M1 边界贴合度: 深度梯度与图像梯度的空间重合率（高=撕缝贴轮廓, 撕裂判据可靠）
#   M2 物体内噪声: SAM2 mask 内深度高频残差 std（低=脸/椅子不抖动）
#   M3 平面平整度: 低纹理区平面拟合残差 RMS（低=墙/地板不波动）
#   M4 断层锐度:   深度断层(foreground 边界)两侧过渡带宽（窄=分层利落）
# 综合分 = 0.35·M1 + 0.25·(1−M2) + 0.2·(1−M3) + 0.2·(1−M4过渡)   → 0..1 越高越好
# 用法: python depth-score.py <image.jpg> <depth.npy或png> [sam2]
import os
import sys
import json
import numpy as np

IMG, DEPTH = sys.argv[1], sys.argv[2]
USE_SAM = len(sys.argv) > 3 and sys.argv[3] == 'sam2'

from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, sobel, label

img = np.asarray(Image.open(IMG).convert('L').resize(
    (np.asarray(Image.open(IMG)).shape[1], np.asarray(Image.open(IMG)).shape[0]))).astype(np.float32) / 255.0

if DEPTH.endswith('.npy'):
    depth = np.load(DEPTH).astype(np.float32)
else:
    depth = np.asarray(Image.open(DEPTH).convert('L')).astype(np.float32) / 255.0
if depth.shape[0] != img.shape[0] or depth.shape[1] != img.shape[1]:
    from PIL import Image as P
    depth = np.asarray(P.fromarray((depth * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.LANCZOS)).astype(np.float32) / 255.0
H, W = depth.shape
print(f'depth {W}x{H}')

def norm(x):
    x = x - x.min()
    return x / max(1e-6, x.max())

# ── M1 边界贴合度: 深度梯度与图像梯度的重合率 ──
gy, gx = np.gradient(gaussian_filter(img, 1.0))
e_img = norm(np.hypot(gx, gy))
dy, dx = np.gradient(gaussian_filter(depth, 1.0))
e_dep = norm(np.hypot(dx, dy))
e_img_t = e_img > np.percentile(e_img, 90)
e_dep_t = e_dep > np.percentile(e_dep, 90)
from scipy.ndimage import binary_dilation, maximum_filter
e_img_d = maximum_filter(e_img_t, size=5)
hit = (e_dep_t & e_img_d).sum() / max(1, e_dep_t.sum())
m1 = float(hit)

# ── M2 物体内噪声: SAM2 mask 内深度高频残差 ──
m2 = None
if USE_SAM:
    from ultralytics import SAM
    from PIL import Image as P
    sam = SAM('sam2.1_s.pt')
    res = sam(P.fromarray((img * 255).astype(np.uint8)), verbose=False)
    noises = []
    if res and res[0].masks is not None:
        for m in res[0].masks.data.cpu().numpy():
            mi = np.asarray(P.fromarray((m * 255).astype(np.uint8)).resize((W, H), P.LANCZOS)) > 0.35
            if mi.sum() < 2000:
                continue
            d = depth.copy()
            sm = gaussian_filter(depth, 4.0)
            hf = np.abs(d - sm)[mi]
            noises.append(float(np.std(hf)))
    m2 = float(np.mean(noises)) if noises else None

# ── M3 平面平整度: 低纹理区 16×16 tile 平面拟合残差 ──
gimg = gaussian_filter(img, 2.0)
tiles = []
T = 32
for ty in range(0, H - T, T):
    for tx in range(0, W - T, T):
        tile = gimg[ty:ty + T, tx:tx + T]
        if tile.std() < 0.02:  # 低纹理区
            tiles.append((ty, tx))
resids = []
for ty, tx in tiles[:200]:
    patch = depth[ty:ty + T, tx:tx + T]
    yy, xx = np.mgrid[0:T, 0:T]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(T * T)], 1)
    sol, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    resid = patch.ravel() - A @ sol
    resids.append(float(np.sqrt((resid ** 2).mean())))
m3 = float(np.mean(resids)) if resids else None

# ── M4 断层过渡带宽: 强深度边缘处的过渡像素比例（窄=分层利落） ──
e_dep_d = maximum_filter(e_dep_t, size=3)
band = e_dep_d & (~e_dep_t)
edge_band_ratio = float(band.sum() / max(1, e_dep_t.sum()))
m4 = edge_band_ratio  # 越低越利落

score = 0.35 * m1 + 0.25 * (1 - (m2 or 0)) + 0.2 * (1 - (m3 or 0)) + 0.2 * (1 - min(1, (m4 or 0) * 2))
result = dict(
    M1_edge_align=round(m1, 4),
    M2_intra_noise=round(m2, 4) if m2 is not None else None,
    M3_planar_resid=round(m3, 4) if m3 is not None else None,
    M4_tear_band=round(edge_band_ratio, 4),
    SCORE=round(score, 4),
)
print(json.dumps(result, indent=1))
json.dump(result, open(os.path.splitext(DEPTH)[0] + '.score.json', 'w'), indent=1)
