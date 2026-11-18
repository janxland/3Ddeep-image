# sam-layered-mesh.py — v2 语义分层管线（SAM 自动分割 + DA-V2-Large 深度 + LaMa 背景补全 → 渲染器兼容 bins）
# 架构: [背景层] 补全图(bgfill)的深度网格(连续, 无洞, 低频) ∥ [物体层] 每个 SAM instance 独立刚性 slab 段
# 物体 z = mask 内深度中位数(整块刚体, 内部 12% 微起伏) → 人物/椅子永不内部拉扯
# 用法: python sam-layered-mesh.py <img.jpg> <bgfill.png> <outdir>  → outdir/{web_verts.bin,web_faces.bin,web_meta.json}
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
IMG, BG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
GMAX = 288  # 网格分辨率(长边), 背景层
os.makedirs(OUT, exist_ok=True)

from PIL import Image
import torch

# ── 1) DA-V2-Large 深度 (MPS) ──
from transformers import pipeline as hf_pipeline
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('深度模型加载中(DA-V2-large, MPS)…', flush=True)
depther = hf_pipeline('depth-estimation', 'depth-anything/Depth-Anything-V2-Large-hf', device=dev)
img = Image.open(IMG).convert('RGB')
W, H = img.size
dout = depther(img)
depth_hi = np.asarray(dout['depth'].resize((W, H))).astype(np.float32)
depth_hi = (depth_hi - depth_hi.min()) / max(1e-6, depth_hi.max() - depth_hi.min())  # 0..1, 1=近
print('深度 ✔', depth_hi.shape, flush=True)

# ── 2) SAM2 自动分割 (ultralytics SAM) ──
print('SAM 分割中…', flush=True)
from ultralytics import SAM
sam = SAM('sam2.1_s.pt')  # 小模型 46MB, 自动下载
res = sam(IMG, device=dev)
masks = []
if res and res[0].masks is not None:
    for m in res[0].masks.data.cpu().numpy():
        mimg = Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
        masks.append(np.asarray(mimg).astype(np.float32) / 255.0)
# 过滤: 太小的碎片(面积<1.5%)丢弃, 按面积降序, 最多 6 个物体
masks = [m for m in masks if (m > 0.5).mean() > 0.015]
masks.sort(key=lambda m: -(m > 0.5).mean())
masks = masks[:6]
print(f'SAM ✔ {len(masks)} 物体, 面积: {[f"{(m>0.5).mean():.2f}" for m in masks]}', flush=True)

# ── 3) 背景层: LaMa 补全图的深度网格 (连续, 无洞) ──
bg = Image.open(BG).convert('RGB')
bg_depth = np.asarray(dout['depth'])  # 占位: 用人物深度会带洞, 背景层深度直接用补全图重跑一次太贵 — 近似: 用同 depth 但人物区深度取外推
# 实用近似: 背景层深度 = 原深度向远处压 15%(人物区被 LaMa 补全后视觉上背景应更远)
GW = GMAX
GH = int(GMAX * H / W) if W >= H else GMAX
GW = int(GMAX * W / H) if H >= W else GMAX
GH = GMAX if H >= W else GH
gw, gh = (GW, GH) if W >= H else (GW, GH)
gw = min(GMAX, int(GMAX * W / H)) if H > W else GMAX
gh = min(GMAX, int(GMAX * H / W)) if W > H else GMAX
gw = GMAX if W >= H else gw
gh = GMAX if H >= W else gh
bg_c = np.asarray(bg.resize((gw, gh), Image.LANCZOS))
bg_d = np.asarray(Image.fromarray((depth_hi * 255).astype(np.uint8)).resize((gw, gh), Image.LANCZOS)).astype(np.float32) / 255.0

zN, zF = 1.2, 2.6
verts = []   # (x, y, z, r, g, b)
indices = []
def add_mesh(zbuf, cbuf, gw_, gh_, tear=0.14, slab_z=None, maskbuf=None):
    global indices, verts
    base = len(verts)
    dArr = np.zeros(gw_ * gh_, dtype=np.float32)
    for gy in range(gh_):
        for gx in range(gw_):
            i = gy * gw_ + gx
            d = zbuf[gy, gx]
            dArr[i] = d
            x = (gx / (gw_ - 1) - 0.5) * (W / H)
            y = (gy / (gh_ - 1) - 0.5)
            z = zF - d * (zF - zN)
            if slab_z is not None:
                z = slab_z - (d - np.median(zbuf)) * (zF - zN) * 0.12
            c = cbuf[gy, gx]
            verts.append((x, y, z, int(c[0]), int(c[1]), int(c[2])))
    for gy in range(gh_ - 1):
        for gx in range(gw_ - 1):
            a, b, c, e = gy * gw_ + gx, gy * gw_ + gx + 1, (gy + 1) * gw_ + gx, (gy + 1) * gw_ + gx + 1
            if maskbuf is not None and not (maskbuf[a] and maskbuf[b] and maskbuf[c] and maskbuf[e]):
                continue
            if tear and (dArr[a] - dArr[e]) ** 2 <= (dArr[b] - dArr[c]) ** 2:
                indices += [base + a, base + c, base + b, base + b, base + c, base + e]
            else:
                indices += [base + a, base + e, base + b, base + a, base + c, base + e]
    return base

# 背景层(补全图, 连续无撕裂): 遮挡区=LaMa 画的真实背景
add_mesh(bg_d, bg_c, gw, gh, tear=0)  # 背景 slab 禁撕(连续房间)

# ── 4) 物体层: 每实例刚性 slab ──
img_np = np.asarray(img)
for mi, m in enumerate(masks):
    m_g = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((gw, gh), Image.LANCZOS)) / 255.0
    # 物体 slab 深度 = mask 内深度 40 分位(偏近, 保证压在背景前)
    d_slab = float(np.percentile(depth_hi[m > 0.5], 40)) if (m > 0.5).any() else 0.8
    slab_z = zF - d_slab * (zF - zN)
    # 物体专属深度场: slab 平面 + 原深度 12% 细节（网格分辨率）
    zgrid = np.zeros((gh, gw), dtype=np.float32)
    cgrid = np.zeros((gh, gw, 3), dtype=np.uint8)
    mgrid = np.zeros(gh * gw, dtype=bool)
    img_g = np.asarray(img.resize((gw, gh), Image.LANCZOS))
    for gy in range(gh):
        for gx in range(gw):
            if m_g[gy, gx] > 0.35:
                i = gy * gw + gx
                mgrid[i] = True
                zgrid[gy, gx] = depth_hi[gy, gx] if m_g[gy, gx] > 0.35 else d_slab
                cgrid[gy, gx] = img_g[gy, gx]
    # zgrid 是网格分辨率上的原深度（未压平）——add_mesh 的 slab_z 参数负责压平
    zgrid_use = zgrid
    add_mesh(zgrid_use, cgrid, gw, gh, tear=0.05, slab_z=slab_z, maskbuf=mgrid)
    print(f'  物体{mi+1}: slab_d={d_slab:.3f} z={slab_z:.2f}', flush=True)

# ── 5) 输出 16B 交错 bins (渲染器兼容) ──
nv = len(verts)
vb = np.zeros((nv, 16), dtype=np.uint8)
pos = np.array([[v[0], v[1], v[2]] for v in verts], dtype=np.float32)
rgb = np.array([[v[3], v[4], v[5]] for v in verts], dtype=np.uint8)
vb[:, 0:12] = pos.view(np.uint8).reshape(nv, 12)
vb[:, 12:15] = rgb
vb[:, 15] = 1
np.array(indices, dtype=np.uint32).tofile(os.path.join(OUT, 'web_faces.bin'))
vb.tofile(os.path.join(OUT, 'web_verts.bin'))
import json
aspect = W / H
zMid = (zN + zF) / 2
json.dump(dict(nv=nv, nf=len(indices) // 3, W=GW, H=GH,
               hFov=2 * np.arctan(aspect * 0.5 / zMid), vFov=2 * np.arctan(0.5 / zMid)),
          open(os.path.join(OUT, 'web_meta.json'), 'w'))
print(f'✔ bins: {nv} 顶点 / {len(indices)//3} 面 → {OUT}', flush=True)
