# ply-semantic-smooth.py — 对 CVPR20 管线 ply 做语义级处理（不动拓扑，只修 z 与补全区颜色）
# ① SAM2 物体 mask(640×960) 按顶点 uv 归属 → 物体内 z 向中位数收敛(刚体化 85%) → 脸/椅子不再各自乱动
# ② 补全区顶点(flag≥2) 颜色重采样自 LaMa bgfill → 遮挡区纹理从「糊」变「净」
# 用法: python ply-semantic-smooth.py <ply> <bgfill.png> <matte-person.png> <outdir> <原图.jpg>
import sys
import os
import json
import numpy as np
from PIL import Image

PLY, BGFILL, PERSON_MATTE, OUT, ORIG, PERSON_INST = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
os.makedirs(OUT, exist_ok=True)

# ── ply 解析（numpy） ──
hdr = []
with open(PLY) as f:
    while True:
        line = f.readline().strip()
        hdr.append(line)
        if line == 'end_header':
            break
nv = nf = W = H = 0
hfov = vfov = 0.0
for l in hdr:
    if l.startswith('element vertex'):
        nv = int(l.split()[-1])
    elif l.startswith('element face'):
        nf = int(l.split()[-1])
    elif l.startswith('comment H '):
        H = int(l.split()[-1])
    elif l.startswith('comment W '):
        W = int(l.split()[-1])
    elif l.startswith('comment hFov'):
        hfov = float(l.split()[-1])
    elif l.startswith('comment vFov'):
        vfov = float(l.split()[-1])
vhdr = len(hdr)
print(f'ply: {nv}v {nf}f {W}x{H}')

with open(PLY) as f:
    for _ in range(vhdr):
        f.readline()
    vdata = np.loadtxt(f, max_rows=nv, dtype=np.float32)
pos = np.ascontiguousarray(vdata[:, :3])
rgb = vdata[:, 3:6].astype(np.uint8)
flag = vdata[:, 6].astype(np.int32)
with open(PLY) as f:
    for _ in range(vhdr + nv):
        f.readline()
    fdata = np.loadtxt(f, max_rows=nf, dtype=np.int64)
idx = fdata[:, 1:4].astype(np.uint32)

# ── uv 反解（已用首顶点+发色验证）: u = W/2 + x/z·f, v = H/2 + y/z·f, f=(H/2)/tan(vFov/2) ──
focal = (H / 2) / np.tan(vfov / 2)
z = pos[:, 2]
u = W / 2 + pos[:, 0] / z * focal
v = H / 2 + pos[:, 1] / z * focal
ui = np.clip(np.round(u).astype(int), 0, W - 1)
vi = np.clip(np.round(v).astype(int), 0, H - 1)
print(f'uv 反解 ✔ (u∈[{u.min():.0f},{u.max():.0f}] v∈[{v.min():.0f},{v.max():.0f}])')

# ── SAM2 物体分割本轮不需要（三组深度带用 matte-person + salient 即可） ──

# ── ① 三组深度带: 人物层(matte-person 实例) / 家具层(salient−person=垫子椅子腿) / 背景(不动)
#    每组各自收敛到自己的中位数带（±0.15 去抖）→ 人物动的时候垫子/椅子腿不再跟着动
from scipy.ndimage import binary_dilation
person_m = np.asarray(Image.open(PERSON_INST).convert('L').resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
person_bin = person_m > 0.35
person_bin = binary_dilation(person_bin, iterations=2)  # 外扩 2px 吸收 matte 边缘误差
salient = np.asarray(Image.open(PERSON_MATTE).convert('L').resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
furn_m = (salient > 0.35) & (~person_bin)  # 家具区: 显著前景里不属于人物实例的部分（垫子/椅子腿/凳子）

z_new = z.copy()
obj_id = np.zeros(nv, dtype=np.int32)

def band_smooth(sel, label):
    if sel.sum() < 500:
        print(f'  {label}: 顶点不足({sel.sum()}), 跳过', flush=True)
        return None
    med = np.median(z[sel])
    d = np.clip(z[sel] - med, -0.15, 0.15)
    z_new[sel] = med + d
    print(f'  {label}: {sel.sum()} 顶点 z_med={med:.2f}', flush=True)
    return med

sel_person = person_bin[vi, ui] & (flag == 1)
person_med = band_smooth(sel_person, '人物层')
if person_med is not None:
    sel2 = sel_person & (np.abs(z_new - person_med) > 0.05)
    z_new[sel2] = person_med + np.sign(z_new[sel2] - person_med) * 0.05  # 人物带硬收敛(纯刚体)

sel_furn = furn_m[vi, ui] & (flag == 1) & (~person_bin[vi, ui])
furn_med = band_smooth(sel_furn, '家具层(垫子/椅子腿)')

moved = np.abs(z_new - z) > 0.01
print(f'z 调整顶点: {moved.sum()}/{nv}')

# ── ② 补全区颜色重采样自 LaMa bgfill ──
bfill = np.asarray(Image.open(BGFILL).convert('RGB').resize((W, H), Image.LANCZOS))
inp = flag >= 2
rgb_new = rgb.copy()
rgb_new[inp] = bfill[vi[inp], ui[inp]]
print(f'补全区颜色精修: {inp.sum()} 顶点')

# ── 输出 16B bins ──
vb = np.zeros((nv, 16), dtype=np.uint8)
vb[:, 0:12] = pos.view(np.uint8).reshape(nv, 12)
vb[:, 12:15] = rgb_new
vb[:, 15] = np.clip(flag, 1, 255).astype(np.uint8)
vb.tofile(os.path.join(OUT, 'web_verts.bin'))
idx.tofile(os.path.join(OUT, 'web_faces.bin'))
json.dump(dict(nv=int(nv), nf=int(nf), W=W, H=H, hFov=hfov, vFov=vfov,
               semantic_smooth=True), open(os.path.join(OUT, 'web_meta.json'), 'w'))
print('✔', OUT, f'{os.path.getsize(os.path.join(OUT, "web_verts.bin"))/1e6:.1f}MB + {os.path.getsize(os.path.join(OUT, "web_faces.bin"))/1e6:.1f}MB')
