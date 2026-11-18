# ply2bin.py — ASCII ply → WebGL 二进制（顶点交错 16B: xyz float32 + rgb flag uint8；面片 Uint32×3）
# 用法: python ply2bin.py <ply路径> <输出目录>
import sys
import os
import json
import numpy as np
import pandas as pd

PLY = sys.argv[1] if len(sys.argv) > 1 else 'mesh/w167-0.ply'
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else 'mesh'
os.makedirs(OUTDIR, exist_ok=True)

hdr = []
with open(PLY) as f:
    while True:
        line = f.readline().strip()
        hdr.append(line)
        if line == 'end_header':
            break
nv = nf = H = W = 0
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
print(f'{PLY}: nv={nv} nf={nf} {W}x{H} hFov={hfov:.3f} vFov={vfov:.3f}')

vhdr_lines = len(hdr)
vdf = pd.read_csv(PLY, skiprows=vhdr_lines, nrows=nv, sep=r'\s+', header=None, dtype=np.float32)
assert vdf.shape == (nv, 7), f'vertex 段异常: {vdf.shape}'
pos = np.ascontiguousarray(vdf.iloc[:, :3].to_numpy(np.float32))
rgb = vdf.iloc[:, 3:6].to_numpy(np.int32).astype(np.uint8)
flag = vdf.iloc[:, 6].to_numpy(np.int32).astype(np.uint8)

fdf = pd.read_csv(PLY, skiprows=vhdr_lines + nv, nrows=nf, sep=r'\s+', header=None, dtype=np.int64)
assert fdf.shape == (nf, 4), f'face 段异常: {fdf.shape}'
assert (fdf.iloc[:, 0] == 3).all(), '存在非三角形面'
idx = fdf.iloc[:, 1:4].to_numpy(np.int64)
assert idx.min() >= 0 and idx.max() < nv
idx = idx.astype(np.uint32)

out_v = np.empty((nv, 16), dtype=np.uint8)
out_v[:, 0:12] = pos.view(np.uint8).reshape(nv, 12)
out_v[:, 12:15] = rgb
out_v[:, 15] = flag
out_v.tofile(os.path.join(OUTDIR, 'web_verts.bin'))
idx.tofile(os.path.join(OUTDIR, 'web_faces.bin'))

meta = dict(
    nv=int(nv), nf=int(nf), H=H, W=W, hFov=hfov, vFov=vfov,
    bboxMin=[float(x) for x in pos.min(0)],
    bboxMax=[float(x) for x in pos.max(0)],
    zMin=float(pos[:, 2].min()), zMax=float(pos[:, 2].max()),
)
json.dump(meta, open(os.path.join(OUTDIR, 'web_meta.json'), 'w'), indent=1)
print('✔', os.path.join(OUTDIR, 'web_verts.bin'), os.path.getsize(os.path.join(OUTDIR, 'web_verts.bin')) / 1e6, 'MB')
print('✔', os.path.join(OUTDIR, 'web_faces.bin'), os.path.getsize(os.path.join(OUTDIR, 'web_faces.bin')) / 1e6, 'MB')
