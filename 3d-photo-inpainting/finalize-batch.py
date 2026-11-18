# finalize-batch.py — 批量装配线：mesh/*.ply → 每图组件资产包 + 深度 png + 库存清单
# 用法: python finalize-batch.py [输出根目录]  (默认 ../component-assets)
# 产出结构:
#   <out>/<stem>/web_verts.bin web_faces.bin web_meta.json   ← <depth-viewer mesh-src="<CDN>/<stem>">
#   <out>/depth/<stem>.depth.png                             ← <depth-viewer depth-src="...">
#   <out>/inventory.json                                     ← 库存清单（哪些 stem 有 mesh/depth/color）
import os
import sys
import glob
import json
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, '..', 'component-assets')

plies = sorted(glob.glob(os.path.join(REPO, 'mesh', '*.ply')))
npy_s = sorted(glob.glob(os.path.join(REPO, 'depth', '*.npy')))
imgs = sorted(glob.glob(os.path.join(REPO, 'image', '*.jpg')))
print(f'输入: {len(plies)} ply / {len(npy_s)} npy / {len(imgs)} 原图')

inventory = {}
for ply in plies:
    stem = os.path.splitext(os.path.basename(ply))[0]
    outdir = os.path.join(OUT, stem)
    os.makedirs(outdir, exist_ok=True)
    # 解析 ply 头
    hdr = []
    with open(ply) as f:
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
    vhdr = len(hdr)
    vdf = pd.read_csv(ply, skiprows=vhdr, nrows=nv, sep=r'\s+', header=None, dtype=np.float32)
    if vdf.shape != (nv, 7):
        print('✘', stem, 'vertex 段异常', vdf.shape); continue
    pos = np.ascontiguousarray(vdf.iloc[:, :3].to_numpy(np.float32))
    rgb = vdf.iloc[:, 3:6].to_numpy(np.int32).astype(np.uint8)
    flag = vdf.iloc[:, 6].to_numpy(np.int32).astype(np.uint8)
    fdf = pd.read_csv(ply, skiprows=vhdr + nv, nrows=nf, sep=r'\s+', header=None, dtype=np.int64)
    if fdf.shape != (nf, 4) or not (fdf.iloc[:, 0] == 3).all():
        print('✘', stem, 'face 段异常'); continue
    idx = fdf.iloc[:, 1:4].to_numpy(np.int64)
    if idx.min() < 0 or idx.max() >= nv:
        print('✘', stem, '索引越界'); continue
    # 顶点量化 float32→int16（bbox 归一，精度 ~1/65535 × 跨度，视觉无损）
    mn, mx = pos.min(0), pos.max(0)
    span = np.maximum(mx - mn, 1e-6)
    q = np.round((pos - mn) / span * 65535).astype(np.int32)
    assert q.min() >= 0 and q.max() <= 65535
    q16 = q.astype(np.uint16)
    # 交错: qxyz(uint16×3=6B) + rgb+flag(4B) = 10B stride（GPU 端按 meta 反量化）
    out_v = np.empty((nv, 10), dtype=np.uint8)
    out_v[:, 0:6] = q16.view(np.uint8).reshape(nv, 6)
    out_v[:, 6:9] = rgb
    out_v[:, 9] = flag
    out_v.tofile(os.path.join(outdir, 'web_verts.bin'))
    idx.astype(np.uint32).tofile(os.path.join(outdir, 'web_faces.bin'))
    json.dump(dict(
        nv=int(nv), nf=int(nf), H=H, W=W, hFov=hfov, vFov=vfov,
        bboxMin=[float(x) for x in mn], bboxMax=[float(x) for x in mx],
        quantized='uint16',
    ), open(os.path.join(outdir, 'web_meta.json'), 'w'))
    print('✔', stem, f'{nv}v/{nf}f verts 10B/点 = {os.path.getsize(os.path.join(outdir, "web_verts.bin")) / 1e6:.1f}MB')

# 深度 png（亮=近）
os.makedirs(os.path.join(OUT, 'depth'), exist_ok=True)
for npy in npy_s:
    stem = os.path.splitext(os.path.basename(npy))[0]
    d = np.load(npy).astype(np.float32)
    d = (d - d.min()) / max(1e-6, d.max() - d.min())
    cv2png = os.path.join(OUT, 'depth', stem + '.depth.png')
    try:
        import cv2
        cv2.imwrite(cv2png, (d * 255).astype(np.uint8))
    except Exception:
        # 无 cv2 时用纯 numpy + zlib PNG
        import struct, zlib
        h, w = d.shape
        g = (d * 255).astype(np.uint8)
        raw = b''.join(b'\x00' + g[y].tobytes() for y in range(h))
        def chunk(t, dd): return struct.pack('>I', len(dd)) + t + dd + struct.pack('>I', zlib.crc32(t + dd))
        png = (b'\x89PNG\r\n\x1a\n'
               + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(raw, 6))
               + chunk(b'IEND', b''))
        open(cv2png, 'wb').write(png)
    print('✔ depth', stem)

# 库存清单
for img in imgs:
    stem = os.path.splitext(os.path.basename(img))[0]
    inv = inventory.setdefault(stem, {'color': 'image/' + os.path.basename(img)})
    if os.path.isdir(os.path.join(OUT, stem)):
        inv['mesh'] = stem + '/web_meta.json'
    if os.path.exists(os.path.join(OUT, 'depth', stem + '.depth.png')):
        inv['depth'] = 'depth/' + stem + '.depth.png'
json.dump(inventory, open(os.path.join(OUT, 'inventory.json'), 'w'), indent=1, ensure_ascii=False)
print('✔ inventory.json:', len(inventory), '项')
print('完成。输出目录:', os.path.abspath(OUT))
