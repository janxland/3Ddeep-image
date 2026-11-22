# tune-person-depth.py — 人物内部景深的数学闭环优化
# 问题: 端内轨分层把人物压平(fgD 常数) → 手/脚/头"贴"在同一深度。
# 方案: 人物区内深度 K 层分位切片(层内取中位数消噪, 层间保层次) + 层间自然撕缝。
# 闭环: 参数网格(K, sigma, edge) × 四维评分(M5 层次保真/M2' 层内噪声/M1 边界/M4 撕裂带)
#       → 自动选最优 → 输出参数表, 供 buildLayeredMeshCore 采用。
# 用法: python tune-person-depth.py <image.jpg> <depth.npy|png> <matte.png>
import os, sys, json, itertools
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, sobel, maximum_filter

IMG, DEPTH, MATTE = sys.argv[1], sys.argv[2], sys.argv[3]

# ── 载入(depth 统一 0..1 亮=近, 与管线约定一致) ──
im = Image.open(IMG).convert('RGB')
W, H = im.size
if DEPTH.endswith('.npy'):
    dep = np.load(DEPTH).astype(np.float32)
else:
    dep = np.asarray(Image.open(DEPTH).convert('L')).astype(np.float32) / 255.0
dep = np.asarray(Image.fromarray((dep * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
matte = np.asarray(Image.open(MATTE).convert('L').resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
gray = np.asarray(im.convert('L')).astype(np.float32) / 255.0
person = matte > 0.5
print(f'image {W}x{H}, person {person.mean()*100:.1f}%')

def norm(x):
    x = x - x.min(); return x / max(1e-6, x.max())

# ── 分层深度合成: 参数 (K, sigma, edge) ──
def layered_depth(K, sigma, edge_keep):
    """返回合成深度场 + 调试信息。
    K: 人物区分层数(1=旧版全压平)
    sigma: 切片前人物区高斯平滑(消噪, 保留大结构)
    edge_keep: 边缘过渡带保留原始深度的强度(1=不融合, 0=融合到层值)"""
    ds = gaussian_filter(dep, sigma) if sigma > 0 else dep.copy()
    out = dep.copy()
    p = person
    if K == 1:
        fgD = float(np.median(ds[p]))
        out[p] = ds[p] * (1 - (1 - edge_keep)) + fgD * (1 - edge_keep)
        return out, {'layers': [fgD]}
    # K 层: 人物区深度分位数切片
    vals = ds[p]
    qs = [np.quantile(vals, i / K) for i in range(1, K)]
    labels = np.digitize(ds, qs)  # 0..K-1
    centers = [float(np.median(ds[p & (labels == k)])) for k in range(K)]
    out[p] = np.array(centers)[labels[p]]  # 层内替换为层中位数
    # 边缘过渡带(人物边缘 6px 羽化区): 原始深度与层值按 edge_keep 混合, 保轮廓撕缝自然
    edge = np.asarray(Image.fromarray((person * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(13))).astype(np.float32) / 255.0
    band = (edge > 0.05) & (~p)
    out[band] = dep[band] * edge_keep + out[band] * (1 - edge_keep)
    return out, {'layers': [round(c, 3) for c in centers], 'quantiles': [round(q, 3) for q in qs]}

def evaluate(out):
    """四维评分(无需真值)"""
    # M5 人物内层次保真: 输出与原始深度(σ=0 参考)在人物区的 Pearson 相关
    a, b = out[person], dep[person]
    if a.std() < 1e-6 or b.std() < 1e-6:
        m5 = 0.0
    else:
        m5 = float(np.corrcoef(a, b)[0, 1])
    # M2' 层内噪声: 人物区高频残差 std(层内中位数替换应显著降低)
    sm = gaussian_filter(out, 4.0)
    m2 = float(np.std(np.abs(out - sm)[person]))
    # M1 边界贴合: 深度梯度与图像梯度重合率
    gy, gx = np.gradient(gaussian_filter(gray, 1.0)); e_img = norm(np.hypot(gx, gy))
    dy, dx = np.gradient(gaussian_filter(out, 1.0)); e_dep = norm(np.hypot(dx, dy))
    ei = e_img > np.percentile(e_img, 90); ed = e_dep > np.percentile(e_dep, 90)
    m1 = float((ed & maximum_filter(ei, size=5)).sum() / max(1, ed.sum()))
    # M4 撕裂过渡带
    band = maximum_filter(ed, size=3) & (~ed)
    m4 = float(band.sum() / max(1, ed.sum()))
    score = 0.30 * m5 + 0.25 * (1 - min(1, m2 * 8)) + 0.25 * m1 + 0.20 * (1 - min(1, m4 * 2))
    return dict(M5=m5, M2=m2, M1=m1, M4=m4, SCORE=score)

def evaluate(out, K=None):
    """四维评分(无需真值)。K>1 时 M2 只统计层内残差(层间跳变是分层 feature, 不罚)。"""
    # M5 人物内层次保真: 输出与原始深度(σ=0 参考)在人物区的 Pearson 相关
    a, b = out[person], dep[person]
    if a.std() < 1e-6 or b.std() < 1e-6:
        m5 = 0.0
    else:
        m5 = float(np.corrcoef(a, b)[0, 1])
    # M2' 层内噪声: 人物区高频残差 std; K 层时排除层间边界 ±2px
    sm = gaussian_filter(out, 4.0)
    hf = np.abs(out - sm)
    if K and K > 1:
        vals = dep[person]
        qs = [np.quantile(vals, i / K) for i in range(1, K)]
        labels = np.digitize(dep, qs)
        inner = np.zeros_like(person)
        for k in range(K):
            seg = person & (labels == k)
            inner |= seg & (~np.asarray(Image.fromarray((seg * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))).astype(bool))
        zone = inner if inner.sum() > 500 else person
    else:
        zone = person
    m2 = float(np.std(hf[zone]))
    # M1 边界贴合: 深度梯度与图像梯度重合率
    gy, gx = np.gradient(gaussian_filter(gray, 1.0)); e_img = norm(np.hypot(gx, gy))
    dy, dx = np.gradient(gaussian_filter(out, 1.0)); e_dep = norm(np.hypot(dx, dy))
    ei = e_img > np.percentile(e_img, 90); ed = e_dep > np.percentile(e_dep, 90)
    m1 = float((ed & maximum_filter(ei, size=5)).sum() / max(1, ed.sum()))
    # M4 撕裂过渡带
    band = maximum_filter(ed, size=3) & (~ed)
    m4 = float(band.sum() / max(1, ed.sum()))
    score = 0.30 * m5 + 0.25 * (1 - min(1, m2 * 8)) + 0.25 * m1 + 0.20 * (1 - min(1, m4 * 2))
    return dict(M5=m5, M2=m2, M1=m1, M4=m4, SCORE=score)

# ── 网格搜索 ──
grid = list(itertools.product([1, 2, 3, 4], [0, 3, 5, 7], [1.0, 0.6, 0.0]))
rows = []
for K, sg, ek in grid:
    out, info = layered_depth(K, sg, ek)
    m = evaluate(out, K)
    rows.append(dict(K=K, sigma=sg, edge_keep=ek, **{k: round(v, 4) for k, v in m.items()}, layers=info['layers']))
    print(f"K={K} σ={sg} edge={ek}: M5={m['M5']:.3f} M2={m['M2']:.4f} M1={m['M1']:.3f} M4={m['M4']:.3f} SCORE={m['SCORE']:.4f}")

best = max(rows, key=lambda r: r['SCORE'])
print('\n=== BEST ===')
print(json.dumps(best, indent=1))
json.dump({'best': best, 'all': rows}, open(os.path.join(os.path.dirname(DEPTH), 'tune-person-depth.json'), 'w'), indent=1)

# ── 可视化对比: 原始 / 旧压平 / 最优分层 ──
flat, _ = layered_depth(1, 7, 0.92)
bestd, _ = layered_depth(best['K'], best['sigma'], best['edge_keep'])
cmp = Image.new('RGB', (W * 3, H))
for i, dd in enumerate([dep, flat, bestd]):
    v = (np.clip(dd, 0, 1) * 255).astype(np.uint8)
    cmp.paste(Image.fromarray(v).convert('RGB'), (W * i, 0))
cmp.save(os.path.join(os.path.dirname(DEPTH), 'tune-person-depth-cmp.png'))
print('✔ cmp saved: raw | flat | best-layered')
