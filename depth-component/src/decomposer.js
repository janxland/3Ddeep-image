// decomposer.js — 主体/背景两层软分解(对齐 spatial-scene-android SoftLayerDecomposer, Apple 同款推测)
// 范式(他们的踩坑结论, 全部移植):
//   ① 主体遮罩: Otsu 深度阈值 + 最大连通域 + 闭运算 + 洞填充("宁可多包")
//   ② 前景 alpha = 硬遮罩不羽化(羽化会把两层不同位移内容混合 → "液态玻璃"重影)
//   ③ push-pull 几何预填整个主体脚印(给 AI "背景状"上下文, 避免它延续主体纹理)
//   ④ MI-GAN 只补脚印(hole=脚印, 基于预填上下文 → 内容更实)
//   ⑤ 脚印内柔化(fill×0.35 + blur×0.65), 脚印外永远是原图 → 静态画质无损
//   ⑥ 背景层深度: nearestFill 平坦外延 × BG_DEPTH_SCALE(背景不被"透镜弯")
//   ⑦ 前景层深度: 遮罩内原始深度, 遮罩外 nearestFill 边缘外延(两层内部都无悬崖 → 无拉伸条纹)
//   ⑧ 静止合成 = 原图
// 输入: imageRGBA/depthGray(Float32 0..1 亮=近)/w/h(+可选 aiInpaint)
// 输出: { bgImage, bgDepth, fgImage, fgDepth, fgAlpha, note } 全部 RGBA Uint8ClampedArray

const PACK_LONG_SIDE = 1024
const BG_DEPTH_SCALE = 0.3
const MIN_MASK_FRACTION = 0.03
const MAX_MASK_FRACTION = 0.90

function boxBlurGray(src, w, h, r) {
  if (r <= 0) return src.slice()
  const out = new Float32Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let acc = 0, n = 0
    for (let dy = -r; dy <= r; dy++) {
      const yy = Math.min(h - 1, Math.max(0, y + dy))
      acc += src[yy * w + x]; n++
    }
    out[y * w + x] = acc / n
  }
  const out2 = new Float32Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let acc = 0, n = 0
    for (let dx = -r; dx <= r; dx++) {
      const xx = Math.min(w - 1, Math.max(0, x + dx))
      acc += out[y * w + xx]; n++
    }
    out2[y * w + x] = acc / n
  }
  return out2
}

function boxBlurRgb(src, w, h, r) {
  const g0 = new Float32Array(w * h), g1 = new Float32Array(w * h), g2 = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) { g0[i] = src[i * 3]; g1[i] = src[i * 3 + 1]; g2[i] = src[i * 3 + 2] }
  const b0 = boxBlurGray(g0, w, h, r), b1 = boxBlurGray(g1, w, h, r), b2 = boxBlurGray(g2, w, h, r)
  const out = new Float32Array(src.length)
  for (let i = 0; i < w * h; i++) { out[i * 3] = b0[i]; out[i * 3 + 1] = b1[i]; out[i * 3 + 2] = b2[i] }
  return out
}

function otsu(d) {
  const hist = new Array(256).fill(0)
  for (let i = 0; i < d.length; i++) hist[Math.min(255, Math.max(0, Math.round(d[i] * 255)))]++
  const total = d.length
  let sum = 0
  for (let t = 0; t < 256; t++) sum += t * hist[t]
  let sumB = 0, wB = 0, best = 0, thr = 128
  for (let t = 0; t < 256; t++) {
    wB += hist[t]
    if (!wB) continue
    const wF = total - wB
    if (!wF) break
    sumB += t * hist[t]
    const mB = sumB / wB, mF = (sum - sumB) / wF
    const between = wB * wF * (mB - mF) * (mB - mF)
    if (between > best) { best = between; thr = t }
  }
  return thr / 255
}

// 最大连通域(4 邻域 BFS), 过滤小于 minSize 的碎片
function largestComponents(mask, w, h, minSize) {
  const lbl = new Int32Array(w * h).fill(-1)
  const sizes = []
  const queue = new Int32Array(w * h)
  let comps = 0
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || lbl[s] >= 0) continue
    let head = 0, tail = 0
    queue[tail++] = s
    lbl[s] = comps
    let size = 0
    while (head < tail) {
      const i = queue[head++]; size++
      const x = i % w, y = (i / w) | 0
      if (x > 0 && mask[i - 1] && lbl[i - 1] < 0) { lbl[i - 1] = comps; queue[tail++] = i - 1 }
      if (x < w - 1 && mask[i + 1] && lbl[i + 1] < 0) { lbl[i + 1] = comps; queue[tail++] = i + 1 }
      if (y > 0 && mask[i - w] && lbl[i - w] < 0) { lbl[i - w] = comps; queue[tail++] = i - w }
      if (y < h - 1 && mask[i + w] && lbl[i + w] < 0) { lbl[i + w] = comps; queue[tail++] = i + w }
    }
    sizes.push(size); comps++
  }
  const keep = sizes.map((sz, ci) => (sz >= minSize ? ci : -1))
  const out = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) if (lbl[i] >= 0 && keep[lbl[i]] >= 0) out[i] = 1
  return out
}

// min/max filter(可分离 1D): 方形核 = 水平 pass + 垂直 pass, 精确等价且 O(N·r) — 比 O(N·r²) 快 30 倍+
function filterMask(srcF, w, h, r, isMin) {
  const tmp = new Float32Array(w * h)
  // 水平 pass
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      let v = isMin ? 1 : 0
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r)
      for (let xx = x0; xx <= x1; xx++) {
        const s = srcF[row + xx]
        v = isMin ? Math.min(v, s) : Math.max(v, s)
      }
      tmp[row + x] = v
    }
  }
  // 垂直 pass
  const out = new Float32Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = isMin ? 1 : 0
    const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r)
    for (let yy = y0; yy <= y1; yy++) {
      const sv = tmp[yy * w + x]
      v = isMin ? Math.min(v, sv) : Math.max(v, sv)
    }
    out[y * w + x] = v
  }
  return out
}

// 洞填充: 不与画面边缘连通的背景块并入主体(阴影造成的主体内背景缺口)
function fillEnclosedBackground(mask, w, h) {
  const n = w * h
  const reachable = new Uint8Array(n)
  const queue = new Int32Array(n)
  let head = 0, tail = 0
  for (let x = 0; x < w; x++) {
    if (!mask[x] && !reachable[x]) { reachable[x] = 1; queue[tail++] = x }
    const b = (h - 1) * w + x
    if (!mask[b] && !reachable[b]) { reachable[b] = 1; queue[tail++] = b }
  }
  for (let y = 0; y < h; y++) {
    const l = y * w, r = y * w + w - 1
    if (!mask[l] && !reachable[l]) { reachable[l] = 1; queue[tail++] = l }
    if (!mask[r] && !reachable[r]) { reachable[r] = 1; queue[tail++] = r }
  }
  while (head < tail) {
    const i = queue[head++]
    const x = i % w, y = (i / w) | 0
    if (x > 0 && !mask[i - 1] && !reachable[i - 1]) { reachable[i - 1] = 1; queue[tail++] = i - 1 }
    if (x < w - 1 && !mask[i + 1] && !reachable[i + 1]) { reachable[i + 1] = 1; queue[tail++] = i + 1 }
    if (y > 0 && !mask[i - w] && !reachable[i - w]) { reachable[i - w] = 1; queue[tail++] = i - w }
    if (y < h - 1 && !mask[i + w] && !reachable[i + w]) { reachable[i + w] = 1; queue[tail++] = i + w }
  }
  const out = mask.slice()
  for (let i = 0; i < n; i++) if (!mask[i] && !reachable[i]) out[i] = 1
  return out
}

// push-pull 简化替代: 多源 BFS 均值扩散(O(N) 一次, rgb) — 大洞层级扩散收敛
function pushPull(rgb, valid, w, h) {
  const out = rgb.slice()
  const known = valid.slice()
  const queue = new Int32Array(w * h)
  let head = 0, tail = 0
  for (let i = 0; i < w * h; i++) if (known[i]) queue[tail++] = i
  while (head < tail) {
    const i = queue[head++]
    const x = i % w, y = (i / w) | 0
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue
      const xx = Math.min(w - 1, Math.max(0, x + dx)), yy = Math.min(h - 1, Math.max(0, y + dy))
      const j = yy * w + xx
      if (!known[j]) {
        // 取邻域已知均值(标记时邻域尚在扩散中, 用当前 out 值即可 — 波前保证单调)
        let r = 0, g = 0, b = 0, n = 0
        for (let dy2 = -1; dy2 <= 1; dy2++) for (let dx2 = -1; dx2 <= 1; dx2++) {
          const xx2 = Math.min(w - 1, Math.max(0, x + dx2)), yy2 = Math.min(h - 1, Math.max(0, y + dy2))
          const j2 = yy2 * w + xx2
          if (known[j2] || j2 === i) { }
          if (known[j2]) { r += out[j2 * 3]; g += out[j2 * 3 + 1]; b += out[j2 * 3 + 2]; n++ }
        }
        if (n >= 1) {
          out[j * 3] = r / n; out[j * 3 + 1] = g / n; out[j * 3 + 2] = b / n
          known[j] = 1; queue[tail++] = j
        }
      }
    }
  }
  return out
}

// Chamfer 距离变换(两遍扫描, 欧氏近似): 每像素到前景 mask 的距离(px)
// O(N) — 比 filterMask 多轮膨胀快得多且精确; 用于"前景位移会扫过的区域"= 补全区
function chamferDist(mask, w, h) {
  const INF = 1e9
  const d = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) d[i] = mask[i] ? 0 : INF
  // 前向(左上→右下): 邻域 (−1,0)(0,−1)(−1,−1)(1,−1) 权重 1/1/1.4/1.4
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x
    if (d[i] === 0) continue
    let v = d[i]
    if (x > 0) v = Math.min(v, d[i - 1] + 1)
    if (y > 0) v = Math.min(v, d[i - w] + 1)
    if (x > 0 && y > 0) v = Math.min(v, d[i - w - 1] + 1.4)
    if (x < w - 1 && y > 0) v = Math.min(v, d[i - w + 1] + 1.4)
    d[i] = v
  }
  // 反向(右下→左上)
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x
    if (d[i] === 0) continue
    let v = d[i]
    if (x < w - 1) v = Math.min(v, d[i + 1] + 1)
    if (y < h - 1) v = Math.min(v, d[i + w] + 1)
    if (x < w - 1 && y < h - 1) v = Math.min(v, d[i + w + 1] + 1.4)
    if (x > 0 && y < h - 1) v = Math.min(v, d[i + w - 1] + 1.4)
    d[i] = v
  }
  return d
}

// 最近值平坦外延(单通道): 多源 BFS 波前扩散(O(N) 一次) — 最近值填充, 无"碗形"
function nearestFill(d, valid, w, h) {
  const out = d.slice()
  const known = valid.slice()
  const queue = new Int32Array(w * h)
  let head = 0, tail = 0
  for (let i = 0; i < w * h; i++) if (known[i]) queue[tail++] = i
  while (head < tail) {
    const i = queue[head++]
    const x = i % w, y = (i / w) | 0
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue
      const xx = Math.min(w - 1, Math.max(0, x + dx)), yy = Math.min(h - 1, Math.max(0, y + dy))
      const j = yy * w + xx
      if (!known[j]) { out[j] = out[i]; known[j] = 1; queue[tail++] = j }
    }
  }
  return out
}

export async function decompose(imageRGBA, depthGray, w0, h0, { aiInpaint = null, onStage, maskOverride = null, maxShiftRatio = 0.25 } = {}) {
  // ① pack 到 1024 长边
  const longSide = Math.max(w0, h0)
  const scale = Math.min(1, PACK_LONG_SIDE / longSide)
  const w = Math.max(8, Math.round(w0 * scale))
  const h = Math.max(8, Math.round(h0 * scale))
  const n = w * h
  const img = new Float32Array(n * 3)
  const d = new Float32Array(n)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const sx = Math.min(w0 - 1, Math.round(x / scale)), sy = Math.min(h0 - 1, Math.round(y / scale))
    const si = sy * w0 + sx, di = y * w + x
    img[di * 3] = imageRGBA[si * 4]; img[di * 3 + 1] = imageRGBA[si * 4 + 1]; img[di * 3 + 2] = imageRGBA[si * 4 + 2]
    d[di] = depthGray[si]
  }
  const dd = boxBlurGray(d, w, h, 1)
  // 1024² 有百万元素, 不能展开给 Math.max(栈溢出) — 循环求极值
  let dMax = -Infinity, dMin = Infinity
  for (let i = 0; i < n; i++) { if (dd[i] > dMax) dMax = dd[i]; if (dd[i] < dMin) dMin = dd[i] }
  const depthRange = dMax - dMin

  onStage && onStage('st:mask')
  // ② mask: 用户画笔修正优先(若提供) → Otsu + 最大连通域(比例越界 → 0.70 分位 fallback)
  let mask
  if (maskOverride) {
    // 用户 mask(RGBA, A 通道)直接下采样到 pack 尺寸
    mask = new Uint8Array(n)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const sx = Math.min(w0 - 1, Math.round(x / scale)), sy = Math.min(h0 - 1, Math.round(y / scale))
      mask[y * w + x] = maskOverride[sy * w0 + sx] > 128 ? 1 : 0
    }
  } else if (depthRange < 0.04) {
    mask = new Uint8Array(n)
  } else {
    const t = otsu(dd)
    mask = largestComponents(Uint8Array.from(dd, (v) => (v > t ? 1 : 0)), w, h, Math.round(n * 0.002))
  }
  let maskCount = mask.reduce((a, b) => a + b, 0)
  if (maskCount < n * MIN_MASK_FRACTION || maskCount > n * MAX_MASK_FRACTION) {
    const sorted = Array.from(dd).sort((a, b) => a - b)
    const cut = sorted[Math.min(n - 1, Math.floor(n * 0.7))]
    const fallback = largestComponents(Uint8Array.from(dd, (v) => (v > cut ? 1 : 0)), w, h, Math.round(n * 0.002))
    if (fallback.reduce((a, b) => a + b, 0) > n * 0.01) { mask = fallback; maskCount = fallback.reduce((a, b) => a + b, 0) }
  }
  let maskF = Float32Array.from(mask)

  onStage && onStage('st:close')
  // ③ 闭运算 + 洞填充("宁可多包")
  const closeR = Math.max(4, Math.min(32, Math.round(0.02 * Math.max(w, h))))
  let mf = filterMask(maskF, w, h, closeR, false)
  mf = filterMask(mf, w, h, closeR, true)
  mask = fillEnclosedBackground(Uint8Array.from(mf, (v) => v > 0.5), w, h)
  // 外扩 6px: 细发梢通常在深度阈值 mask 之外 → 落入背景层与前景位移不同 → 边缘撕裂。
  // "宁可多包"(前景边缘带 1-2px 背景色, 远好于发丝被撕成波纹)
  mask = Uint8Array.from(filterMask(Float32Array.from(mask), w, h, 6, false))
  maskF = Float32Array.from(mask)

  onStage && onStage('st:alpha')
  // ④ 前景 alpha = 硬遮罩不羽化(放大双线性自带 ~1px AA; 羽化 → 液态玻璃重影)
  const alpha = maskF

  onStage && onStage('st:prefill')
  // ⑤ push-pull 预填整个脚印(脚印 = mask 膨胀 2px; AI 看到的洞 = 整个脚印)
  const colors = new Float32Array(n * 3)
  for (let i = 0; i < n * 3; i++) colors[i] = img[i]
  // 补全区 = 前景位移扫过的全部区域: chamfer(mask) ≤ maxShiftPx
  // (脚印只膨胀 2px 时, 大位移会露出"原图人物残影"带)
  const maxShiftPx = Math.max(16, Math.round(maxShiftRatio * Math.min(w, h)))
  const distMap = chamferDist(mask, w, h)
  const aiHole = new Uint8Array(n)
  for (let i = 0; i < n; i++) aiHole[i] = distMap[i] <= maxShiftPx ? 1 : 0
  const validFoot = Float32Array.from(aiHole, (v) => (v ? 0 : 1))
  let fill = pushPull(colors, validFoot, w, h)

  onStage && onStage('st:ai')
  // ⑥ MI-GAN 只补脚印(洞小 → 内容实; 基于"背景状"预填上下文)
  let aiFilled = false
  if (aiInpaint) {
    onStage && onStage('MI-GAN 补全…')
    const out = await aiInpaint(fill, aiHole, w, h) // aiAdapter 是 async(MI-GAN ONNX run)
    if (out) { fill = out; aiFilled = true }
  }

  onStage && onStage('st:soften')
  // ⑦ 脚印内柔化(fill×0.35 + blur×0.65), 脚印外永远原图
  const softened = boxBlurRgb(fill, w, h, 2)
  const bgColor = fill.slice()
  for (let i = 0; i < n; i++) {
    if (aiHole[i]) {
      bgColor[i * 3] = fill[i * 3] * 0.35 + softened[i * 3] * 0.65
      bgColor[i * 3 + 1] = fill[i * 3 + 1] * 0.35 + softened[i * 3 + 1] * 0.65
      bgColor[i * 3 + 2] = fill[i * 3 + 2] * 0.35 + softened[i * 3 + 2] * 0.65
    } else {
      bgColor[i * 3] = img[i * 3]; bgColor[i * 3 + 1] = img[i * 3 + 1]; bgColor[i * 3 + 2] = img[i * 3 + 2]
    }
  }

  onStage && onStage('st:bgdepth')
  // ⑧ 背景层深度: nearestFill 平坦外延 × 0.3(push-pull 是碗形 → 透镜弯折; 最近值不弯)
  const validBg = Float32Array.from(aiHole, (v) => (v ? 0 : 1))
  const bgDepth = Float32Array.from(nearestFill(dd, validBg, w, h), (v) => v * BG_DEPTH_SCALE)

  onStage && onStage('st:fgdepth')
  // ⑨ 前景层深度: 遮罩内原始深度, 遮罩外 nearestFill 外延(无悬崖)
  const fgDepth = nearestFill(dd, Float32Array.from(maskF), w, h)

  onStage && onStage('st:output')
  // ⑩ 输出 RGBA
  const toRGBA = (rgb, gray) => {
    const o = new Uint8ClampedArray(n * 4)
    for (let i = 0; i < n; i++) {
      o[i * 4] = rgb ? rgb[i * 3] : gray[i] * 255
      o[i * 4 + 1] = rgb ? rgb[i * 3 + 1] : gray[i] * 255
      o[i * 4 + 2] = rgb ? rgb[i * 3 + 2] : gray[i] * 255
      o[i * 4 + 3] = 255
    }
    return o
  }
  const maskRGBA = new Uint8ClampedArray(n * 4)
  for (let i = 0; i < n; i++) maskRGBA[i * 4 + 3] = maskF[i] * 255 // A 通道(编辑器约定)
  return {
    w, h,
    bgImage: toRGBA(bgColor, null),
    bgDepth: toRGBA(null, bgDepth),
    fgImage: toRGBA(img, null),   // 前景颜色 = 原图(alpha 纹理控制显示范围)
    fgDepth: toRGBA(null, fgDepth),
    fgAlpha: toRGBA(null, alpha),
    maskRGBA,
    note: `主体 ${Math.round(maskCount / n * 100)}%${aiFilled ? ' · AI已补' : ' · 几何预填'}`,
  }
}
