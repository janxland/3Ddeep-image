// occlusion.js — SLIDE 遮挡掩码(Jampani ICCV 2021)纯函数模块, 零依赖
// 论文: 硬规则 D(x,y) − D(xi,yj) > ρ·K(xi,yj)(K=欧氏距离)→ 软化 S=tanh(γ·max(...))
// 高效实现: 逐行/列单向扫描, O(N) 均摊——每个背景像素只与"最近前景像素"比较一次。
// 输入: 深度场(Float32Array 0..1 亮=近)+ 前景 mask(Uint8Array 0/1)
// 输出: 补全掩码 Uint8Array(0/255, 背景侧被前景遮挡的区域 = 待生成区)
export function occlusionMask(depthF, fgMask, GW, GH, { rho = 0.14, gamma = 4.0, dilate = 2 } = {}) {
  const N = GW * GH
  const occ = new Uint8Array(N)
  const strength = new Float32Array(N)
  // 单向行扫描(左→右 + 右→左): 背景像素与其行内最近前景像素比深度
  // 前景像素行进时携带 (深度, 距离); 背景像素检查 D_bg − D_fg > ρ·dist
  const scanRow = (y, dir) => {
    let fgd = -1, fgx = -1
    for (let step = 0; step < GW; step++) {
      const x = dir > 0 ? step : GW - 1 - step
      const i = y * GW + x
      if (fgMask[i]) { fgd = depthF[i]; fgx = x; continue }
      if (fgd < 0) continue
      const dist = Math.abs(x - fgx)
      if (dist > 40) { fgd = -1; continue } // 超出最大视线距离重置
      const diff = fgd - depthF[i] // 论文语义: D前景 − D背景 > ρ·K(前景 disparity 更大, 背景被遮挡); 我们的深度 0..1 亮=近 = disparity 语义一致
      if (diff > rho * dist) {
        const s = Math.tanh(gamma * (diff - rho * dist))
        if (s > strength[i]) strength[i] = s
      }
    }
  }
  const scanCol = (x, dir) => {
    let fgd = -1, fgy = -1
    for (let step = 0; step < GH; step++) {
      const y = dir > 0 ? step : GH - 1 - step
      const i = y * GW + x
      if (fgMask[i]) { fgd = depthF[i]; fgy = y; continue }
      if (fgd < 0) continue
      const dist = Math.abs(y - fgy)
      if (dist > 40) { fgd = -1; continue }
      const diff = fgd - depthF[i]
      if (diff > rho * dist) {
        const s = Math.tanh(gamma * (diff - rho * dist))
        if (s > strength[i]) strength[i] = s
      }
    }
  }
  for (let y = 0; y < GH; y++) { scanRow(y, 1); scanRow(y, -1) }
  for (let x = 0; x < GW; x++) { scanCol(x, 1); scanCol(x, -1) }
  // 强度 → 二值 + dilate(窄带)
  for (let i = 0; i < N; i++) if (strength[i] > 0.15) occ[i] = 255
  if (dilate > 0) {
    for (let round = 0; round < dilate; round++) {
      const cur = occ.slice()
      for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
        const i = y * GW + x
        if (cur[i]) continue
        for (let dy = -1; dy <= 1 && !occ[i]; dy++) for (let dx = -1; dx <= 1 && !occ[i]; dx++) {
          const xx = Math.min(GW - 1, Math.max(0, x + dx)), yy = Math.min(GH - 1, Math.max(0, y + dy))
          if (cur[yy * GW + xx]) occ[i] = 255
        }
      }
    }
  }
  // 前景本身不算补全区
  for (let i = 0; i < N; i++) if (fgMask[i]) occ[i] = 0
  return occ
}
