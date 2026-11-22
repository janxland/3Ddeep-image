// gpu-pipeline.js — CPU 密集管线, 默认跑在 inline Worker, 主线程回退共用同一份函数
// 职责: ① 深度推理(ORT Web: WebGPU→WASM SIMD, 质量阶梯) ② mini-LDI 网格构建 ③ meshopt 资产解码
// 编排(缓存层/IO)在 depth-scene.js; 本文件只做纯计算。config 由调用方注入, 不读 globalThis。

import { MeshoptDecoder } from 'meshoptimizer/decoder' // wasm 内嵌自包含(32KB), 打进 bundle 零外部依赖

export const QUALITY_LADDER = [
  { id: 'onnx-community/depth-anything-v2-base', dtype: 'fp16', device: 'webgpu', bytes: 194.6e6 },
  { id: 'onnx-community/depth-anything-v2-small', dtype: 'fp16', device: 'webgpu', bytes: 49.6e6 },
  { id: 'onnx-community/depth-anything-v2-small', dtype: 'q8', device: 'wasm', bytes: 27.3e6 },
]

let tfsModule = null
const pipes = {}

// transformers.js 加载: cfg.tfsSources 顺序尝试(注入变量 → 脚本同目录 vendor → 3 CDN)
async function getTfs(cfg) {
  if (!tfsModule) {
    let lastErr = null
    for (const src of cfg.tfsSources) {
      try { tfsModule = await import(/* @vite-ignore */ src); break } catch (e) { lastErr = e }
    }
    if (!tfsModule) throw lastErr || new Error('transformers.js 所有源不可达')
    if (cfg.modelsLocal) {
      tfsModule.env.allowLocalModels = true
      tfsModule.env.localModelPath = cfg.modelsLocal
      if (cfg.wasmLocal) tfsModule.env.backends.onnx.wasm.wasmPaths = cfg.wasmLocal
    } else {
      tfsModule.env.remoteHost = cfg.remoteHost || 'https://hf-mirror.com'
      tfsModule.env.allowLocalModels = false // 模型文件自带浏览器 Cache(回访免下载)
    }
  }
  return tfsModule
}

// 质量阶梯择优: 同源本地模型锁定 q8; fast 跳过 base; best 只 base→q8
async function getPipe(cfg, onStage) {
  const q = cfg.quality || 'auto'
  let ladder
  if (cfg.modelsLocal) ladder = [QUALITY_LADDER[2]] // 同源本地模式: 只有 q8, 无视 quality 锁定
  else if (q === 'fast') ladder = QUALITY_LADDER.slice(1)
  else if (q === 'best') ladder = [QUALITY_LADDER[0], QUALITY_LADDER[2]]
  else ladder = QUALITY_LADDER
  if (!pipes[q]) {
    pipes[q] = (async () => {
      const tfs = await getTfs(cfg)
      let lastErr = null
      for (const step of ladder) {
        try {
          onStage && onStage(step.device + '·' + step.dtype + '·' + Math.round(step.bytes / 1e6) + 'MB')
          return await tfs.pipeline('depth-estimation', step.id, { dtype: step.dtype, device: step.device })
        } catch (e) { lastErr = e }
      }
      throw lastErr || new Error('模型加载失败')
    })()
    pipes[q].catch(() => { delete pipes[q] }) // 失败允许重试
  }
  return pipes[q]
}

// ① 端内推理: fetch 图 → DA-V2 → 灰度深度 PNG blob(亮=近)。worker/主线程通用。
export async function inferDepth(url, cfg, onStage) {
  const pipe = await getPipe(cfg, onStage)
  const resp = await fetch(url, { mode: 'cors' })
  if (!resp.ok) throw new Error('图片加载失败 ' + resp.status)
  const tfs = await getTfs(cfg)
  const img = await tfs.RawImage.fromBlob(await resp.blob())
  const out = await pipe(img)
  const d = out.depth
  const cnv = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(d.width, d.height)
    : Object.assign(document.createElement('canvas'), { width: d.width, height: d.height })
  const ctx = cnv.getContext('2d')
  const idata = ctx.createImageData(d.width, d.height)
  for (let i = 0; i < d.width * d.height; i++) {
    const v = d.data[i * d.channels]
    idata.data[i * 4] = v; idata.data[i * 4 + 1] = v; idata.data[i * 4 + 2] = v; idata.data[i * 4 + 3] = 255
  }
  ctx.putImageData(idata, 0, 0)
  return typeof cnv.convertToBlob === 'function'
    ? await cnv.convertToBlob({ type: 'image/png' })
    : await new Promise(res => cnv.toBlob(res, 'image/png'))
}

// ② mini-LDI 网格构建(纯 TypedArray 计算): 深度→z 网格, 轮廓小差对角撕裂, 顶点带色
// 输入像素包: colorRGBA / depthRGBA (可选 matteRGBA 1=前景), 输出 {buffer, indices, meta}
export function buildLayeredMeshCore({ colorRGBA, depthRGBA, matteRGBA, GW, GH, K }) {
  const cpx = colorRGBA, dpx = depthRGBA, apx = matteRGBA || null
  const aspect = GW / GH
  let fgD = 0.8, bgMean = [96, 88, 80]
  if (apx) {
    let r = 0, g = 0, b = 0, n = 0; const ds = []
    for (let i = 0; i < GW * GH; i++) {
      const a = apx[i * 4]
      if (a < 64) { r += cpx[i * 4]; g += cpx[i * 4 + 1]; b += cpx[i * 4 + 2]; n++ }
      if (a > 210) ds.push(dpx[i * 4] / 255)
    }
    if (n > 0) bgMean = [r / n, g / n, b / n]
    if (ds.length) { ds.sort((x, y) => x - y); fgD = ds[Math.floor(ds.length * 0.9)] }
  }
  const GW2 = GW + 2, GH2 = GH + 2, nv = GW2 * GH2
  const zN = 1.2, zF = 2.6
  const vb = new Uint8Array(nv * 2 * 16) // [0,nv)=主层 [nv,2nv)=背景填充层(仅主体区索引)
  const vf = new Float32Array(vb.buffer)
  const dArr = new Float32Array(nv)
  const aArr = new Uint8Array(nv)
  const fgZ = zF - fgD * (zF - zN)
  const fillZ = Math.min(zF + 0.6, fgZ + 0.55) // 背景填充层: 主体后 0.55
  for (let gy2 = 0; gy2 < GH2; gy2++) for (let gx2 = 0; gx2 < GW2; gx2++) {
    const gx = Math.min(GW - 1, Math.max(0, gx2 - 1)), gy = Math.min(GH - 1, Math.max(0, gy2 - 1))
    const si = gy * GW + gx, i = gy2 * GW2 + gx2
    let d = dpx[si * 4] / 255
    if (K > 0) d = Math.floor(d * K + 0.5) / K
    // 语义融合: 主体区深度压平到主体带(刚性), 边缘 a 过渡自然撕出轮廓断层
    let a = 0
    if (apx) {
      const a0 = apx[si * 4] / 255
      a = Math.min(1, Math.max(0, (a0 - 0.45) / 0.35))
      d = d * (1 - a) + fgD * a * 0.92
    }
    dArr[i] = d; aArr[i] = apx ? apx[si * 4] : 0
    vf[i * 4] = ((gx2 + 0.5) / GW2 - 0.5) * aspect * (GW2 / GW)
    vf[i * 4 + 1] = ((gy2 + 0.5) / GH2 - 0.5) * (GH2 / GH)
    vf[i * 4 + 2] = zF - d * (zF - zN)
    vb[i * 16 + 12] = cpx[si * 4]; vb[i * 16 + 13] = cpx[si * 4 + 1]; vb[i * 16 + 14] = cpx[si * 4 + 2]; vb[i * 16 + 15] = 1
    const j = nv + i
    vf[j * 4] = vf[i * 4]; vf[j * 4 + 1] = vf[i * 4 + 1]; vf[j * 4 + 2] = fillZ
    vb[j * 16 + 12] = bgMean[0]; vb[j * 16 + 13] = bgMean[1]; vb[j * 16 + 14] = bgMean[2]; vb[j * 16 + 15] = 1
  }
  const tris = []
  for (let gy = 0; gy < GH2 - 1; gy++) for (let gx = 0; gx < GW2 - 1; gx++) {
    const a = gy * GW2 + gx, b = a + 1, c = a + GW2, e = c + 1
    if ((dArr[a] - dArr[e]) ** 2 <= (dArr[b] - dArr[c]) ** 2) tris.push(a, c, b, b, c, e)
    else tris.push(a, e, b, a, c, e)
    if (apx && Math.max(aArr[a], aArr[b], aArr[c], aArr[e]) > 100) {
      const na = nv + a, nb = nv + b, nc = nv + c, ne = nv + e
      tris.push(na, nc, nb, nb, nc, ne)
    }
  }
  const zMid = (zN + zF) / 2
  return {
    buffer: vb.buffer,
    indices: new Uint32Array(tris).buffer,
    meta: { nv, nf: tris.length / 3, W: GW2, H: GH2, hFov: 2 * Math.atan(aspect * 0.5 / zMid), vFov: 2 * Math.atan(0.5 / zMid) },
  }
}

// ③ meshopt 资产解码(顶点 + 索引); decoder 打进 bundle, worker/主线程共用
export async function decodeMeshopt({ verts, faces, vStride, nv, nf }) {
  await MeshoptDecoder.ready
  const vOut = new Uint8Array(nv * vStride)
  MeshoptDecoder.decodeVertexBuffer(vOut, nv, vStride, new Uint8Array(verts))
  const fOut = new Uint8Array(nf * 3 * 4)
  MeshoptDecoder.decodeIndexBuffer(fOut, nf * 3, 4, new Uint8Array(faces))
  return { verts: vOut.buffer, faces: fOut.buffer }
}

// ── Worker 入口(vite ?worker&inline 打包; 主线程回退不经过这里) ──
/* eslint-disable no-undef */
if (typeof self !== 'undefined' && typeof self.postMessage === 'function' && typeof importScripts === 'function') {
  let CFG = null
  self.onmessage = async (e) => {
    const { type, job } = e.data
    try {
      if (type === 'init') { CFG = e.data.cfg; return self.postMessage({ job, ok: true }) }
      if (type === 'inferDepth') {
        const t0 = performance.now()
        const blob = await inferDepth(e.data.url, CFG, (stage) => self.postMessage({ job, stage }))
        self.postMessage({ job, blob, ms: Math.round(performance.now() - t0) })
      } else if (type === 'buildMesh') {
        const t0 = performance.now()
        const out = buildLayeredMeshCore(e.data.pack)
        self.postMessage({ job, ...out, ms: Math.round(performance.now() - t0) },
          [out.buffer, out.indices])
      } else if (type === 'decodeBins') {
        const t0 = performance.now()
        const out = await decodeMeshopt(e.data)
        self.postMessage({ job, ...out, ms: Math.round(performance.now() - t0) }, [out.verts, out.faces])
      }
    } catch (err) {
      self.postMessage({ job, error: String(err && err.message || err), stack: String(err && err.stack ? err.stack.split('\n').slice(0, 4).join(' | ') : '') })
    }
  }
}
