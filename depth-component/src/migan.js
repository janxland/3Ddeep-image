// migan.js — MI-GAN 端内图像补全封装(ICCV 2023 Picsart, MIT; pipeline ONNX 26.8MB)
// 解耦: 仅暴露 createMigan({ ortSources, modelSources, cacheName }) → { inpaint }
// 模型: andraniksargsyan/migan pipeline 版(内嵌 bbox crop + 512 resize + 羽化合成, 动态 H/W)
// 输入约定: image (1,3,H,W) uint8 NCHW + mask (1,1,H,W) uint8(255=known, 0=hole 待生成)
// 输出: result (1,3,H,W) uint8 已合成(原区域保留 + 生成区替换)
const ORT_VERSION = '1.21.0'

async function loadOrt(sources) {
  for (const src of sources) {
    try { return await import(/* @vite-ignore */ src) } catch (e) { /* 下一个源 */ }
  }
  throw new Error('onnxruntime-web 所有源不可达')
}

async function fetchModelCached(url, cacheName, onProgress) {
  try {
    const hit = await caches.open(cacheName).then(c => c.match(url))
    if (hit) return await hit.arrayBuffer()
  } catch (_) { /* Cache API 不可用则直接下载 */ }
  const resp = await fetch(url, { mode: 'cors' })
  if (!resp.ok) throw new Error('模型下载失败 ' + resp.status)
  const buf = await resp.arrayBuffer()
  try { await (await caches.open(cacheName)).put(url, new Response(buf.slice(0))) } catch (_) {}
  return buf
}

export async function createMigan({ ortSources, modelSources, cacheName = 'depth-scene-models', onProgress } = {}) {
  const ort = await loadOrt(ortSources)
  if (ort.env?.wasm?.wasmPaths === undefined) {
    // wasm runtime 文件与 js 入口同源(第一个可达 CDN)
    const base = (ortSources.find(Boolean) || '').replace(/ort[^/]*\.mjs.*$/, '')
    if (base) ort.env.wasm.wasmPaths = base
  }
  let lastErr = null
  let session = null
  for (const url of modelSources) {
    try {
      onProgress && onProgress('下载补全模型(26.8MB)…')
      const buf = await fetchModelCached(url, cacheName, onProgress)
      onProgress && onProgress('初始化补全模型…')
      // 强制 WASM: uint8 NCHW 在 WebGPU EP 下输出崩坏(彩虹噪声实测), WASM CPU 一次推理 ~500ms 足够
      session = await ort.InferenceSession.create(new Uint8Array(buf),
        { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
      break
    } catch (e) { lastErr = e }
  }
  if (!session) throw lastErr || new Error('MI-GAN 会话创建失败')

  // RGBA(全图) + maskRGBA(A 通道: 255=hole 待生成) → MI-GAN(NCHW uint8 平面布局!, mask 0=hole) → 补全 RGBA
  async function inpaint(imageRGBA, maskRGBA, w, h) {
    const plane = w * h
    const image = new Uint8Array(3 * plane) // NCHW: R 平面 + G 平面 + B 平面
    const mask = new Uint8Array(plane)
    let holes = 0
    for (let i = 0; i < plane; i++) {
      image[i] = imageRGBA[i * 4]; image[plane + i] = imageRGBA[i * 4 + 1]; image[2 * plane + i] = imageRGBA[i * 4 + 2]
      const hole = maskRGBA[i * 4 + 3] > 128 ? 0 : 255 // 0=hole, 255=known
      mask[i] = hole
      if (!hole) holes++
    }
    if (holes < 64) return null // 没有有效补全区
    const t0 = performance.now()
    const results = await session.run({
      image: new ort.Tensor('uint8', image, [1, 3, h, w]),
      mask: new ort.Tensor('uint8', mask, [1, 1, h, w]),
    })
    const res = results.result.data // (1,3,h,w) uint8 NCHW
    const outRGBA = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < plane; i++) {
      outRGBA[i * 4] = res[i]; outRGBA[i * 4 + 1] = res[plane + i]; outRGBA[i * 4 + 2] = res[2 * plane + i]; outRGBA[i * 4 + 3] = 255
    }
    return { rgba: outRGBA, ms: Math.round(performance.now() - t0) }
  }

  return { inpaint }
}
