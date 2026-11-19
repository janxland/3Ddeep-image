// <depth-viewer> — 景深壁纸 Web Component（三档质量模式，自动择优）
//
// 模式优先级（属性即开关）:
//   1. mesh-src  真几何多层补全 mesh（3d-photo-inpainting 产物转二进制）—— 无水滴感, 质量最高
//   2. depth-src 预计算深度 png —— 单层 warp + 前景刚性遮罩 + 引导双边滤波
//   3. 实时模式   访客浏览器内 DA-V2（质量阶梯 base-fp16→small-fp16→small-q8; WebGPU→WASM）
//                 + cache-endpoint 云端深度缓存池（KV Worker, 第一个访客算完所有人共享）
// 共同: 懒加载(可见才初始化, 同步检查兜底) / 小椭圆运镜(幅度自适应图幅比例) / 滚轮变焦 / I键补全区高亮 / F键翻转
// 事件: depth-ready {ms, device} / error {message}
// 依赖约定(对齐 music-component/ppt-component): transformers.js 不进包, CDN 动态 import + 多 CDN 容错

const TFS_VERSION = '3.7.6'
const TFS_CDNS = [
  'https://fastly.jsdelivr.net/npm/@huggingface/transformers@' + TFS_VERSION,
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@' + TFS_VERSION,
  'https://unpkg.com/@huggingface/transformers@' + TFS_VERSION,
]
const MODEL_HOSTS = ['https://hf-mirror.com', 'https://huggingface.co'] // 国内镜像优先, 失败回落官方
const DEPTH_CACHE = 'depth-viewer-depth-v1'

// ── 深度模型管线单例（页面多实例共享; 质量阶梯） ────────────────────
let tfsModule = null
const pipes = {}

const QUALITY_LADDER = [
  { id: 'onnx-community/depth-anything-v2-base', dtype: 'fp16', device: 'webgpu', bytes: 194.6e6 },
  { id: 'onnx-community/depth-anything-v2-small', dtype: 'fp16', device: 'webgpu', bytes: 49.6e6 },
  { id: 'onnx-community/depth-anything-v2-small', dtype: 'q8', device: 'wasm', bytes: 27.3e6 },
]

async function getTfs() {
  if (!tfsModule) {
    let lastErr = null
    // 加载顺序: 页面注入的本地 vendor（demo/内网场景, 绕 CDN）→ 三个公共 CDN
    const sources = [...(globalThis.__DEPTH_VIEWER_TFS__ ? [globalThis.__DEPTH_VIEWER_TFS__] : []), ...TFS_CDNS]
    for (const src of sources) {
      try {
        tfsModule = await import(/* @vite-ignore */ src)
        break
      } catch (e) { lastErr = e }
    }
    if (!tfsModule) throw lastErr || new Error('所有 CDN 均不可达')
    if (globalThis.__DEPTH_VIEWER_LOCAL_MODELS__) {
      // 同源本地模型 + 本地 wasm（完全离线, 无跨域限制）
      tfsModule.env.allowLocalModels = true
      tfsModule.env.localModelPath = globalThis.__DEPTH_VIEWER_LOCAL_MODELS__
      if (globalThis.__DEPTH_VIEWER_WASM__) tfsModule.env.backends.onnx.wasm.wasmPaths = globalThis.__DEPTH_VIEWER_WASM__
    } else {
      tfsModule.env.remoteHost = MODEL_HOSTS[0]
      tfsModule.env.allowLocalModels = false // 模型文件自带浏览器 Cache（回访免下载）
    }
  }
  return tfsModule
}

async function getPipe(onStage, quality) {
  const q = quality || 'auto'
  // 同源本地模式: 只有 small-q8 本地模型, 直接锁定
  const localOnly = !!globalThis.__DEPTH_VIEWER_LOCAL_MODELS__
  let ladder = localOnly ? [QUALITY_LADDER[2]] : QUALITY_LADDER
  if (q === 'fast') ladder = QUALITY_LADDER.slice(1)
  else if (q === 'best') ladder = [QUALITY_LADDER[0], QUALITY_LADDER[2]]
  if (!pipes[q]) {
    pipes[q] = (async () => {
      const tfs = await getTfs()
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

async function sha256hex(s) {
  const d = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s))
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, '0')).join('')
}

// 实时计算深度; 三级缓存: 本地 Cache API → 云端缓存池(cache-endpoint) → 实时计算
async function computeDepthLive(src, onStage, endpoint, quality) {
  const key = 'd/' + await sha256hex(src)
  // ① 本地 Cache API（回访 0 计算）
  const cacheUrl = 'depth-cache:' + src
  try {
    const hit = await caches.open(DEPTH_CACHE).then(c => c.match(cacheUrl))
    if (hit) {
      onStage && onStage('local-cache')
      return await createImageBitmap(await hit.blob())
    }
  } catch (_) { /* Cache API 不可用则跳过 */ }
  // ② 云端缓存池（第一个访客算完写入, 之后所有访客秒取）
  if (endpoint) {
    try {
      const r = await fetch(endpoint + '/' + key, { mode: 'cors' })
      if (r.ok) {
        onStage && onStage('cloud-cache')
        const blob = await r.blob()
        const bmp = await createImageBitmap(blob.slice(0))
        try {
          await (await caches.open(DEPTH_CACHE)).put(cacheUrl, new Response(blob, { headers: { 'Content-Type': 'image/png' } }))
        } catch (_) {}
        return bmp
      }
    } catch (_) { /* 云端不可达则实时算 */ }
  }
  // ③ 实时计算
  const pipe = await getPipe(onStage, quality)
  const resp = await fetch(src, { mode: 'cors' })
  if (!resp.ok) throw new Error('图片加载失败 ' + resp.status)
  const blob = await resp.blob()
  const tfs = await getTfs()
  const img = await tfs.RawImage.fromBlob(blob)
  const out = await pipe(img)
  // 灰度深度图(亮=近) → PNG blob → ImageBitmap + 写缓存
  const d = out.depth
  const cnv = document.createElement('canvas')
  cnv.width = d.width; cnv.height = d.height
  const ctx = cnv.getContext('2d')
  const idata = ctx.createImageData(d.width, d.height)
  for (let i = 0; i < d.width * d.height; i++) {
    const v = d.data[i * d.channels]
    idata.data[i * 4] = v; idata.data[i * 4 + 1] = v; idata.data[i * 4 + 2] = v; idata.data[i * 4 + 3] = 255
  }
  ctx.putImageData(idata, 0, 0)
  const dblob = await new Promise(res => cnv.toBlob(res, 'image/png'))
  try {
    await (await caches.open(DEPTH_CACHE)).put(cacheUrl, new Response(dblob.slice(0), { headers: { 'Content-Type': 'image/png' } }))
  } catch (_) { /* 忽略缓存失败 */ }
  if (endpoint) fetch(endpoint + '/' + key, { method: 'PUT', body: dblob.slice(0), headers: { 'Content-Type': 'image/png' } }).catch(() => {})
  return await createImageBitmap(dblob)
}

// ── 人像抠图管线（ModNet, 25.9MB; 语义分层用） ──────────────────────
let matteModelPromise = null
async function computeMatteLive(src, onStage) {
  const cacheUrl = 'matte-cache:' + src
  try {
    const hit = await caches.open(DEPTH_CACHE).then(c => c.match(cacheUrl))
    if (hit) return await createImageBitmap(await hit.blob())
  } catch (_) {}
  if (!matteModelPromise) {
    matteModelPromise = (async () => {
      const tfs = await getTfs()
      const device = typeof navigator.gpu !== 'undefined' ? 'webgpu' : 'wasm'
      onStage && onStage('加载人像分割模型…')
      const model = await tfs.AutoModel.from_pretrained('Xenova/modnet', { dtype: 'fp32', device })
      const processor = await tfs.AutoProcessor.from_pretrained('Xenova/modnet')
      return { model, processor, tfs }
    })()
    matteModelPromise.catch(() => { matteModelPromise = null })
  }
  const { model, processor, tfs } = await matteModelPromise
  const resp = await fetch(src, { mode: 'cors' })
  if (!resp.ok) throw new Error('图片加载失败 ' + resp.status)
  onStage && onStage('主体识别中…')
  const image = await tfs.RawImage.fromBlob(await resp.blob())
  const { pixel_values } = await processor(image)
  const { output } = await model({ input: pixel_values })
  const mask = await tfs.RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height)
  // → png blob 缓存
  const cnv = document.createElement('canvas')
  cnv.width = mask.width; cnv.height = mask.height
  const ctx = cnv.getContext('2d')
  const idata = ctx.createImageData(mask.width, mask.height)
  for (let i = 0; i < mask.width * mask.height; i++) {
    const v = mask.data[i * mask.channels]
    idata.data[i * 4] = v; idata.data[i * 4 + 1] = v; idata.data[i * 4 + 2] = v; idata.data[i * 4 + 3] = 255
  }
  ctx.putImageData(idata, 0, 0)
  const blob = await new Promise(res => cnv.toBlob(res, 'image/png'))
  try { await (await caches.open(DEPTH_CACHE)).put(cacheUrl, new Response(blob.slice(0), { headers: { 'Content-Type': 'image/png' } })) } catch (_) {}
  return await createImageBitmap(blob)
}

async function fetchDepthPng(url) {
  const resp = await fetch(url, { mode: 'cors' })
  if (!resp.ok) throw new Error('深度图加载失败 ' + resp.status)
  return await createImageBitmap(await resp.blob())
}

// ── WebGL 着色器 ──────────────────────────────────────────────
// warp 模式: 单层深度位移 + 引导双边滤波 + 前景刚性遮罩（缓解水滴感）
const WARP_VS = `attribute vec2 aPos; varying vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; vUv.y = 1.0-vUv.y; gl_Position = vec4(aPos,0.0,1.0); }`
const WARP_FS = `precision highp float;
uniform sampler2D uTex; uniform sampler2D uDepth;
uniform vec2 uMouse; uniform float uAmp; uniform float uZoom; uniform float uFgCut;
uniform vec2 uCoverScale; uniform vec2 uCoverOff; uniform vec2 uTexel;
varying vec2 vUv;
float guidedDepth(vec2 uv, vec2 guide) {
  float sum = 0.0; float wsum = 0.0001;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2 su = clamp(uv + vec2(float(dx), float(dy)) * uTexel * 1.5, vec2(0.001), vec2(0.999));
      vec3 cv = texture2D(uTex, su).rgb;
      float dv = texture2D(uDepth, su).r;
      vec2 e = cv.xy - guide;
      float w = exp(-14.0 * dot(e, e));
      sum += dv * w; wsum += w;
    }
  }
  return sum / wsum;
}
void main(){
  vec2 m = (vUv - 0.5) / uZoom + 0.5;
  vec2 iuv = m * uCoverScale + uCoverOff;
  float d = guidedDepth(iuv, texture2D(uTex, iuv).xy);
  float fg = smoothstep(uFgCut, uFgCut + 0.22, d);   // 前景刚性遮罩: 主体不参与形变(消除水滴游动感)
  vec2 base = (iuv - 0.5) * (1.0 - 2.5 * uAmp) + 0.5;
  vec2 disp = uMouse * uAmp * (1.0 - d) * (1.0 - fg);
  vec2 uv = clamp(base + disp, vec2(0.001), vec2(0.999));
  gl_FragColor = vec4(texture2D(uTex, uv).rgb, 1.0);
}`

// mesh 模式: 真几何多层补全网格（无水滴感, 背景为神经网络补全的实几何）
// 两个顶点格式: float32(16B stride) 与 uint16 量化(10B stride, bbox 反解)
const MESH_VS = `attribute vec3 aPos; attribute vec3 aCol; attribute float aFlag;
uniform mat4 uMVP; uniform float uFlipY;
varying vec3 vCol; varying float vFlag;
void main(){ vec3 p = aPos; p.y *= uFlipY; gl_Position = uMVP * vec4(p, 1.0); vCol = aCol; vFlag = aFlag; }`
const MESH_VS_Q = `attribute vec3 aPos; attribute vec3 aCol; attribute float aFlag;
uniform mat4 uMVP; uniform float uFlipY;
uniform vec3 uBMin; uniform vec3 uBSpan;
varying vec3 vCol; varying float vFlag;
void main(){
  vec3 p = uBMin + aPos * uBSpan;   // uint16 归一化坐标 → bbox 反解
  p.y *= uFlipY;
  gl_Position = uMVP * vec4(p, 1.0);
  vCol = aCol; vFlag = aFlag;
}`
const MESH_FS = `precision highp float;
varying vec3 vCol; varying float vFlag;
uniform float uShowInp;
void main(){
  vec3 c = vCol;
  if (uShowInp > 0.5 && vFlag >= 1.5) c = mix(c, vec3(1.0, 0.25, 0.2), 0.6);
  gl_FragColor = vec4(c, 1.0);
}`

// ── mat4 最小集（列主序; 相机看 +z, 世界 y 向下） ──────────────────
function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2)
  const A = (far + near) / (far - near), B = -2 * far * near / (far - near)
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, -f, 0, 0,
    0, 0, A, 1,
    0, 0, B, 0])
}
function mat4RotX(a) { const c = Math.cos(a), s = Math.sin(a)
  return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]) }
function mat4RotY(a) { const c = Math.cos(a), s = Math.sin(a)
  return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]) }
function mat4T(x, y, z) { return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]) }
function mat4Mul(a, b) {
  const o = new Float32Array(16)
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]
    o[c * 4 + r] = s
  }
  return o
}

class DepthViewer extends HTMLElement {
  static get observedAttributes() {
    return ['src', 'depth-src', 'mesh-src', 'cutout-src', 'layers-src', 'bg-src', 'amp', 'zoom-max', 'idle-seconds', 'quality', 'cache-endpoint', 'fg-cut']
  }

  constructor() {
    super()
    this._root = this.attachShadow({ mode: 'open' })
    this._root.innerHTML = `
      <style>
        :host{display:block;position:relative;background:#000;overflow:hidden;contain:content}
        canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:crosshair}
        img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}
        #st{position:absolute;top:8px;left:10px;color:#9fd;font:11px/1.5 monospace;
            text-shadow:0 1px 6px #000;pointer-events:none;user-select:none}
        #hint{position:absolute;left:50%;bottom:5vh;transform:translateX(-50%);color:#fff;
            font:12px/1.6 system-ui;opacity:.75;transition:opacity 1.2s;text-shadow:0 1px 8px #000;
            pointer-events:none;user-select:none;white-space:nowrap}
      </style>
      <canvas></canvas><div id="st"></div><div id="hint"></div>`
    this._canvas = this._root.querySelector('canvas')
    this._st = this._root.querySelector('#st')
    this._hint = this._root.querySelector('#hint')
    this._mx = 0; this._my = 0; this._tx = 0; this._ty = 0
    this._idle = 0; this._last = performance.now()
    this._zoom = 1; this._ready = false; this._started = false
    this._flipY = 1; this._showInp = 0; this._hovered = false
    this._raf = 0
    this._onFrame = this._frame.bind(this)
    this._onMove = (x, y) => {
      const r = this.getBoundingClientRect()
      this._tx = ((x - r.left) / Math.max(1, r.width)) * 2 - 1
      this._ty = ((y - r.top) / Math.max(1, r.height)) * 2 - 1
      this._idle = 0; this._hint.style.opacity = 0
    }
    this._onWheel = (e) => {
      const max = parseFloat(this.getAttribute('zoom-max')) || 3.5
      this._zoom = Math.min(max, Math.max(1, this._zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12)))
      this._idle = 0
    }
    this._onKey = (e) => {
      if (!this._hovered) return
      const k = e.key.toLowerCase()
      if (k === 'i') this._showInp = 1 - this._showInp
      if (k === 'f') this._flipY = -this._flipY
    }
  }

  connectedCallback() {
    this._io = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting) && !this._started) {
        this._started = true
        this._init()
      }
    }, { rootMargin: '200px' })
    this._io.observe(this)
    // 同步可见性检查（不依赖 IO/定时器/rAF —— 异常 webview 也即点即用）
    const r = this.getBoundingClientRect()
    if (r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < (window.innerHeight || 800)) {
      this._started = true
      this._init()
    }
    this._canvas.addEventListener('mousemove', e => this._onMove(e.clientX, e.clientY))
    this._canvas.addEventListener('mouseenter', () => { this._hovered = true })
    this._canvas.addEventListener('mouseleave', () => { this._hovered = false })
    this._canvas.addEventListener('touchmove', e => { if (e.touches[0]) this._onMove(e.touches[0].clientX, e.touches[0].clientY) }, { passive: true })
    this._canvas.addEventListener('wheel', this._onWheel, { passive: true })
    window.addEventListener('keydown', this._onKey)
    if (!this._raf) this._raf = requestAnimationFrame(this._onFrame)
  }

  disconnectedCallback() {
    this._io && this._io.disconnect()
    window.removeEventListener('keydown', this._onKey)
    cancelAnimationFrame(this._raf); this._raf = 0
  }

  attributeChangedCallback(name, oldV, newV) {
    if (oldV === newV || !this._started) return
    if (name === 'src' || name === 'depth-src' || name === 'mesh-src') {
      this._ready = false; this._init()
    }
  }

  async _init() {
    const meshSrc = this.getAttribute('mesh-src')
    const src = this.getAttribute('src')
    const cutSrc = this.getAttribute('cutout-src')
    if (this._loading) return
    if (!meshSrc && !cutSrc && !src) return
    this._loading = true
    this._ready = false
    this._st.style.color = '#9fd'
    this._st.textContent = '准备中…'
    try {
      const gl = this._canvas.getContext('webgl', { antialias: true, preserveDrawingBuffer: true })
        || this._canvas.getContext('experimental-webgl', { antialias: true, preserveDrawingBuffer: true })
      if (!gl) throw new Error('浏览器不支持 WebGL')
      this._gl = gl

      if (cutSrc) {
        // ── 抠图双层模式: 人物 RGBA 平面(近) ∥ LaMa 背景平面(远) ──
        this._mode = 'cutout'
        this._st.textContent = '加载双层资产…'
        this._setupCutout(gl)
        const layersSrc = this.getAttribute('layers-src')
        const layers = layersSrc ? await fetch(layersSrc).then(r => r.json()) : { personZ: 1.6, bgZ: 2.4, f: 960 }
        const [cb, cf] = await Promise.all([
          fetch(cutSrc).then(r => r.arrayBuffer()),
          fetch(this.getAttribute('bg-src') || '').then(r => r.ok ? r.arrayBuffer() : null),
        ])
        this._uploadCutout(gl, cutSrc, this.getAttribute('bg-src'), cb, cf, layers)
        this._ready = true
        this._st.textContent = '✔ 抠图双层 · 人物z' + layers.personZ + '/背景z' + layers.bgZ
      } else if (meshSrc) {
        // ── 真几何模式 ──
        this._mode = 'mesh'
        this._st.textContent = '加载真几何 mesh…'
        const meta = await fetch(meshSrc + '/web_meta.json').then(r => r.json())
        this._setupMesh(gl, meshSrc, meta)
        const fetchBuf = async (url) => {
          const r = await fetch(url, { mode: 'cors' })
          if (meta.compressed === 'deflate') return await new Response(r.body.pipeThrough(new DecompressionStream('deflate'))).arrayBuffer()
          return await r.arrayBuffer()
        }
        const [vb, fb] = await Promise.all([
          fetchBuf(meshSrc + '/web_verts.bin' + (meta.compressed === 'deflate' ? '.z' : '')),
          fetchBuf(meshSrc + '/web_faces.bin' + (meta.compressed === 'deflate' ? '.z' : '')),
        ])
        this._uploadMesh(gl, meta, vb, fb)
        this._t0 = performance.now()
        this._ready = true
        this._st.textContent = '✔ 真几何 ' + meta.nv.toLocaleString() + '顶点'
      } else {
        // ── 深度模式（depth-src 预计算 | 实时） ──
        this._mode = 'warp'
        this._setupWarp(gl)
        const t0 = performance.now()
        const depthSrc = this.getAttribute('depth-src')
        let device = 'precomputed'
        let depthBmp
        if (depthSrc) {
          this._st.textContent = '加载预计算深度…'
          depthBmp = await fetchDepthPng(depthSrc)
        } else {
          device = typeof navigator.gpu !== 'undefined' ? 'webgpu' : 'wasm'
          const endpoint = this.getAttribute('cache-endpoint') || ''
          const quality = this.getAttribute('quality') || 'auto'
          depthBmp = await computeDepthLive(src, (dev) => { this._st.textContent = '深度计算中(' + dev + ')…' }, endpoint, quality)
        }
        const ms = Math.round(performance.now() - t0)
        this._ms = ms; this._device = device
        // 渲染路径: layered(默认, 浏览器现场生成分层网格, 刚性无水滴) | warp(旧单层位移)
        if ((this.getAttribute('render') || 'layered') === 'layered') {
          this._mode = 'mesh'
          // 语义分层: matte-src 预计算 | ModNet 实时人像抠图 | 失败则纯深度
          let matteBmp = null
          const matteSrc = this.getAttribute('matte-src')
          if (matteSrc) matteBmp = await fetchDepthPng(matteSrc)
          else if (this.getAttribute('matte') !== 'off') {
            try { matteBmp = await computeMatteLive(src, (s) => { this._st.textContent = s }) } catch (e) { /* 无 matte 退化 */ }
          }
          const built = await this._buildLayered(depthBmp, src, matteBmp)
          this._setupMesh(gl, '', built.meta)
          this._uploadMesh(gl, built.meta, built.buffer, built.indices)
          this._st.textContent = '✔ ' + device + '+分层 ' + built.meta.nf.toLocaleString() + '面 · ' + ms + 'ms'
        } else {
          this._mode = 'warp'
          this._setupWarp(gl)
          const depthUrl = this._bmpToUrl(depthBmp)
          await Promise.all([this._loadTex(src, 0), this._loadTex(depthUrl, 1)])
          this._st.textContent = '✔ ' + device + ' · ' + ms + 'ms'
        }
        this._ready = true
      }
      setTimeout(() => { this._st.style.opacity = 0.4 }, 2500)
      this._hint.textContent = '滚轮变焦 · 移动鼠标环视 · I键补全区 · F键翻转 · 空闲自动运镜'
      this.dispatchEvent(new CustomEvent('depth-ready', {
        detail: { mode: this._mode, ms: this._ms || 0, device: this._device || 'mesh' }, bubbles: true,
      }))
    } catch (e) {
      this._st.textContent = '✘ ' + (e.message || e)
      this._st.style.color = '#f99'
      const fallback = src || this.getAttribute('src')
      if (fallback && !this._root.querySelector('img')) {
        const img = document.createElement('img')
        img.src = fallback
        this._root.insertBefore(img, this._canvas)
        this._canvas.remove()
      }
      this.dispatchEvent(new CustomEvent('error', { detail: { message: String(e.message || e) }, bubbles: true }))
    } finally {
      this._loading = false
    }
  }

  // 浏览器现场生成分层网格（mini-LDI: 深度→z 网格, 轮廓处沿小差对角撕裂, 顶点带色）
  // 浏览器现场生成分层网格（mini-LDI: 深度→z 网格, 轮廓处沿小差对角撕裂, 顶点带色）
  // 语义版: matte(1=人物) → ①主体深度压平 ②轮廓撕裂 ③背景填充层（低频均值色, z=人物后0.55）
  async _buildLayered(depthBmp, colorUrl, matteBmp) {
    const GMAX = 224
    const cim = await new Promise((res, rej) => {
      const im = new Image(); im.crossOrigin = 'anonymous'
      im.onload = () => res(im); im.onerror = () => rej(new Error('颜色图加载失败'))
      im.src = colorUrl
    })
    const aspect = cim.naturalWidth / cim.naturalHeight
    let GH = GMAX, GW = Math.round(GMAX * aspect)
    if (aspect > 1) { GW = GMAX; GH = Math.round(GMAX / aspect) }
    // 边缘外扩一圈: 环视时边缘露出拉伸背景而非黑边（数据 clamp 采样, 位置真实外延）
    const dc = document.createElement('canvas'); dc.width = GW; dc.height = GH
    const dctx = dc.getContext('2d', { willReadFrequently: true })
    dctx.drawImage(depthBmp, 0, 0, GW, GH)
    const dpx = dctx.getImageData(0, 0, GW, GH).data
    const cc = document.createElement('canvas'); cc.width = GW; cc.height = GH
    const cctx = cc.getContext('2d', { willReadFrequently: true })
    cctx.drawImage(cim, 0, 0, GW, GH)
    const cpx = cctx.getImageData(0, 0, GW, GH).data
    // ── 语义层: matte(1=人物) → 主体深度带压平 + 轮廓撕裂 + 背景填充层 ──
    let apx = null, fgD = 0.8, bgMean = [96, 88, 80]
    if (matteBmp) {
      const mc = document.createElement('canvas'); mc.width = GW; mc.height = GH
      const mctx = mc.getContext('2d', { willReadFrequently: true })
      mctx.drawImage(matteBmp, 0, 0, GW, GH)
      apx = mctx.getImageData(0, 0, GW, GH).data
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
    const K = parseInt(this.getAttribute('layers')) || 0 // layers=N: 深度量化 N 档更刚性
    const vb = new Uint8Array(nv * 2 * 16) // [0,nv)=主层 [nv,2nv)=背景填充层(仅主体区被索引)
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
      // 语义融合: 主体区深度压平到主体带（刚性）, 边缘 a 过渡自然撕出轮廓断层
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
      // 背景填充层顶点（低频均值色, 仅主体区索引引用）
      const j = nv + i
      vf[j * 4] = vf[i * 4]; vf[j * 4 + 1] = vf[i * 4 + 1]; vf[j * 4 + 2] = fillZ
      vb[j * 16 + 12] = bgMean[0]; vb[j * 16 + 13] = bgMean[1]; vb[j * 16 + 14] = bgMean[2]; vb[j * 16 + 15] = 1
    }
    const tris = []
    for (let gy = 0; gy < GH2 - 1; gy++) for (let gx = 0; gx < GW2 - 1; gx++) {
      const a = gy * GW2 + gx, b = a + 1, c = a + GW2, e = c + 1
      if ((dArr[a] - dArr[e]) ** 2 <= (dArr[b] - dArr[c]) ** 2) tris.push(a, c, b, b, c, e)
      else tris.push(a, e, b, a, c, e)
      // 背景填充层: 仅主体区 quad（绕到后层顶点段）
      if (apx && Math.max(aArr[a], aArr[b], aArr[c], aArr[e]) > 100) {
        const na = nv + a, nb = nv + b, nc = nv + c, ne = nv + e
        tris.push(na, nc, nb, nb, nc, ne)
      }
    }
    const zMid = (zN + zF) / 2
    const meta = {
      nv, nf: tris.length / 3, W: GW2, H: GH2,
      hFov: 2 * Math.atan(aspect * 0.5 / zMid), vFov: 2 * Math.atan(0.5 / zMid),
    }
    return { buffer: vb.buffer, indices: new Uint32Array(tris).buffer, meta }
  }

  _bmpToUrl(bmp) {
    const cnv = document.createElement('canvas')
    cnv.width = bmp.width; cnv.height = bmp.height
    cnv.getContext('2d').drawImage(bmp, 0, 0)
    return cnv.toDataURL('image/png')
  }

  // ── warp 程序（单层深度位移） ──────────────────────────────
  _setupWarp(gl) {
    const prog = this._makeProgram(gl, WARP_VS, WARP_FS)
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW)
    const aPos = gl.getAttribLocation(prog, 'aPos')
    gl.enableVertexAttribArray(aPos)
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0)
    this._u = {
      mouse: gl.getUniformLocation(prog, 'uMouse'),
      amp: gl.getUniformLocation(prog, 'uAmp'),
      zoom: gl.getUniformLocation(prog, 'uZoom'),
      fg: gl.getUniformLocation(prog, 'uFgCut'),
      cs: gl.getUniformLocation(prog, 'uCoverScale'),
      co: gl.getUniformLocation(prog, 'uCoverOff'),
      texel: gl.getUniformLocation(prog, 'uTexel'),
    }
    gl.uniform1i(gl.getUniformLocation(prog, 'uTex'), 0)
    gl.uniform1i(gl.getUniformLocation(prog, 'uDepth'), 1)
    this._fit = () => {
      const dpr = Math.min(2, devicePixelRatio || 1)
      const r = this.getBoundingClientRect()
      this._canvas.width = Math.max(2, Math.floor(r.width * dpr))
      this._canvas.height = Math.max(2, Math.floor(r.height * dpr))
      gl.viewport(0, 0, this._canvas.width, this._canvas.height)
      if (this._iw) {
        const s = Math.max(this._canvas.width / this._iw, this._canvas.height / this._ih)
        const fx = this._canvas.width / (s * this._iw), fy = this._canvas.height / (s * this._ih)
        gl.uniform2f(this._u.cs, fx, fy)
        gl.uniform2f(this._u.co, (1 - fx) / 2, (1 - fy) / 2)
      }
    }
    this._ro = new ResizeObserver(() => this._fit())
    this._ro.observe(this)
  }

  // ── mesh 程序（真几何） ────────────────────────────────────
  _setupMesh(gl, meshSrc, meta) {
    this._meshSrc = meshSrc
    this._quantized = meta.quantized === 'uint16'
    gl.getExtension('OES_element_index_uint') // WebGL1 下 100 万+ 顶点索引必需（WebGL2 原生支持）
    const vs = this._quantized ? MESH_VS_Q : MESH_VS
    const prog = this._makeProgram(gl, vs, MESH_FS)
    this._u = {
      mvp: gl.getUniformLocation(prog, 'uMVP'),
      flip: gl.getUniformLocation(prog, 'uFlipY'),
      inp: gl.getUniformLocation(prog, 'uShowInp'),
      bmin: gl.getUniformLocation(prog, 'uBMin'),
      bspan: gl.getUniformLocation(prog, 'uBSpan'),
    }
    this._fit = () => {
      const dpr = Math.min(2, devicePixelRatio || 1)
      const r = this.getBoundingClientRect()
      this._canvas.width = Math.max(2, Math.floor(r.width * dpr))
      this._canvas.height = Math.max(2, Math.floor(r.height * dpr))
      gl.viewport(0, 0, this._canvas.width, this._canvas.height)
    }
    this._ro = new ResizeObserver(() => this._fit())
    this._ro.observe(this)
  }

  _uploadMesh(gl, meta, vb, fb) {
    this._meta = meta
    this._nFaces = meta.nf
    const vbuf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, vbuf)
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(vb), gl.STATIC_DRAW)
    const prog = this._prog
    const aPos = gl.getAttribLocation(prog, 'aPos')
    const aCol = gl.getAttribLocation(prog, 'aCol')
    const aFlag = gl.getAttribLocation(prog, 'aFlag')
    gl.enableVertexAttribArray(aPos)
    gl.enableVertexAttribArray(aCol)
    gl.enableVertexAttribArray(aFlag)
    if (this._quantized) {
      // uint16 量化: stride 10B = qxyz(2B×3) + rgb(3B) + flag(1B); 顶点着色器内 bbox 反解
      gl.vertexAttribPointer(aPos, 3, gl.UNSIGNED_SHORT, true, 10, 0)
      gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, 10, 6)
      gl.vertexAttribPointer(aFlag, 1, gl.UNSIGNED_BYTE, false, 10, 9)
      const mn = meta.bboxMin, mx = meta.bboxMax
      gl.uniform3f(this._u.bmin, mn[0], mn[1], mn[2])
      gl.uniform3f(this._u.bspan, mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    } else {
      // float32: stride 16B
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 16, 0)
      gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, 16, 12)
      gl.vertexAttribPointer(aFlag, 1, gl.UNSIGNED_BYTE, false, 16, 15)
    }
    const ibuf = gl.createBuffer()
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(fb), gl.STATIC_DRAW)
    this._fit()
  }

  _makeProgram(gl, vs, fs) {
    const sh = (type, s) => {
      const o = gl.createShader(type)
      gl.shaderSource(o, s); gl.compileShader(o)
      if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(o))
      return o
    }
    const prog = gl.createProgram()
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs))
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs))
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog))
    gl.useProgram(prog)
    this._prog = prog
    return prog
  }

  _loadTex(url, unit) {
    return new Promise((res, rej) => {
      const gl = this._gl
      const im = new Image()
      im.crossOrigin = 'anonymous'
      im.onload = () => {
        const t = gl.createTexture()
        gl.activeTexture(gl.TEXTURE0 + unit)
        gl.bindTexture(gl.TEXTURE_2D, t)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, im)
        if (unit === 0) {
          this._iw = im.naturalWidth; this._ih = im.naturalHeight; this._fit()
          gl.uniform2f(this._u.texel, 1 / im.naturalWidth, 1 / im.naturalHeight)
        }
        res()
      }
      im.onerror = () => rej(new Error('贴图加载失败: ' + url))
      im.src = url
    })
  }

  _frame(t) {
    const idleSec = parseFloat(this.getAttribute('idle-seconds')) || 4
    const dt = Math.min(0.05, (t - this._last) / 1000); this._last = t; this._idle += dt
    const gl = this._gl
    if (!(this._ready && gl)) { this._raf = requestAnimationFrame(this._onFrame); return }
    const amp = parseFloat(this.getAttribute('amp')) || 0.045
    // 空闲自动椭圆（利萨如）+ 鼠标接管
    const kx = this._idle > idleSec ? Math.sin(t / 1000 * 0.6) * 0.85 : this._tx
    const ky = this._idle > idleSec ? Math.cos(t / 1000 * 0.45) * 0.65 : -this._ty
    this._mx += (kx - this._mx) * 0.07; this._my += (ky - this._my) * 0.07
    gl.clearColor(0.02, 0.02, 0.03, 1)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.enable(gl.DEPTH_TEST)
    const aspect = this._canvas.width / this._canvas.height

    if (this._mode === 'mesh' && this._meta) {
      // 小椭圆相机（幅度自适应图幅比例 = 各轴半FOV×35%）+ 长焦变焦
      const yawAmp = this._meta.hFov * 0.175, pitchAmp = this._meta.vFov * 0.175
      const yaw = this._mx * yawAmp, pitch = this._my * pitchAmp
      const v = mat4Mul(mat4T(-yaw * 0.35, -pitch * 0.25, 0), mat4Mul(mat4RotX(pitch), mat4RotY(yaw)))
      const p = mat4Perspective(this._meta.vFov / this._zoom, aspect, 0.05, 40)
      gl.useProgram(this._prog)
      gl.uniform1f(this._u.flip, this._flipY)
      gl.uniform1f(this._u.inp, this._showInp)
      gl.uniformMatrix4fv(this._u.mvp, false, mat4Mul(p, v))
      gl.drawElements(gl.TRIANGLES, this._nFaces * 3, gl.UNSIGNED_INT, 0)
    } else if (this._mode === 'warp') {
      gl.useProgram(this._prog)
      gl.uniform2f(this._u.mouse, this._mx, this._my)
      gl.uniform1f(this._u.amp, amp)
      gl.uniform1f(this._u.zoom, this._zoom)
      gl.uniform1f(this._u.fg, parseFloat(this.getAttribute('fg-cut')) || 0.55)
      gl.drawArrays(gl.TRIANGLES, 0, 6)
    }
    this._raf = requestAnimationFrame(this._onFrame)
  }
}

if (!customElements.get('depth-viewer')) customElements.define('depth-viewer', DepthViewer)
export default DepthViewer
