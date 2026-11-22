// <depth-scene> — 一图转 3D 景深原生组件(对齐 ppt-component / music-component 交付模式)
//
// 用法(页面零配置, 单 script 即用):
//   <script src="depth-scene.js"></script>
//   <depth-scene src="photo.jpg"></depth-scene>            → 端内轨: DA-V2 推理 + 分层网格(WebGPU→WASM)
//   <depth-scene src="a.jpg" depth-src="a.depth.png"></depth-scene> → 预计算深度轨
//   <depth-scene mesh-src="mesh/w167-3"></depth-scene>     → 真几何资产轨(meshopt/deflate bins, 质量上限)
//   <depth-scene src="a.jpg" render="warp"></depth-scene>  → 轻量单层位移
// 属性: src / depth-src / mesh-src / amp / zoom-max / idle-seconds / quality(auto|fast|best)
//        cache-endpoint(云端深度缓存池 KV Worker) / fg-cut / layers / render(mesh|warp)
// 事件: d3d-progress {stage} / d3d-ready {mode, ms, device, verts, faces} / d3d-error {message}
//        (bubbles + composed)  另支持拖放图片到组件上即转
// 隔离: Shadow DOM(open) + --d3d-* CSS 变量主题穿透(music-playlist 同款约定)
// 并行: 推理/网格构建/meshopt 解码跑 inline Worker, 主线程只做 DOM/IO/rAF; Worker 不可用时自动回退
import PipelineWorker from './gpu-pipeline.js?worker&inline'
import { createRenderer, mat4Perspective, mat4RotX, mat4RotY, mat4T, mat4Mul } from './render-core.js'
import { loadBins } from './asset-bins.js'
import { inferDepth, buildLayeredMeshCore, decodeMeshopt } from './gpu-pipeline.js'
import { decompose } from './decomposer.js'
import { createMaskEditor } from './mask-editor.js'
import { createMigan } from './migan.js'

const SELF_URL = (document.currentScript && document.currentScript.src) || ''
const TFS_VERSION = '3.7.6'
const TFS_CDNS = [
  'https://fastly.jsdelivr.net/npm/@huggingface/transformers@' + TFS_VERSION,
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@' + TFS_VERSION,
  'https://unpkg.com/@huggingface/transformers@' + TFS_VERSION,
]
const MODEL_HOSTS = 'https://huggingface.co'
const ORT_VERSION = '1.21.0'
const ORT_CDNS = [
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@' + ORT_VERSION + '/dist/ort.webgpu.mjs',
  'https://fastly.jsdelivr.net/npm/onnxruntime-web@' + ORT_VERSION + '/dist/ort.webgpu.mjs',
]
const MIGAN_MODEL_SOURCES = [
  'https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx',
  'https://hf-mirror.com/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx',
] // hf-mirror 无 CORS 头, 浏览器 fetch 会被拦 — 必须官方源
const DEPTH_CACHE = 'depth-scene-v1'
const GMAX = 224 // 现场分层网格分辨率上限

// vendor 源(页面可注入 __DEPTH_SCENE_TFS__ / __DEPTH_SCENE_MODELS__ / __DEPTH_SCENE_WASM__ 覆盖)
function vendorCfg(quality) {
  const injected = globalThis.__DEPTH_SCENE_TFS__
  const modelsLocal = globalThis.__DEPTH_SCENE_MODELS__ || ''
  return {
    tfsSources: [...(injected ? [injected] : []), ...TFS_CDNS],
    modelsLocal: modelsLocal || null,
    wasmLocal: globalThis.__DEPTH_SCENE_WASM__ || null,
    remoteHost: MODEL_HOSTS,
    quality: quality || 'auto',
    miganOrt: ORT_CDNS,
    miganModel: MIGAN_MODEL_SOURCES,
    miganCache: 'depth-scene-models',
  }
}

async function sha256hex(s) {
  const d = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s))
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, '0')).join('')
}

// 主线程像素解码(canvas 2d, willReadFrequently)
async function pixelData(url, w, h) {
  const blob = await fetch(url, { mode: 'cors' }).then(r => { if (!r.ok) throw new Error('加载失败 ' + r.status + ': ' + url); return r.blob() })
  const bmp = await createImageBitmap(blob)
  const cnv = document.createElement('canvas')
  cnv.width = w; cnv.height = h
  const ctx = cnv.getContext('2d', { willReadFrequently: true })
  ctx.drawImage(bmp, 0, 0, w, h)
  return ctx.getImageData(0, 0, w, h).data
}

class DepthScene extends HTMLElement {
  static get observedAttributes() {
    return ['src', 'depth-src', 'mesh-src', 'amp', 'zoom-max', 'idle-seconds', 'quality', 'cache-endpoint', 'fg-cut', 'layers', 'render']
  }

  constructor() {
    super()
    this._root = this.attachShadow({ mode: 'open' })
    this._root.innerHTML = `
      <style>
        :host{display:block;position:relative;overflow:hidden;contain:content;
              background:var(--d3d-surface, transparent);
              border-radius:var(--d3d-radius, 0)}
        canvas{position:absolute;inset:0;width:100%;height:100%;display:block;
               cursor:var(--d3d-cursor, crosshair)}
        img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}
        #st{position:absolute;top:8px;left:10px;color:var(--d3d-status, #9fd);
            font:11px/1.5 monospace;text-shadow:0 1px 6px rgba(0,0,0,.8);
            pointer-events:none;user-select:none;transition:opacity .8s}
        #hint{position:absolute;left:50%;bottom:5vh;transform:translateX(-50%);
            color:var(--d3d-hint, #fff);font:12px/1.6 system-ui;opacity:.75;transition:opacity 1.2s;
            text-shadow:0 1px 8px rgba(0,0,0,.8);pointer-events:none;user-select:none;white-space:nowrap}
        /* 启动门: 默认封面+按钮, 点击才开始加载(内容流多组件/低配机器友好) */
        #gate{position:absolute;inset:0;display:flex;flex-direction:column;gap:14px;
              align-items:center;justify-content:center;cursor:pointer;border:0;
              background:transparent;color:#fff;font:14px/1.6 system-ui;
              text-shadow:0 2px 10px rgba(0,0,0,.7)}
        #gate img{opacity:.92}
        #gate .play{width:58px;height:58px;border-radius:50%;display:flex;align-items:center;justify-content:center;
              background:rgba(10,14,24,.55);border:1px solid rgba(255,255,255,.5);
              backdrop-filter:blur(6px);font-size:22px;transition:transform .2s, background .2s}
        #gate:hover .play{transform:scale(1.08);background:rgba(40,90,180,.6)}
        #gate .urlrow{display:flex;gap:6px;margin-top:2px;width:min(420px, 90%)}
        #gate .urlrow input{flex:1;padding:8px 10px;border-radius:9px;border:1px solid rgba(255,255,255,.4);
              background:rgba(10,14,24,.72);color:#fff;font:12px system-ui;outline:none;
              backdrop-filter:blur(6px)}
        #gate .urlrow input:focus{border-color:rgba(125,208,255,.8)}
        #gate .urlrow button{padding:8px 12px;border-radius:9px;border:1px solid rgba(125,208,255,.6);
              background:rgba(40,90,180,.55);color:#fff;font:12px system-ui;cursor:pointer;white-space:nowrap}
        #gate.hide{display:none}
      </style>
      <canvas></canvas><div id="st"></div><div id="hint"></div><button id="gate"></button>
      <button id="edit" style="position:absolute;right:10px;top:8px;display:none;border:0;
        background:rgba(10,14,24,.6);color:#dbe4f0;font:11px system-ui;padding:4px 9px;
        border-radius:8px;cursor:pointer;backdrop-filter:blur(6px)">✏ 修正抠图</button>`
    this._canvas = this._root.querySelector('canvas')
    this._st = this._root.querySelector('#st')
    this._ver = 'v20261122-fix'
    this._hint = this._root.querySelector('#hint')
    this._gate = this._root.querySelector('#gate')
    this._activated = false // 启动门: 点击封面/autoplay 才真正加载
    this._gate.addEventListener('click', () => this._activate())
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
      if (!this._ready) return // 未加载: 不拦截
      e.preventDefault() // 组件上一律拦截: 滚轮只做变焦, 绝不带动文章滚动
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
    this._onDrop = (e) => {
      e.preventDefault()
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]
      if (f && /^image\//.test(f.type)) {
        const url = URL.createObjectURL(f)
        this.setAttribute('src', url)
        this.removeAttribute('depth-src'); this.removeAttribute('mesh-src')
      }
    }
    this._onDrag = (e) => { e.preventDefault() }
  }

  connectedCallback() {
    this._io = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting) && !this._started) {
        if (!this._maybeStart()) this._deployGate()
      }
    }, { rootMargin: '200px' })
    this._io.observe(this)
    // 同步可见性检查(不依赖 IO/定时器 —— 异常 webview 也即点即用)
    const r = this.getBoundingClientRect()
    if (r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < (window.innerHeight || 800)) {
      if (!this._maybeStart()) this._deployGate()
    } else {
      // IO 回调不可靠的 webview 兜底: 短轮询直到可见
      const tryStart = () => {
        if (this._started || this._gateDeployed) return
        const rr = this.getBoundingClientRect()
        if (rr.width > 0 && rr.height > 0 && rr.bottom > 0 && rr.top < (window.innerHeight || 800)) {
          this._gateDeployed = true
          if (!this._maybeStart()) this._deployGate()
          return
        }
        if ((this._tries = (this._tries || 0) + 1) < 200) setTimeout(tryStart, 120)
      }
      setTimeout(tryStart, 100)
    }
    this._canvas.addEventListener('mousemove', e => this._onMove(e.clientX, e.clientY))
    this._canvas.addEventListener('mouseenter', () => { this._hovered = true; this._lockScroll(this._ready) })
    this._canvas.addEventListener('mouseleave', () => { this._hovered = false; this._lockScroll(false) })
    this._canvas.addEventListener('touchmove', e => { if (e.touches[0]) this._onMove(e.touches[0].clientX, e.touches[0].clientY) }, { passive: true })
    this.addEventListener('wheel', this._onWheel, { passive: false }) // 组件全域拦截(冒泡捕获)
    this.addEventListener('drop', this._onDrop)
    this.addEventListener('dragover', this._onDrag)
    // 粘贴图片(Ctrl+V): clipboard 的图片 blob 直接端内轨 — 零网络依赖, 免疫一切 CORS/防盗链
    this._onPaste = (e) => {
      const items = (e.clipboardData && e.clipboardData.items) || []
      for (const it of items) {
        if (it.type && it.type.startsWith('image/')) {
          e.preventDefault()
          const blob = it.getAsFile()
          if (!blob) return
          const url = URL.createObjectURL(blob)
          this._gate && this._gate.classList.add('hide')
          this._activated = true
          this.removeAttribute('depth-src'); this.removeAttribute('mesh-src')
          this.setAttribute('src', url)
          this._maybeStart()
          return
        }
      }
    }
    window.addEventListener('paste', this._onPaste)
    window.addEventListener('keydown', this._onKey)
    if (!this._raf) this._raf = requestAnimationFrame(this._onFrame)
  }

  // ── 滚动锁: 鼠标在组件内时, 任何 scroll 事件立即回弹到锁定位置 —
  // 物理兜底(preventDefault 被站点脚本绕过时, 页面也无法移动) ──
  _lockScroll(on) {
    if (on) {
      this._lockY = window.scrollY
      if (!this._scrollLock) {
        this._scrollLock = () => window.scrollTo(0, this._lockY)
        window.addEventListener('scroll', this._scrollLock, { passive: true })
      }
      window.scrollTo(0, this._lockY)
    } else if (this._scrollLock) {
      window.removeEventListener('scroll', this._scrollLock)
      this._scrollLock = null
    }
  }

  disconnectedCallback() {
    this._io && this._io.disconnect()
    window.removeEventListener('keydown', this._onKey)
    window.removeEventListener('paste', this._onPaste)
    this._lockScroll(false)
    cancelAnimationFrame(this._raf); this._raf = 0
  }

  // 可见性/自动启动逻辑: autoplay=true → 可见即加载; 否则布置封面等待点击
  _maybeStart() {
    if (this._started) return
    if (this.getAttribute('autoplay') === 'true' || this._activated) {
      this._started = true
      this._init()
      return true
    }
    return false
  }

  // 启动门: 布置封面(有 src 用图, 无 src 用纯提示), 点击后进入加载
  _deployGate() {
    if (this._gateDeployed && this._gateBuiltFor === this.getAttribute('src')) return
    this._gateDeployed = true
    this._gateBuiltFor = this.getAttribute('src')
    const src = this.getAttribute('src')
    const cover = src ? `<img src="${src}" alt="">` : ''
    const label = this.getAttribute('mesh-src') ? '点击加载 3D 真几何场景' : '点击加载 3D 景深(首次需下载深度模型)'
    this._gate.innerHTML = `${cover}<div class="play">▶</div><div>${label}</div>
      <div class="urlrow"><input id="gateurl" type="text" placeholder="或粘贴图片 URL 直接实验"><button id="gateurlgo">转 3D</button></div>`
    const inp = this._gate.querySelector('#gateurl')
    const goBtn = this._gate.querySelector('#gateurlgo')
    for (const el of [inp, goBtn]) {
      el.addEventListener('click', (e) => e.stopPropagation())
      el.addEventListener('keydown', (e) => e.stopPropagation())
      el.addEventListener('wheel', (e) => e.stopPropagation())
    }
    const submit = () => this._loadUrl(inp.value)
    goBtn.addEventListener('click', submit)
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit() })
  }

  // 外部图片 URL → CORS 代理(wsrv.nl: 加 CORS 头 + 限宽 1920)→ 端内轨加载
  _loadUrl(url) {
    url = String(url || '').trim()
    if (!/^https?:\/\//.test(url)) return
    this._gate.classList.add('hide')
    this._activated = true
    // 同源 imgproxy(nginx → wsrv → 目标图): 外部图无 CORS 也能 fetch, 服务器出海取图
    let proxied = 'https://www.roginx.ink/imgproxy/?url=' + encodeURIComponent(url.replace(/^https?:\/\//, '')) + '&w=1920'
    this.removeAttribute('depth-src'); this.removeAttribute('mesh-src')
    this.setAttribute('src', proxied)
    this._maybeStart()
  }

  _activate() {
    if (this._activated) return
    this._activated = true
    this._gate.classList.add('hide')
    this._maybeStart()
  }

  attributeChangedCallback(name, oldV, newV) {
    if (oldV === newV || !this._started) {
      if (!this._started && name === 'src') this._deployGate() // 未激活时更新封面
      return
    }
    if (name === 'autoplay' && newV === 'true' && !this._started) { this._maybeStart(); return }
    if (name === 'src' || name === 'depth-src' || name === 'mesh-src') {
      this._ready = false
      this._init()
    }
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail, bubbles: true, composed: true }))
  }
  _progress(stage) {
    this._st.style.opacity = 1
    this._st.textContent = stage
    this._emit('d3d-progress', { stage })
  }

  // ── Worker 编排: job 协议; 构造失败/初始化超时 → 主线程回退(同一份管线函数) ──
  _rpc(type, payload, transfer, timeoutMs, onStage) {
    if (!this._worker) return this._runLocal(type, payload, onStage)
    return new Promise((res, rej) => {
      const job = ++this._jobSeq
      const timer = timeoutMs ? setTimeout(() => { this._jobs.delete(job); rej(new Error(type + ' 超时')) }, timeoutMs) : 0
      this._jobs.set(job, { res, rej, timer, onStage })
      try { this._worker.postMessage({ type, job, ...payload }, transfer || []) }
      catch (e) { this._jobs.delete(job); this._worker = null; this._runLocal(type, payload, onStage).then(res, rej) }
    })
  }
  async _ensureWorker() {
    if (this._workerState) return this._workerState
    this._workerState = (async () => {
      const jobs = new Map()
      const w = new PipelineWorker() // inline blob worker; IAB 等异常环境抛错 → 回退
      w.onmessage = (e) => {
        const { job, ok, stage, error, ...rest } = e.data
        const j = jobs.get(job)
        if (!j) return
        if (stage !== undefined) { j.onStage && j.onStage(stage); return }
        if (j.timer) clearTimeout(j.timer)
        jobs.delete(job)
        if (error) j.rej(new Error(error))
        else j.res(rest)
      }
      w.onerror = () => { /* worker 崩溃: 后续 rpc 失败自然回退 */ }
      this._worker = w; this._jobs = jobs; this._jobSeq = 0
      await this._rpc('init', { cfg: vendorCfg(this.getAttribute('quality')) }, null, 6000)
      return w
    })()
    try { await this._workerState } catch (e) {
      this._worker = null; this._jobs = new Map(); this._jobSeq = 0
      this._workerState = null // 回退主线程
    }
    return this._workerState
  }
  async _runLocal(type, payload, onStage) {
    if (type === 'init') return { ok: true }
    if (type === 'inferDepth') {
      const t0 = performance.now()
      const blob = await inferDepth(payload.url, vendorCfg(this.getAttribute('quality')), onStage)
      return { blob, ms: Math.round(performance.now() - t0) }
    }
    if (type === 'buildMesh') {
      const t0 = performance.now()
      const out = buildLayeredMeshCore(payload.pack)
      return { ...out, ms: Math.round(performance.now() - t0) }
    }
    if (type === 'decodeBins') {
      const out = await decodeMeshopt(payload)
      return { ...out }
    }
    throw new Error('未知任务 ' + type)
  }

  // ── 深度获取三级缓存: 本地 Cache API → 云端缓存池 → worker 实时推理 ──
  async _depthLive(src, onStage) {
    const key = 'd/' + await sha256hex(src)
    const cacheUrl = 'depth-cache:' + src
    try {
      const hit = await caches.open(DEPTH_CACHE).then(c => c.match(cacheUrl))
      if (hit) { onStage('local-cache'); return await hit.blob() }
    } catch (_) { /* Cache API 不可用则跳过 */ }
    const endpoint = this.getAttribute('cache-endpoint') || ''
    if (endpoint) {
      try {
        const r = await fetch(endpoint + '/' + key, { mode: 'cors' })
        if (r.ok) {
          onStage('cloud-cache')
          const blob = await r.blob()
          try { await (await caches.open(DEPTH_CACHE)).put(cacheUrl, blob.slice(0)) } catch (_) {}
          return blob
        }
      } catch (_) { /* 云端不可达则实时算 */ }
    }
    await this._ensureWorker()
    let imgBlob = null
    try { imgBlob = await this._fetchImageBlob(src) } catch (_) { /* worker 自行 fetch 兜底 */ }
    const payload = imgBlob ? { blob: imgBlob, url: new URL(src, location.href).href } : { url: new URL(src, location.href).href }
    const out = await this._rpc('inferDepth', payload, null, 0, onStage)
    const blob = out.blob
    try { await (await caches.open(DEPTH_CACHE)).put(cacheUrl, blob.slice(0)) } catch (_) {}
    if (endpoint) fetch(endpoint + '/' + key, { method: 'PUT', body: blob.slice(0), headers: { 'Content-Type': 'image/png' } }).catch(() => {})
    return blob
  }

  // 主线程 fetch 图片(同源直连; 跨域走 imgproxy/wsrv 代理链)——blob 传 worker
  // (blob worker 的 fetch 受 IAB 网络限制, 统一在主线程取图)
  async _fetchImageBlob(src) {
    const tries = [src]
    if (/^https?:/.test(src) && !src.includes(location.hostname)) {
      tries.push('https://www.roginx.ink/imgproxy/?url=' + encodeURIComponent(src.replace(/^https?:\/\//, '')) + '&w=1920')
      tries.push('https://images.weserv.nl/?url=' + encodeURIComponent(src.replace(/^https?:\/\//, '')) + '&w=1920')
    }
    let lastErr = null
    for (const u of tries) {
      try {
        const r = await fetch(u, { mode: 'cors' })
        if (!r.ok) throw new Error('HTTP ' + r.status)
        return await r.blob()
      } catch (e) { lastErr = e }
    }
    throw lastErr || new Error('图片加载失败')
  }

  // ── 主流程 ──
  async _init() {
    const meshSrc = this.getAttribute('mesh-src')
    const src = this.getAttribute('src')
    if (this._loading) return
    if (!meshSrc && !src) return
    this._loading = true
    this._ready = false
    this._st.style.color = 'var(--d3d-status, #9fd)'
    this._st.textContent = this._ver + ' 加载中…'
    const t0 = performance.now()
    try {
      const gl = this._canvas.getContext('webgl', { antialias: true, preserveDrawingBuffer: true })
        || this._canvas.getContext('experimental-webgl', { antialias: true, preserveDrawingBuffer: true })
      if (!gl) throw new Error('浏览器不支持 WebGL')
      this._R = createRenderer(this._canvas, gl)

      if (meshSrc) {
        // ── 真几何资产轨(meshopt / deflate bins) ──
        this._progress('加载真几何 mesh…')
        await this._ensureWorker()
        const { meta, verts, faces } = await loadBins(meshSrc, {
          rpc: (type, payload, transfer) => this._rpc(type, payload, transfer, 0, s => this._progress(s + '…')),
          onProgress: (s) => this._progress(s + '…'),
        })
        this._R.setMesh(meta)
        this._R.uploadMesh(verts, faces)
        if (src) { // 背景兜底平面: 撕缝黑条全部被原图兜住
          try { this._R.setBackTex(await createImageBitmap(await fetch(src, { mode: 'cors' }).then(r => r.blob()))) } catch (_) {}
        }
        this._fit()
        this._mode = 'mesh'
        this._detail = { mode: 'mesh', device: 'asset', verts: meta.nv, faces: meta.nf }
      } else {
        // ── 深度轨(depth-src 预计算 | 实时推理) → mesh(默认) / warp ──
        const depthSrc = this.getAttribute('depth-src')
        let depthBlob = null
        let device = 'precomputed'
        if (depthSrc) {
          this._progress('加载预计算深度…')
          depthBlob = await fetch(depthSrc, { mode: 'cors' }).then(r => { if (!r.ok) throw new Error('深度图加载失败 ' + r.status); return r.blob() })
        } else {
          device = typeof navigator.gpu !== 'undefined' ? 'webgpu' : 'wasm'
          const inferT = performance.now()
          depthBlob = await this._depthLive(src, (s) => this._progress('深度计算中(' + s + ')…'))
          this._inferMs = Math.round(performance.now() - inferT)
        }
        this._device = device

        if ((this.getAttribute('render') || 'mesh') === 'warp') {
          // warp: 深度图直接当纹理, 引导滤波在着色器里
          this._mode = 'warp'
          const dUrl = URL.createObjectURL(depthBlob)
          const cb = await fetch(src, { mode: 'cors' }).then(r => r.blob()).catch(() => this._fetchImageBlob(src))
          const [cBmp, dBmp] = await Promise.all([createImageBitmap(cb), createImageBitmap(depthBlob)])
          this._iw = cBmp.width; this._ih = cBmp.height
          this._R.setWarp(cBmp, dBmp, [1 / cBmp.width, 1 / cBmp.height])
          this._fit()
        } else {
          // photo 两层范式(对齐 SoftLayerDecomposer): 硬遮罩 + push-pull 预填 + (MI-GAN)脚印补全
          // + nearestFill 深度外延 — 片元逐像素位移(零网格伪影), 静止合成 = 原图
          this._progress('两层软分解…')
          const cim = await createImageBitmap(await fetch(src, { mode: 'cors' }).then(r => r.blob()))
          const aspect = cim.width / cim.height
          const longSide = Math.max(cim.width, cim.height)
          const pw = longSide > 1024 ? Math.round(cim.width * 1024 / longSide) : cim.width
          const ph = longSide > 1024 ? Math.round(cim.height * 1024 / longSide) : cim.height
          const [ipx, dpx] = await Promise.all([
            this._px(cim, pw, ph),
            this._px(depthBlob, pw, ph),
          ])
          const depthGray = new Float32Array(pw * ph)
          for (let i = 0; i < pw * ph; i++) depthGray[i] = dpx[i * 4] / 255
          await this._ensureWorker()
          this._lastPack = { imageRGBA: ipx, depthGray, w0: pw, h0: ph, generative: this.getAttribute('generative') === 'true' }
          const out = await this._rpc('decompose', {
            imageRGBA: ipx, depthGray, w0: pw, h0: ph,
            generative: this._lastPack.generative,
            maxShiftRatio: (parseFloat(this.getAttribute('amp')) || 0.28) * 0.92,
          }, null, 0, s => this._progress(s))
          this._R.setPhoto(out, pw / ph, pw, ph)
          this._fit()
          this._mode = 'photo'
          this._maskRGBA = out.maskRGBA
          this._detail = { mode: 'photo', device, inferMs: this._inferMs || 0, buildMs: out.ms, note: out.note }
        }
        this._detail = this._detail || { mode: this._mode, device, inferMs: this._inferMs || 0 }
      }
      this._ready = true
      if (this._hovered) this._lockScroll(true) // 鼠标已在组件内: 立即上锁
      const totalMs = Math.round(performance.now() - t0)
      this._ms = totalMs
      this._st.textContent = this._stText()
      // photo 模式: 显示画笔修正入口(mask 编辑 → decompose 重建)
      const editBtn = this._root.querySelector('#edit')
      if (this._mode === 'photo' && this._maskRGBA) {
        editBtn.style.display = 'block'
        editBtn.onclick = () => {
          if (!this._editor) {
            this._editor = createMaskEditor({
              root: this._root,
              GW: this._lastPack.w0, GH: this._lastPack.h0,
              matte: this._maskRGBA,
              onApply: (newMatte) => this._rebuildWithMatte(newMatte),
            })
          }
          this._editor.enter()
        }
      } else {
        editBtn.style.display = 'none'
      }
      setTimeout(() => { this._st.style.opacity = 0.4 }, 2500)
      this._hint.textContent = '滚轮变焦 · 移动鼠标环视 · I键补全区 · F键翻转 · 拖入图片即转'
      this._emit('d3d-ready', { ...this._detail, ms: totalMs })
    } catch (e) {
      this._st.textContent = '✘ ' + (e.message || e) + (e.stack ? ' @' + String(e.stack).split('\n')[1].trim().slice(0, 80) : '')
      this._st.style.color = 'var(--d3d-error, #f99)'
      if (src && !this._root.querySelector('img')) {
        const img = document.createElement('img')
        img.src = src
        this._root.insertBefore(img, this._canvas)
        this._canvas.remove()
      }
      this._emit('d3d-error', { message: String(e.message || e) })
    } finally {
      this._loading = false
    }
  }

  _stText() {
    const d = this._detail || {}
    if (d.mode === 'mesh' && d.faces) {
      const parts = []
      if (d.device === 'asset') parts.push('真几何')
      else { parts.push(d.device + '+分层'); if (d.inferMs) parts.push('推理' + d.inferMs + 'ms'); if (d.buildMs) parts.push('网格' + d.buildMs + 'ms') }
      return '✔ ' + parts.join(' ') + ' · ' + d.faces.toLocaleString() + '面'
    }
    return '✔ ' + (d.device || '') + ' · ' + (this._ms || 0) + 'ms'
  }

  async _px(srcOrBlob, w, h) {
    // 接受 Blob / ImageBitmap / URL —— ImageBitmap 直接用(否则 fetch(ImageBitmap) 会 404 空响应)
    const bmp = srcOrBlob instanceof Blob ? await createImageBitmap(srcOrBlob)
      : (srcOrBlob && typeof srcOrBlob !== 'string' ? srcOrBlob
        : await createImageBitmap(await fetch(srcOrBlob, { mode: 'cors' }).then(r => { if (!r.ok) throw new Error('加载失败 ' + r.status + ': ' + srcOrBlob); return r.blob() })))
    const cnv = document.createElement('canvas')
    cnv.width = w; cnv.height = h
    const ctx = cnv.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(bmp, 0, 0, w, h)
    return ctx.getImageData(0, 0, w, h).data
  }

  _fit() {
    const dpr = Math.min(2, devicePixelRatio || 1)
    const r = this.getBoundingClientRect()
    const w = Math.max(2, Math.floor(r.width * dpr)), h = Math.max(2, Math.floor(r.height * dpr))
    this._R.setViewport(w, h)
    if (this._R.kind === 'warp' && this._iw) {
      const s = Math.max(w / this._iw, h / this._ih)
      const fx = w / (s * this._iw), fy = h / (s * this._ih)
      this._R.gl.useProgram(this._R.warpProg)
      this._R.gl.uniform2f(this._R.u.cs, fx, fy)
      this._R.gl.uniform2f(this._R.u.co, (1 - fx) / 2, (1 - fy) / 2)
    }
  }

  _frame(t) {
    const idleSec = parseFloat(this.getAttribute('idle-seconds')) || 4
    const dt = Math.min(0.05, (t - this._last) / 1000); this._last = t; this._idle += dt
    const R = this._R
    if (!(this._ready && R)) { this._raf = requestAnimationFrame(this._onFrame); return }
    // 空闲自动椭圆(利萨如) + 鼠标接管
    const kx = this._idle > idleSec ? Math.sin(t / 1000 * 0.6) * 0.85 : this._tx
    const ky = this._idle > idleSec ? Math.cos(t / 1000 * 0.45) * 0.65 : -this._ty
    this._mx += (kx - this._mx) * 0.07; this._my += (ky - this._my) * 0.07
    const aspect = this._canvas.width / this._canvas.height
    R.clear()
    if (R.kind === 'photo' && R.photoTex) {
      // photo 两层(PhotoProgram 对齐): 背景层(bgDepth 位移) → 前景层(POM 连续深度)
      const cam = [this._mx, this._my]
      const parallax = parseFloat(this.getAttribute('amp')) || 0.28
      const fgGamma = parseFloat(this.getAttribute('depth-gamma')) || 0.55
      R.drawPhotoLayer({ imageTex: R.photoTex.bgImage, depthTex: R.photoTex.bgDepth, alphaTex: R.photoTex.fgAlpha }, cam, parallax, { useAlpha: false })
      R.drawPhotoLayer({ imageTex: R.photoTex.fgImage, depthTex: R.photoTex.fgDepth, alphaTex: R.photoTex.fgAlpha }, cam, parallax, { pom: true, height: 0.3, steady: 0.5, gamma: fgGamma })
    } else if (R.kind === 'mesh' && R.meta) {
      // 小椭圆相机(幅度 = 各轴半FOV×35%) + 长焦变焦
      const yawAmp = R.meta.hFov * 0.175, pitchAmp = R.meta.vFov * 0.175
      const yaw = this._mx * yawAmp, pitch = this._my * pitchAmp
      const v = mat4Mul(mat4T(-yaw * 0.35, -pitch * 0.25, 0), mat4Mul(mat4RotX(pitch), mat4RotY(yaw)))
      const p = mat4Perspective(R.meta.vFov / this._zoom, aspect, 0.05, 40)
      R.drawBack(mat4Mul(p, v)) // 背景兜底平面: 撕缝黑条被原图兜住
      R.drawMesh(mat4Mul(p, v), this._flipY, this._showInp)
    } else if (R.kind === 'warp') {
      const amp = parseFloat(this.getAttribute('amp')) || 0.045
      R.drawWarp(this._mx, this._my, amp, this._zoom, parseFloat(this.getAttribute('fg-cut')) || 0.55)
    }
    this._raf = requestAnimationFrame(this._onFrame)
  }
}

if (!customElements.get('depth-scene')) customElements.define('depth-scene', DepthScene)
export default DepthScene
