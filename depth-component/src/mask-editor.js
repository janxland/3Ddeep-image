// mask-editor.js — 人物抠图修正编辑器(解耦模块, 仅依赖 matte 数据流)
// 端内轨 ready 后可进入: 半透明 overlay 显示当前人物区, 画笔=加入 / 橡皮=移除,
// 应用 → 新 matteRGBA → 组件 7ms 重建分层网格, 即时看到修正效果。
// 数据模型: matte 数组(A 通道 = 人物值)是唯一事实源; overlay 只是可视化, 每笔后从 matte 重绘。
// 与组件的契约: createMaskEditor({ root, GW, GH, matte: Uint8ClampedArray, onApply(newMatte) })
export function createMaskEditor({ root, GW, GH, matte, onApply }) {
  const view = document.createElement('canvas')
  view.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:none;cursor:crosshair;touch-action:none'
  view.width = GW; view.height = GH
  const vctx = view.getContext('2d')

  const bar = document.createElement('div')
  bar.style.cssText = `position:absolute;left:50%;bottom:12px;transform:translateX(-50%);display:none;gap:6px;
    padding:6px 10px;border-radius:12px;background:rgba(10,14,24,.82);border:1px solid rgba(148,163,184,.35);
    align-items:center;user-select:none;font:12px/1 system-ui;color:#dbe4f0;backdrop-filter:blur(8px)`
  const btn = (label, title, fn) => {
    const b = document.createElement('button')
    b.textContent = label; b.title = title
    b.style.cssText = 'border:0;background:rgba(148,163,184,.16);color:#eef;font:12px system-ui;padding:5px 10px;border-radius:8px;cursor:pointer'
    b.addEventListener('click', fn)
    return b
  }
  const size = document.createElement('input')
  size.type = 'range'; size.min = '6'; size.max = '60'; size.value = '24'
  size.style.cssText = 'width:90px;accent-color:#7dd0ff'
  const modeLabel = document.createElement('span')
  modeLabel.textContent = '画笔'
  bar.append(btn('✏ 画笔', '涂抹加入人物区', () => { tool = 'add'; modeLabel.textContent = '画笔' }),
    btn('◻ 橡皮', '擦除人物区', () => { tool = 'erase'; modeLabel.textContent = '橡皮' }),
    size, modeLabel,
    btn('↩ 撤销', '撤销上一笔', undo),
    btn('✔ 应用', '应用修正并重建场景', apply),
    btn('✕', '放弃修正', cancel))
  root.append(view, bar)

  let tool = 'add'
  let drawing = false
  let last = null
  const undoStack = []
  let liveMatte = null
  let origMatte = null // 进入编辑时的快照, cancel 回滚用

  // 从 matte(A 通道)重绘可视化: 人物区红色 45%
  function repaint() {
    vctx.clearRect(0, 0, GW, GH)
    const im = vctx.createImageData(GW, GH)
    for (let i = 0; i < GW * GH; i++) {
      const a = liveMatte[i * 4 + 3]
      if (a > 8) {
        im.data[i * 4] = 255; im.data[i * 4 + 1] = 40; im.data[i * 4 + 2] = 60
        im.data[i * 4 + 3] = Math.round(a * 0.45)
      }
    }
    vctx.putImageData(im, 0, 0)
  }

  // 笔刷: 直接写 matte A 通道(圆形区域), matte 是唯一事实源
  function stroke(p0, p1) {
    const r = +size.value / 2
    const steps = Math.max(1, Math.ceil(Math.hypot(p1.x - p0.x, p1.y - p0.y) / (r * 0.6)))
    for (let s = 0; s <= steps; s++) {
      const cx = p0.x + (p1.x - p0.x) * (s / steps), cy = p0.y + (p1.y - p0.y) * (s / steps)
      const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(GW - 1, Math.ceil(cx + r))
      const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(GH - 1, Math.ceil(cy + r))
      const rr = r * r
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const dx = x - cx, dy = y - cy
        if (dx * dx + dy * dy <= rr) liveMatte[(y * GW + x) * 4 + 3] = tool === 'add' ? 255 : 0
      }
    }
    matte.set(liveMatte)
    repaint()
  }

  function pt(e) {
    const r = view.getBoundingClientRect()
    return {
      x: Math.round(((e.clientX - r.left) / Math.max(1, r.width)) * GW),
      y: Math.round(((e.clientY - r.top) / Math.max(1, r.height)) * GH),
    }
  }

  view.addEventListener('pointerdown', (e) => {
    if (!drawing) undoStack.push(new Uint8ClampedArray(liveMatte))
    if (undoStack.length > 12) undoStack.shift()
    drawing = true
    view.setPointerCapture(e.pointerId)
    last = pt(e)
    stroke(last, last)
  })
  view.addEventListener('pointermove', (e) => {
    if (!drawing) return
    const p = pt(e)
    stroke(last, p)
    last = p
  })
  view.addEventListener('pointerup', () => { drawing = false })
  view.addEventListener('wheel', (e) => e.preventDefault(), { passive: false }) // 编辑中不让页面滚

  function undo() {
    const prev = undoStack.pop()
    if (!prev) return
    liveMatte.set(prev)
    matte.set(prev)
    repaint()
  }

  function enter() {
    liveMatte = new Uint8ClampedArray(matte)
    origMatte = new Uint8ClampedArray(matte)
    undoStack.length = 0
    repaint()
    view.style.display = 'block'
    bar.style.display = 'flex'
  }

  function apply() {
    view.style.display = 'none'
    bar.style.display = 'none'
    onApply(new Uint8ClampedArray(liveMatte))
  }

  function cancel() {
    matte.set(origMatte) // 回滚到进入编辑前
    view.style.display = 'none'
    bar.style.display = 'none'
  }

  return { enter, destroy: () => { view.remove(); bar.remove() } }
}
