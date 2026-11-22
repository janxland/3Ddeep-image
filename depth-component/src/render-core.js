// render-core.js — 纯 WebGL 渲染器, 零 DOM 依赖(canvas 可来自主线程或 OffscreenCanvas)
// 职责单一: 着色器/矩阵/缓冲上传/绘制。DOM、事件、属性、加载全部在 depth-scene.js。

// ── warp 着色器: 单层深度位移 + 引导双边滤波 + 前景刚性遮罩(消水滴感) ──
export const WARP_VS = `attribute vec2 aPos; varying vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; vUv.y = 1.0-vUv.y; gl_Position = vec4(aPos,0.0,1.0); }`
export const WARP_FS = `precision highp float;
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
  float fg = smoothstep(uFgCut, uFgCut + 0.22, d);
  vec2 base = (iuv - 0.5) * (1.0 - 2.5 * uAmp) + 0.5;
  vec2 disp = uMouse * uAmp * (1.0 - d) * (1.0 - fg);
  vec2 uv = clamp(base + disp, vec2(0.001), vec2(0.999));
  gl_FragColor = vec4(texture2D(uTex, uv).rgb, 1.0);
}`

// ── mesh 着色器: 真几何多层网格; 两种顶点格式 f32(16B) / uint16 量化(10B, bbox 反解) ──
export const MESH_VS = `attribute vec3 aPos; attribute vec3 aCol; attribute float aFlag;
uniform mat4 uMVP; uniform float uFlipY;
varying vec3 vCol; varying float vFlag;
void main(){ vec3 p = aPos; p.y *= uFlipY; gl_Position = uMVP * vec4(p, 1.0); vCol = aCol; vFlag = aFlag; }`
export const MESH_VS_Q = `attribute vec3 aPos; attribute vec3 aCol; attribute float aFlag;
uniform mat4 uMVP; uniform float uFlipY;
uniform vec3 uBMin; uniform vec3 uBSpan;
varying vec3 vCol; varying float vFlag;
void main(){
  vec3 p = uBMin + aPos * uBSpan;
  p.y *= uFlipY;
  gl_Position = uMVP * vec4(p, 1.0);
  vCol = aCol; vFlag = aFlag;
}`
export const MESH_FS = `precision highp float;
varying vec3 vCol; varying float vFlag;
uniform float uShowInp;
void main(){
  vec3 c = vCol;
  if (uShowInp > 0.5 && vFlag >= 1.5) c = mix(c, vec3(1.0, 0.25, 0.2), 0.6);
  gl_FragColor = vec4(c, 1.0);
}`

// ── 背景兜底平面: 完整原图贴最远层(z=2.85), mesh 撕缝/洞全部被原图兜住 ──
export const BACK_VS = `attribute vec2 aPos; uniform mat4 uMVP; uniform float uAspect;
varying vec2 vUv;
void main(){
  vUv = aPos * 0.5 + 0.5;
  gl_Position = uMVP * vec4(aPos.x*uAspect*1.12, aPos.y*1.12, 2.85, 1.0);
}`
export const BACK_FS = `precision highp float;
uniform sampler2D uTex; varying vec2 vUv;
void main(){ gl_FragColor = vec4(texture2D(uTex, vUv).rgb, 1.0); }`

export const PHOTO_VS = `attribute vec2 aPos;
uniform vec2 uScale; uniform vec2 uOffset;
varying vec2 vNdc;
void main(){
  vec2 p = aPos * uScale + uOffset;
  vNdc = p;
  gl_Position = vec4(p, 0.0, 1.0);
}`
// 背景层: 常规逐像素位移(补全背景纹理)
export const PHOTO_BG_FS = `precision highp float;
varying vec2 vNdc;
uniform vec2 uScale; uniform vec2 uOffset;
uniform vec2 uCam; uniform float uParallax; uniform float uGamma;
uniform sampler2D uImageTex; uniform sampler2D uDepthTex; uniform sampler2D uAlphaTex;
uniform float uUseAlpha; uniform float uConstantDepth;
vec2 ndcToTex(vec2 ndc) {
  vec2 aPos = (ndc - uOffset) / uScale;
  vec2 uv = aPos * 0.5 + 0.5;
  return vec2(uv.x, 1.0 - uv.y);
}
void main(){
  vec2 t0 = clamp(ndcToTex(vNdc), vec2(0.0), vec2(1.0));
  float d = uConstantDepth > 0.0 ? uConstantDepth : pow(texture2D(uDepthTex, t0).r, uGamma);
  vec2 tuv = clamp(ndcToTex(vNdc - uCam * d * uParallax), vec2(0.001), vec2(0.999));
  vec3 col = texture2D(uImageTex, tuv).rgb;
  float a = uUseAlpha > 0.5 ? texture2D(uAlphaTex, tuv).r : 1.0;
  gl_FragColor = vec4(col, a);
}`

// 前景层: POM 光线步进(对齐 DepthFlow DepthMake)——光线从相机经当前像素出发,
// 与前景深度高度场两阶段求交(粗探+回退精化) → 交点 UV 采样原图。
// 数学上正确的 3D 重投影: 遮挡自动正确、连续深度、无网格伪影。未命中 → alpha 0 露出背景层。
export const PHOTO_FG_FS = `precision highp float;
varying vec2 vNdc;
uniform vec2 uScale; uniform vec2 uOffset;
uniform vec2 uCam; uniform float uParallax;
uniform float uHeight; uniform float uSteady;
uniform sampler2D uImageTex; uniform sampler2D uDepthTex; uniform sampler2D uAlphaTex;
vec2 ndcToTex(vec2 ndc) {
  vec2 aPos = (ndc - uOffset) / uScale;
  vec2 uv = aPos * 0.5 + 0.5;
  return vec2(uv.x, 1.0 - uv.y);
}
vec2 uvOf(vec2 p) {
  return clamp(vec2(p.x*0.5+0.5, 0.5-p.y*0.5), vec2(0.001), vec2(0.999));
}
void main(){
  vec2 gluv0 = vNdc;                       // 焦平面(z=1)交点 = 当前像素(未位移)
  vec2 camXY = uCam * uParallax;           // 相机平移(场景空间)
  float relSteady = uSteady * uHeight;
  // sticky 枢轴: 交点目标随相机反向延伸
  vec2 target = gluv0 - camXY / (1.0 - relSteady);
  float safe = 1.0 - uHeight;
  float walk = 0.0;
  float value = 0.0, last = 0.0, steepD = 0.0;
  vec2 gluv = gluv0;
  bool hit = false;
  // FORWARD 粗探(probe 步长): 第一次光线进入高度面
  float probe = 1.0 / 32.0;
  for (int i = 0; i < 48; i++) {
    walk += probe;
    if (walk >= 1.0) break;
    vec2 p = mix(gluv0, target, mix(safe, 1.0, walk));
    float ceiling = 1.0 - mix(safe, 1.0, walk);
    value = texture2D(uDepthTex, uvOf(p)).r;
    if (ceiling < uHeight * value) { hit = true; gluv = p; break; }
  }
  // BACKWARD 精化(quality 小步回退): 交点精确化 + 陡峭度
  if (hit) {
    float quality = 1.0 / 128.0;
    for (int i = 0; i < 160; i++) {
      walk -= quality;
      if (walk <= 0.0) { gluv = gluv0; break; }
      vec2 p = mix(gluv0, target, mix(safe, 1.0, walk));
      float ceiling = 1.0 - mix(safe, 1.0, walk);
      last = value;
      value = texture2D(uDepthTex, uvOf(p)).r;
      steepD = (last - value) / quality;
      if (ceiling >= uHeight * value) { gluv = p; break; }
      gluv = p;
    }
  }
  vec2 tuv = uvOf(gluv);
  vec3 col = texture2D(uImageTex, tuv).rgb;
  float a = hit ? texture2D(uAlphaTex, tuv).r : 0.0; // 未命中(出界/洞) → 露出背景层
  gl_FragColor = vec4(col, a);
}`


// ── mat4 最小集(列主序; 相机看 +z, 世界 y 向下) ──
export function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2)
  const A = (far + near) / (far - near), B = -2 * far * near / (far - near)
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, -f, 0, 0,
    0, 0, A, 1,
    0, 0, B, 0])
}
export function mat4RotX(a) { const c = Math.cos(a), s = Math.sin(a)
  return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]) }
export function mat4RotY(a) { const c = Math.cos(a), s = Math.sin(a)
  return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]) }
export function mat4T(x, y, z) { return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]) }
export function mat4Mul(a, b) {
  const o = new Float32Array(16)
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]
    o[c * 4 + r] = s
  }
  return o
}

// ── 渲染器工厂: getContext 由调用方决定(canvas 或 offscreen), core 只管 GL 状态 ──
export function createRenderer(canvas, gl) {
  const R = { canvas, gl }

  function makeProgram(vsSrc, fsSrc) {
    const sh = (type, s) => {
      const o = gl.createShader(type)
      gl.shaderSource(o, s); gl.compileShader(o)
      if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(o))
      return o
    }
    const prog = gl.createProgram()
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, vsSrc))
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fsSrc))
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog))
    return prog
  }

  function makeTex(src) {
    const t = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, t)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, src)
    return t
  }

  // ── warp 路径 ──
  R.setWarp = (colorSrc, depthSrc, texel) => {
    R.kind = 'warp'
    const prog = R.warpProg = makeProgram(WARP_VS, WARP_FS)
    gl.useProgram(prog)
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW)
    const aPos = gl.getAttribLocation(prog, 'aPos')
    gl.enableVertexAttribArray(aPos)
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0)
    R.u = {
      mouse: gl.getUniformLocation(prog, 'uMouse'),
      amp: gl.getUniformLocation(prog, 'uAmp'),
      zoom: gl.getUniformLocation(prog, 'uZoom'),
      fg: gl.getUniformLocation(prog, 'uFgCut'),
      cs: gl.getUniformLocation(prog, 'uCoverScale'),
      co: gl.getUniformLocation(prog, 'uCoverOff'),
      texel: gl.getUniformLocation(prog, 'uTexel'),
    }
    gl.activeTexture(gl.TEXTURE0)
    makeTex(colorSrc)
    gl.uniform1i(gl.getUniformLocation(prog, 'uTex'), 0)
    gl.activeTexture(gl.TEXTURE1)
    makeTex(depthSrc)
    gl.uniform1i(gl.getUniformLocation(prog, 'uDepth'), 1)
    gl.uniform2f(R.u.texel, texel[0], texel[1])
  }

  // ── 背景兜底平面方法 ──
  R.setBackTex = (src) => {
    if (!R.backProg) {
      R.backProg = makeProgram(BACK_VS, BACK_FS)
      R.backQuadBuf = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, R.backQuadBuf)
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW)
      R.backAPos = gl.getAttribLocation(R.backProg, 'aPos')
      gl.useProgram(R.backProg) // uniform 必须在 useProgram 之后设置
      gl.uniform1i(gl.getUniformLocation(R.backProg, 'uTex'), 0)
      R.backU = {
        mvp: gl.getUniformLocation(R.backProg, 'uMVP'),
        aspect: gl.getUniformLocation(R.backProg, 'uAspect'),
      }
    }
    if (R.backTex) gl.deleteTexture(R.backTex)
    R.backTex = makeTex(src)
  }
  R.drawBack = (mvp) => {
    if (!R.backTex) return
    gl.useProgram(R.backProg)
    gl.bindBuffer(gl.ARRAY_BUFFER, R.backQuadBuf)
    gl.enableVertexAttribArray(R.backAPos)
    gl.vertexAttribPointer(R.backAPos, 2, gl.FLOAT, false, 0, 0)
    gl.uniformMatrix4fv(R.backU.mvp, false, mvp)
    gl.uniform1f(R.backU.aspect, R.meta ? (R.meta.W / R.meta.H) : 1)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, R.backTex)
    gl.disable(gl.DEPTH_TEST)
    gl.depthMask(false)
    gl.disable(gl.CULL_FACE)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
    gl.enable(gl.CULL_FACE)
    gl.depthMask(true)
    gl.enable(gl.DEPTH_TEST)
  }

  // ── mesh 路径(meta.quantized 决定顶点格式) ──
  R.setMesh = (meta) => {
    R.kind = 'mesh'
    R.meta = meta
    R.nFaces = meta.nf
    R.quantized = meta.quantized === 'uint16'
    gl.getExtension('OES_element_index_uint') // WebGL1 下 100 万+顶点索引必需
    const vs = R.quantized ? MESH_VS_Q : MESH_VS
    const prog = R.meshProg = makeProgram(vs, MESH_FS)
    gl.useProgram(prog)
    R.u = {
      mvp: gl.getUniformLocation(prog, 'uMVP'),
      flip: gl.getUniformLocation(prog, 'uFlipY'),
      inp: gl.getUniformLocation(prog, 'uShowInp'),
      bmin: gl.getUniformLocation(prog, 'uBMin'),
      bspan: gl.getUniformLocation(prog, 'uBSpan'),
    }
  }

  R.uploadMesh = (verts, faces) => {
    gl.useProgram(R.meshProg)
    const vbuf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, vbuf)
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(verts), gl.STATIC_DRAW)
    const prog = R.meshProg
    const aPos = gl.getAttribLocation(prog, 'aPos')
    const aCol = gl.getAttribLocation(prog, 'aCol')
    const aFlag = gl.getAttribLocation(prog, 'aFlag')
    gl.enableVertexAttribArray(aPos)
    gl.enableVertexAttribArray(aCol)
    gl.enableVertexAttribArray(aFlag)
    if (R.quantized) {
      // uint16 量化: stride = meta.vStride(12B meshopt 格式 / 10B 旧格式) = qxyz(2B×3) + rgb(3B) + flag(1B) + pad
      const stride = R.meta.vStride || 10
      gl.vertexAttribPointer(aPos, 3, gl.UNSIGNED_SHORT, true, stride, 0)
      gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, stride, 6)
      gl.vertexAttribPointer(aFlag, 1, gl.UNSIGNED_BYTE, false, stride, 9)
      const mn = R.meta.bboxMin, mx = R.meta.bboxMax
      gl.uniform3f(R.u.bmin, mn[0], mn[1], mn[2])
      gl.uniform3f(R.u.bspan, mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    } else {
      // float32: stride 16B
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 16, 0)
      gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, 16, 12)
      gl.vertexAttribPointer(aFlag, 1, gl.UNSIGNED_BYTE, false, 16, 15)
    }
    const ibuf = gl.createBuffer()
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(faces), gl.STATIC_DRAW)
  }

  R.drawMesh = (mvp, flipY, showInp) => {
    gl.useProgram(R.meshProg)
    gl.uniform1f(R.u.flip, flipY)
    gl.uniform1f(R.u.inp, showInp)
    gl.uniformMatrix4fv(R.u.mvp, false, mvp)
    gl.drawElements(gl.TRIANGLES, R.nFaces * 3, gl.UNSIGNED_INT, 0)
  }

  R.drawWarp = (mx, my, amp, zoom, fgCut) => {
    gl.useProgram(R.warpProg)
    gl.uniform2f(R.u.mouse, mx, my)
    gl.uniform1f(R.u.amp, amp)
    gl.uniform1f(R.u.zoom, zoom)
    gl.uniform1f(R.u.fg, fgCut)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }

  R.clear = () => {
    gl.clearColor(0, 0, 0, 0) // 透明清屏: 预乘 alpha 下 a=0 时 RGB 必须为 0
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.enable(gl.DEPTH_TEST)
  }

  R.setViewport = (w, h) => { canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h) }

  // ── photo 两层渲染(PhotoProgram 对齐): 全屏 quad + 三纹理 + 片元逐像素位移 ──
  R.setPhoto = (layers, aspect, w, h) => {
    if (!R.photoProg) {
      R.photoProg = makeProgram(PHOTO_VS, PHOTO_BG_FS)
      R.photoFgProg = makeProgram(PHOTO_VS, PHOTO_FG_FS)
      R.photoQuadBuf = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, R.photoQuadBuf)
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW)
      R.photoAPos = gl.getAttribLocation(R.photoProg, 'aPos')
    }
    R.kind = 'photo'
    R.photoAspect = aspect
    const mk = (rgba) => {
      // rgba: Uint8ClampedArray(RGBA) — 走 9 参重载(6 参重载需要 TexImageSource)
      const t = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, t)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba)
      return t
    }
    R.photoTex = {
      fgImage: mk(layers.fgImage), fgDepth: mk(layers.fgDepth), fgAlpha: mk(layers.fgAlpha),
      bgImage: mk(layers.bgImage), bgDepth: mk(layers.bgDepth),
    }
  }
  R.drawPhotoLayer = (layer, cam, parallax, opts = {}) => {
    // layer: {imageTex, depthTex, alphaTex}; opts: {useAlpha, pom, height, steady, gamma}
    const prog = opts.pom ? R.photoFgProg : R.photoProg
    gl.useProgram(prog)
    gl.bindBuffer(gl.ARRAY_BUFFER, R.photoQuadBuf)
    gl.enableVertexAttribArray(R.photoAPos)
    gl.vertexAttribPointer(R.photoAPos, 2, gl.FLOAT, false, 0, 0)
    // cover: 图幅 contain 进视口(uScale/uOffset 由 shader 内 ndcToTex 反算)
    const vw = R.canvas.width, vh = R.canvas.height
    const ia = R.photoAspect
    const sX = vw / vh > ia ? 1 : ia * (vh / vw)
    const sY = vw / vh > ia ? (vw / vh) / ia : 1
    gl.uniform2f(gl.getUniformLocation(prog, 'uScale'), sX, sY)
    gl.uniform2f(gl.getUniformLocation(prog, 'uOffset'), 0, 0)
    gl.uniform2f(gl.getUniformLocation(prog, 'uCam'), cam[0], cam[1])
    gl.uniform1f(gl.getUniformLocation(prog, 'uParallax'), parallax)
    if (opts.pom) {
      gl.uniform1f(gl.getUniformLocation(prog, 'uHeight'), opts.height ?? 0.3)
      gl.uniform1f(gl.getUniformLocation(prog, 'uSteady'), opts.steady ?? 0.5)
    } else {
      gl.uniform1f(gl.getUniformLocation(prog, 'uGamma'), opts.gamma ?? 1.0)
      gl.uniform1f(gl.getUniformLocation(prog, 'uUseAlpha'), opts.useAlpha ? 1 : 0)
      gl.uniform1f(gl.getUniformLocation(prog, 'uConstantDepth'), opts.constantDepth || 0)
    }
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, layer.imageTex)
    gl.uniform1i(gl.getUniformLocation(prog, 'uImageTex'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, layer.depthTex)
    gl.uniform1i(gl.getUniformLocation(prog, 'uDepthTex'), 1)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, layer.alphaTex)
    gl.uniform1i(gl.getUniformLocation(prog, 'uAlphaTex'), 2)
    gl.disable(gl.DEPTH_TEST)
    gl.disable(gl.CULL_FACE)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
    gl.enable(gl.DEPTH_TEST)
    gl.enable(gl.CULL_FACE)
  }

  // ── 背景兜底平面: 完整原图贴最远层(z=2.85, 比网格/填充层远), 盖住一切洞 ──
  R.setBackTex = (src) => {
    if (!R.backProg) {
      R.backProg = makeProgram(BACK_VS, BACK_FS)
      R.backQuadBuf = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, R.backQuadBuf)
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW)
      R.backAPos = gl.getAttribLocation(R.backProg, 'aPos')
      gl.useProgram(R.backProg) // uniform 必须在 useProgram 之后设置
      gl.uniform1i(gl.getUniformLocation(R.backProg, 'uTex'), 0)
      R.backU = {
        mvp: gl.getUniformLocation(R.backProg, 'uMVP'),
        aspect: gl.getUniformLocation(R.backProg, 'uAspect'),
      }
    }
    if (R.backTex) gl.deleteTexture(R.backTex)
    R.backTex = makeTex(src)
  }
  R.drawBack = (mvp) => {
    if (!R.backTex) return
    gl.useProgram(R.backProg)
    gl.bindBuffer(gl.ARRAY_BUFFER, R.backQuadBuf)
    gl.enableVertexAttribArray(R.backAPos)
    gl.vertexAttribPointer(R.backAPos, 2, gl.FLOAT, false, 0, 0)
    gl.uniformMatrix4fv(R.backU.mvp, false, mvp)
    gl.uniform1f(R.backU.aspect, R.meta ? (R.meta.W / R.meta.H) : 1)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, R.backTex)
    gl.disable(gl.DEPTH_TEST) // 背景平面永远画在最底(先画, depth 无关)
    gl.depthMask(false)
    gl.disable(gl.CULL_FACE) // 投影矩阵 y 取负翻转 winding → quad 会被当 back 剔除, 局部禁用
    gl.drawArrays(gl.TRIANGLES, 0, 6)
    gl.enable(gl.CULL_FACE)
    gl.depthMask(true)
    gl.enable(gl.DEPTH_TEST)
  }

  R.setViewport = (w, h) => {
    canvas.width = w; canvas.height = h
    gl.viewport(0, 0, w, h)
    if (R.fxaaOn) {
      // 屏幕尺寸变化 → 重建 FBO 颜色纹理与深度 renderbuffer
      gl.bindTexture(gl.TEXTURE_2D, R.sceneTex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
      gl.bindRenderbuffer(gl.RENDERBUFFER, R.depthRB)
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h)
    }
  }


  return R
}
