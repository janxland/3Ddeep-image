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
    gl.clearColor(0.02, 0.02, 0.03, 1)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.enable(gl.DEPTH_TEST)
  }

  R.setViewport = (w, h) => { canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h) }

  return R
}
