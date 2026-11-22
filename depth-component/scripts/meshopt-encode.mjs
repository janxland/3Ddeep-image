// meshopt-encode.mjs — mesh bins → meshopt 编码资产(顶点重排 + 位打包 + deflate)
// 输入: demo/img/mesh/<stem>/ {web_meta.json, web_verts.bin[.z], web_faces.bin[.z]}
// 输出: 同目录 web_verts.mo.z / web_faces.mo.z + meta 更新 {format:'meshopt', quantized:'uint16', vStride:12, deflate:true}
// 顶点统一为 12B stride(4 字节对齐是 meshopt 硬要求): qxyz u16×3 + rgb 3B + flag 1B + pad 2B
// 两种源格式都收: float32(16B stride) 先量化; 已 uint16(10B) 重排为 12B
// 用法: node scripts/meshopt-encode.mjs <stem|目录名>...
import { readFileSync, writeFileSync, existsSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { inflateSync, deflateSync } from 'zlib'
import { MeshoptEncoder } from 'meshoptimizer/encoder'

const HERE = dirname(fileURLToPath(import.meta.url))
const MESH = resolve(HERE, '../demo/img/mesh')

function loadBuf(dir, base, meta) {
  let p = resolve(dir, base + (meta.compressed === 'deflate' ? '.z' : ''))
  if (!existsSync(p)) p = resolve(dir, base)
  const raw = readFileSync(p)
  return meta.compressed === 'deflate' ? inflateSync(raw) : raw
}

// 源顶点 → 12B stride 量化顶点(f32 16B 或 u16 10B 均可)
function toStride12(vertsBuf, srcStride) {
  const isF32 = srcStride === 16
  const nv = vertsBuf.byteLength / srcStride
  const f = isF32
    ? new Float32Array(vertsBuf.buffer, vertsBuf.byteOffset, vertsBuf.byteLength / 4)
    : new Uint16Array(vertsBuf.buffer, vertsBuf.byteOffset, vertsBuf.byteLength / 2)
  let mn, mx, span
  if (isF32) {
    mn = [Infinity, Infinity, Infinity]; mx = [-Infinity, -Infinity, -Infinity]
    for (let i = 0; i < nv; i++) for (let c = 0; c < 3; c++) {
      const v = f[i * 4 + c]
      if (v < mn[c]) mn[c] = v
      if (v > mx[c]) mx[c] = v
    }
    span = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]]
  } else {
    // 已量化: bbox 从 meta 由调用方传入(mn/mx/span 为 null 时按 0..65535 全域)
    mn = null; mx = null; span = null
  }
  const out = new Uint8Array(nv * 12)
  const q = new Uint16Array(out.buffer)
  const srcQ = !isF32 ? new Uint16Array(vertsBuf.buffer, vertsBuf.byteOffset, vertsBuf.byteLength / 2) : null
  for (let i = 0; i < nv; i++) {
    if (isF32) {
      for (let c = 0; c < 3; c++) q[i * 6 + c] = Math.round((f[i * 4 + c] - mn[c]) / Math.max(1e-9, span[c]) * 65535)
      out[i * 12 + 6] = vertsBuf[i * 16 + 12]
      out[i * 12 + 7] = vertsBuf[i * 16 + 13]
      out[i * 12 + 8] = vertsBuf[i * 16 + 14]
      out[i * 12 + 9] = vertsBuf[i * 16 + 15]
    } else {
      // u16 源(10B): qxyz@0..5 + rgb@6..8 + flag@9 → 平移到 12B
      out[i * 12 + 6] = vertsBuf[i * 10 + 6]
      out[i * 12 + 7] = vertsBuf[i * 10 + 7]
      out[i * 12 + 8] = vertsBuf[i * 10 + 8]
      out[i * 12 + 9] = vertsBuf[i * 10 + 9]
      for (let c = 0; c < 3; c++) q[i * 6 + c] = srcQ[i * 5 + c]
    }
  }
  return { out, mn, mx }
}

function permuteVertices(verts, vStride, remap) {
  const nv = remap.length
  const out = new Uint8Array(nv * vStride)
  for (let old = 0; old < nv; old++) {
    const n = remap[old]
    if (n === 0xffffffff) continue // 未引用顶点悬空无妨(索引不指向)
    const o = old * vStride, d = n * vStride
    for (let b = 0; b < vStride; b++) out[d + b] = verts[o + b]
  }
  return out
}

async function encode(dirName) {
  const dir = resolve(MESH, dirName)
  const metaPath = resolve(dir, 'web_meta.json')
  if (!existsSync(metaPath)) { console.error(`✘ ${dirName}: 无 web_meta.json`); return }
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'))
  const t0 = Date.now()
  await MeshoptEncoder.ready

  // ① 读原始数据 → 统一 12B stride u16 量化顶点
  const vertsRaw = loadBuf(dir, 'web_verts.bin', meta)
  const facesRaw = loadBuf(dir, 'web_faces.bin', meta)
  const srcStride = meta.quantized === 'uint16' ? 10 : 16
  const q = toStride12(vertsRaw, srcStride)
  let bboxMin = meta.bboxMin || q.mn, bboxMax = meta.bboxMax || q.mx
  if (srcStride === 10 && !meta.bboxMin) { bboxMin = [0, 0, 0]; bboxMax = [1, 1, 1] } // 旧 u16 资产 meta 无 bbox 时按全域
  const nv = q.out.length / 12
  const indices = new Uint32Array(facesRaw.buffer, facesRaw.byteOffset, facesRaw.byteLength / 4)
  const nf = indices.length / 3

  // ② 网格重排(索引原地重排 + 顶点按 remap permute)
  const [remap, unique] = MeshoptEncoder.reorderMesh(indices, true, true)
  const vertsSorted = permuteVertices(q.out, 12, remap)

  // ③ 位打包 + deflate 组合编码
  const encV = MeshoptEncoder.encodeVertexBuffer(vertsSorted, unique, 12)
  const encI = MeshoptEncoder.encodeIndexBuffer(new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength), indices.length, 4)
  const vZ = deflateSync(encV, { level: 9 })
  const iZ = deflateSync(encI, { level: 9 })
  writeFileSync(resolve(dir, 'web_verts.mo.z'), vZ)
  writeFileSync(resolve(dir, 'web_faces.mo.z'), iZ)

  // ④ meta 更新(format: meshopt; 原 bins 保留)
  writeFileSync(metaPath, JSON.stringify({
    ...meta, format: 'meshopt', quantized: 'uint16', vStride: 12, deflate: true,
    nv: unique, nf, bboxMin, bboxMax,
  }, null, 1))
  const inSize = (vertsRaw.byteLength + facesRaw.byteLength) / 1e6
  const outSize = (vZ.length + iZ.length) / 1e6
  console.log(`[${dirName}] ✔ ${(Date.now() - t0) / 1000}s  ${inSize.toFixed(1)}MB → ${outSize.toFixed(1)}MB (${(inSize / outSize).toFixed(1)}x)  verts ${(vZ.length / 1e6).toFixed(1)} + faces ${(iZ.length / 1e6).toFixed(1)}  (unique ${unique.toLocaleString()}/${nv.toLocaleString()})`)
}

for (const name of process.argv.slice(2)) await encode(name)
