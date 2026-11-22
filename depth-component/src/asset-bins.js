// asset-bins.js — 真几何资产加载器: meta → 顶点/索引 → {meta, verts, faces}
// 双格式: ① format:'meshopt'(web_*.mo.z = 位打包+deflate, 解码在 worker) ② compressed:'deflate'(.z)
// IO(deflate 流)在调用线程; meshopt 位解码是 CPU 密集, 走 rpc(默认 worker)。
import { decodeMeshopt } from './gpu-pipeline.js'

async function fetchBuf(url, inflate) {
  const r = await fetch(url, { mode: 'cors' })
  if (!r.ok) throw new Error('资产加载失败 ' + r.status + ': ' + url)
  if (!inflate) return await r.arrayBuffer()
  return await new Response(r.body.pipeThrough(new DecompressionStream('deflate'))).arrayBuffer()
}

// rpc: (type, payload, transferList) => Promise<result> — 由组件提供(worker 编排); 不传则本线程解码
export async function loadBins(base, { rpc, onProgress } = {}) {
  const meta = await fetch(base + '/web_meta.json', { mode: 'cors' }).then(r => {
    if (!r.ok) throw new Error('meta 加载失败 ' + r.status)
    return r.json()
  })
  let verts, faces
  if (meta.format === 'meshopt') {
    onProgress && onProgress('下载 meshopt 资产')
    const [vb, fb] = await Promise.all([
      fetchBuf(base + '/web_verts.mo.z', true),
      fetchBuf(base + '/web_faces.mo.z', true),
    ])
    onProgress && onProgress('解码中')
    const payload = { verts: vb, faces: fb, vStride: meta.vStride || 10, nv: meta.nv, nf: meta.nf }
    if (rpc) {
      const out = await rpc('decodeBins', payload, [vb, fb])
      verts = out.verts; faces = out.faces
    } else {
      const out = await decodeMeshopt(payload)
      verts = out.verts; faces = out.faces
    }
  } else if (meta.compressed === 'deflate') {
    onProgress && onProgress('下载压缩资产')
    const [vb, fb] = await Promise.all([
      fetchBuf(base + '/web_verts.bin.z', true),
      fetchBuf(base + '/web_faces.bin.z', true),
    ])
    verts = vb; faces = fb
  } else {
    onProgress && onProgress('下载资产')
    const [vb, fb] = await Promise.all([
      fetchBuf(base + '/web_verts.bin'),
      fetchBuf(base + '/web_faces.bin'),
    ])
    verts = vb; faces = fb
  }
  return { meta, verts, faces }
}
