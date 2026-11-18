// depth-cache Worker — KV 深度缓存池（GET 取 / PUT 存，全 CORS）
// key 约定: d/<sha256(原图URL)>；值: 灰度深度 PNG（亮=近），≤3MB
const cors = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,HEAD,PUT,POST,OPTIONS',
  'Access-Control-Allow-Headers': '*',
  'Access-Control-Max-Age': '86400',
}

export default {
  async fetch(req, env) {
    if (req.method === 'OPTIONS') return new Response(null, { headers: cors })
    const url = new URL(req.url)
    const key = url.pathname.replace(/^\/+/, '')
    if (!key.startsWith('d/') || key.length < 10) {
      return new Response(JSON.stringify({ ok: false, error: 'key 需为 d/<hash>' }), { status: 400, headers: { ...cors, 'Content-Type': 'application/json' } })
    }

    if (req.method === 'GET' || req.method === 'HEAD') {
      const v = await env.DEPTH.getWithMetadata(key, { type: 'arrayBuffer' })
      if (!v.value) return new Response(null, { status: 404, headers: cors })
      return new Response(v.value, {
        headers: { ...cors, 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=86400' },
      })
    }

    if (req.method === 'PUT' || req.method === 'POST') {
      const buf = await req.arrayBuffer()
      if (buf.byteLength === 0 || buf.byteLength > 3_000_000) {
        return new Response('size out of range', { status: 413, headers: cors })
      }
      const b = new Uint8Array(buf)
      if (!(b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47)) {
        return new Response('not a png', { status: 400, headers: cors })
      }
      await env.DEPTH.put(key, buf)
      return new Response(JSON.stringify({ ok: true, key, bytes: buf.byteLength }), {
        headers: { ...cors, 'Content-Type': 'application/json' },
      })
    }

    return new Response('method not allowed', { status: 405, headers: cors })
  },
}
