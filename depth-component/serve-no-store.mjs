// serve-no-store.mjs — 无缓存静态服务器（调试专用，杜绝浏览器启发式缓存）
// 用法: node serve-no-store.mjs <rootDir> <port>
import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'

const root = path.resolve(process.argv[2] || '.')
const port = Number(process.argv[3]) || 8798
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.json': 'application/json',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
  '.gif': 'image/gif', '.webp': 'image/webp', '.bin': 'application/octet-stream',
  '.css': 'text/css; charset=utf-8', '.ply': 'application/octet-stream',
  '.gz': 'application/gzip', '.br': 'application/x-brotli',
}

http.createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split('?')[0])
  const fp = path.join(root, urlPath)
  if (!fp.startsWith(root)) { res.writeHead(403); res.end('403'); return }
  fs.readFile(fp, (err, data) => {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain' }); res.end('404 ' + urlPath); return }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(fp).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'Access-Control-Allow-Origin': '*',
    })
    res.end(data)
  })
}).listen(port, () => console.log(`no-store static: ${root} → :${port}`))
