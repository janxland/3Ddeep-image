// compress-assets.mjs — 资产无损压缩实验：索引+顶点 deflate level9（浏览器 DecompressionStream 原生解码）
// 用法: node compress-assets.mjs <component-assets根目录> <stem>...
import fs from 'node:fs'
import path from 'node:path'
import zlib from 'node:zlib'

const root = path.resolve(process.argv[2] || '.')
const stems = process.argv.slice(3)
for (const stem of stems) {
  const dir = path.join(root, stem)
  const meta = JSON.parse(fs.readFileSync(path.join(dir, 'web_meta.json'), 'utf8'))
  let total0 = 0, total1 = 0
  for (const name of ['web_verts.bin', 'web_faces.bin']) {
    const raw = fs.readFileSync(path.join(dir, name))
    const z = zlib.deflateSync(raw, { level: 9, memLevel: 9 })
    fs.writeFileSync(path.join(dir, name + '.z'), z)
    total0 += raw.length; total1 += z.length
    console.log(`  ${name}: ${(raw.length / 1e6).toFixed(1)}MB → ${(z.length / 1e6).toFixed(1)}MB (${(raw.length / z.length).toFixed(1)}x)`)
  }
  meta.compressed = 'deflate'
  fs.writeFileSync(path.join(dir, 'web_meta.json'), JSON.stringify(meta))
  console.log(`✔ ${stem}: ${meta.nv}v/${meta.nf}f  ${(total0 / 1e6).toFixed(1)}MB → ${(total1 / 1e6).toFixed(1)}MB`)
}
