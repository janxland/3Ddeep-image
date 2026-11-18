# 3Ddeep-image — 单图 → 可交互 3D 景深

一张图变成浏览器里可鼠标环视的 3D 景深场景。三档模式，质量/体积可调。

## 目录
- `depth-component/` — `<depth-viewer>` Web Component（核心交付物）
  - `src/depth-viewer.js` 源码；`dist/depth-component/depth-viewer.js` 构建产物（~10KB IIFE）
  - `demo/` 本地 demo（`node serve-no-store.mjs . 8798` 后访问 `/demo/`）
  - `vendor/`+`models/` — 同源依赖（transformers.js + ModNet + DA-V2 q8），受限 webview 也能跑
- `3d-photo-inpainting/` — 生产端管线（vendored 自 [vt-vl-lab/3d-photo-inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting) CVPR 2020，含本地兼容补丁）
  - `ply2bin.py` ply→WebGL 二进制；`finalize-batch.py` 批量装配
- `inpaint-bg.py` — Workers AI SD-inpainting 背景补全（`bg-src`）
- `pano-out.py` — Workers AI 画布外扩全景长条（`pano-src` 盒壁纹理，当前未接入渲染）
- `compress-assets.mjs` — 资产 deflate 压缩（57MB→20MB，DecompressionStream 原生解码）
- `depth-cache-worker/` — KV 深度缓存池 Worker（众包缓存）
- `EXPLORATION-20261118.md` — 技术探索报告

## 快速开始（demo）
```bash
cd depth-component
npm install && npx vite build          # 构建组件
node serve-no-store.mjs . 8798         # 起本地服务
open http://localhost:8798/demo/index.html
```
## 生产管线（单图→mesh，约 3 分钟，Mac MPS 自动加速）
```bash
cd 3d-photo-inpainting
sh download.sh                         # 下载 checkpoints（HF 镜像见 git log 说明）
.venv/bin/python main.py --config argument.yml
.venv/bin/python ply2bin.py mesh/<stem>.ply <输出目录>
```
## `<depth-viewer>` 属性
`mesh-src`（真几何 20MB）> `depth-src`+`matte-src`+`bg-src`（分层 0.4MB）> `src`（全实时 DA-V2+ModNet）；另有 `amp` / `zoom-max` / `idle-seconds` / `layers` / `quality` / `render` / `cache-endpoint`；事件 `depth-ready` / `error`
