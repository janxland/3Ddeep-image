// depth-demo.mjs — 单图转景深：Depth Anything V2 (ONNX, transformers.js 本地推理) → 深度图 + 视差 wiggle GIF
// 用法: node depth-demo.mjs <img1> [img2 ...]
import fs from 'node:fs';
import path from 'node:path';
import { pipeline, env } from '@huggingface/transformers';
import gifenc from 'gifenc';
import sharp from 'sharp';
const { GIFEncoder, quantize, applyPalette } = gifenc;

const ROOT = path.dirname(new URL(import.meta.url).pathname);
const MODELS = path.join(ROOT, 'models');
const OUT = path.join(ROOT, 'out');
const MAXW = 800;      // 处理宽度
const FRAMES = 12;     // 动图帧数
const AMP = 10;        // 视差幅度(px)
const DELAY = 90;      // 帧间隔 ms
const INVERT = false;  // 若近/远方向反了改 true

env.allowRemoteModels = false;
env.localModelPath = MODELS;
fs.mkdirSync(OUT, { recursive: true });

const onnxDir = path.join(MODELS, 'onnx-community/depth-anything-v2-small/onnx');
const dtype = fs.existsSync(path.join(onnxDir, 'model_quantized.onnx')) ? 'q8'
  : fs.existsSync(path.join(onnxDir, 'model_fp16.onnx')) ? 'fp16' : 'fp32';
console.log('dtype =', dtype);
const depther = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', { dtype });

const clamp = (v, a, b) => v < a ? a : v > b ? b : v;

function boxBlur(d, w, h, passes = 2) {
  for (let p = 0; p < passes; p++) {
    const cp = Float32Array.from(d);
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        d[i] = (cp[i] + cp[i - 1] + cp[i + 1] + cp[i - w] + cp[i + w]
              + cp[i - w - 1] + cp[i - w + 1] + cp[i + w - 1] + cp[i + w + 1]) / 9;
      }
  }
  return d;
}

for (const src of process.argv.slice(2)) {
  console.log('→', src);
  const stem = path.basename(src).replace(/\.[^.]+$/, '');

  // 推理：原图路径直喂（内部 518² 推理后回插原尺寸），depth 亮度=近
  const { depth } = await depther(src);
  const depthPath = path.join(OUT, stem + '.depth.png');
  await sharp(Buffer.from(depth.data), {
    raw: { width: depth.width, height: depth.height, channels: depth.channels },
  }).png().toFile(depthPath);

  // 原图 RGBA @MAXW
  const { data: rgba, info } = await sharp(src)
    .resize({ width: MAXW, withoutEnlargement: true })
    .ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const w = info.width, h = info.height, n = w * h;

  // 深度对齐到处理尺寸（sharp）+ 归一化 + 平滑
  const { data: dBuf, info: dInfo } = await sharp(Buffer.from(depth.data), {
    raw: { width: depth.width, height: depth.height, channels: depth.channels },
  }).resize(w, h, { fit: 'fill' }).greyscale().raw().toBuffer({ resolveWithObject: true });
  const dn = new Float32Array(n);
  for (let i = 0; i < n; i++) dn[i] = dBuf[i * dInfo.channels] / 255;
  boxBlur(dn, w, h, 2);
  let mn = 1, mx = 0;
  for (const v of dn) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const span = Math.max(1e-6, mx - mn);
  for (let i = 0; i < n; i++) dn[i] = (dn[i] - mn) / span;
  if (INVERT) for (let i = 0; i < n; i++) dn[i] = 1 - dn[i];

  // 视差 wiggle：近处锚定，远处按深度摆动
  const gif = GIFEncoder();
  for (let t = 0; t < FRAMES; t++) {
    const ph = Math.sin(2 * Math.PI * t / FRAMES);
    const frame = new Uint8Array(n * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const sx = clamp(Math.round(x + AMP * (1 - dn[i]) * ph), 0, w - 1);
        const s = (y * w + sx) * 4;
        frame[i * 4] = rgba[s];
        frame[i * 4 + 1] = rgba[s + 1];
        frame[i * 4 + 2] = rgba[s + 2];
        frame[i * 4 + 3] = 255;
      }
    }
    const pal = quantize(frame, 256);
    gif.writeFrame(applyPalette(frame, pal), w, h, { palette: pal, delay: DELAY });
  }
  gif.finish();
  fs.writeFileSync(path.join(OUT, stem + '.wiggle.gif'), gif.bytes());
  console.log('   ✔', `${w}x${h}`, '→', depthPath, '+', path.join(OUT, stem + '.wiggle.gif'));
}
process.exit(0); // 防 onnxruntime-node 退出期 teardown 崩溃
