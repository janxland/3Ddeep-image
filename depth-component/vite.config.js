// vite.config.js — <depth-scene> 单文件 IIFE 构建(对齐 ppt-component / music-component 约定)
// 产物: dist/depth-component/depth-scene.js (自注册 <depth-scene>, 无外部 CSS/依赖)
// 推理/网格构建/meshopt 解码跑 inline Worker(?worker&inline → base64 内嵌, 单文件承诺不破坏)
import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  publicDir: false,
  worker: {
    format: 'iife', // inline worker 也必须是经典脚本
  },
  build: {
    emptyOutDir: true,
    outDir: resolve(__dirname, 'dist/depth-component'),
    lib: {
      entry: resolve(__dirname, 'src/depth-scene.js'),
      name: 'DepthScene',
      formats: ['iife'],
      fileName: () => 'depth-scene',
    },
    rollupOptions: {
      output: { entryFileNames: 'depth-scene.js' },
      // transformers.js / onnxruntime-web 不进包: 运行时按需从 CDN 动态 import(music-component 懒加载约定)
      external: [/^@huggingface\//, /^onnxruntime-web/],
    },
    minify: 'terser',
    terserOptions: { compress: { drop_console: true, drop_debugger: true } },
    target: 'es2020',
  },
})
