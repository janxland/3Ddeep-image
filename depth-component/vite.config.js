// vite.config.js — <depth-viewer> 单文件 IIFE 构建（对齐 ppt-component / music-component 约定）
// 产物: dist/depth-component/depth-viewer.js （自注册 <depth-viewer>，无外部 CSS/依赖）
import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  publicDir: false,
  build: {
    emptyOutDir: true,
    outDir: resolve(__dirname, 'dist/depth-component'),
    lib: {
      entry: resolve(__dirname, 'src/depth-viewer.js'),
      name: 'DepthViewer',
      formats: ['iife'],
      fileName: () => 'depth-viewer',
    },
    rollupOptions: {
      output: { entryFileNames: 'depth-viewer.js' },
      // transformers.js / onnxruntime-web 不进包：运行时按需从 CDN 动态 import（music-component 的懒加载约定）
      external: [/^@huggingface\//, /^onnxruntime-web/],
    },
    minify: 'terser',
    terserOptions: { compress: { drop_console: true, drop_debugger: true } },
    target: 'es2020',
  },
})
