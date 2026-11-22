# 一图转 3D 景深：把单张照片变成浏览器里的立体空间

> 无需安装、无需服务器渲染——访客打开网页，一张普通照片就能环视、变焦、有真实的景深立体感。这篇文章记录我把它做成一个**原生 Web Component** 的全过程：双轨渲染架构、Worker 化的 GPU 管线，以及 meshopt 带来的 20 倍资产压缩。

<!-- more -->

## 先看效果

下面是一张 4K 高清示例（人物、扶手椅、房间、窗外街景四层纵深）。组件默认**不加载任何东西**——只显示一张封面图，点击才开始加载（对内容流多组件页面和低配机器都友好）：

<depth-scene src="https://cos.roginx.ink/www/index/depth-scene/img/tifa.jpg" mesh-src="https://cos.roginx.ink/www/index/depth-scene/mesh/tifa" style="display:block;width:100%;height:480px;border-radius:14px;overflow:hidden"></depth-scene>

加载后：移动鼠标环视，滚轮变焦，空闲几秒它会自己慢慢"呼吸"；按 `I` 键可以高亮神经网络补全的区域，按 `F` 翻转场景。

这个组件还有一条**端内轨**——不加 `mesh-src` 时，任意一张图在浏览器里实时推理深度并生成分层网格（WebGPU 优先，WASM 回退），或者直接把图片**拖进组件**就能转。本文为了不卡你的电脑，只放这一个演示。

## 问题：单图怎么可能"3D"？

一张照片只有一个视角，没有几何信息。要让它看起来立体，需要三件事：

1. **猜出深度**——每个像素离相机多远（单目深度估计）
2. **分层**——前景/背景沿深度切开，遮挡区域要有"内容"可以转出来
3. **渲染**——视差位移或真网格，让鼠标移动时有空间感

这个领域 2020 年的 CVPR 论文（3D Photography using Context-Aware Layered Depth Inpainting）给出了当时最好的答案：MiDaS 深度 → 分层深度图（LDI）→ 三个补全网络 → 多层网格。我把它完整跑在了本地，作为**质量上限轨**——上面演示的真几何多层 mesh 就是它产出的，没有"纸片感"，代价是单图预处理要二十多分钟。

## 双轨设计

一个组件，两种质量的来源，按需选择：

| | 资产轨 | 端内轨 |
|---|---|---|
| 触发 | `mesh-src` 指向预计算资产 | `src` 任意图片 |
| 深度来源 | 服务端管线（DA3/MiDaS + 补全网络） | 浏览器内 DA-V2（WebGPU→WASM） |
| 质量 | 真多层几何，带遮挡补全 | 单层偏移 + 分层网格 + 引导滤波 |
| 首屏 | 3MB bins，几百毫秒出画面 | 模型 27MB 起缓存，推理 1-6s |
| 适用 | 精选壁纸、作品展示 | 用户随手拖一张图 |

## 架构：主线程零计算

组件是标准 Web Component（Shadow DOM + CSS 变量主题 + composed 事件），内部拆成四个纯模块：

- `depth-scene.js` 入口：DOM 粘合 / 缓存编排（本地 Cache API → 云端缓存池 → 实时推理）/ **启动门**（默认封面+点击加载，`autoplay` 可选自动）
- `render-core.js` 纯 WebGL 渲染器：零 DOM 依赖，可跑在主线程也能跑在 OffscreenCanvas
- `gpu-pipeline.js` inline Worker：深度推理 + mini-LDI 网格构建 + meshopt 解码
- `asset-bins.js` 资产加载：deflate / meshopt 双格式

所有 CPU 密集的活（推理、网格生成、解码）都在 Worker 里，主线程只做 DOM、IO 和 rAF。推理期间页面滚动不掉帧——这是对"GPU 充分利用"的第一层回答。

第二层是**推理本身的降级链**：WebGPU EP 优先（现代 Chromium/Safari 26+），回退 WASM SIMD + 多线程，再回退单线程 WASM。浏览器能力探测在运行时完成，同一份代码覆盖所有环境。

## meshopt：66MB → 3MB 的关键

真几何网格很大：一张图导出后是 50 万顶点 + 500 万三角面，原始二进制 66MB——这是这个方案最早不可用的原因。

解法是游戏工业的标准工具链 **meshoptimizer**：`reorderMesh` 重排顶点（缓存亲和）→ 顶点/索引位打包编码（配合 4 字节对齐的 12B stride 顶点格式）→ 再叠一层 deflate。组合下来 **17~25 倍压缩**，66MB → 3MB。解码器是自包含 WASM（~32KB），直接打进组件包，加载时在 Worker 里毫秒级解完——下载 + 解码 + GPU 上传全链路几百毫秒出画面。

## 交付形态：一行 script

对齐我站上已有的 ppt-component / music-component 的约定，这个组件也是单文件 IIFE：

```
&lt;script src="https://cos.roginx.ink/www/index/depth-scene/depth-scene.js"&gt;&lt;/script&gt;

&lt;depth-scene src="封面图.jpg" mesh-src="https://cos.roginx.ink/www/index/depth-scene/mesh/e1d28550"
             style="display:block;width:100%;height:480px"&gt;&lt;/depth-scene&gt;
```

页面零配置，CDN 容错链内置；Shadow DOM 隔离样式，`--d3d-surface/--d3d-radius/--d3d-status` 等变量穿透主题；事件（`d3d-ready/d3d-progress/d3d-error`）composed，宿主页面可监听。默认封面+点击加载，加 `autoplay="true"` 变为进入视口即自动加载。

## 数字小结

| 指标 | 之前 | 现在 |
|---|---|---|
| 资产体积 | 66MB/图 | **3-4.7MB/图**（17-25x） |
| 资产轨首帧 | ~20s（下载+解析） | **270ms** |
| 端内推理 | — | WebGPU <2s / WASM 1-6s |
| 网格构建 | — | Worker 内 **7ms** |
| 主线程 | 推理卡页面 | **零计算** |
| 内容流加载 | 进页面全加载 | **封面+点击，按需启动** |

## 后续

- WebGPU compute shader 版的引导滤波与网格生成（现在是 Worker JS，compute 版作为 quality=best 分支）
- 遮挡补全区质量升级（LaMa 替换 2020 pix2pix）
- 全景扩展：背景 360° 包围感的真全景层

组件源码：[github.com/janxland/3Ddeep-image](https://github.com/janxland/3Ddeep-image)
