# 3D 景深壁纸 · 方案定稿文档

> 状态：2026-11-19 定稿。当前方案 = CVPR20 完整管线（质量天花板，已复现锁定）。
> 本文档是唯一权威方案记录；探索过程见 `EXPLORATION-20261118.md`。

## 一、定稿方案（当前交付）

```
单图 → MiDaS 深度(640, fp32) → LDI 分层 + 3 个补全网络(真·画出遮挡背景)
     → 多层 mesh.ply(像素级) → ply2bin.py → WebGL 渲染器
```

- **质量**：遮挡区有神经网络补全的真实内容，人物/家具保形——管线可复现已验证（重跑渲染逐像素一致）
- **资产**：66MB/图（float bins）→ `compress-assets.mjs` deflate 可压到 20MB（浏览器 DecompressionStream 原生解码）
- **成本**：Mac MPS ~3 分钟/张（SAM 关闭时）
- **交付物**：`wallpaper.html` 全屏壁纸页（按图比例 contain、极简切换、滚轮变焦/鼠标环视/I 键补全高亮）
- **用法**：
  ```bash
  cd 3d-photo-inpainting && .venv/bin/python main.py --config argument.yml   # 跑管线
  .venv/bin/python ply2bin.py mesh/<stem>.ply ../depth-component/demo/img/mesh/<stem>
  # demo 或 wallpaper.html 里 mesh-src="./img/mesh/<stem>"
  ```

## 二、已落地的增量改进（非质变，保留）

### SAM 引导深度平滑（sigma=7 预处理）
- `sam_depth_smooth.py`：SAM2 automask → mask 内大核平滑（物体内部消噪 → 不撕脸/椅子），mask 外原样（物体间跳变→轮廓撕缝照常），边缘 feather 过渡
- 注入点：`main.py` 的 `sam_smooth: True` 开关（write_ply 前），多花 ~100s/张（SAM 推理）
- **结论：撕扯有减轻但非质变**——撕裂判据的本质（深度突变）未变，只是噪声被压低
- 旋钮：`sigma`（撕得多→调大；轮廓糊→调小）

### 其他已落地
- LaMa 背景补全（`lama-bgfill.py`，isnet matte + big-lama，离线 3s/张）——供 `bg-src` 分层模式
- isnet-general-use 抠图（替换 u2net/BiRefNet-portrait）
- KV 深度缓存池 Worker（`depth-cache-worker/`，CN 直连待 2 号账号绑路由）
- 资产量化 uint16（`finalize-batch.py`）与 deflate 压缩

## 三、已验证失败的路线（不再走）

| 路线 | 失败原因 |
|---|---|
| v2 slab 压平（SAM mask 内压单一平面） | 立体人物压成纸片=形体塌陷；背景层深度残留人物近深度=层间打架；288 网格锯齿 |
| BiRefNet-portrait 抠图 | matte 大面积溢出（比 u2net 还差） |
| u2net_human_seg / isnet 裸用 | 白衬衫贴过曝背景会被丢进背景（须配 alpha 重映射 `clip((a-12)/45*255)`） |
| 浏览器内 CVPR20 图算法 | networkx 图算法不可移植（pyodide 慢 100 倍） |
| Workers AI 深度/分割 | 目录 75 模型无深度估计/分割/3D |

**教训**：SAM 给的是「哪些像素同属一个物体」，不提供形状。用它当「撕裂裁判」（约束判据）是对的；用它当「几何工匠」（压 slab/替深度）必败。

## 四、未来立项（按价值排序）

1. **语义引导撕裂 surgery**（SAM 的正确深用法）：改 `mesh.py` 的 `tear_edges`/edge 处理——mask 内部禁撕、撕缝吸附 SAM 轮廓。风险：千行图算法 deep surgery，需独立分支慢慢调。预期：撕扯问题的根治。
2. **DA3 深度替换 MiDaS**（[ByteDance-Seed/Depth-Anything-3](https://github.com/bytedance-seed/depth-anything-3)，2025）：空间一致几何，拉扯源头再减。
3. **Flash3D / 3DGS 范式**（[eldar/flash3d](https://github.com/eldar/flash3d)）：单图→高斯泼溅，质量上限最高，需换渲染范式（浏览器 3DGS viewer）。
4. **扩图全景背景**：GPT-Image-2/Seedream 把背景外扩成全景（盒壁贴真实延续内容）。
5. **CN 直连**：2 号账号登录 → worker 绑 `img.roginx.ink/depth-cache/*`。

## 五、环境速查

- 本地服务：`:8798` 组件 demo + wallpaper（`node serve-no-store.mjs . 8798`）、`:8799` mesh 渲染器
- 仓库：github.com/janxland/3Ddeep-image（main，大权重/venv/bins 不入库，见 .gitignore）
- checkpoints：HF 镜像 `ai-minamo/3d-photo-inpainting`（原 VT filebox 已死）
- demo 模型：`depth-component/demo/{vendor,models}`（同源化，受限 webview 可用）
- 组件属性：`mesh-src` / `depth-src`+`matte-src`+`bg-src` / `src`（全实时）；`amp` `zoom-max` `layers` `quality` `render` `cache-endpoint`
