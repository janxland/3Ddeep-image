# 轻资产探索报告（2026-11-18 深夜轮）

> ## 🌙 第二轮（语义分层 + 360° 包围 + 头动视差）——已落地
>
> 用户裁定：①人物与背景必须独立处理 ②背景要 360° 包围。已全部实现：
>
> ### 语义双层分离（重写 _buildLayered）
> - **ModNet 人像抠图**（Xenova/modnet 25.9MB，vendor 同源 + WebGPU 实测 6.5-7.8s 全实时；rembg/u2net 预计算 matte 39-122KB）
> - **背景层**：全格点独立几何；人物区 z 垫高 + 低频渐变色填充（上下亮度渐变）——人物身后不再是黑洞
> - **人物层**：独立顶点段（alpha>0.35 撕出），深度带内 12% 微起伏=刚性，颜色用原像素
> - 轮廓撕裂：背景层跳过混合 quad，人物层只连纯 fg quad → 断层严格沿 matte 轮廓
> - 深度获取链: matte-src 预计算 / ModNet 实时 / matte=off 关闭
>
> ### 360° 全景包围盒（程序生成零资产）
> - 5 面内壁（左右壁用原图边缘背景色带均值，后壁/顶/底用全局低频色），7×7 网格/面，相机任意朝向都有内容
> - `meta.surround=true` 标记
>
> ### 相机模型修正（重要教训）
> - 旋转主导 = 整屏均匀平移（无相对视差，"看呆照"）——已废弃
> - **平移主导（头动视差）**: 眼睛在椭圆上平移（ex=±0.22, ey=±0.16 世界单位）、看向前方 → 近景位移>远景（物理正确）≈1.9× 差
> - 屏幕定点采样法验证视差有噪声（内容随位移跨界），几何层物理正确性由静态相机 16×16 采样 nonRedRatio=1.00 背书
>
> ### IAB（内嵌浏览器）限制清单（真实浏览器无此问题）
> 1. 跨域动态 import 被 CSP 拦 → 组件支持 `__DEPTH_VIEWER_TFS__` 同源 vendor 注入
> 2. hf-mirror 网络断 → `__DEPTH_VIEWER_LOCAL_MODELS__` 同源模型注入
> 3. wasmPaths 相对路径被拼到 CDN 基址 → 必须绝对 URL
> 4. IO/setTimeout 触发器可能哑火 → 同步可见性检查兜底（已入组件）
> 5. 自动化标签页 rAF 可能不泵帧 → 验证时手动泵帧（`el._frame(t)` 循环）
> 6. 跨域 mesh 资产被 CSP 拦 → demo 全部同源
>
> ### 实测数据
> - 预计算链: 83ms 初始化（图+深度+matte 三件套 0.4MB）
> - 全实时链: 6.5-7.8s（DA-V2 q8 + ModNet fp32 双推理, WebGPU, 模型同源）
> - 面数: 65k（分层）/ 990k（预计算 mesh）
>
> ### 待办
> - matte 边缘羽化（shader 3 行）视用户反馈
> - CN 直连缓存池（2 号账号登录 + img.roginx.ink/depth-cache/* 路由）
> - GPT-Image-2 补全区精修（提质不减肥）



目标：不降质量，压缩浏览器端资产。现状基线：预计算真几何 57-62MB/图。

## ✅ 已实测落地（今晚）

### 实验 A+B：deflate 压缩 + 浏览器原生解码 → **57MB → 20MB（2.9×）**
- 索引 deflate 4.1×（51.6→12.5MB），顶点仅 1.1-1.2×（量化后熵高，符合预期）
- 解码端 `DecompressionStream('deflate')` 全主流浏览器原生，**零依赖零 wasm**
- 组件已接入：`web_meta.json` 标记 `compressed:'deflate'` 自动走解压路径；demo 第一格已用 20MB 资产验证（像素回读正常）
- 脚本：`compress-assets.mjs`；装配线 finalize-batch.py 后续可并此步

### 实验 C：浏览器分层模式边缘外扩 → 消黑边
- `_buildLayered` 网格外扩一圈（数据 clamp 采样、位置真实外延）→ 环视边缘露拉伸背景
- IAB 实测 67,500 面 64ms 构建完成，画面正常

## 📋 下一步路线（按性价比排序，均为估算）

| 路线 | 预期资产 | 质量 | 成本 |
|---|---|---|---|
| ① 索引重建式压缩 | **~10-12MB** | 无损 | 生产端输出「撕裂掩码」bitmap（大部分面片是规则网格 quad，索引可从掩码+网格尺寸重建）；掩码 PNG ~100KB；改造 mesh.py 或后处理推导 |
| ② 顶点纹理化 | 顶点 10→8B 且颜色质量↑（原分辨率 jpeg ~0.2MB 替代顶点色） | 更好（撕裂边颜色不再混叠） | shader 加 uTex，uv 由 aPos 反算（bbox 在 meta）；~50 行 |
| ③ meshopt/Draco | 索引再压 30-50%（12.5→7-9MB），顶点重排后总 ~14-16MB | 无损 | 引入 wasm 解码器 ~85KB；仅当①不做时值得 |
| ④ 远景抽稀+简化 | 面片 4.3M→1.5-2M，索引 12.5→5-6MB，总 ~12-14MB | 远景轻微简化（环视幅度小，几乎不可见） | quadric decimation（three.js SimplifyModifier 或 meshopt simplify） |
| ⑤ 深度模型升级 | 不减资产但提质：MiDaS v2 → **MoGe-2 / DA-V2-large**（生产端重跑） | 深度边缘/平面度显著提升 | Mac 重跑管线即可，资产结构不变 |
| ⑥ 补全区精修 | 不减资产但提质：GPT-Image-2 只重绘 flag≥2 区域 | 补全背景从「糊」变「真」 | 一次 API 调用/图；需 key |
| ⑦ 2 号账号 CN 直连 | 缓存池路由 img.roginx.ink/depth-cache/* | — | 需登录 2 号账号（一次授权） |

**理论极限**：①+②+④ 叠加 ≈ **8-10MB/图**（无损路线）或 ~6MB（④远景微损）。当前 20MB 已可用。

## 组件现状属性速记
- `mesh-src`（真几何, 支持 deflate 压缩资产）/ `src`+`depth-src`（分层, 默认）/ `src`（全实时 DA-V2 阶梯）
- `render="layered|warp"`、`layers="N"`、`amp`、`zoom-max`、`cache-endpoint`、`quality`
- 本地服务：`:8799` mesh 渲染器、`:8798` 组件 demo（nohup 常驻）
