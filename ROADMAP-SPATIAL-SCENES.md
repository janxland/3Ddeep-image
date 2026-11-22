# ROADMAP — iOS 26 空间场景复刻(下阶段方向, 实验已暂停)

> 2026-11-22 立项。
> **实验结论(用户裁定)**: 实时 photo 两层范式(POM/软分层/MI-GAN)的视觉质量
> 远低于 CVPR20 资产轨(离线多层 mesh+补全网络), 已否决回退(24140ea)。
> 差距本质: 离线管线是逐像素多层几何+神经补全, 实时两层近似只有贴图位移。
> 下阶段若继续此方向, 需先解决实时多层几何的质量问题(WebGPU compute 离线化缓存),
> 而非继续在两层近似上打补丁。研究结论(公式/复刻/坑)保留如下, 仍有参考价值。本文档是原理研究与演进路线的持久记录。
> 触发:用户反馈"背景有洞/边缘锯齿"→ 联网研究 iOS 26 空间场景(Spatial Scenes)原理。

## 一、Apple 官方原理链(已证实)

iOS 26 「空间场景 / Spatial Scenes」= Vision Pro 技术下放 iPhone:

```
单图 → 神经引擎端侧单目度量深度 → 深度图分层 → 陀螺仪/重力向量驱动视差渲染
```

- 深度模型推断为 Apple **Depth Pro** 系列:
  - Depth Pro(2024,开源,零样本度量深度,锐利边界,亚秒级 2.25MP)
  - **Depth Pro 2**(2025,视频级时序一致深度,专供空间场景)— https://huggingface.co/apple/DepthPro2
- 锁屏时钟可被前景主体遮挡(iOS 16 depth effect)+ iOS 26 新增 3D 视差
- 陀螺仪改为**重力向量**驱动 + 静止回中(复刻者 M5 实测经验)

## 二、逆向复刻印证(GitHub)

**DanXinG-bit/spatial-scene-android**(Android 复刻,里程碑 = 答案书):

| 里程碑 | 做法 | 对照我们 |
|---|---|---|
| M1 | Depth Anything V2 Small fp16 端侧 | ✅ 完全一致 |
| M2 | 引导滤波边缘精修 | ✅ guidedDepth 同款 |
| M3 | LDI 前景刚体 + 金字塔补全 | ✅ 走过 |
| M4 | 4 层 k-means 切片 + 逐层补全 | ✅ tune 脚本 K 层 |
| **M5** | **放弃硬分层 → SLIDE 式软分层** | 🎯 下一步 |
| **M6** | **MI-GAN 端侧补全(28MB)** | 🎯 生成式补洞 |

## 三、两个关键技术

### 1. SLIDE 软分层(锯齿/波浪纹根治)— Jampani et al., ICCV 2021
- 论文: https://openaccess.thecvf.com/content/ICCV2021/papers/Jampani_SLIDE_Single_Image_3D_Photography_With_Soft_Layering_and_Depth-Aware_ICCV_2021_paper.pdf
- **精确公式(arXiv 2109.01068 方法节)**:
  - 可见度: `A = exp(−β·‖∇D‖²)`(∇ 为 Sobel, β 标量; 视差梯度越大→前景越透明→透过不连续看背景)
  - 两层分解: 前景(I, D, A) + 背景(补全后 Ĩ, D̃)
  - 遮挡推理: 硬规则 `D(x,y) − D(xi,yj) > ρ·K(xi,yj)`(K=欧氏距离) → 软化 `S = tanh(γ·max(D−D−ρK))`(扫描线邻域卷积, 可降采样算再上采样)——**这个 S 就是 MI-GAN 的补全 mask 生成器**
  - 双层合成: `I* = A·I_T + (1−A)·Ĩ_T`(前景按可见度与背景混合——我们已实现的 blend 是它的运行时形态)
  - Matting 修正: `A' = A·(1−(M̄−M)·(1−Ŝ))`(U2Net+FBA matte 膨胀修正, 保发丝且尊重深度不连续)
- **放弃硬撕缝**:渲染无"撕"动作 → 波浪纹/锯齿从根源消失
- 我们的"边缘 alpha 软混合"是 A 的简化版; 完整版 = 逐像素可见度场( worker Sobel+exp 毫秒级)

### 2. MI-GAN 端侧生成式补全(洞的生成式方案)— ICCV 2023, Picsart, MIT 许可
- 论文: https://openaccess.thecvf.com/content/ICCV2023/papers/Sargsyan_MI-GAN_A_Simple_Baseline_for_Image_Inpainting_on_Mobile_Devices_ICCV_2023_paper.pdf
- **web 化已被完整验证**: [huangyongking418/p-inpaint](https://github.com/huangyongking418/p-inpaint) — 浏览器内 MI-GAN inpainting
  (ONNX Runtime Web WASM 推理, 画笔 mask, IndexedDB 缓存模型, 与我们组件的模型缓存策略同构)
- **模型确切来源与大小**(MI-GAN 作者官方 HF):
  - `andraniksargsyan/migan` → `migan.onnx` **28.2MB**(raw 版: uint8 RGB+mask 输入, 自写前后处理)
  - `migan_pipeline_v2.onnx`(pipeline 版: 内嵌前后处理, p-inpaint 采用)
- 限制: 单图 ≤400 万像素(浏览器内存); 复杂纹理/大面积缺失会糊
- 我们的接入点: SLIDE 遮挡掩码 S → 窄带 mask → MI-GAN patch(512 crop)→ 真生成背景纹理
  (现有背景兜底平面保证无洞, MI-GAN 作为 quality=best 的生成式升级)

### 3. Web 端生图参考
- MLC web-stable-diffusion(WebGPU 全端内, https://github.com/mlc-ai/web-stable-diffusion)— 但 SD 级模型 2GB+,过重;MI-GAN 28MB 是正确量级

## 四、当前已落地的相关能力

- 背景兜底平面(BACK_VS/BACK_FS, z=2.85 最远层, 完整原图)✅
- 边缘色腐蚀(羽化带顶点色向内 5px 重采样×2 轮)✅
- 边缘 alpha 软混合(主层 flag 字节=matte alpha, MESH_FS uEdgeAlpha smoothstep)✅
- FXAA 后处理(FBO+全屏 pass)✅
- mask 画笔修正(端内轨 matte 编辑→7ms 重建)✅
- 数学闭环 tune-person-depth.py(K/sigma/edge 网格×四维评分)✅

## 五、演进路线(按 ROI, 2026-11-22 探索后修订)

1. ✅ **SLIDE 可见度场 A** 进 buildLayeredMeshCore(ea3c9e2, worker Sobel+exp 毫秒级):
   三段语义(人物核心恒不透明 / 羽化带 min(matte,A) / 背景区 A); 线上 2.4x 实测撕缝软过渡
2. ✅ **SLIDE 遮挡掩码 S**(occlusion.js, 扫描线 O(N)): 前景 disparity−背景 > ρK + tanh 软化
3. ✅ **MI-GAN web 化**(d1b5d64): pipeline ONNX 26.8MB(作者官方 HF), ORT Web **WASM**(WebGPU
   uint8 崩坏已绕开, WASM 一次推理 ~0.5-3.8s)→ 背景平面升级为生成纹理;
   hole = 人物区 ∪ 遮挡区(纯背景语义, 否则人物鬼影)
4. ⏭ Depth Pro 2 ONNX(后置, ViT 大模型浏览器吃力)
5. ⏭ 重力向量式运镜(静止回中)对齐 iOS 手感
6. ⏭ β/rho/gamma 参数进 tune 数学闭环

### 踩坑记录(2026-11-22 MI-GAN 接入)
- ORT Web **WebGPU EP 对 uint8 输入输出崩坏**(彩虹噪声) → MI-GAN 强制 WASM(一次性推理够快)
- torch.onnx 导出 NCHW: **喂 HWC 交错数据会通道错位**(同一彩噪症状) → 严格平面布局
- pipeline ONNX 动态 H/W: 内嵌 bbox crop + resize + 羽化合成, web 端只喂 image+mask uint8
- mask 语义: **255=known(保留), 0=hole(生成)**; composed = image*mask + output*(1−mask)
- 主层网格**背面无剔除**会在大视差时露出镜像成像 → CULL_FACE BACK; 投影矩阵 y 取负
  翻转 winding → 背景平面 quad 需局部禁 cull
