# 当前产出细节问题清单与优化路线（持久记录）

> 管线：单图 → MiDaS 深度(640 fp32) → LDI 分层 + 3 补全网络 → 多层 mesh → uint16 量化 → WebGL
> 基线资产：e1d28550 / w167-2 / w167-3（CVPR20 管线，已复现锁定）
> 本文档持续更新：每个问题 → 根因 → 前沿方案 → 试错状态

## 一、现存细节问题（按用户观察排序）

### P1 人物边缘拉扯/撕裂伪影 ⭐核心
- **现象**：环视时人物轮廓边缘出现拉丝、碎片、锯齿；面部分块错位
- **根因**：撕裂判据 = 深度突变（`tear_edges`, threshold 0.00025）。MiDaS v2(2020) 深度在物体
  内部有高频噪声 → 物体内部被误撕（脸变形）；物体边界处深度渐变 → 该撕的不撕（边缘拉丝）
- **语义盲区**：撕裂判据不知道"这些像素是同一个人"——人物内部噪声=背景噪声同罪

### P2 物体间粘连（人物-椅子-垫子）
- **现象**：人物坐在椅子上 → 椅子/垫子/扶手与人物深度连续 → 整体被当作一层
- **根因**：LDI 分层只看深度连续性，无实例语义。人物与接触物深度物理上就连续
- **SAM2 验证结论**：SAM2 实例分割能准确分出椅子/人物（4am 验证+tri 图），但"压 slab"用法失败

### P3 遮挡补全区质量糊
- **现象**：环视露出的补全背景是 2020 pix2pix 水准的低频糊块
- **根因**：color-model/depth-model 是 2020 年 GAN，分辨率与先验远弱于现代扩散/LaMa

### P4 深度图整体质量
- **现象**：MiDaS v2 边界糊、平面不平（墙面/地板有波动）
- **根因**：2020 模型 + affine-invariant disparity 目标（视觉优先，几何精度次之）

### P5 资产体积
- 66MB/图（量化后），deflate 可至 20MB；移动端偏重

## 二、前沿方案映射（2025-11 检索）

| 问题 | 方案 | 来源 | 状态 |
|---|---|---|---|
| P1/P4 | **DA3（Depth Anything 3）** 替换 MiDaS：spatially consistent geometry，边界更净，平坦面更平；`pip install depth-anything-3`，DA3-BASE 单图即用 | [ByteDance-Seed/Depth-Anything-3](https://github.com/bytedance-seed/depth-anything-3) / [arXiv:2511.10647](https://arxiv.org/html/2511.10647v1) | 🔄 试错中(T1) |
| P1/P2 | **SAM2 mask 引导分层**：auto 实例 → 深度 k-means 分箱按物体 → 撕缝吸附轮廓（2025 已有同思路管线） | 检索见 EXPLORATION-20261118.md | 📋 设计中(T2) |
| P1 | **MaskDepth: Layered Depth Refinement with Mask Guidance** —— 与本问题几乎同构的 mask 引导分层精化 | [sooyekim.github.io/MaskDepth](https://sooyekim.github.io/MaskDepth) | 📋 参考 |
| P1 | **Occlusion Boundary and Depth: Mutual Enhancement**（2025-05）遮挡边界与深度互相增强 | [arXiv:2505.21231](https://arxiv.org/html/2505.21231v1) | 📋 参考 |
| P2/P3 | **Amodal 完成**：[Open-World Amodal Appearance Completion (CVPR25)](https://github.com/saraao/amodal) 输出完整 RGBA 物体（含被遮挡部分） | CVPR25 | 📋 二期 |
| P3 | **LaMa 替换 pix2pix color-model**：补全区纹理质量升级（已在 bgfill 验证过 LaMa 质量） | simple-lama-inpainting | 🔄 试错中(T3) |
| 全部 | **Flash3D 3DGS 范式**（牛津/Niantic, 159引）：单图→per-pixel 高斯 | [eldar/flash3d](https://github.com/eldar/flash3d) | 二期（换渲染范式） |

## 三、试错计划（持续更新）

### T1：DA3 深度替换 MiDaS（进行中）
- 步骤：装 depth-anything-3 → w167-3 出 DA3 深度 → 转管线 npy 格式 → 重跑 mesh → IAB 对比
- 验收：边界干净度 / 平面平整度 / 人物完整性 三项对照
- 风险：DA3 输出是 metric/relative 混合格式，需适配管线深度约定（near=1 归一）

### T2：SAM2 mask 深度预处理（已完成 v1，保留为叠加项）
- `sam_depth_smooth.py`：mask 内高斯平滑消内部噪声（已验证有效，sigma=7）
- 进阶：mask 内深度中位数替换（更激进，v2 slab 教训=不能压平，只消噪）

### T3：补全网络换 LaMa（设计）
- mesh.py 的 rgb_model(color-model) 输出 → 改调 LaMa 对撕裂区修复
- 收益：补全区纹理质量; 风险：接口适配

## 四、验收基准
- 格 3（w167-3 办公椅）环视：脸不变形 / 椅子不拉扯 / 补全区可用
- 格 1（e1d28550）：不回退
- 资产 ≤ 25MB/图（deflate 后）
