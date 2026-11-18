# 人物/物体实例分离方案（PERSON-SEG-PLAN）

> 病根定稿：rembg/isnet 系是「显著前景分割」——人物坐着时，人物+凳子+扶手是同一个显著
> 连通域，模型没义务拆开。这是语义错位，不是模型调参问题。三张图的症状全部由此解释。

## 一、方案（三层分离管线 v3）

```
① 人物实例 mask: YOLO(person) 检测框 → SAM2 box-prompt → 人物实例轮廓
   （实例级分割：即使人物坐在凳子上，凳子也不会进人物 mask）
② 轮廓细节: isnet alpha 重映射 matte 与人物实例 mask 取交
   （实例保证"不连错"，isnet 保证发丝/边缘细节）
③ 其他物体实例（凳子/扶手/沙发）: SAM2 剩余实例各自独立 → 深度带独立 → 不随人物运动
④ 背景层: LaMa 补全（mask = 人物实例 ∪ 其他被移动物体实例, 各自或合并跑）
⑤ 分层构建: 沿用现有 _buildLayered（v2 失败教训遵守——不压 slab, 保留 CVPR20 像素深度细节,
   只换 mask 来源 + 层归属）
```

**v2 失败教训吸收**：不动几何深度（保留像素级细节），只修「mask 语义」（谁跟谁一层）。

## 二、学术前沿（2025, 学名 Amodal Segmentation）

| 工作 | 内容 | 对本方案的意义 |
|---|---|---|
| [Open-World Amodal Appearance Completion](https://github.com/saraao/amodal) (CVPR 2025) | 分割+遮挡分析+补全统一, 输出**完整 RGBA 物体** | 终极形态: 人物被凳子挡住的腿也能补全 → 视差时人物层无洞 |
| [Segment Anything, Even Occluded](https://arxiv.org/abs/2503.06261) (CVPR 2025) | SAM 系 amodal 实例分割(可见+不可见部分) | SAM2 后处理升级件 |
| [pix2gestalt](https://github.com/saraao/amodal) 系 | 零样本 amodal 合成 | 备选 |
| YOLO→SAM2 box-prompt 管线 | [Det-SAM2](https://github.com/motern88/Det-SAM2) / [yolo-sam2](https://github.com/praveena396/yolo-sam2-detection-segmentation) | 工程链现成, ultralytics 原生支持 box prompt |

## 三、凌晨 4 点验证任务（自动执行）

`person-seg-validate.py` 对三张演示图:
1. YOLO person 检测 → SAM2 box-prompt → 人物实例 mask（对比 isnet 旧 matte: IoU+并排差异图）
2. 重跑 LaMa bgfill（用人物实例 mask）
3. 覆盖 `demo/img/*.matte.png` / `*.bgfill.png`（旧文件备份 `.isnet` 后缀）→ demo 打开即新版
4. 报告 `PERSON-SEG-VALIDATION.md` + git commit+push

验收（醒来后浏览器看三格）：人物身后露出的背景是 LaMa 补全（人物不含凳子）→ 凳子留在背景层
按背景深度运动 → 「凳子和人物弄在一起」根治。
