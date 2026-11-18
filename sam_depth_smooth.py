# sam_depth_smooth.py — SAM2 mask 引导的深度图平滑（CVPR20 管线预处理步骤）
# 原理: 撕裂判据=深度突变。SAM 说"这些像素是同一个物体" → mask 内大核平滑消噪
#       (物体内部不再产生伪突变 → 不撕脸/不撕椅子), mask 外保留原深度(物体间跳变→轮廓撕缝照常)
# 用法: from sam_depth_smooth import sam_smooth_depth; depth = sam_smooth_depth(depth, image_rgb, config)
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def sam_smooth_depth(depth, image, config=None, sigma=7.0, feather=4, min_area=0.02, max_masks=8):
    """depth: (h,w) float 归一化深度; image: (h,w,3) uint8 → 平滑后 depth"""
    from ultralytics import SAM
    from PIL import Image as PImage
    h, w = depth.shape
    sam = SAM('sam2.1_s.pt')
    res = sam(PImage.fromarray(image), device='mps')
    smooth = depth.astype(np.float32).copy()
    applied = 0
    if res and res[0].masks is not None:
        raw = res[0].masks.data.cpu().numpy()
        masks = []
        for m in raw:
            mm = np.asarray(PImage.fromarray((m * 255).astype(np.uint8)).resize((w, h), PImage.LANCZOS)).astype(np.float32) / 255.0
            if (mm > 0.35).mean() < min_area:
                continue
            masks.append(mm)
        masks.sort(key=lambda m: -(m > 0.35).mean())
        for mm in masks[:max_masks]:
            mbin = (mm > 0.35).astype(np.float32)
            sm = gaussian_filter(depth, sigma=sigma)
            # 权重: 物体内部=1(全平滑), mask 边缘 feather 环=0.5 过渡, 背景=0(原深度)
            dil = maximum_filter(mbin, size=feather * 2 + 1)
            wgt = np.clip(mbin + (dil - mbin) * 0.5, 0, 1)
            smooth = smooth * (1 - wgt) + sm * wgt
            applied += 1
    print(f'[sam-smooth] {applied} 物体平滑应用 ✔', flush=True)
    return smooth
