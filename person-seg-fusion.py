# person-seg-fusion.py v3 — 人物 matte 定稿: salient(完整人物) − 椅子连通域(大件家具)
# 原理: isnet salient 人物完整(含头/发)但粘家具; SAM2 auto 家具实例分割准但人物有洞。
#       furn = salient − SAM2box人物实例 → 连通域分析 → 只减「大件家具组件」(≥8% 画幅),
#       小碎片(头发/边缘噪声)保留在人物层。→ 人物完整 + 家具干净, 两全。
# 用法: python person-seg-fusion.py [stem...]
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(REPO, 'depth-component', 'demo', 'img')

def main(stems):
    from ultralytics import YOLO, SAM
    from simple_lama_inpainting import SimpleLama
    from PIL import Image, ImageFilter
    import numpy as np
    from scipy.ndimage import label, binary_fill_holes

    yolo = YOLO('yolo11n.pt')
    sam = SAM('sam2.1_s.pt')
    lama = SimpleLama()

    for stem in stems:
        print(f'[{stem}]', flush=True)
        img = Image.open(os.path.join(DEMO, stem + '.jpg')).convert('RGB')
        W, H = img.size

        # ① salient 基底: isnet(alpha 重映射) — 人物完整含头/发
        sal = np.asarray(Image.open(os.path.join(DEMO, stem + '.matte.png')).convert('L').resize((W, H))).astype(np.float32) / 255.0

        # ② 人物实例参考: SAM2 box-prompt（YOLO person 框）— 身体准、头部可能有洞
        det = yolo(img, classes=[0], verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy().tolist() if det.boxes is not None else []
        pref = np.zeros((H, W), dtype=np.float32)
        if boxes:
            res = sam(img, bboxes=boxes, verbose=False)
            if res and res[0].masks is not None:
                for m in res[0].masks.data.cpu().numpy():
                    mi = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
                    pref = np.maximum(pref, mi)

        # ③ furn = salient − person实例 → 连通域 → 大件家具(≥8%画幅)才减
        furn = np.clip(sal - pref, 0, 1) > 0.4
        lbl, n = label(furn)
        chair = np.zeros((H, W), dtype=np.float32)
        areas = []
        for ci in range(1, n + 1):
            comp = lbl == ci
            a = comp.mean()
            if a >= 0.08:
                chair = np.maximum(chair, comp.astype(np.float32))
                areas.append(round(a * 100, 1))
        print(f'  大件家具组件: {areas}% 画幅', flush=True)

        # ④ final = salient − chair(外扩6px+羽化) → 人物完整, 家具干净剥离
        chair_d = np.asarray(Image.fromarray((chair * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(13))).astype(np.float32) / 255.0
        final = np.clip(sal - chair_d, 0, 1)
        # 内部小洞填充（人物身上的低对比碎片）
        solid = binary_fill_holes(final > 0.35).astype(np.float32)
        final = np.maximum(final, solid)  # 内部洞归回人物层(满强度)
        final_u8 = (np.clip(final, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(final_u8).save(os.path.join(DEMO, stem + '.matte.png'))

        # ⑤ 调试页: salient / person实例 / chair / final
        dbg = Image.new('RGB', (W * 4, H))
        dbg.paste(Image.fromarray((sal * 255).astype(np.uint8)).convert('RGB'), (0, 0))
        dbg.paste(Image.fromarray((pref * 255).astype(np.uint8)).convert('RGB'), (W, 0))
        dbg.paste(Image.fromarray((chair * 255).astype(np.uint8)).convert('RGB'), (W * 2, 0))
        dbg.paste(Image.fromarray(final_u8).convert('RGB'), (W * 3, 0))
        dbg.save(os.path.join(DEMO, stem + '.dbg-fusion.png'))

        # ⑥ LaMa bgfill 重生成（人物+大件家具整体移除）
        mask = Image.fromarray((np.clip(sal, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(6))
        out = lama(img, mask).resize(img.size, Image.LANCZOS)
        out.save(os.path.join(DEMO, stem + '.bgfill.png'))
        print(f'  ✔ matte 定稿 + bgfill', flush=True)

if __name__ == '__main__':
    main(sys.argv[1:] or ['w167-0', 'w167-2', 'w167-3'])
