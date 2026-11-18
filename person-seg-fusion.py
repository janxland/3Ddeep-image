# person-seg-fusion.py — 人物 matte 定稿生成（SAM2 box-prompt mask + 内部洞填充）
# 原理: SAM2 box-prompt 人物实例(精准, 无家具) → binary_fill_holes 填衣服内部洞(黑裙条纹等)
#       → 人物层 matte；LaMa 以此为 mask 补全背景(人物+凳子整体移除) → 背景层纹理
# 用法: python person-seg-fusion.py [stem...]  （需已有 <stem>.matte-person.png 或现场生成）
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(REPO, 'depth-component', 'demo', 'img')

def main(stems):
    from ultralytics import YOLO, SAM
    from simple_lama_inpainting import SimpleLama
    from PIL import Image, ImageFilter
    import numpy as np
    from scipy.ndimage import binary_fill_holes

    yolo = YOLO('yolo11n.pt')
    sam = SAM('sam2.1_s.pt')
    lama = SimpleLama()

    for stem in stems:
        print(f'[{stem}]', flush=True)
        img = Image.open(os.path.join(DEMO, stem + '.jpg')).convert('RGB')
        W, H = img.size
        pref_path = os.path.join(DEMO, stem + '.matte-person.png')
        if os.path.exists(pref_path):
            pref = np.asarray(Image.open(pref_path).convert('L').resize((W, H))).astype(np.float32) / 255.0
        else:
            # 现场生成: YOLO person → SAM2 box-prompt
            det = yolo(img, classes=[0], verbose=False)[0]
            boxes = det.boxes.xyxy.cpu().numpy().tolist()
            assert boxes, '未检测到人物'
            res = sam(img, bboxes=boxes, verbose=False)
            pref = np.zeros((H, W), dtype=np.float32)
            if res and res[0].masks is not None:
                for m in res[0].masks.data.cpu().numpy():
                    mi = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
                    pref = np.maximum(pref, mi)

        # 人物 = SAM2 mask 填内部洞（黑裙条纹等低对比区）; isnet 提供边缘细节（∪ 边界带）
        pb = pref > 0.35
        person = binary_fill_holes(pb).astype(np.float32)
        isnet = np.asarray(Image.open(os.path.join(DEMO, stem + '.matte.isnet.png') if os.path.exists(os.path.join(DEMO, stem + '.matte.isnet.png')) else os.path.join(DEMO, stem + '.matte-person.png')).convert('L').resize((W, H))).astype(np.float32) / 255.0
        # isnet ∩ 人物外扩区: 发丝细节并入（isnet 在人物边界附近的细节纹理）
        from scipy.ndimage import binary_dilation
        band = binary_dilation(person > 0, iterations=6)
        person = np.maximum(person, (isnet > 0.45) * band)

        final_u8 = (np.clip(person, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(final_u8).save(os.path.join(DEMO, stem + '.matte.png'))
        dbg = Image.new('RGB', (W * 3, H))
        dbg.paste(Image.fromarray((pref * 255).astype(np.uint8)).convert('RGB'), (0, 0))
        dbg.paste(Image.fromarray(final_u8).convert('RGB'), (W, 0))
        dbg.paste(Image.fromarray((pref * 255).astype(np.uint8)).convert('RGB'), (W * 2, 0))
        dbg.save(os.path.join(DEMO, stem + '.dbg-fusion.png'))

        # LaMa: 人物+接触物整体移除 → 背景层纹理
        mask = Image.fromarray(final_u8).filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(6))
        out = lama(img, mask).resize(img.size, Image.LANCZOS)
        out.save(os.path.join(DEMO, stem + '.bgfill.png'))
        print(f'  ✔ matte(填洞+边缘细节) + bgfill 重生成', flush=True)

if __name__ == '__main__':
    main(sys.argv[1:] or ['w167-0', 'w167-2', 'w167-3'])
