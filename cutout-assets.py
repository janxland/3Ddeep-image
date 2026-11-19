# cutout-assets.py v3 — 语义实例分层资产
# 人物 = isnet 软 matte 完整保留(含头/白衬衫) − 检测到的家具实例(SAM2 box-prompt 精确减除)
# 家具 = YOLO 检测 chair(56)/couch(57)/dining table(60) → 每框 SAM2 box-prompt
# 背景层 = LaMa 补全(人物+家具整体移除)
# 用法: python cutout-assets.py <stem>...
import os
import sys
import json

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(REPO, 'depth-component', 'demo', 'img')

def main(stems):
    from ultralytics import YOLO, SAM
    from simple_lama_inpainting import SimpleLama
    from PIL import Image, ImageFilter
    import numpy as np
    from rembg import remove, new_session

    yolo = YOLO('yolo11n.pt')
    sam = SAM('sam2.1_s.pt')
    lama = SimpleLama()
    isnet_session = new_session('isnet-general-use')

    for stem in stems:
        print(f'[{stem}]', flush=True)
        img = Image.open(os.path.join(DEMO, stem + '.jpg')).convert('RGB')
        W, H = img.size

        # ① 软 matte（isnet, 连续 0..1, 含头/白衬衫——完整人物）
        soft = np.asarray(remove(img, session=isnet_session, only_mask=True).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        # ② YOLO 只检测家具: chair=56, couch=57, dining table=60
        det = yolo(img, classes=[56, 57, 60], conf=0.1, iou=0.3, agnostic_nms=False, max_det=20, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy().tolist() if det.boxes is not None else []
        print(f'  家具检测: {len(boxes)} 个')

        # ③ 家具框 → SAM2 box-prompt → 家具实例 mask
        furn_m = np.zeros((H, W), dtype=np.float32)
        for pb in boxes:
            mi = np.zeros((H, W), dtype=np.float32)
            res = sam(img, bboxes=[pb], verbose=False)
            if not (res and res[0].masks is not None):
                continue
            for m in res[0].masks.data.cpu().numpy():
                mi = np.maximum(mi, np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0)
            furn_m = np.maximum(furn_m, mi)

        # ④ 人物 alpha = soft − 家具实例羽化（人物完整保留, 家具精确剥离）
        def feather(m, blur=4):
            return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))).astype(np.float32) / 255.0
        fu_f = feather(furn_m, 3)
        alpha = np.clip(soft - fu_f * 0.9, 0, 1)

        # ⑤ RGBA 抠图
        rgba = np.dstack([np.asarray(img), (alpha * 255).astype(np.uint8)])
        Image.fromarray(rgba, 'RGBA').save(os.path.join(DEMO, stem + '.cutout.png'))

        # ⑥ 背景层 = LaMa(人物 ∪ 家具 整体移除)
        mask_all = np.clip((soft > 0.35).astype(np.float32) + furn_m, 0, 1)
        mask_img = Image.fromarray((np.asarray(Image.fromarray((mask_all * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(13))).astype(np.float32) / 255.0 * 255).astype(np.uint8))
        bg = lama(img, mask_img).resize(img.size, Image.LANCZOS)
        bg.save(os.path.join(DEMO, stem + '.bgfill.png'))

        # ⑦ 层深
        dp = os.path.join(DEMO, stem + '.depth.png')
        depth = np.asarray(Image.open(dp).convert('L')).astype(np.float32) / 255.0
        pmask = alpha > 0.5
        d_person = float(np.median(depth[pmask])) if pmask.any() else 0.8
        bgm = (furn_m > 0.35) | (alpha < 0.4)
        d_bg = float(np.median(depth[bgm])) if bgm.any() else 0.3
        zN, zF = 1.2, 2.6
        person_z = float(zF - d_person * (zF - zN))
        bg_z = float(max(person_z + 0.7, zF - d_bg * (zF - zN)))
        json.dump(dict(W=W, H=H, personZ=round(person_z, 3), bgZ=round(bg_z, 3),
                       f=round((H / 2) / 0.5, 1)),
                  open(os.path.join(DEMO, stem + '.layers.json'), 'w'))
        print(f'  ✔ personZ={person_z:.2f} bgZ={bg_z:.2f} 人物区{int((alpha > 0.5).sum() / (W * H) * 100)}%')

if __name__ == '__main__':
    main(sys.argv[1:] or ['w167-2', 'w167-3'])
