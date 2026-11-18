# person-seg-validate.py — 人物实例分割验证（YOLO person → SAM2 box-prompt → LaMa bgfill）
# 用法: python person-seg-validate.py [stem...]   （默认三张演示图; 需 3d-photo-inpainting/.venv）
# 产出: demo/img/<stem>.matte-person.png, .bgfill-person.png, 对比图, 旧文件备份 .isnet 后缀
#       报告 PERSON-SEG-VALIDATION.md（IoU/覆盖/结论）
import os
import sys
import json

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(REPO, 'depth-component', 'demo', 'img')
STEMS = sys.argv[1:] or ['w167-0', 'w167-2', 'w167-3']
IOU_THRESHOLD = 0.35  # 新旧 matte IoU 高于此值=质量相当(不覆盖), 低于=旧 matte 有粘连(覆盖)

def main():
    from ultralytics import YOLO, SAM
    from simple_lama_inpainting import SimpleLama
    from PIL import Image, ImageFilter, ImageChops
    import numpy as np

    print('[1/4] 加载模型…', flush=True)
    yolo = YOLO('yolo11n.pt')  # COCO person=0
    sam = SAM('sam2.1_s.pt')
    lama = SimpleLama()

    report = {'images': {}, 'verdict': {}}
    for stem in STEMS:
        print(f'[{stem}] 处理中…', flush=True)
        img = Image.open(os.path.join(DEMO, stem + '.jpg')).convert('RGB')
        W, H = img.size
        old_matte = Image.open(os.path.join(DEMO, stem + '.matte.png')).convert('L').resize((W, H))

        # ① 人物检测（COCO class 0 = person）
        det = yolo(img, classes=[0], verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else []
        if len(boxes) == 0:
            print(f'  ✘ 未检测到人物, 跳过（保留原 matte）', flush=True)
            report['images'][stem] = {'status': 'no-person'}
            continue
        print(f'  人物框: {len(boxes)} 个', flush=True)

        # ② SAM2 box-prompt → 人物实例 mask（可能多人, 取并集）
        res = sam(img, bboxes=boxes.tolist(), verbose=False)
        person = np.zeros((H, W), dtype=np.float32)
        if res and res[0].masks is not None:
            for m in res[0].masks.data.cpu().numpy():
                mi = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
                person = np.maximum(person, mi)
        person_u8 = (person * 255).astype(np.uint8)
        Image.fromarray(person_u8).save(os.path.join(DEMO, stem + '.matte-person.png'))

        # ③ 对比指标: 新旧 matte IoU + 变化面积
        inter = np.minimum(person_u8, np.asarray(old_matte)).astype(np.float32).sum() / 255.0
        union = np.maximum(person_u8, np.asarray(old_matte)).astype(np.float32).sum() / 255.0
        iou = float(inter / max(union, 1))
        changed_px = float((np.abs(person_u8.astype(np.float32) - np.asarray(old_matte).astype(np.float32)) > 64).mean())
        print(f'  IoU(新vs旧)={iou:.3f} | 变化面积={changed_px:.1%}', flush=True)

        # ④ 并排差异图（左=旧, 中=新, 右=差异）
        diff = Image.fromarray(np.abs(person_u8.astype(np.int16) - np.asarray(old_matte).astype(np.int16)).astype(np.uint8))
        side = Image.new('RGB', (W * 3, H))
        side.paste(old_matte.convert('RGB'), (0, 0))
        side.paste(Image.fromarray(person_u8).convert('RGB'), (W, 0))
        side.paste(diff.convert('RGB'), (W * 2, 0))
        side.save(os.path.join(DEMO, stem + '.matte-compare.png'))

        # ⑤ 覆盖决策: IoU 低于阈值=旧 matte 粘连严重 → 用人物实例 mask 重跑 LaMa 并覆盖线上文件
        use_new = iou < IOU_THRESHOLD
        if use_new:
            os.replace(os.path.join(DEMO, stem + '.matte.png'), os.path.join(DEMO, stem + '.matte.isnet.png'))
            os.replace(os.path.join(DEMO, stem + '.bgfill.png'), os.path.join(DEMO, stem + '.bgfill.isnet.png'))
            Image.fromarray(person_u8).save(os.path.join(DEMO, stem + '.matte.png'))
            mimg = Image.fromarray(person_u8).filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(6))
            out = lama(img, mimg).resize(img.size, Image.LANCZOS)
            out.save(os.path.join(DEMO, stem + '.bgfill.png'))
            print(f'  ✔ IoU={iou:.3f}<{IOU_THRESHOLD} → 采用人物实例 matte, LaMa bgfill 重生成', flush=True)
        else:
            print(f'  - IoU={iou:.3f}≥{IOU_THRESHOLD} → 保留原 matte（差异: {changed_px:.1%}）', flush=True)

        report['images'][stem] = {'person_boxes': len(boxes), 'iou': round(iou, 3),
                                  'changed': round(changed_px, 3), 'use_new': use_new}
        report['verdict'][stem] = 'new-adopted' if use_new else 'kept'

    # ⑥ 报告 + 提交
    with open(os.path.join(REPO, 'PERSON-SEG-VALIDATION.md'), 'w') as f:
        f.write('# 人物实例分割验证报告（凌晨 4 点自动任务）\n\n')
        f.write(json.dumps(report, indent=1, ensure_ascii=False))
        f.write('\n\n对比图: demo/img/<stem>.matte-compare.png（左旧/中新/右差）\n')
        f.write('采用新 matte 的图: demo 打开即新效果（人物不含凳子/扶手）\n')
    print('✔ PERSON-SEG-VALIDATION.md', flush=True)

if __name__ == '__main__':
    main()
