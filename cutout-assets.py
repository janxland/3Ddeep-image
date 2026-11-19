# cutout-assets.py — 双层合成资产: 人物 RGBA 抠图(全分辨率软边) + LaMa 背景 + 层深参数
# 用法: python cutout-assets.py <stem>...  （需 .jpg/.depth.png/.matte-person.png；.bgfill.png 自动生成）
import os
import sys
import json

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(REPO, 'depth-component', 'demo', 'img')

def main(stems):
    from rembg import remove, new_session
    from simple_lama_inpainting import SimpleLama
    from PIL import Image, ImageFilter
    import numpy as np

    session = new_session('isnet-general-use')
    lama = SimpleLama()

    for stem in stems:
        print(f'[{stem}]', flush=True)
        img = Image.open(os.path.join(DEMO, stem + '.jpg')).convert('RGB')
        W, H = img.size

        # ① 软 matte（isnet, 0..1 连续）— 不二值化
        soft = remove(img, session=session, only_mask=True)
        soft = np.asarray(soft.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        # ② 人物实例硬 mask（SAM2 box-prompt, 排除凳子/椅子）→ 羽化后作 alpha 上限
        pref_p = os.path.join(DEMO, stem + '.matte-person.png')
        if os.path.exists(pref_p):
            pref = np.asarray(Image.open(pref_p).convert('L').resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
            pref_f = np.asarray(Image.fromarray((pref * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255.0
            alpha = soft * np.clip(pref_f * 3.0, 0, 1)  # 实例外(凳子)的软 matte 压到 0
        else:
            alpha = soft
        # alpha 曲线: 收紧半透明边缘, 去白边 halo
        alpha = np.clip((alpha - 0.12) / 0.55, 0, 1)

        # ③ RGBA 抠图
        rgba = np.dstack([np.asarray(img), (alpha * 255).astype(np.uint8)])
        Image.fromarray(rgba, 'RGBA').save(os.path.join(DEMO, stem + '.cutout.png'))

        # ④ 背景层: LaMa（mask=人物实例硬 mask 外扩——人物+直接接触物整体移除, 露出补全背景）
        if not os.path.exists(os.path.join(DEMO, stem + '.bgfill.png')):
            hard = (pref > 0.35).astype(np.uint8) * 255 if os.path.exists(pref_p) else (alpha > 0.4).astype(np.uint8) * 255
            hm = Image.fromarray(hard).filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(6))
            bg = lama(img, hm).resize(img.size, Image.LANCZOS)
            bg.save(os.path.join(DEMO, stem + '.bgfill.png'))

        # ⑤ 层深: 人物带中位深度 / 背景带中位深度（深度 png: 1=近）
        dp = os.path.join(DEMO, stem + '.depth.png')
        depth = np.asarray(Image.open(dp).convert('L')).astype(np.float32) / 255.0
        pm = pref > 0.35 if os.path.exists(pref_p) else alpha > 0.5
        d_person = np.median(depth[pm]) if pm.any() else 0.8
        d_bg = np.median(depth[~pm]) if (~pm).any() else 0.3
        zN, zF = 1.2, 2.6
        person_z = float(zF - d_person * (zF - zN))
        bg_z = float(max(person_z + 0.7, zF - d_bg * (zF - zN)))
        json.dump(dict(W=W, H=H, personZ=round(person_z, 3), bgZ=round(bg_z, 3),
                       f=round((H / 2) / 0.5, 1)),  # f=焦距(像素, 按竖图半高 0.5 归一)
                  open(os.path.join(DEMO, stem + '.layers.json'), 'w'))
        print(f'  ✔ cutoutRGBA + bgfill + layers: personZ={person_z:.2f} bgZ={bg_z:.2f}')

if __name__ == '__main__':
    main(sys.argv[1:] or ['w167-2', 'w167-3'])
