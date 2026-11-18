# lama-bgfill.py — isnet-general-use matte + LaMa 修复 → 背景补全（离线, Mac MPS）
# 用法: python lama-bgfill.py <stem>...   （需 demo/img/<stem>.jpg；输出 .matte.png + .bgfill.png）
# 依赖: rembg, simple-lama-inpainting（3d-photo-inpainting/.venv）
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(REPO, 'depth-component', 'demo', 'img')

def main(stems, session_name='isnet-general-use'):
    from rembg import remove, new_session
    from simple_lama_inpainting import SimpleLama
    from PIL import Image, ImageFilter
    import numpy as np
    session = new_session(session_name)
    lama = SimpleLama()
    for stem in stems:
        img = Image.open(os.path.join(DEMO, stem + '.jpg')).convert('RGB')
        matte = remove(img, session=session, only_mask=True)
        # alpha 重映射: 亮色衣物(白衬衫贴亮背景)在 matte 里是 0.15~0.6 中间值 → 放宽阈值收进主体
        arr = np.asarray(matte).astype(np.float32)
        arr = np.clip((arr - 12.0) / 45.0 * 255.0, 0, 255).astype(np.uint8)
        matte = Image.fromarray(arr)
        matte.save(os.path.join(DEMO, stem + '.matte.png'))
        mask = matte.filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(6))
        out = lama(img, mask).resize(img.size, Image.LANCZOS)
        out.save(os.path.join(DEMO, stem + '.bgfill.png'))
        print('✔', stem, 'matte(alpha重映射) + lama bgfill', out.size)

if __name__ == '__main__':
    main(sys.argv[1:] or ['w167-0'])
