# lama-bgfill.py — isnet-general-use matte + LaMa 修复 → 背景补全（离线, Mac MPS）
# 用法: python lama-bgfill.py <stem>...   （需 demo/img/<stem>.jpg；输出 .matte.png + .bgfill.png）
# 依赖: rembg, simple-lama-inpainting（3d-photo-inpainting/.venv）
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(REPO, 'depth-component', 'demo', 'img')

def main(stems):
    from rembg import remove, new_session
    from simple_lama_inpainting import SimpleLama
    from PIL import Image, ImageFilter
    session = new_session('isnet-general-use')
    lama = SimpleLama()
    for stem in stems:
        img = Image.open(os.path.join(DEMO, stem + '.jpg')).convert('RGB')
        matte = remove(img, session=session, only_mask=True)
        matte.save(os.path.join(DEMO, stem + '.matte.png'))
        mask = matte.filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(6))
        out = lama(img, mask).resize(img.size, Image.LANCZOS)
        out.save(os.path.join(DEMO, stem + '.bgfill.png'))
        print('✔', stem, 'isnet matte + lama bgfill', out.size)

if __name__ == '__main__':
    main(sys.argv[1:] or ['w167-0'])
