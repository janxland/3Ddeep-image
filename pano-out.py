# pano-out.py — CF inpainting 画布外扩 → 全景长条（背景专用: 输入为已抠人物后的 bgfill）
# 原理: 把 bgfill 放到 512×512 画布半边, 另半边白蒙版 → SD-inpainting 生成延续内容 → 拼接
# 用法: python pano-out.py <stem>   → 输出 demo/img/<stem>.pano.png (1024×512)
import base64
import io
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(REPO, 'depth-component', 'demo', 'img')
ACC = '1938b1d00c93863f2ca5350273b9be18'
TOK = None
for line in open(os.path.expanduser('~/Library/Preferences/.wrangler/config/default.toml')):
    if line.startswith('oauth_token'):
        TOK = line.split('"')[1]
URL = f'https://api.cloudflare.com/client/v4/accounts/{ACC}/ai/run/@cf/runwayml/stable-diffusion-v1-5-inpainting'

def ai_inpaint_png(img_png, mask_png, prompt):
    body = json.dumps({'prompt': prompt, 'image': list(img_png), 'mask': list(mask_png)}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        'Authorization': 'Bearer ' + TOK, 'Content-Type': 'application/json', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            ctype = r.headers.get('Content-Type', '')
            return r.read() if 'json' not in ctype else None
    except urllib.error.HTTPError as e:
        print('✘', e.code, e.read().decode()[:200]); return None

def run(stem):
    from PIL import Image, ImageFilter
    base = Image.open(os.path.join(DEMO, stem + '.bgfill.png')).convert('RGB').resize((512, 512), Image.LANCZOS)
    prompt = 'seamless room interior continuation, walls and furniture, photorealistic, no people'
    pieces = []
    # 左扩: base 贴右半, 左半生成
    canvas = Image.new('RGB', (512, 512))
    canvas.paste(base, (256, 0))
    mask = Image.new('L', (512, 512), 0)
    mask.paste(255, (0, 0, 256, 512))
    mask = mask.filter(ImageFilter.GaussianBlur(10))
    out = ai_inpaint_png(canvas.tobytes() and _to_png(canvas), _to_png(mask), prompt)
    if out:
        left = Image.open(io.BytesIO(out)).convert('RGB').crop((0, 0, 256, 512))
        pieces.append(left)
        print('✔ 左扩 256px')
    pieces.append(base)
    # 右扩: base 贴左半
    canvas = Image.new('RGB', (512, 512))
    canvas.paste(base, (0, 0))
    mask = Image.new('L', (512, 512), 0)
    mask.paste(255, (256, 0, 512, 512))
    mask = mask.filter(ImageFilter.GaussianBlur(10))
    out = ai_inpaint_png(_to_png(canvas), _to_png(mask), prompt)
    if out:
        right = Image.open(io.BytesIO(out)).convert('RGB').crop((256, 0, 512, 512))
        pieces.append(right)
        print('✔ 右扩 256px')
    pano = Image.new('RGB', (sum(p.width for p in pieces), 512))
    x = 0
    for p in pieces:
        pano.paste(p, (x, 0)); x += p.width
    pano.save(os.path.join(DEMO, stem + '.pano.png'))
    print('✔', stem, '全景长条', pano.size, '→', stem + '.pano.png')

def _to_png(im):
    b = io.BytesIO(); im.save(b, 'PNG'); return b.getvalue()

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'w167-0')
