# inpaint-bg.py — Workers AI SD-v1.5-inpainting 背景补全
# 用法: python inpaint-bg.py <stem>...  （读 demo/img/<stem>.jpg + .matte.png → 写 <stem>.bgfill.png）
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
MODEL = '@cf/runwayml/stable-diffusion-v1-5-inpainting'
URL = f'https://api.cloudflare.com/client/v4/accounts/{ACC}/ai/run/{MODEL}'

PROMPTS = {
    'w167-0': 'empty hotel room interior with a striped armchair, large window with sheer white curtains, soft daylight, wooden floor, photorealistic, no people',
    'e1d28550': 'cozy wooden cabin interior, large window overlooking green forest, warm afternoon light, sofa, photorealistic, no people',
}
DEFAULT_PROMPT = 'empty room interior, soft natural light, photorealistic, no people'

def ai_inpaint(img_b64, mask_b64, prompt):
    body = json.dumps({'prompt': prompt, 'image': img_b64, 'mask': mask_b64}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        'Authorization': 'Bearer ' + TOK,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    })
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    if not d.get('success'):
        raise RuntimeError(str(d.get('errors'))[:300])
    img64 = d['result']['image']
    if ',' in img64[:64]:
        img64 = img64.split(',', 1)[1]
    return base64.b64decode(img64)

def ai_inpaint_raw(img_bytes, mask_bytes, prompt):
    body = json.dumps({'prompt': prompt, 'image': list(img_bytes), 'mask': list(mask_bytes)}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        'Authorization': 'Bearer ' + TOK,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            ctype = r.headers.get('Content-Type', '')
            if 'json' in ctype:
                return json.loads(r.read())
            return {'png': r.read()}  # 二进制 PNG 直出
    except urllib.error.HTTPError as e:
        return {'http': e.code, 'body': e.read().decode()[:500]}

def main(stems):
    from PIL import Image, ImageFilter
    for stem in stems:
        ip = os.path.join(DEMO, stem + '.jpg')
        mp = os.path.join(DEMO, stem + '.matte.png')
        out = os.path.join(DEMO, stem + '.bgfill.png')
        img = Image.open(ip).convert('RGB')
        matte = Image.open(mp).convert('L')
        W, H = 512, 768  # SD1.5 友好尺寸（保持竖图比例）
        img_s = img.resize((W, H), Image.LANCZOS)
        m_s = matte.resize((W, H), Image.LANCZOS)
        # 蒙版外扩+羽化: 白=重绘区（人物）, 周边一圈过渡
        mask = m_s.filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(6))
        mb = io.BytesIO(); mask.save(mb, 'PNG')
        ib = io.BytesIO(); img_s.save(ib, 'PNG')
        prompt = PROMPTS.get(stem, DEFAULT_PROMPT)
        print(f'→ {stem}: 请求生图 ({W}x{H}, prompt={prompt[:40]}…)')
        resp = ai_inpaint_raw(ib.getvalue(), mb.getvalue(), prompt)
        if resp.get('http'):
            print('✘ API', resp['http'], resp['body'])
            continue
        if resp.get('png'):
            png = resp['png']
        else:
            png = base64.b64decode(resp['result']['image'].split(',', 1)[-1])
        filled = Image.open(io.BytesIO(png)).convert('RGB')
        print(f'   返回尺寸 {filled.size}')
        if filled.size != (W, H):
            filled = filled.resize((W, H), Image.LANCZOS)  # 模型内部压方 → 还原纵横比
        filled = filled.resize(img.size, Image.LANCZOS)
        filled.save(out)
        print('✔', out, os.path.getsize(out) // 1024, 'KB')

if __name__ == '__main__':
    main(sys.argv[1:] or ['w167-0'])
