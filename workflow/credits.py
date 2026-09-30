"""
汉化组信息：合成 PDF 时插入信息页、给页面打水印。

只作用于 PDF，不修改 output/ 里生成好的页面图片，因此改设置后只需「重新合成 PDF」。
设置保存在 config.json 的 credits 段：

    "credits": {
      "page_position": "last",          # 信息页位置：none / first / last / both
      "page_image": "",                 # 自备信息页图片；为空时按下面的文字自动生成
      "group": "某某汉化组",
      "lines": "翻译：xx\\n嵌字：xx",      # 自动生成信息页的正文，每行一条
      "watermark": {
        "mode": "none",                 # none / text / image
        "text": "",
        "image": "",                    # logo 图片，支持透明 png
        "pages": "all",                 # all / first / last / first_last
        "position": "br",               # tl / tr / bl / br / center
        "opacity": 0.35,
        "size": 0.15                    # 文字：字高占页高 ×0.2；图片：宽度占页宽的比例
      }
    }
"""
import os

from PIL import Image, ImageDraw

DEFAULTS = {
    'page_position': 'none',
    'page_image': '',
    'group': '',
    'lines': '',
    'watermark': {'mode': 'none', 'text': '', 'image': '', 'pages': 'all',
                  'position': 'br', 'opacity': 0.35, 'size': 0.15},
}


def settings(cfg):
    """从 config.json 内容取出 credits 设置（补全默认值）"""
    raw = (cfg or {}).get('credits') or {}
    st = dict(DEFAULTS)
    st.update({k: v for k, v in raw.items() if k != 'watermark'})
    wm = dict(DEFAULTS['watermark'])
    wm.update(raw.get('watermark') or {})
    st['watermark'] = wm
    return st


def active(st):
    return (st['page_position'] != 'none' and (st['page_image'] or st['group'] or st['lines'])) or \
        (st['watermark']['mode'] == 'text' and st['watermark']['text'].strip()) or \
        (st['watermark']['mode'] == 'image' and os.path.isfile(st['watermark']['image']))


def _font(size):
    import typeset
    return typeset._font(typeset.resolve_font(''), max(8, int(size)))


# ================================================================ 信息页

def make_credit_page(size, st):
    """生成信息页（RGB）。size=(宽, 高) 与漫画页一致"""
    W, H = size
    img_path = st.get('page_image') or ''
    if img_path and os.path.isfile(img_path):
        src = Image.open(img_path).convert('RGBA')
        scale = W / src.width
        src = src.resize((W, max(1, round(src.height * scale))), Image.LANCZOS)
        page = Image.new('RGB', (W, max(H, src.height)), 'white')
        page.paste(src, (0, (page.height - src.height) // 2), src)
        return page
    page = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(page)
    group = (st.get('group') or '').strip()
    lines = [l.strip() for l in (st.get('lines') or '').splitlines() if l.strip()]
    gsize, lsize = W * 0.075, W * 0.035
    blocks = []
    if group:
        blocks.append((group, _font(gsize), gsize * 1.8))
    for l in lines:
        blocks.append((l, _font(lsize), lsize * 1.7))
    total = sum(h for _, _, h in blocks)
    y = (H - total) / 2
    for text, f, h in blocks:
        # 太长时缩小到页宽的 85%
        w = f.getlength(text)
        if w > W * 0.85:
            f = _font(f.size * W * 0.85 / w)
            w = f.getlength(text)
        d.text(((W - w) / 2, y), text, font=f, fill=(20, 20, 20))
        y += h
    return page


# ================================================================ 水印

def _pages_for(which, n):
    return {'all': set(range(n)), 'first': {0}, 'last': {n - 1},
            'first_last': {0, n - 1}}.get(which, set())


def _place(W, H, w, h, position):
    m = int(W * 0.025)
    return {'tl': (m, m), 'tr': (W - w - m, m), 'bl': (m, H - h - m),
            'center': ((W - w) // 2, (H - h) // 2)}.get(position, (W - w - m, H - h - m))


def watermark(img, wm):
    """在一页上打水印，返回新图（RGB）"""
    mode = wm.get('mode')
    base = img.convert('RGBA')
    W, H = base.size
    opacity = max(0.0, min(1.0, float(wm.get('opacity', 0.35))))
    size = float(wm.get('size', 0.15))
    if mode == 'text' and (wm.get('text') or '').strip():
        text = wm['text'].strip()
        fs = max(10, H * size * 0.2)
        f = _font(fs)
        sw = max(1, round(fs * 0.08))
        w = int(f.getlength(text)) + sw * 2
        h = int(fs * 1.3) + sw * 2
        layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((sw, sw), text, font=f, fill=(0, 0, 0, 255),
                                   stroke_width=sw, stroke_fill=(255, 255, 255, 255))
    elif mode == 'image' and os.path.isfile(wm.get('image') or ''):
        layer = Image.open(wm['image']).convert('RGBA')
        w = max(8, int(W * size))
        h = max(1, round(layer.height * w / layer.width))
        layer = layer.resize((w, h), Image.LANCZOS)
    else:
        return img.convert('RGB')
    alpha = layer.getchannel('A').point(lambda a: int(a * opacity))
    layer.putalpha(alpha)
    x, y = _place(W, H, layer.width, layer.height, wm.get('position', 'br'))
    base.alpha_composite(layer, (max(0, x), max(0, y)))
    return base.convert('RGB')


# ================================================================ 合成

def compose(image_paths, st, tmp_dir):
    """返回用于合成 PDF 的图片路径列表：水印页写到 tmp_dir，原图不动"""
    if not image_paths or not active(st):
        return list(image_paths)
    n = len(image_paths)
    wm = st['watermark']
    marked = _pages_for(wm.get('pages', 'all'), n) if wm.get('mode') in ('text', 'image') else set()
    out = []
    for i, p in enumerate(image_paths):
        if i in marked:
            q = os.path.join(tmp_dir, f'wm_{i:05d}.jpg')
            watermark(Image.open(p), wm).save(q, quality=95)
            out.append(q)
        else:
            out.append(p)
    pos = st.get('page_position', 'none')
    if pos != 'none' and (st.get('page_image') or st.get('group') or st.get('lines')):
        with Image.open(image_paths[0]) as first:
            size = first.size
        cp = os.path.join(tmp_dir, 'credit_page.png')
        make_credit_page(size, st).save(cp)
        if pos in ('first', 'both'):
            out.insert(0, cp)
        if pos in ('last', 'both'):
            out.append(cp)
    return out
