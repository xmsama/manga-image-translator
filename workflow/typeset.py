"""
漫画气泡嵌字器：替换 manga-image-translator 默认渲染，解决字号不统一、溢出气泡、不居中、断行难看等问题。

对每个文本区域：
  1. 在擦字后的图上找气泡：从原文位置向外泛洪，得到封闭气泡的形状；
     找不到封闭气泡（字直接压在画面上）时，退回到原文区域并给文字加描边。
  2. 字号：以整本书的基准字号为上限（由导出时记录的原文字号中位数得出），
     放不下才逐级缩小，从不放大，因此全书字号统一。
  3. 按气泡形状排版：每行（列）可用宽度取气泡在该高度（该列）处的宽度，
     整体居中到气泡的“视觉中心”（离边缘最远的点），行长自动均衡。
  4. 中文断行规则：句读、右括号等不出现在行首，左括号不出现在行尾，英文单词和数字不拆开。
  5. 原文竖排则竖排，标点换成竖排字形；原文横排则横排。
"""
import os
import re
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(ROOT, 'fonts')
_FALLBACK_CANDIDATES = [
    os.path.join(FONT_DIR, '思源黑体-Medium.otf'),
    # 没运行 install_fonts.py 时，退回 manga-image-translator 自带的字体
    os.path.join(ROOT, 'manga-image-translator', 'fonts', 'msyh.ttc'),
    os.path.join(os.path.dirname(ROOT), 'fonts', 'msyh.ttc'),
]
FALLBACK_FONT = next((p for p in _FALLBACK_CANDIDATES if os.path.exists(p)), _FALLBACK_CANDIDATES[0])

DEFAULTS = {
    'font': '源柔黑体-Medium.ttf',   # fonts/ 目录下的文件名
    'size_scale': 1.0,               # 基准字号倍率
    'min_scale': 0.55,               # 放不下时最多缩到基准的多少
    'line_spacing': 1.18,            # 行距（倍字号）
    'margin': 0.10,                  # 气泡内边距（占气泡短边比例）
    'stroke': 'auto',                # auto: 气泡外的文字才描边；always / never
    'direction': 'auto',             # auto: 跟随原文；h 横排；v 竖排
}

# 不能出现在行首 / 行尾的字符
NO_START = set('，。、．！？!?：；:;,.)）」』】》〉”’…—～~%％·・ー')
NO_END = set('（(「『【《〈“‘')

VERTICAL_FORMS = {
    '，': '︐', '、': '︑', '。': '︒', '：': '︓', '；': '︔', '！': '︕', '？': '︖',
    '…': '︙', '‥': '︰', '—': '︱', '–': '︲', '（': '︵', '）': '︶', '(': '︵', ')': '︶',
    '【': '︻', '】': '︼', '《': '︽', '》': '︾', '〈': '︿', '〉': '﹀',
    '「': '﹁', '」': '﹂', '『': '﹃', '』': '﹄', '[': '﹇', ']': '﹈', '{': '︷', '}': '︸',
    '“': '﹁', '”': '﹂', '‘': '﹃', '’': '﹄', '!': '︕', '?': '︖', ',': '︐', ':': '︓', ';': '︔',
}
ROTATE_IN_VERTICAL = set('～~ー-－=＝<>＜＞')


# ================================================================ 字体

@lru_cache(maxsize=16)
def _cmap(path):
    try:
        from fontTools.ttLib import TTFont
        return frozenset(TTFont(path, lazy=True).getBestCmap().keys())
    except Exception:
        return None


@lru_cache(maxsize=256)
def _font(path, size):
    return ImageFont.truetype(path, max(4, int(size)))


class Fonts:
    def __init__(self, main_path):
        self.main = main_path if os.path.exists(main_path) else FALLBACK_FONT
        self.main_cmap = _cmap(self.main)
        self.fallback = FALLBACK_FONT if os.path.exists(FALLBACK_FONT) else self.main

    def for_char(self, ch, size):
        if self.main_cmap is None or ord(ch) in self.main_cmap or ch.isspace():
            return _font(self.main, size)
        return _font(self.fallback, size)

    def has(self, ch):
        return self.main_cmap is None or ord(ch) in self.main_cmap

    def text_width(self, text, size):
        return sum(self.for_char(c, size).getlength(c) for c in text)


def resolve_font(name):
    if not name:
        name = DEFAULTS['font']
    p = name if os.path.isabs(name) else os.path.join(FONT_DIR, name)
    return p if os.path.exists(p) else FALLBACK_FONT


def list_fonts():
    if not os.path.isdir(FONT_DIR):
        return []
    return sorted(f for f in os.listdir(FONT_DIR) if f.lower().endswith(('.ttf', '.otf', '.ttc')))


# ================================================================ 分词与断行

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’.\-][A-Za-z0-9]+)*|\s+|……|——|.", re.S)


def tokenize(text):
    """把译文切成不可再分的排版单元，并按禁则把标点粘到相邻单元上"""
    raw = [t for t in _TOKEN.findall(norm_text(text))]
    units = []
    pending_open = ''
    for t in raw:
        if t.isspace():
            if units and not pending_open:
                units[-1] += ' '
            continue
        if all(c in NO_END for c in t):
            pending_open += t
            continue
        t = pending_open + t
        pending_open = ''
        if units and all(c in NO_START for c in t):
            units[-1] = units[-1].rstrip() + t   # 行首禁则：并到前一个单元
        else:
            units.append(t)
    if pending_open:
        if units:
            units[-1] += pending_open
        else:
            units.append(pending_open)
    return units


def norm_text(text):
    text = (text or '').replace('─', '—').replace('...', '…').replace('。。。', '…')
    return ' '.join(text.split())


def greedy_lines(units, caps, measure):
    """按每行容量依次填充，放不下返回 None"""
    lines, cur, i = [], '', 0
    for u in units:
        while True:
            if i >= len(caps):
                return None
            cand = (cur + u) if cur else u.lstrip()
            if measure(cand.rstrip()) <= caps[i] + 0.5:
                cur = cand
                break
            if not cur:  # 单个单元都放不下
                return None
            lines.append(cur.rstrip())
            cur, i = '', i + 1
    if cur:
        lines.append(cur.rstrip())
    return lines


# ================================================================ 气泡检测

def _fill_holes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


def find_bubble(gray, box, tol=26):
    """返回 (气泡掩码, 是否封闭气泡)。掩码为整图尺寸的 uint8"""
    H, W = gray.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    inner = gray[y1:y2 + 1, x1:x2 + 1]
    if inner.size == 0:
        return None, False
    seed_val = int(np.median(inner))
    for expand in (2.0, 4.5):
        pad = int(max(bw, bh) * expand)
        wx1, wy1 = max(0, x1 - pad), max(0, y1 - pad)
        wx2, wy2 = min(W, x2 + pad + 1), min(H, y2 + pad + 1)
        win = gray[wy1:wy2, wx1:wx2]
        near = (np.abs(win.astype(np.int16) - seed_val) <= tol).astype(np.uint8) * 255
        near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(near, connectivity=4)
        # 选与文本框重叠最多的连通域
        sub = labels[y1 - wy1:y2 - wy1 + 1, x1 - wx1:x2 - wx1 + 1]
        ids, cnt = np.unique(sub[sub > 0], return_counts=True)
        if len(ids) == 0:
            return None, False
        comp = ids[np.argmax(cnt)]
        m = (labels == comp).astype(np.uint8) * 255
        m = _fill_holes(m)
        sx, sy, sw, sh, area = stats[comp]
        touches = sx <= 0 or sy <= 0 or sx + sw >= win.shape[1] or sy + sh >= win.shape[0]
        at_image_edge = (wx1 == 0 and sx <= 0) or (wy1 == 0 and sy <= 0) or \
                        (wx2 == W and sx + sw >= win.shape[1]) or (wy2 == H and sy + sh >= win.shape[0])
        if touches and not at_image_edge:
            continue  # 窗口太小或不是封闭区域，扩大再试
        covered = m[y1 - wy1:y2 - wy1 + 1, x1 - wx1:x2 - wx1 + 1].mean() / 255
        box_area = bw * bh
        area = int(np.count_nonzero(m))
        if covered < 0.75 or area < box_area * 0.9 or area > box_area * 25:
            return None, False
        ys, xs = np.nonzero(m)
        mw, mh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if mw > W * 0.55 or mh > H * 0.55:
            return None, False          # 比半页还大：泄漏到了画面背景
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        if area / max(1.0, cv2.contourArea(hull)) < 0.72:
            return None, False          # 形状很不规则：不是气泡
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        tcx, tcy = (x1 + x2) // 2 - wx1, (y1 + y2) // 2 - wy1
        if dist[tcy, tcx] < 0.25 * dist.max():
            return None, False          # 原文贴着边缘：多半不是它的气泡
        full = np.zeros_like(gray, dtype=np.uint8)
        full[wy1:wy2, wx1:wx2] = m
        return full, True
    return None, False


def split_shared_bubbles(masks, boxes):
    """同一个气泡里有多个文本区域时，按离哪个原文更近来划分气泡"""
    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = masks[i], masks[j]
            if a is None or b is None:
                continue
            inter = np.count_nonzero(a & b)
            if inter < 0.5 * min(np.count_nonzero(a), np.count_nonzero(b)):
                continue
            ys, xs = np.nonzero(a | b)
            ca = np.array([(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2])
            cb = np.array([(boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2])
            da = (xs - ca[0]) ** 2 + (ys - ca[1]) ** 2
            db = (xs - cb[0]) ** 2 + (ys - cb[1]) ** 2
            union = a | b
            na, nb = np.zeros_like(a), np.zeros_like(b)
            na[ys[da <= db], xs[da <= db]] = 255
            nb[ys[da > db], xs[da > db]] = 255
            masks[i], masks[j] = na & union, nb & union
    return masks


# ================================================================ 排版

class Area:
    """可用区域（气泡内缩后的掩码），提供按行/按列查询可用宽度"""

    def __init__(self, mask):
        self.mask = mask > 0
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        self.cx, self.cy = int(cx), int(cy)
        ys, xs = np.nonzero(self.mask)
        self.bbox = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (cx, cy, cx, cy)

    def _run(self, line, c):
        """一行（或一列）像素里包含位置 c 的连续可用区间"""
        if c < 0 or c >= len(line) or not line[c]:
            return None
        l = c
        while l > 0 and line[l - 1]:
            l -= 1
        r = c
        while r < len(line) - 1 and line[r + 1]:
            r += 1
        return l, r

    def h_span(self, y0, y1):
        """水平条带 [y0,y1] 内，包含中心列的可用横向区间（各行取交集）"""
        y0, y1 = int(np.floor(y0)), int(np.ceil(y1))
        if y0 < 0 or y1 >= self.mask.shape[0]:
            return None
        lo, hi = -1, 10 ** 9
        for y in range(y0, y1 + 1, max(1, (y1 - y0) // 6)):
            run = self._run(self.mask[y], self.cx)
            if run is None:
                return None
            lo, hi = max(lo, run[0]), min(hi, run[1])
        return (lo, hi) if hi > lo else None

    def v_span(self, x0, x1):
        x0, x1 = int(np.floor(x0)), int(np.ceil(x1))
        if x0 < 0 or x1 >= self.mask.shape[1]:
            return None
        lo, hi = -1, 10 ** 9
        for x in range(x0, x1 + 1, max(1, (x1 - x0) // 6)):
            run = self._run(self.mask[:, x], self.cy)
            if run is None:
                return None
            lo, hi = max(lo, run[0]), min(hi, run[1])
        return (lo, hi) if hi > lo else None


def layout_horizontal(units, fonts, size, area, spacing):
    measure = lambda s: fonts.text_width(s, size)
    lh = size * spacing
    total_w = measure(''.join(units))
    max_lines = max(1, int((area.bbox[3] - area.bbox[1]) / lh) + 1)
    for n in range(1, max_lines + 1):
        block_h = (n - 1) * lh + size
        top = area.cy - block_h / 2
        spans = [area.h_span(top + i * lh, top + i * lh + size) for i in range(n)]
        if any(s is None for s in spans):
            # 中心附近放不下时，整体上下微调再试
            ok = False
            for shift in (-0.25, 0.25, -0.5, 0.5):
                t2 = top + shift * lh
                s2 = [area.h_span(t2 + i * lh, t2 + i * lh + size) for i in range(n)]
                if all(s is not None for s in s2):
                    top, spans, ok = t2, s2, True
                    break
            if not ok:
                continue
        caps = [s[1] - s[0] for s in spans]
        if sum(caps) < total_w * 0.98:
            continue
        lines = greedy_lines(units, caps, measure)
        if lines is None:
            continue
        # 均衡行长：在仍能放下的前提下尽量缩小每行的最大占用比例
        lo, hi, best = 0.3, 1.0, lines
        for _ in range(8):
            mid = (lo + hi) / 2
            cand = greedy_lines(units, [c * mid for c in caps], measure)
            if cand is not None and len(cand) <= n:
                best, hi = cand, mid
            else:
                lo = mid
        placed = []
        for i, text in enumerate(best):
            l, r = spans[i]
            w = measure(text)
            placed.append((text, (l + r) / 2 - w / 2, top + i * lh))
        # 实际行数少于 n 时重新垂直居中
        if len(best) < n:
            k = len(best)
            new_top = area.cy - ((k - 1) * lh + size) / 2
            placed = [(t, x, new_top + i * lh) for i, (t, x, _) in enumerate(placed)]
        return placed
    return None


def vertical_chars(line):
    """竖排：拆成逐字，标点换成竖排字形；两位以内数字横放（纵中横）"""
    out, i = [], 0
    while i < len(line):
        m = re.match(r'[0-9]{1,2}(?![0-9])', line[i:])
        if m and (i == 0 or not line[i - 1].isdigit()):
            out.append(m.group())
            i += len(m.group())
            continue
        out.append(line[i])
        i += 1
    return out


def layout_vertical(units, fonts, size, area, spacing):
    adv = size * 1.02
    col_w = size * spacing * 1.08
    measure = lambda s: len(vertical_chars(s.replace(' ', ''))) * adv
    total_h = measure(''.join(units))
    max_cols = max(1, int((area.bbox[2] - area.bbox[0]) / col_w) + 1)
    for n in range(1, max_cols + 1):
        block_w = (n - 1) * col_w + size
        right = area.cx + block_w / 2
        spans = [area.v_span(right - i * col_w - size, right - i * col_w) for i in range(n)]
        if any(s is None for s in spans):
            continue
        caps = [s[1] - s[0] for s in spans]
        if sum(caps) < total_h * 0.98:
            continue
        cols = greedy_lines(units, caps, measure)
        if cols is None:
            continue
        lo, hi, best = 0.3, 1.0, cols
        for _ in range(8):
            mid = (lo + hi) / 2
            cand = greedy_lines(units, [c * mid for c in caps], measure)
            if cand is not None and len(cand) <= n:
                best, hi = cand, mid
            else:
                lo = mid
        k = len(best)
        right = area.cx + ((k - 1) * col_w + size) / 2
        placed = []
        for i, text in enumerate(best):
            t, b = spans[i]
            h = measure(text)
            placed.append((text.replace(' ', ''), right - i * col_w - size, (t + b) / 2 - h / 2))
        return placed
    return None


# ================================================================ 绘制

def draw_horizontal(draw, placed, fonts, size, fill, stroke_w, stroke_fill):
    for text, x, y in placed:
        cx = x
        for ch in text:
            f = fonts.for_char(ch, size)
            draw.text((cx, y), ch, font=f, fill=fill, stroke_width=stroke_w, stroke_fill=stroke_fill)
            cx += f.getlength(ch)


def draw_vertical(img, draw, placed, fonts, size, fill, stroke_w, stroke_fill):
    adv = size * 1.02
    for text, x, y in placed:
        cy = y
        for ch in vertical_chars(text):
            if len(ch) > 1:  # 纵中横
                f = fonts.for_char(ch[0], size * 0.8)
                w = sum(f.getlength(c) for c in ch)
                draw.text((x + (size - w) / 2, cy + size * 0.1), ch, font=f, fill=fill,
                          stroke_width=stroke_w, stroke_fill=stroke_fill)
            elif ch in VERTICAL_FORMS and fonts.has(VERTICAL_FORMS[ch]):
                g = VERTICAL_FORMS[ch]
                f = fonts.for_char(g, size)
                draw.text((x, cy), g, font=f, fill=fill, stroke_width=stroke_w, stroke_fill=stroke_fill)
            elif ch in ROTATE_IN_VERTICAL or ch in VERTICAL_FORMS or ch.isascii() and ch.isalpha():
                _draw_rotated(img, ch, fonts.for_char(ch, size), size, (x, cy), fill, stroke_w, stroke_fill)
            else:
                f = fonts.for_char(ch, size)
                w = f.getlength(ch)
                draw.text((x + (size - w) / 2, cy), ch, font=f, fill=fill,
                          stroke_width=stroke_w, stroke_fill=stroke_fill)
            cy += adv


def _draw_rotated(img, ch, font, size, pos, fill, stroke_w, stroke_fill):
    s = int(size * 1.6) + stroke_w * 2
    tile = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    w = font.getlength(ch)
    d.text(((s - w) / 2, (s - size) / 2), ch, font=font, fill=tuple(fill) + (255,),
           stroke_width=stroke_w, stroke_fill=tuple(stroke_fill) + (255,) if stroke_fill else None)
    tile = tile.rotate(-90, resample=Image.BICUBIC)
    img.paste(tile, (int(pos[0] + (size - s) / 2), int(pos[1] + (size - s) / 2)), tile)


# ================================================================ 气泡清理

def clean_bubbles(img, found, keep_boxes, base, logger=None):
    """擦掉封闭气泡里引擎漏擦的小块墨迹（注音假名、残字、杂点）。

    只处理底色均匀的气泡（网点、渐变、有画面的气泡跳过）；
    只擦小于约 1.6 倍字号的墨迹块，气泡里画的图形、线条保留；
    气泡里有留空（保留原文）的文字时整个气泡跳过。
    """
    H, W = img.shape[:2]
    limit = max(12.0, base * 1.6)
    done, total = [], 0
    k3 = np.ones((3, 3), np.uint8)
    for mask, enclosed in found:
        if mask is None or not enclosed:
            continue
        if any(np.count_nonzero(mask & d) > 0.8 * np.count_nonzero(mask) for d in done):
            continue                                  # 同一个气泡已处理过
        done.append(mask)
        inner = cv2.erode(mask, k3, iterations=2)
        if any(inner[max(0, y1):y2 + 1, max(0, x1):x2 + 1].any() for x1, y1, x2, y2 in keep_boxes):
            continue
        sel = inner > 0
        px = img[sel].astype(np.int16)
        if len(px) < 200:
            continue
        bg = np.median(px, axis=0)
        diff = np.zeros((H, W), np.int16)
        diff[sel] = np.abs(px - bg).max(axis=1)
        if (diff[sel] < 30).mean() < 0.9:
            continue                                  # 底色不均匀
        ink = ((diff > 28) & sel).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        erase = np.zeros((H, W), np.uint8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if max(w, h) <= limit:
                erase[labels == i] = 255
                total += 1
        if not erase.any():
            continue
        erase = cv2.dilate(erase, k3, iterations=2) & mask
        soft = cv2.GaussianBlur(erase.astype(np.float32) / 255, (5, 5), 0)[..., None]
        img[:] = (img * (1 - soft) + bg.reshape(1, 1, 3) * soft).astype(np.uint8)
    if total and logger:
        logger.info(f'清理气泡内残留：{total} 处（注音 / 残字 / 杂点）')
    return img


# ================================================================ 入口

SWITCH_GAIN = 1.15   # 另一方向的字号至少大 15% 才换方向，避免同页横竖来回跳


def fit_size(layout, units, fonts, base, inner, full, st):
    """从基准字号往下找能放下的最大字号；缩到下限仍放不下时改用未内缩的完整气泡。返回 (字号, 排版结果)"""
    area = inner
    lo_size = max(8.0, base * st['min_scale'])
    size, widened = base, False
    while size >= 8:
        placed = layout(units, fonts, size, area, st['line_spacing'])
        if placed is not None:
            return size, placed
        size -= max(1.0, size * 0.05)
        if size < lo_size and full is not None and not widened:
            area, widened = full, True
    return size, None


def _colors(region, bubble_gray):
    try:
        fg, bg = region.get_font_colors()
        fg, bg = tuple(int(v) for v in fg), tuple(int(v) for v in bg)
    except Exception:
        fg, bg = (0, 0, 0), (255, 255, 255)
    # 保证文字与气泡底色有足够对比度
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    if bubble_gray is not None and abs(lum(fg) - bubble_gray) < 90:
        fg = (0, 0, 0) if bubble_gray > 127 else (255, 255, 255)
    if abs(lum(fg) - lum(bg)) < 90:
        bg = (255, 255, 255) if lum(fg) < 128 else (0, 0, 0)
    return fg, bg


def _is_vertical_source(region):
    try:
        lines = region.lines.reshape(-1, 4, 2)
        tall = wide = 0.0
        for q in lines:
            w = np.linalg.norm(q[1] - q[0])
            h = np.linalg.norm(q[3] - q[0])
            if h > w * 1.2:
                tall += w * h
            else:
                wide += w * h
        return tall > wide
    except Exception:
        return False


def typeset_page(img, regions, settings=None, base_size=None, target_lang='CHS', logger=None, keep_boxes=()):
    """img: 擦字后的 RGB 图（numpy）。返回嵌字后的 RGB 图"""
    st = dict(DEFAULTS)
    st.update({k: v for k, v in (settings or {}).items() if v is not None})
    regions = [r for r in regions if (getattr(r, 'translation', '') or '').strip()]
    if not regions:
        return img
    fonts = Fonts(resolve_font(st['font']))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    H, W = gray.shape

    sizes = [r.font_size for r in regions if r.font_size and r.font_size > 0]
    page_base = float(np.median(sizes)) if sizes else (H + W) / 80
    base = float(base_size or page_base) * float(st['size_scale'])
    cjk_target = target_lang in ('CHS', 'CHT', 'JPN', 'KOR')

    boxes = [tuple(int(v) for v in r.xyxy) for r in regions]
    found = [find_bubble(gray, b) for b in boxes]
    if st.get('clean', True):
        img = clean_bubbles(np.ascontiguousarray(img).copy(), found, list(keep_boxes), base, logger)
    masks = split_shared_bubbles([m for m, _ in found], boxes)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for r, box, mask, (_, enclosed) in zip(regions, boxes, masks, found):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if mask is not None and enclosed:
            ys, xs = np.nonzero(mask)
            short = min(xs.max() - xs.min(), ys.max() - ys.min())
            m = max(2, int(short * st['margin']))
            area_mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=max(1, m // 1))
            if np.count_nonzero(area_mask) < 50:
                area_mask = mask
            bubble_gray = float(np.median(gray[mask > 0]))
        else:
            # 没有封闭气泡：在原文区域基础上略微放宽
            area_mask = np.zeros_like(gray, dtype=np.uint8)
            ex, ey = int(bw * 0.12) + 4, int(bh * 0.12) + 4
            cv2.rectangle(area_mask, (max(0, x1 - ex), max(0, y1 - ey)),
                          (min(W - 1, x2 + ex), min(H - 1, y2 + ey)), 255, -1)
            bubble_gray = None
            enclosed = False
        area = Area(area_mask)

        units = tokenize(r.translation)
        fg, bg = _colors(r, bubble_gray)
        stroke_mode = st['stroke']
        use_stroke = stroke_mode == 'always' or (stroke_mode == 'auto' and not enclosed)

        # 描边会向外扩，先把可用区域相应内缩
        inner = area
        sw_ref = max(1, round(base * 0.12)) if use_stroke else 0
        if sw_ref:
            eroded = cv2.erode(area_mask, np.ones((3, 3), np.uint8), iterations=sw_ref)
            if np.count_nonzero(eroded) >= 50:
                inner = Area(eroded)
        full = Area(mask) if (enclosed and mask is not None) else None

        # 候选方向：跟随原文时，先试原文方向；放不满基准字号才试另一方向
        direction = st['direction']
        src_vertical = _is_vertical_source(r)
        can_vertical = cjk_target and not re.search(r'[A-Za-z]{4,}', r.translation)
        if direction == 'h' or not can_vertical:
            candidates = [False]
        elif direction == 'v':
            candidates = [True]
        else:
            candidates = [src_vertical, not src_vertical]

        tried = []
        for vert in candidates:
            layout = layout_vertical if vert else layout_horizontal
            size, placed = fit_size(layout, units, fonts, base, inner, full, st)
            tried.append((vert, size, placed))
            if placed is not None and size >= base * 0.97:
                break           # 该方向已能用全书统一字号，无需再试
        vertical, size, placed = tried[0]
        for vert, sz, pl in tried[1:]:
            if pl is not None and (placed is None or sz >= size * SWITCH_GAIN):
                if logger:
                    logger.info(f'换成{"竖排" if vert else "横排"}更合适（字号 {size:.0f}→{sz:.0f}）：{r.translation[:16]}')
                vertical, size, placed = vert, sz, pl

        if placed is None:
            # 兜底：在原文区域（放宽）内强制排版，允许最小字号
            rect = np.zeros_like(gray, dtype=np.uint8)
            ex, ey = int(bw * 0.25) + 6, int(bh * 0.25) + 6
            cv2.rectangle(rect, (max(0, x1 - ex), max(0, y1 - ey)),
                          (min(W - 1, x2 + ex), min(H - 1, y2 + ey)), 255, -1)
            area = Area(rect)
            for vert in candidates:
                layout = layout_vertical if vert else layout_horizontal
                size = base
                while size >= 6 and placed is None:
                    placed = layout(units, fonts, size, area, st['line_spacing'])
                    if placed is None:
                        size -= max(1.0, size * 0.05)
                if placed is not None:
                    vertical = vert
                    break
            if placed is not None and logger:
                logger.warning(f'气泡内放不下，已在原文区域内缩小排版：{r.translation[:20]}')
        if placed is None:
            if logger:
                logger.warning(f'排版失败：{r.translation[:20]} | 封闭气泡={enclosed} 原文框={box} '
                               f'基准字号={base:.1f}')
            continue
        size = int(round(size))
        sw = max(1, round(size * 0.12)) if use_stroke else 0
        if vertical:
            draw_vertical(pil, draw, placed, fonts, size, fg, sw, bg if sw else None)
        else:
            draw_horizontal(draw, placed, fonts, size, fg, sw, bg if sw else None)
    return np.array(pil)
