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


@lru_cache(maxsize=16)
def _loadable_path(path):
    """Windows 上 Pillow 遇到含非 ASCII 字符的字体路径（如 fonts/源柔黑体-Medium.ttf）
    会把整个字体文件读进内存再交给 FreeType，而且每个字号各一份（中文字体约 14MB×2）。
    嵌字时字号很多，GPU 进程的内存会随页数一路上涨。
    复制到纯 ASCII 路径后，FreeType 直接按需读取文件，每个字号只占很少内存。"""
    try:
        path.encode('ascii')
        return path
    except UnicodeEncodeError:
        pass
    import hashlib
    import shutil
    import tempfile
    try:
        st = os.stat(path)
        key = hashlib.md5(f'{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}'.encode('utf-8')).hexdigest()[:16]
        folder = os.path.join(tempfile.gettempdir(), 'manga_typeset_fonts')
        dst = os.path.join(folder, key + os.path.splitext(path)[1])
        dst.encode('ascii')
        if not os.path.exists(dst):
            os.makedirs(folder, exist_ok=True)
            tmp = f'{dst}.{os.getpid()}.part'
            shutil.copyfile(path, tmp)
            os.replace(tmp, dst)
        return dst
    except (OSError, UnicodeEncodeError):
        return path     # 临时目录不可用时退回原路径（仍能用，只是占内存）


@lru_cache(maxsize=64)
def _font(path, size):
    return ImageFont.truetype(_loadable_path(path), max(4, int(size)))


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


# 掩码都是整页大小，但有效部分只是一个气泡。以下运算都先裁到包围框再算，
# 否则文字多的页面（几十个文本框、两两比较）会在整页数组上反复运算，非常慢。
_RECTS = {}


def _rect(m):
    """掩码的包围框 (x0, y0, x1, y1, 像素数)，x1/y1 不含。同一页内按对象缓存（typeset_page 结束时清空）"""
    hit = _RECTS.get(id(m))
    if hit is not None and hit[0] is m:
        return hit[1]
    x, y, w, h = cv2.boundingRect(m)
    r = (x, y, x + w, y + h, int(np.count_nonzero(m[y:y + h, x:x + w])))
    _RECTS[id(m)] = (m, r)
    return r


def _overlap(a, b):
    """两个掩码重叠的像素数"""
    ra, rb = _rect(a), _rect(b)
    x0, y0, x1, y1 = max(ra[0], rb[0]), max(ra[1], rb[1]), min(ra[2], rb[2]), min(ra[3], rb[3])
    if x0 >= x1 or y0 >= y1:
        return 0
    return int(np.count_nonzero(a[y0:y1, x0:x1] & b[y0:y1, x0:x1]))


def same_bubble(a, b):
    if a is None or b is None:
        return False
    return _overlap(a, b) >= 0.5 * min(_rect(a)[4], _rect(b)[4])


def _erode(mask, iterations):
    """只在包围框附近腐蚀（结果与整页腐蚀相同）"""
    x0, y0, x1, y1, _ = _rect(mask)
    H, W = mask.shape
    x0, y0, x1, y1 = max(0, x0 - 1), max(0, y0 - 1), min(W, x1 + 1), min(H, y1 + 1)
    out = np.zeros_like(mask)
    # 默认边界（图外视为可用）与整页腐蚀一致：裁剪边外扩的 1 像素本身是 0，照样会从那里向内腐蚀
    out[y0:y1, x0:x1] = cv2.erode(mask[y0:y1, x0:x1], np.ones((3, 3), np.uint8), iterations=iterations)
    return out


FRAGMENT_CHARS = 3   # 原文（不计标点）不超过这么多字的框视为碎片


def _box_gap(a, b):
    return max(a[0] - b[2], b[0] - a[2], 0) + max(a[1] - b[3], b[1] - a[3], 0)


def merge_bubble_fragments(regions, boxes, found):
    """OCR 有时把气泡里一句话的末尾几个字（如「じゃ」「ね」）识别成单独的框。
    这种碎片与同一气泡里最近的框合成一段排版，否则碎片的译文会被单独挤到气泡一角。
    返回 [(区域, 框, 译文, (气泡掩码, 是否封闭))]，区域取较长的那段（决定方向、颜色）"""
    items = [[r, b, r.translation, f, i] for i, (r, b, f) in enumerate(zip(regions, boxes, found))]
    count = lambda r: len(re.sub(r'[\W_…‥ー〜～]', '', getattr(r, 'text', '') or ''))
    changed = True
    while changed:
        changed = False
        for frag in items:
            if not frag[3][1] or count(frag[0]) > FRAGMENT_CHARS:
                continue
            mates = [o for o in items if o is not frag and o[3][1] and same_bubble(o[3][0], frag[3][0])]
            if not mates:
                continue
            other = min(mates, key=lambda o: _box_gap(o[1], frag[1]))
            first, second = (frag, other) if frag[4] < other[4] else (other, frag)
            a, b = first[2].rstrip(), second[2].lstrip()
            sep = ' ' if a[-1:].isascii() and a[-1:].isalnum() and b[:1].isascii() and b[:1].isalnum() else ''
            main = other if count(other[0]) >= count(frag[0]) else frag
            box = (min(frag[1][0], other[1][0]), min(frag[1][1], other[1][1]),
                   max(frag[1][2], other[1][2]), max(frag[1][3], other[1][3]))
            merged = [main[0], box, a + sep + b, main[3], first[4]]
            items[items.index(other)] = merged
            items.remove(frag)
            changed = True
            break
    return [(r, b, t, f) for r, b, t, f, _ in items]


def _rect_dist(xs, ys, box, center=False):
    """各像素到矩形框的距离平方（框内为 0）；center=True 时改为到框中心的距离平方"""
    x1, y1, x2, y2 = box
    if center:
        return (xs - (x1 + x2) / 2) ** 2 + (ys - (y1 + y2) / 2) ** 2
    dx = np.maximum(np.maximum(x1 - xs, xs - x2), 0)
    dy = np.maximum(np.maximum(y1 - ys, ys - y2), 0)
    return dx.astype(np.int64) ** 2 + dy.astype(np.int64) ** 2


def split_shared_bubbles(masks, boxes):
    """同一个气泡里有多个文本区域时，按离哪个原文更近来划分气泡"""
    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = masks[i], masks[j]
            if not same_bubble(a, b):
                continue
            ra, rb = _rect(a), _rect(b)
            x0, y0 = min(ra[0], rb[0]), min(ra[1], rb[1])
            x1, y1 = max(ra[2], rb[2]), max(ra[3], rb[3])
            ys, xs = np.nonzero(a[y0:y1, x0:x1] | b[y0:y1, x0:x1])
            xs, ys = xs + x0, ys + y0
            # 按到原文框（矩形）的距离划分：原文字所在的地方一定归它自己。
            # 按中心点距离划分时，斜着相邻的两个框会被一条斜线切开，
            # 把某个框原文顶部的空间划给另一个框，译文只好往下挪
            da, db = _rect_dist(xs, ys, boxes[i]), _rect_dist(xs, ys, boxes[j])
            ca, cb = _rect_dist(xs, ys, boxes[i], True), _rect_dist(xs, ys, boxes[j], True)
            to_a = (da < db) | ((da == db) & (ca <= cb))   # 两框重叠处按中心距离
            na, nb = np.zeros_like(a), np.zeros_like(b)
            na[ys[to_a], xs[to_a]] = 255
            nb[ys[~to_a], xs[~to_a]] = 255
            masks[i], masks[j] = na, nb
    return masks


# ================================================================ 排版

class Area:
    """可用区域（气泡内缩后的掩码），提供按行/按列查询可用宽度"""

    def __init__(self, mask, anchor=None):
        """anchor: 希望文字居中的位置（如原文中心）。离边缘太近时向「视觉中心」靠拢。
        内部只保存包围框（外扩 1 像素）内的部分，对外的坐标仍是整页坐标"""
        H, W = mask.shape
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            x, y, w, h = 0, 0, W, H
        x0, y0, x1, y1 = max(0, x - 1), max(0, y - 1), min(W, x + w + 1), min(H, y + h + 1)
        self.ox, self.oy, self.shape = x0, y0, (H, W)
        crop = np.ascontiguousarray(mask[y0:y1, x0:x1])
        self.mask = crop > 0
        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        self.cx, self.cy = int(cx) + x0, int(cy) + y0
        self.anchored = False       # True：以原文位置为准，不再在气泡里重新居中
        if anchor is not None:
            best = dist[cy, cx]
            for t in (1.0, 0.75, 0.5, 0.25):
                ax = int(round(self.cx + (anchor[0] - self.cx) * t)) - x0
                ay = int(round(self.cy + (anchor[1] - self.cy) * t)) - y0
                if 0 <= ay < dist.shape[0] and 0 <= ax < dist.shape[1] and dist[ay, ax] >= 0.5 * best:
                    self.cx, self.cy = ax + x0, ay + y0
                    self.anchored = True
                    break
        ys, xs = np.nonzero(self.mask)
        self.bbox = (xs.min() + x0, ys.min() + y0, xs.max() + x0, ys.max() + y0) if len(xs) \
            else (self.cx, self.cy, self.cx, self.cy)

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
        if y0 < 0 or y1 >= self.shape[0]:
            return None
        lo, hi = -1, 10 ** 9
        for y in range(y0, y1 + 1, max(1, (y1 - y0) // 6)):
            if not 0 <= y - self.oy < self.mask.shape[0]:
                return None                 # 包围框外的行没有可用像素
            run = self._run(self.mask[y - self.oy], self.cx - self.ox)
            if run is None:
                return None
            lo, hi = max(lo, run[0]), min(hi, run[1])
        return (lo + self.ox, hi + self.ox) if hi > lo else None

    def v_span(self, x0, x1):
        x0, x1 = int(np.floor(x0)), int(np.ceil(x1))
        if x0 < 0 or x1 >= self.shape[1]:
            return None
        lo, hi = -1, 10 ** 9
        for x in range(x0, x1 + 1, max(1, (x1 - x0) // 6)):
            if not 0 <= x - self.ox < self.mask.shape[1]:
                return None
            run = self._run(self.mask[:, x - self.ox], self.cy - self.oy)
            if run is None:
                return None
            lo, hi = max(lo, run[0]), min(hi, run[1])
        return (lo + self.oy, hi + self.oy) if hi > lo else None


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
        # 各行共用同一条居中轴，气泡边缘弯曲处放不下时才平移，避免斜气泡里行行错开
        placed = []
        for i, text in enumerate(best):
            l, r = spans[i]
            w = measure(text)
            placed.append((text, min(max(area.cx - w / 2, l), r - w), top + i * lh))
        # 实际行数少于 n 时重新垂直居中
        if len(best) < n:
            k = len(best)
            new_top = area.cy - ((k - 1) * lh + size) / 2
            placed = [(t, x, new_top + i * lh) for i, (t, x, _) in enumerate(placed)]
        if not area.anchored:
            placed = _recenter_horizontal(placed, measure, size, lh, area) or placed
        return placed
    return None


def _recenter_horizontal(placed, measure, size, lh, area):
    """横排整块在气泡里重新居中：上下按文字所占宽度范围内的留白相等，左右按文字所占高度范围内居中。
    （内切圆圆心在不对称的气泡里会偏向一侧）放不下返回 None"""
    widths = [measure(t) for t, _, _ in placed]
    xl = min(x for _, x, _ in placed)
    xr = max(x + w for (_, x, _), w in zip(placed, widths))
    col = area.v_span(xl, xr)
    if not col:
        return None
    block_h = (len(placed) - 1) * lh + size
    top = (col[0] + col[1]) / 2 - block_h / 2
    row = area.h_span(top, top + block_h)
    if not row:
        return None
    axis = (row[0] + row[1]) / 2
    out = []
    for (text, _, _), w, i in zip(placed, widths, range(len(placed))):
        span = area.h_span(top + i * lh, top + i * lh + size)
        if not span or span[1] - span[0] < w - 0.5:
            return None
        out.append((text, min(max(axis - w / 2, span[0]), span[1] - w), top + i * lh))
    return out


def vertical_chars(line):
    """竖排：拆成逐字，标点换成竖排字形；两位以内数字横放（纵中横）"""
    out, i = [], 0
    while i < len(line):
        # 「!!」「!?」等两个连续感叹/问号并排放进一格（纵中横），与日漫原版一致
        m = re.match(r'[!?！？]{2}', line[i:])
        if m:
            out.append(m.group().translate(str.maketrans('！？', '!?')))
            i += 2
            continue
        m = re.match(r'[0-9]{1,2}(?![0-9])', line[i:])
        if m and (i == 0 or not line[i - 1].isdigit()):
            out.append(m.group())
            i += len(m.group())
            continue
        out.append(line[i])
        i += 1
    return out


def layout_vertical(units, fonts, size, area, spacing):
    """竖排：所有列共用同一条顶线（与日漫原版一致），整块在气泡里垂直居中。

    顶线不能高于任何一列处的气泡上沿，因此每列的容量按「该列下沿 - 共用顶线」计算，
    圆形气泡两侧的列自然排得短一些，而不是把列往下挪造成参差。
    """
    adv = size * 1.02
    col_w = size * spacing * 1.08
    slack = size * 0.3      # 顶线允许略微伸进内边距（可用区域已按气泡短边内缩过）
    measure = lambda s: len(vertical_chars(s.replace(' ', ''))) * adv
    total_h = measure(''.join(units))
    max_cols = max(1, int((area.bbox[2] - area.bbox[0]) / col_w) + 1)

    def spans_at(right, n):
        spans = [area.v_span(right - i * col_w - size, right - i * col_w) for i in range(n)]
        return None if any(s is None for s in spans) else spans

    def place(cols, right, spans):
        heights = [measure(text) for text in cols]
        lo = max(s[0] for s in spans) - slack           # 顶线下限：每列的气泡上沿
        hi = min(s[1] - h for s, h in zip(spans, heights))  # 顶线上限：每列都不超出气泡下沿
        if lo > hi + 0.5:
            return None
        if area.anchored:
            ideal = area.cy - max(heights) / 2
        else:   # 在这几列所在的范围内上下留白相等（内切圆圆心在上宽下窄的气泡里会偏上）
            ideal = (max(s[0] for s in spans) + hi) / 2
        top = min(max(ideal, lo), hi)
        return [(text.replace(' ', ''), right - i * col_w - size, top) for i, text in enumerate(cols)]

    for n in range(1, max_cols + 1):
        right = area.cx + ((n - 1) * col_w + size) / 2
        spans = spans_at(right, n)
        if spans is None:
            continue
        top_line = max(s[0] for s in spans) - slack
        caps = [s[1] - top_line for s in spans]
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
        # 实际列数少于 n 时按实际列数重新水平居中；居中后放不下就保持原位置
        k = len(best)
        block_w = (k - 1) * col_w + size
        right_k = area.cx + block_w / 2
        spans_k = spans_at(right_k, k)
        placed = place(best, right_k, spans_k) if spans_k else None
        placed = placed or place(best, right, spans[:k])
        if placed and not area.anchored:
            # 再在文字所占的高度范围内左右居中：内切圆圆心不一定在气泡左右正中
            top = placed[0][2]
            row = area.h_span(top, top + max(measure(t) for t, _, _ in placed))
            if row:
                right_c = (row[0] + row[1]) / 2 + block_w / 2
                spans_c = spans_at(right_c, k) if abs(right_c - (placed[0][1] + size)) > 1 else None
                placed = (place(best, right_c, spans_c) if spans_c else None) or placed
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
        if any(_overlap(mask, d) > 0.8 * _rect(mask)[4] for d in done):
            continue                                  # 同一个气泡已处理过
        done.append(mask)
        # 只在气泡包围框（外扩几像素，留给膨胀和模糊）内运算，结果与整页运算相同
        rx0, ry0, rx1, ry1, _ = _rect(mask)
        ox, oy = max(0, rx0 - 6), max(0, ry0 - 6)
        ex, ey = min(W, rx1 + 6), min(H, ry1 + 6)
        m = mask[oy:ey, ox:ex]
        sub = img[oy:ey, ox:ex]
        inner = cv2.erode(m, k3, iterations=2)
        hit = False
        for x1, y1, x2, y2 in keep_boxes:
            ix0, iy0, ix1, iy1 = max(x1, ox, 0), max(y1, oy, 0), min(x2 + 1, ex), min(y2 + 1, ey)
            if ix0 < ix1 and iy0 < iy1 and inner[iy0 - oy:iy1 - oy, ix0 - ox:ix1 - ox].any():
                hit = True
                break
        if hit:
            continue
        sel = inner > 0
        px = sub[sel].astype(np.int16)
        if len(px) < 200:
            continue
        bg = np.median(px, axis=0)
        diff = np.zeros(sel.shape, np.int16)
        diff[sel] = np.abs(px - bg).max(axis=1)
        if (diff[sel] < 30).mean() < 0.9:
            continue                                  # 底色不均匀
        ink = ((diff > 28) & sel).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        small = np.zeros(n, bool)
        small[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) <= limit
        if not small.any():
            continue
        total += int(small.sum())
        erase = np.where(small[labels], 255, 0).astype(np.uint8)
        erase = cv2.dilate(erase, k3, iterations=2) & m
        soft = cv2.GaussianBlur(erase.astype(np.float32) / 255, (5, 5), 0)[..., None]
        sub[:] = (sub * (1 - soft) + bg.reshape(1, 1, 3) * soft).astype(np.uint8)
    if total and logger:
        logger.info(f'清理气泡内残留：{total} 处（注音 / 残字 / 杂点）')
    return img


# ================================================================ 入口

SWITCH_GAIN = 1.3    # 另一方向的字号至少大 30% 才换方向，避免同页横竖来回跳（竖排顶部对齐在圆气泡里本就略吃亏）


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
    try:
        return _typeset_page(img, regions, settings, base_size, target_lang, logger, keep_boxes)
    finally:
        _RECTS.clear()          # 释放本页掩码的引用


def _typeset_page(img, regions, settings, base_size, target_lang, logger, keep_boxes):
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
    items = merge_bubble_fragments(regions, boxes, found)
    if logger and len(items) < len(regions):
        logger.info(f'合并气泡内的碎片文本框：{len(regions)} → {len(items)} 段')
    boxes = [b for _, b, _, _ in items]
    found = [f for _, _, _, f in items]
    masks = split_shared_bubbles([m for m, _ in found], boxes)
    # 与别的文本框共用一个气泡时，各自以原文位置为中心，而不是切开后那一半的几何中心
    shared = [any(j != i and same_bubble(found[i][0], found[j][0]) for j in range(len(found)))
              for i in range(len(found))]

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for (r, box, text, (_, enclosed)), mask, is_shared in zip(items, masks, shared):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        anchor = ((x1 + x2) / 2, (y1 + y2) / 2) if is_shared else None
        if mask is not None and enclosed:
            mx0, my0, mx1, my1, _ = _rect(mask)
            short = min(mx1 - mx0, my1 - my0) - 1
            m = max(2, int(short * st['margin']))
            area_mask = _erode(mask, max(1, m // 1))
            if _rect(area_mask)[4] < 50:
                area_mask = mask
            bubble_gray = float(np.median(gray[my0:my1, mx0:mx1][mask[my0:my1, mx0:mx1] > 0]))
        else:
            # 没有封闭气泡：在原文区域基础上略微放宽
            area_mask = np.zeros_like(gray, dtype=np.uint8)
            ex, ey = int(bw * 0.12) + 4, int(bh * 0.12) + 4
            cv2.rectangle(area_mask, (max(0, x1 - ex), max(0, y1 - ey)),
                          (min(W - 1, x2 + ex), min(H - 1, y2 + ey)), 255, -1)
            bubble_gray = None
            enclosed = False
        area = Area(area_mask, anchor)

        units = tokenize(text)
        fg, bg = _colors(r, bubble_gray)
        stroke_mode = st['stroke']
        use_stroke = stroke_mode == 'always' or (stroke_mode == 'auto' and not enclosed)

        # 描边会向外扩，先把可用区域相应内缩
        inner = area
        sw_ref = max(1, round(base * 0.12)) if use_stroke else 0
        if sw_ref:
            eroded = _erode(area_mask, sw_ref)
            if _rect(eroded)[4] >= 50:
                inner = Area(eroded, anchor)
        full = Area(mask, anchor) if (enclosed and mask is not None) else None

        # 候选方向：跟随原文时，先试原文方向；放不满基准字号才试另一方向
        direction = st['direction']
        src_vertical = _is_vertical_source(r)
        can_vertical = cjk_target and not re.search(r'[A-Za-z]{4,}', text)
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
                    logger.info(f'换成{"竖排" if vert else "横排"}更合适（字号 {size:.0f}→{sz:.0f}）：{text[:16]}')
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
                logger.warning(f'气泡内放不下，已在原文区域内缩小排版：{text[:20]}')
        if placed is None:
            if logger:
                logger.warning(f'排版失败：{text[:20]} | 封闭气泡={enclosed} 原文框={box} '
                               f'基准字号={base:.1f}')
            continue
        size = int(round(size))
        sw = max(1, round(size * 0.12)) if use_stroke else 0
        if vertical:
            draw_vertical(pil, draw, placed, fonts, size, fg, sw, bg if sw else None)
        else:
            draw_horizontal(draw, placed, fonts, size, fg, sw, bg if sw else None)
    return np.array(pil)
