"""
漫画翻译工作流：导入漫画（文件夹或 PDF）-> 导出 txt -> 外部整体翻译 -> 导回 txt -> 自动擦字排版并输出 PDF

命令行（单进程逐页处理）：
    python mit_workflow.py export <图片文件夹 | PDF> [--yes]
    python mit_workflow.py import <翻译好的txt | 工程文件夹 | 原图文件夹 | PDF>
    python mit_workflow.py auto   <图片文件夹 | PDF>
    python mit_workflow.py mt     <工程文件夹>

图形界面使用的并行模式：
    python mit_workflow.py worker
        常驻进程，模型只加载一次。从 stdin 逐行读取单页任务（JSON），
        结果以 "@@RESULT {json}" 输出到 stdout，其余输出均为日志。
        任务: {"id": 1, "mode": "export", "input": 图片路径}
              {"id": 2, "mode": "import", "input": 图片路径, "output": 输出路径,
               "src": [原文...], "trans": {"1": 译文, ...}}
        GUI 启动多个 worker，把所有漫画的所有页面分发给它们并行处理。

工程目录：<漫画文件夹或PDF同名>_翻译工程/
    project.json    工程数据（原图位置、每页原文）
    原图/            PDF 输入时自动提取的页面
    待翻译.txt       导出的原文稿，交给译者
    译文.txt         最近一次导入的译文
    output/          导入后生成的图片
    <名称>_中文.pdf   最终成品
"""
import argparse
import asyncio
import copy
import json
import logging
import os
import re
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_repo():
    """manga-image-translator 所在目录：本目录下的子目录，或（放在仓库 workflow/ 里时）上一级目录"""
    for c in (os.path.join(ROOT, 'manga-image-translator'), os.path.dirname(ROOT)):
        if os.path.isdir(os.path.join(c, 'manga_translator')):
            return c
    return os.path.join(ROOT, 'manga-image-translator')


REPO = _find_repo()
CONFIG_PATH = os.path.join(ROOT, 'config.json')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

EXPORT_FILE = '待翻译.txt'   # 旧版固定文件名，仅用于兼容
TRANS_FILE = '译文.txt'      # 也接受放在工程里、名为「译文.txt」的译文
PROJECT_FILE = 'project.json'
PAGES_DIR = '原图'
OUTPUT_DIR = 'output'
PROJECT_SUFFIX = '_翻译工程'
JPEG_QUALITY = 95
RESULT_PREFIX = '@@RESULT '

LANG_NAMES = {'CHS': '简体中文', 'CHT': '繁体中文', 'ENG': '英文', 'JPN': '日文', 'KOR': '韩文'}

HEADER = """\
# 工程: {project}
# 漫画: {name}
# 来源: {source}
# ============================================================
# 漫画翻译稿：共 {pages} 页、{lines} 句，请全部翻译成{lang}。
# 本文件由程序读回并自动嵌字，请严格遵守以下规则。
#
# 【格式 —— 格式错了，对应句子会丢失或保留原文】
#  1. 第一行「# 工程: ...」必须原样保留（程序靠它找到对应的漫画）
#  2. 「### 文件名」行原样保留，顺序不变，不增不删
#  3. 每行以 <编号> 开头：编号原样保留，只把编号后面的原文换成译文
#  4. 一个编号的译文必须写在同一行，不能换行；不要增加或删除任何行
#  5. 只输出翻译稿本身：不要加解释、注释、标题、总结，也不要写「以下是译文」
#  6. 必须输出全部 {pages} 页，不要省略、不要用「……同上」代替。
#     如果长度受限需要分次输出，每次都从某个「### 文件名」行开始，不要从半页开始
#
# 【每个编号是什么】
#  - 每个 <编号> 对应图上的一个文字框（通常是一个对话气泡），同一页内大致按阅读顺序排列
#  - 一句话可能被拆到相邻的几个编号（几个气泡）里：分别翻译各自那一段，让每段读起来衔接自然；
#    不要把整句挪进一个编号而让其他编号留空 —— 留空的编号会保留原文、不会被擦掉
#
# 【哪些留空】写成「<3>」这样，编号后什么都不写，该处保留原图：
#  - 扫图组致谢、赞助名单、网址水印、作者/出版社署名、宣传广告
#  - 画在画面上的拟声词、效果字（如 ドドド、ゴゴゴ）
#  - 没有意义的识别碎片（如单个乱码字母）
#  除此之外的对白、旁白、心声、标题都要翻译，不要留空
#
# 【译文要求】
#  - 气泡空间有限：译文要简洁口语化，{length}；译文过长会被自动缩小字号
#  - 不要自己加空格、换行来排版，程序会按气泡形状自动排版和换行
#  - {punct}
#  - 人名、称呼、口癖整本统一；语气符合角色身份和场景，整本连贯翻译
#  - 原文是 OCR 识别结果，可能有错字、缺撇号（IM=I'M、DONT=DON'T）、
#    字母数字混淆（6OING=GOING、O8SESSION=OBSESSION），按上下文理解原意后翻译
# ============================================================
# 翻译完成后：以相同文件名「{file}」（UTF-8）保存到与「待翻译」同级的「译文」文件夹
# ============================================================
"""

PUNCT = {
    'CHS': '用中文全角标点：，。！？：；「」『』（），省略号用「…」，破折号用「——」，不要用半角标点',
    'CHT': '用中文全形標點：，。！？：；「」『』（），刪節號用「…」，破折號用「——」，不要用半形標點',
    'JPN': '日本語の全角句読点を使う：、。！？「」『』，三点リーダは「…」',
    'KOR': '한국어 문장부호 사용, 말줄임표는 「…」',
}
LENGTH = {
    'CHS': '中文字数与日文原文大致相当，或约为英文原文字母数的三分之一到一半',
    'CHT': '中文字數與日文原文大致相當，或約為英文原文字母數的三分之一到一半',
}


def make_header(project, pages, lines, lang_code, source='', file=''):
    return HEADER.format(project=project, pages=pages, lines=lines, name=project_name(project),
                         source=source, file=file or os.path.basename(export_path_for(project)),
                         lang=LANG_NAMES.get(lang_code, lang_code),
                         punct=PUNCT.get(lang_code, 'use normal punctuation of the target language; ellipsis as "…"'),
                         length=LENGTH.get(lang_code, '长度与原文相当'))


class WorkflowError(Exception):
    pass


# ================================================================ 纯工具函数（不加载引擎）

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def list_images(folder):
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(IMAGE_EXTS):
                files.append(os.path.relpath(os.path.join(root, n), folder).replace('\\', '/'))
    return sorted(files, key=natural_key)


def norm(text):
    return ' '.join((text or '').split())


_CONFIG_CACHE = {}


def _config():
    """按修改时间缓存的配置（只读，界面每 0.5 秒刷新都会间接读取）"""
    mtime = os.path.getmtime(CONFIG_PATH)
    if _CONFIG_CACHE.get('mtime') != mtime:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            _CONFIG_CACHE.update(mtime=mtime, data=json.load(f))
    return _CONFIG_CACHE['data']


def read_config():
    """返回配置副本，调用方可随意修改后写回"""
    return copy.deepcopy(_config())


def target_lang():
    try:
        return read_config()['translator']['target_lang']
    except Exception:
        return 'CHS'


def typeset_settings():
    """config.json 的 typeset 段；renderer 为 default 时返回 None（使用引擎原渲染）"""
    ts = read_config().get('typeset', {})
    if ts.get('renderer', 'typeset') == 'default':
        return None
    return {k: v for k, v in ts.items() if k != 'renderer' and not k.startswith('_')}


def book_base_size(project):
    """全书基准字号：导出时记录的原文字号中位数（忽略最小的 15%，那多半是拟声词/注音）"""
    sizes = sorted(v for page in (project.get('sizes') or {}).values() for v in page if v and v > 0)
    if not sizes:
        return None
    sizes = sizes[int(len(sizes) * 0.15):] or sizes
    return float(np.median(sizes))


def project_dir_for(src):
    src = os.path.abspath(src).rstrip('\\/')
    base = os.path.splitext(src)[0] if src.lower().endswith('.pdf') else src
    return base + PROJECT_SUFFIX


def project_name(project_dir):
    name = os.path.basename(project_dir)
    return name[:-len(PROJECT_SUFFIX)] if name.endswith(PROJECT_SUFFIX) else name


AI_README = """# 给 AI 的翻译任务说明

本文件夹里的每个 .txt 是一本（一章）漫画的待翻译文本，文件开头的「# 漫画:」写明了是哪本。

任务：
1. 逐个翻译本文件夹里所有 .txt（不包括本说明文件）
2. 每个 txt 开头的注释里写有详细规则（格式、留空规则、标点、长度），必须严格遵守
3. 翻译结果以**相同文件名**、UTF-8 编码写入同级的「译文」文件夹：
   待翻译/Vol. 8.txt  →  译文/Vol. 8.txt
4. 译文文件要保留原文件的所有 # 注释行、### 页标记行和 <编号> 标签，只替换编号后面的文字
5. 不同的 txt 是不同的漫画，各自独立翻译；同一个 txt 内要前后连贯、译名统一
"""


def transfer_dirs():
    """中转目录：(待翻译目录, 译文目录)。可在 config.json 的 transfer_dir 修改"""
    base = _config().get('transfer_dir') or os.path.join(ROOT, '翻译中转')
    return os.path.join(base, '待翻译'), os.path.join(base, '译文')


def ensure_transfer_dirs():
    todo, done = transfer_dirs()
    os.makedirs(todo, exist_ok=True)
    os.makedirs(done, exist_ok=True)
    readme = os.path.join(todo, '00_给AI的说明.md')
    if not os.path.exists(readme):
        with open(readme, 'w', encoding='utf-8') as f:
            f.write(AI_README)
    return todo, done


def _safe_name(name):
    return re.sub(r'[\\/:*?"<>|]+', '_', name).strip(' .') or '未命名'


_OWNER_CACHE = {}


def _owner_of(txt_path):
    """txt 文件头「# 工程:」指向的工程目录（规范化路径），按修改时间缓存"""
    try:
        mtime = os.path.getmtime(txt_path)
    except OSError:
        return None
    hit = _OWNER_CACHE.get(txt_path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        with open(txt_path, 'r', encoding='utf-8-sig') as f:
            head = f.read(4000)
    except OSError:
        return None
    m = re.search(r'^\s*#\s*工程[:：]\s*(.+?)\s*$', head, re.M)
    owner = os.path.normcase(os.path.abspath(m.group(1))) if m else None
    _OWNER_CACHE[txt_path] = (mtime, owner)
    return owner


_EXPORT_NAME_CACHE = {}


def _stored_export_name(project_dir):
    """工程里记录的导出文件名，按 project.json 修改时间缓存。
    project.json 含全书原文，长篇有几百 KB，界面每次刷新都整个解析会很卡"""
    p = os.path.join(project_dir, PROJECT_FILE)
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return None
    hit = _EXPORT_NAME_CACHE.get(p)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        name = load_project(project_dir).get('export_name')
    except Exception:
        name = None
    _EXPORT_NAME_CACHE[p] = (mtime, name)
    return name


def export_name_for(project_dir, create=False):
    """中转目录里这本漫画的 txt 文件名。重名（别的漫画已占用）时加上所在文件夹名区分"""
    name = _stored_export_name(project_dir)
    if name:
        return name
    todo, _ = transfer_dirs()
    base = _safe_name(project_name(project_dir))
    me = os.path.normcase(os.path.abspath(project_dir))
    parent = _safe_name(os.path.basename(os.path.dirname(os.path.abspath(project_dir))))
    for cand in (base, f'{base} ({parent})') + tuple(f'{base} ({parent} {i})' for i in range(2, 100)):
        p = os.path.join(todo, cand + '.txt')
        if not os.path.exists(p) or _owner_of(p) in (None, me):
            return cand + '.txt'
    return base + '.txt'


def export_path_for(project_dir, create=False):
    """中转目录/待翻译/<漫画名>.txt"""
    todo, _ = ensure_transfer_dirs() if create else transfer_dirs()
    return os.path.join(todo, export_name_for(project_dir))


def trans_path_for(project_dir):
    """中转目录/译文/<漫画名>.txt"""
    _, done = transfer_dirs()
    return os.path.join(done, export_name_for(project_dir))


def migrate_legacy(project_dir):
    """旧版工程（待翻译.txt 放在工程里）：把导出稿搬到中转目录"""
    if not os.path.isfile(os.path.join(project_dir, PROJECT_FILE)):
        return
    new = export_path_for(project_dir, create=True)
    if os.path.exists(new):
        return
    for old in (os.path.join(project_dir, f'{project_name(project_dir)}_待翻译.txt'),
                os.path.join(project_dir, EXPORT_FILE)):
        if os.path.exists(old):
            P = load_project(project_dir)
            with open(old, 'r', encoding='utf-8') as f:
                body = f.read()
            body = body[body.index('\n### '):] if '\n### ' in body else body
            header = make_header(project_dir, len(P['pages']), sum(len(v) for v in P['pages'].values()),
                                 target_lang(), P.get('source', ''), os.path.basename(new))
            with open(new, 'w', encoding='utf-8') as f:
                f.write(header + body)
            os.remove(old)
            return


def find_translation(project_dir):
    """这本漫画的译文：中转目录/译文 里同名文件，或文件头「# 工程:」指向本工程的任意 txt；
    兼容旧版放在工程里的 译文.txt。都有时取最新的。没有返回 None"""
    _, done = transfer_dirs()
    me = os.path.normcase(os.path.abspath(project_dir))
    cands = [trans_path_for(project_dir), os.path.join(project_dir, TRANS_FILE),
             os.path.join(project_dir, f'{project_name(project_dir)}_译文.txt')]
    cands += _translations_by_owner(done).get(me, [])
    cands = [p for p in dict.fromkeys(cands) if os.path.isfile(p)]
    return max(cands, key=os.path.getmtime) if cands else None


_TRANS_INDEX = {}


def _translations_by_owner(done):
    """译文目录里 {工程目录: [txt]}。增删、改名文件会改变目录修改时间，此时才重新扫描；
    否则直接复用（界面刷新时每本书都要查，逐个读文件头太慢）"""
    try:
        key = (done, os.path.getmtime(done))
    except OSError:
        return {}
    if _TRANS_INDEX.get('key') != key:
        index = {}
        for f in os.listdir(done):
            if f.lower().endswith('.txt'):
                p = os.path.join(done, f)
                owner = _owner_of(p)
                if owner:
                    index.setdefault(owner, []).append(p)
        _TRANS_INDEX.update(key=key, index=index)
    return _TRANS_INDEX['index']


INDEX_PATH = os.path.join(ROOT, 'projects_index.json')


def register_project(project_dir):
    """把工程位置记入索引，方便以后（即使已从列表移除）找到并清理"""
    p = os.path.abspath(project_dir)
    try:
        with open(INDEX_PATH, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except Exception:
        items = []
    if any(os.path.normcase(x) == os.path.normcase(p) for x in items):
        return
    items.append(p)
    tmp = INDEX_PATH + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    os.replace(tmp, INDEX_PATH)


def known_projects():
    """所有找得到的工程目录：索引里记录的 + 中转目录 txt 文件头「# 工程:」指向的（去重，只保留还存在的）"""
    found = {}
    try:
        with open(INDEX_PATH, 'r', encoding='utf-8') as f:
            for p in json.load(f):
                found[os.path.normcase(os.path.abspath(p))] = os.path.abspath(p)
    except Exception:
        pass
    for d in transfer_dirs():
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith('.txt'):
                    path = os.path.join(d, f)
                    owner = _owner_of(path)
                    if owner and owner not in found:
                        with open(path, 'r', encoding='utf-8-sig') as fh:
                            m = re.search(r'^\s*#\s*工程[:：]\s*(.+?)\s*$', fh.read(4000), re.M)
                        found[owner] = m.group(1) if m else owner
    return sorted((p for p in found.values() if os.path.isfile(os.path.join(p, PROJECT_FILE))),
                  key=natural_key)


def find_projects_under(folder, max_depth=5):
    """在文件夹里查找所有「_翻译工程」目录"""
    out = []
    base = os.path.abspath(folder).rstrip('\\/')
    depth0 = base.count(os.sep)
    for root, dirs, _ in os.walk(base):
        if root.count(os.sep) - depth0 >= max_depth:
            dirs[:] = []
        for d in list(dirs):
            if d.endswith(PROJECT_SUFFIX) and os.path.isfile(os.path.join(root, d, PROJECT_FILE)):
                out.append(os.path.join(root, d))
                dirs.remove(d)
    return out


def scan_translations():
    """扫描中转目录/译文，返回 [(txt路径, 工程目录或 None)]。按文件头定位，失败时按文件名匹配导出稿"""
    todo, done = transfer_dirs()
    out = []
    if not os.path.isdir(done):
        return out
    for f in sorted(os.listdir(done), key=natural_key):
        if not f.lower().endswith('.txt'):
            continue
        p = os.path.join(done, f)
        owner = _owner_of(p)
        if not owner:
            src = os.path.join(todo, f)
            owner = _owner_of(src) if os.path.exists(src) else None
        if owner and not os.path.isfile(os.path.join(owner, PROJECT_FILE)):
            owner = None
        out.append((p, owner))
    return out


def pdf_path_for(project_dir):
    return os.path.join(project_dir, f'{project_name(project_dir)}_中文.pdf')


def load_project(project_dir):
    with open(os.path.join(project_dir, PROJECT_FILE), 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_input(src, project_dir):
    """返回原图所在文件夹。PDF 会把每页图片无损提取到 <工程>/原图/"""
    src = os.path.abspath(src)
    if os.path.isdir(src):
        return src
    if not src.lower().endswith('.pdf'):
        raise WorkflowError(f'不支持的输入（需要图片文件夹或 PDF）: {src}')
    import pymupdf
    out = os.path.join(project_dir, PAGES_DIR)
    os.makedirs(out, exist_ok=True)
    doc = pymupdf.open(src)
    width = max(3, len(str(doc.page_count)))
    for i, page in enumerate(doc, 1):
        name = f'{i:0{width}d}'
        imgs = page.get_images()
        if len(imgs) == 1:  # 扫描版：每页一张图，直接取原始数据，不重新压缩
            x = doc.extract_image(imgs[0][0])
            data, ext = x['image'], x['ext']
            if ext not in ('jpeg', 'jpg', 'png'):
                data, ext = page.get_pixmap(dpi=200).tobytes('png'), 'png'
        else:  # 多图或矢量页面：按 200dpi 渲染
            data, ext = page.get_pixmap(dpi=200).tobytes('png'), 'png'
        with open(os.path.join(out, f'{name}.{ext}'), 'wb') as f:
            f.write(data)
    return out


def write_export(project_dir, src, input_dir, pages, sizes=None):
    """pages: {相对路径: [原文...]}；sizes: {相对路径: [原文字号...]}。写出 project.json 与 待翻译.txt"""
    names = sorted(pages, key=natural_key)
    body, total = [], 0
    for rel in names:
        body.append(f'\n### {rel}')
        for i, text in enumerate(pages[rel], 1):
            body.append(f'<{i}>{text}')
            total += 1
    lang = target_lang()
    export_path = export_path_for(project_dir, create=True)
    header = make_header(project_dir, len(names), total, lang, os.path.abspath(src), os.path.basename(export_path))
    legacy = os.path.join(project_dir, EXPORT_FILE)
    if os.path.exists(legacy):
        os.remove(legacy)
    with open(export_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n'.join(body) + '\n')
    with open(os.path.join(project_dir, PROJECT_FILE), 'w', encoding='utf-8') as f:
        data = {'source': os.path.abspath(src), 'input_dir': input_dir,
                'export_name': os.path.basename(export_path),
                'pages': {r: pages[r] for r in names}}
        if sizes:
            data['sizes'] = {r: sizes.get(r, []) for r in names}
        json.dump(data, f, ensure_ascii=False, indent=1)
    try:
        register_project(project_dir)
    except OSError:
        pass
    return export_path, len(names), total


def locate_project(arg):
    """参数可以是：翻译好的 txt / 工程文件夹 / 原图文件夹 / PDF。返回 (工程目录, 译文路径或 None)"""
    arg = os.path.abspath(arg.strip().strip('"'))
    if os.path.isfile(arg) and arg.lower().endswith('.txt'):
        with open(arg, 'r', encoding='utf-8-sig') as f:
            project = project_from_text(f.read())
        if project:
            return project, arg
        d = os.path.dirname(arg)
        if os.path.isfile(os.path.join(d, PROJECT_FILE)):
            return d, arg
        raise WorkflowError(f'无法从 txt 找到对应工程（缺少「# 工程:」行）: {arg}')
    if os.path.isfile(os.path.join(arg, PROJECT_FILE)):
        return arg, None
    p = project_dir_for(arg)
    if os.path.isfile(os.path.join(p, PROJECT_FILE)):
        return p, None
    raise WorkflowError(f'找不到翻译工程: {arg}')


def project_from_text(text):
    for m in re.finditer(r'^\s*#\s*工程[:：]\s*(.+?)\s*$', text, re.M):
        if os.path.isfile(os.path.join(m.group(1), PROJECT_FILE)):
            return m.group(1)
    return None


def parse_translation(text, pages):
    """解析译文文本，返回 ({页: {编号: 译文}}, [问题])。容忍聊天复制带来的 ``` 围栏"""
    trans, page, errors = {}, None, []
    for ln, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith('```') or (s.startswith('#') and not s.startswith('###')):
            continue
        if s.startswith('###'):
            page = s[3:].strip()
            if page not in pages:
                errors.append(f'第{ln}行: 未知页「{page}」')
                page = None
            else:
                trans.setdefault(page, {})
            continue
        m = re.match(r'^\s*[<＜](\d+)[>＞]\s?(.*)$', line)
        if not m:
            if page is not None:
                errors.append(f'第{ln}行: 缺少 <编号> 标签，已忽略: {s[:40]}')
            continue
        if page is None:
            continue
        trans[page][int(m.group(1))] = m.group(2).strip()
    return trans, errors


def check_translation(trans, pages):
    errors = []
    total = sum(len(v) for v in pages.values())
    same = sum(1 for p, src in pages.items() for i, s in enumerate(src, 1)
               if s and trans.get(p, {}).get(i, '') == s)
    if total and same == total:
        raise WorkflowError('译文与原文完全相同，看起来还没有翻译')
    if total and not any(trans.get(p) for p, src in pages.items() if src):
        raise WorkflowError('译文里的页面与这本漫画一页都对不上，可能是另一本漫画的译文')
    for page, src in pages.items():
        got = trans.get(page, {})
        missing = [i for i in range(1, len(src) + 1) if i not in got]
        if src and page not in trans:
            errors.append(f'{page}: 整页缺失，将保留原文')
        elif missing:
            errors.append(f'{page}: 缺少编号 {missing}，这些句子保留原文')
    return errors


def build_pdf(image_dir, pdf_path, title):
    import img2pdf
    imgs = [os.path.join(image_dir, f) for f in list_images(image_dir)]
    if not imgs:
        return None
    import tempfile
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import credits
    tmp = pdf_path + '.part'
    with tempfile.TemporaryDirectory() as work:
        pages = credits.compose(imgs, credits.settings(read_config()), work)  # 汉化组信息页、水印
        with open(tmp, 'wb') as f:
            f.write(img2pdf.convert(pages, title=title))
    os.replace(tmp, pdf_path)
    return pdf_path


# ================================================================ 碎片文本框自动合并

_CACHED = ('xyxy', 'xywh', 'center', 'unrotated_polygons', 'unrotated_min_rect', 'min_rect',
           'polygon_aspect_ratio', 'unrotated_size', 'aspect_ratio')


def _is_cjk(s):
    return any('぀' <= c <= 'ヿ' or '一' <= c <= '鿿' or '가' <= c <= '힯' for c in s)


def _is_fragment(text):
    """检测把一句话拆出来的孤立短片段，如 "I" / "A" / "OH"。
    只针对拉丁字母：中日韩的一两个字（「白様」「アラ⁉」）通常本身就是一句完整台词"""
    t = norm(text)
    if _is_cjk(t):
        return False
    core = re.sub(r'[\W_]', '', t)
    return 0 < len(core) <= 2 and ' ' not in t


def _gap(a, b):
    """两个框在水平、垂直方向上的间隙（重叠为负）"""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    return max(ax1, bx1) - min(ax2, bx2), max(ay1, by1) - min(ay2, by2)


def _merge_pair(frag, main):
    fx, fy = frag.center
    mx, my = main.center
    gx, gy = _gap(frag, main)
    cjk = _is_cjk(main.text)
    if gy >= gx:          # 上下排列：上面的在前
        first_is_frag = fy < my
    elif cjk:             # 竖排日文左右排列：右边的列在前
        first_is_frag = fx > mx
    else:                 # 横排左右排列：左边的在前
        first_is_frag = fx < mx
    parts = [frag, main] if first_is_frag else [main, frag]
    joiner = '' if cjk else ' '
    new = copy.copy(main)
    for k in _CACHED:
        new.__dict__.pop(k, None)
    new.lines = np.concatenate([p.lines for p in parts]).astype(np.int32)
    new.texts = [t for p in parts for t in (p.texts or [p.text])]
    new.text = joiner.join(norm(p.text) for p in parts)
    new.text_raw = new.text
    new.font_size = max(frag.font_size, main.font_size)
    return new


def merge_fragments(regions, logger=None):
    """把孤立的极短片段并入同一气泡里紧邻的文本框。

    检测模型偶尔把 "I / SEE..." 这类单独成行的短词拆成独立框，
    分开翻译排版后会变成一个挤着小字的框。这里按几何关系自动合并：
    片段与目标框在一个方向上投影重叠、另一个方向的间隙不超过 1.5 倍字号。
    """
    regions = list(regions)
    changed = True
    while changed:
        changed = False
        for frag in regions:
            if not _is_fragment(frag.text):
                continue
            best, best_gap = None, None
            for other in regions:
                if other is frag:
                    continue
                fa, fb = max(frag.font_size, 1), max(other.font_size, 1)
                if max(fa, fb) > 2.0 * min(fa, fb):
                    continue                          # 字号差太多：不是同一句（单个字母的字号估计偏小，故放宽到 2 倍）
                gx, gy = _gap(frag, other)
                fs = max(frag.font_size, other.font_size, 8)
                if gx < 0 and gy <= 1.5 * fs:        # 水平投影重叠，上下相邻
                    g = max(gy, 0)
                elif gy < 0 and gx <= 1.5 * fs:      # 垂直投影重叠，左右相邻
                    g = max(gx, 0)
                else:
                    continue
                if best_gap is None or g < best_gap:
                    best, best_gap = other, g
            if best is None:
                continue
            merged = _merge_pair(frag, best)
            if logger:
                logger.info(f'合并碎片文本框: "{frag.text}" + "{best.text}" -> "{merged.text}"')
            idx = regions.index(best)
            regions[idx] = merged
            regions.remove(frag)
            changed = True
            break
    return regions


# ================================================================ 引擎封装（单页处理）

def _ensure_engine_path():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.chdir(REPO)  # 程序内部按仓库目录找 models/ fonts/ 等资源


def load_config(overrides=None):
    from manga_translator import Config
    data = {k: v for k, v in read_config().items() if not k.startswith('_')}
    for section, values in (overrides or {}).items():
        data.setdefault(section, {}).update(values)
    return Config(**data)


def make_translator_class():
    from manga_translator.manga_translator import MangaTranslator

    class WorkflowTranslator(MangaTranslator):
        """export 模式截获 OCR 结果后中止；import 模式用译文替代翻译器"""
        mode = 'auto'
        current_page = None

        def __init__(self, params):
            super().__init__(params)
            self.exported = {}      # page -> [原文]
            self.exported_sizes = {}  # page -> [原文字号]
            self.translations = {}  # page -> {编号: 译文}
            self.source_texts = {}  # page -> [导出时的原文]
            self.warnings = []
            self.typeset_settings = None  # None = 用引擎默认渲染
            self.base_size = None
            self.cached = None            # 当前页的识别缓存 (文字区域, 文字掩码)
            self.export_capture = {}      # 导出时截获的 (文字区域, 文字掩码)

        def _setup_log_file(self):
            """不用引擎自带的日志文件：它每启动一个进程就在 result/ 新建一个文件、把日志级别调到 DEBUG，
            还把 print 改成每打一行都重新打开文件追加。GPU 进程会定期换新，这些文件只会越积越多；
            进程输出本来就由界面收集。这里只顺手清理旧的引擎日志"""
            _prune_engine_logs()

        def _log(self):
            """本工具自己的排版信息用的 logger（引擎日志只留警告以上，这个保持 INFO）"""
            log = getattr(self, 'log', None)
            if log is None:
                from manga_translator.manga_translator import logger as log
            return log

        async def _run_detection(self, config, ctx):
            if self.cached is not None:
                regions, mask_raw = self.cached
                return list(regions), mask_raw, None
            return await super()._run_detection(config, ctx)

        async def _run_ocr(self, config, ctx):
            if self.cached is not None:
                return ctx.textlines
            return await super()._run_ocr(config, ctx)

        async def _run_textline_merge(self, config, ctx):
            if self.cached is not None:
                return [copy.deepcopy(r) for r in self.cached[0]]
            regions = await super()._run_textline_merge(config, ctx)
            return merge_fragments(regions, self._log())

        async def _run_text_translation(self, config, ctx):
            if self.mode == 'export':
                self.exported[self.current_page] = [norm(r.text) for r in ctx.text_regions]
                self.export_capture[self.current_page] = (copy.deepcopy(list(ctx.text_regions)), ctx.mask_raw)
                self.exported_sizes[self.current_page] = [int(r.font_size) for r in ctx.text_regions]
                return 'cancel'  # 跳过擦字与渲染
            if self.mode == 'import' and self.cached is None:
                # 没有识别缓存时本次重新做了检测和 OCR，记下结果，导入完成后补存缓存
                self.import_capture = ([norm(r.text) for r in ctx.text_regions],
                                       copy.deepcopy(list(ctx.text_regions)), ctx.mask_raw)
            before = list(ctx.text_regions or [])
            result = await super()._run_text_translation(config, ctx)
            # 记下留空（保留原文）的区域，清理气泡时避开它们
            self.keep_boxes = [tuple(int(v) for v in r.xyxy) for r in before
                               if not (getattr(r, 'translation', '') or '').strip()]
            return result

        async def _run_text_rendering(self, config, ctx):
            if self.mode != 'import' or self.typeset_settings is None:
                return await super()._run_text_rendering(config, ctx)
            import typeset
            return typeset.typeset_page(ctx.img_inpainted, ctx.text_regions, self.typeset_settings,
                                        self.base_size, config.translator.target_lang, self._log(),
                                        getattr(self, 'keep_boxes', ()))

        async def _dispatch_with_context(self, config, texts, ctx):
            if self.mode != 'import':
                return await super()._dispatch_with_context(config, texts, ctx)
            page = self.current_page
            src = self.source_texts.get(page, [])
            trans = self.translations.get(page, {})
            result, used = [], set()
            for i, text in enumerate(texts):
                t = norm(text)
                idx = i if i < len(src) and src[i] == t else None
                if idx is None:  # 顺序不一致时按原文内容匹配
                    idx = next((j for j, s in enumerate(src) if s == t and j not in used), None)
                if idx is None:
                    self.warnings.append(f'{page}: 未在导出稿中找到「{t}」，保留原文')
                    result.append('')
                    continue
                used.add(idx)
                # 制表符横线在竖排中不会旋转，统一换成破折号
                result.append(trans.get(idx + 1, '').replace('─', '—'))
            return result

    return WorkflowTranslator


# ---------------------------------------------------------------- 识别缓存
# 导出时把每页的检测 + OCR 结果（文字区域、文字掩码）存进工程，导入时直接读回，
# 跳过第二次检测和 OCR；原图有变化或原文对不上时自动失效，退回重新识别。

CACHE_DIR = '识别缓存'
CACHE_VERSION = 1


def cache_path_for(project_dir, rel):
    return os.path.join(project_dir, CACHE_DIR, rel + '.pkl.gz')


def _image_sig(path):
    st = os.stat(path)
    return st.st_size, int(st.st_mtime)


def save_ocr_cache(path, image_path, texts, regions, mask_raw):
    import gzip
    import pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with gzip.open(tmp, 'wb', compresslevel=3) as f:
        pickle.dump({'v': CACHE_VERSION, 'sig': _image_sig(image_path), 'texts': texts,
                     'regions': regions, 'mask_raw': mask_raw}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_ocr_cache(path, image_path, texts):
    """缓存有效时返回 (文字区域, 文字掩码)，否则 None"""
    if not path or not os.path.isfile(path):
        return None
    try:
        import gzip
        import pickle
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        if data.get('v') != CACHE_VERSION or tuple(data['sig']) != _image_sig(image_path) \
                or list(data['texts']) != list(texts):
            return None
        return data['regions'], data['mask_raw']
    except Exception:
        return None


def open_image(path):
    from PIL import Image
    img = Image.open(path)
    img.verify()
    return Image.open(path)


GRAY_TOLERANCE = 12   # 三通道相差在这以内算黑白（实测黑白页几乎都在 12 以内，彩色页至少 40 以上）


def compact_page(img):
    """引擎输出的页面一律是 RGBA：透明通道全不透明时去掉；黑白页（三通道几乎相同）转成灰度。
    彩色页（封面、彩插、带彩色水印的页）保持彩色。PNG 页面体积小一半多、保存快约 2.5 倍，合成的 PDF 也更小"""
    if img.mode == 'RGBA':
        if img.getchannel('A').getextrema()[0] < 255:
            return img                  # 真有透明部分，原样保留
        img = img.convert('RGB')
    if img.mode == 'RGB':
        a = np.asarray(img).astype(np.int16)
        chroma = np.maximum(np.abs(a[..., 0] - a[..., 1]), np.abs(a[..., 1] - a[..., 2]))
        # 允许零星几个像素略超（擦字、JPEG 噪点），但只要有明显的彩色就保持彩色
        if chroma.max() <= 2 * GRAY_TOLERANCE and np.count_nonzero(chroma > GRAY_TOLERANCE) <= chroma.size // 10000:
            img = img.convert('L')
    return img


def save_page(img, path):
    """保存生成好的页面，格式按扩展名（JPEG 用 JPEG_QUALITY，其余用 Pillow 默认参数，与引擎一致）"""
    img = compact_page(img)
    if os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg'):
        if img.mode not in ('L', 'RGB'):
            img = img.convert('RGB')
        img.save(path, 'JPEG', quality=JPEG_QUALITY)
    else:
        img.save(path)


# ================================================================ 引擎运行时调整
# 只在运行时替换引擎里的个别函数，不改引擎源码（manga-image-translator 是 git 仓库，可照常更新）

class _LazyBilateral:
    """cv2.bilateralFilter 的按需版本，供遮罩细化使用。

    引擎的 complete_mask 先对整页做一次双边滤波（每页约 170ms，占遮罩细化的 3/4），
    之后只取各文字行附近的小块。这里改为取到哪块才滤哪块：每块向外多滤一个滤波半径，
    块内结果与整页滤波逐像素相同。其他用法（整体读取、取属性）退回整页滤波，行为不变。"""

    def __init__(self, cv2, src, d, sigma_color, sigma_space):
        self._cv2, self._src, self._args = cv2, src, (d, sigma_color, sigma_space)
        self.shape, self.dtype, self.ndim = src.shape, src.dtype, src.ndim
        self._pad = (d // 2 if d > 0 else int(round(sigma_space * 1.5))) + 1
        self._full = None

    def _whole(self):
        if self._full is None:
            self._full = self._cv2.bilateralFilter(self._src, *self._args)
        return self._full

    def __array__(self, dtype=None, copy=None):
        a = self._whole()
        return a if dtype is None else a.astype(dtype)

    def __getattr__(self, name):
        return getattr(self._whole(), name)

    def __getitem__(self, key):
        if (self._full is None and isinstance(key, tuple) and len(key) == 2
                and all(isinstance(k, slice) and k.step in (None, 1) for k in key)):
            H, W = self.shape[:2]
            y0, y1, _ = key[0].indices(H)
            x0, x1, _ = key[1].indices(W)
            if y1 > y0 and x1 > x0:
                p = self._pad
                cy0, cx0, cy1, cx1 = max(0, y0 - p), max(0, x0 - p), min(H, y1 + p), min(W, x1 + p)
                out = self._cv2.bilateralFilter(np.ascontiguousarray(self._src[cy0:cy1, cx0:cx1]), *self._args)
                return out[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]
        return self._whole()[key]


class _MaskCv2:
    """遮罩细化模块看到的 cv2：只把 bilateralFilter 换成按需版本，其余原样转发"""

    def __init__(self, cv2):
        self._cv2 = cv2

    def __getattr__(self, name):
        return getattr(self._cv2, name)

    def bilateralFilter(self, src, d, sigmaColor, sigmaSpace, *args, **kwargs):  # noqa: N802,N803
        if args or kwargs or not isinstance(src, np.ndarray):
            return self._cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, *args, **kwargs)
        return _LazyBilateral(self._cv2, src, d, sigmaColor, sigmaSpace)


def _patch_engine():
    """遮罩细化：双边滤波只算文字附近的小块，并关掉它每页一条的进度条"""
    from manga_translator.mask_refinement import text_mask_utils as tmu
    if isinstance(tmu.cv2, _MaskCv2):
        return
    tmu.cv2 = _MaskCv2(tmu.cv2)
    tmu.tqdm = lambda iterable, *args, **kwargs: iterable


ENGINE_LOG_KEEP_DAYS = 7


def _prune_engine_logs():
    """删掉引擎 result/ 目录里 7 天前的 log_*.txt（引擎每启动一次就新建一个）"""
    folder = os.path.join(REPO, 'result')
    limit = time.time() - ENGINE_LOG_KEEP_DAYS * 86400
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for name in names:
        p = os.path.join(folder, name)
        if name.startswith('log_') and name.endswith('.txt'):
            try:
                if os.path.getmtime(p) < limit:
                    os.remove(p)
            except OSError:
                pass


class PageEngine:
    """加载一次模型，逐页处理。命令行与 worker 共用"""

    def __init__(self):
        _ensure_engine_path()
        from manga_translator.args import parser
        from manga_translator.manga_translator import set_main_logger
        from manga_translator.utils import init_logging, set_log_level, get_logger
        init_logging()
        # 引擎每个步骤、OCR 每一行都打 INFO 日志，几个 GPU 进程一起跑时刷屏且拖慢界面，只留警告和错误；
        # 本工具自己的排版信息（清理残留、合并碎片、换方向等）走单独的 logger，照常显示
        set_log_level(logging.WARNING)
        set_main_logger(get_logger('workflow'))
        _patch_engine()
        args = parser.parse_args(['--use-gpu', 'local', '-i', '.'])
        self.t = make_translator_class()(vars(args))
        self.t.log = get_logger('typeset')
        self.t.log.setLevel(logging.INFO)

    @staticmethod
    def export_config():
        from manga_translator.config import Inpainter, Translator
        return load_config({'translator': {'translator': Translator.original.value},
                            'inpainter': {'inpainter': Inpainter.none.value}})

    @staticmethod
    def import_config():
        from manga_translator.config import Translator
        return load_config({'translator': {'translator': Translator.original.value,
                                           'enable_post_translation_check': False}})

    async def export_page(self, image_path, config=None, cache_path=None):
        """返回 (原文列表, 原文字号列表)；给了 cache_path 时顺便保存识别缓存"""
        t = self.t
        t.mode = 'export'
        t.current_page = image_path
        t.exported[image_path] = []
        t.exported_sizes[image_path] = []
        t.cached = None
        await t.translate(open_image(image_path), config or self.export_config(), skip_context_save=True)
        texts, sizes = t.exported.pop(image_path), t.exported_sizes.pop(image_path)
        captured = t.export_capture.pop(image_path, None)
        if cache_path:
            try:
                regions, mask_raw = captured if captured else ([], None)
                save_ocr_cache(cache_path, image_path, texts, regions, mask_raw)
            except Exception as e:  # 缓存失败不影响导出
                print(f'识别缓存保存失败（不影响导出）：{e}', flush=True)
        return texts, sizes

    async def import_page(self, image_path, output_path, src, trans, config=None, base_size=None, cache_path=None):
        """trans: {编号(int): 译文}；base_size: 全书基准字号。返回本页警告"""
        t = self.t
        t.mode = 'import'
        t.current_page = image_path
        t.source_texts = {image_path: src}
        t.translations = {image_path: {int(k): v for k, v in trans.items()}}
        t.warnings = []
        t.typeset_settings = typeset_settings()
        t.base_size = base_size
        config = config or self.import_config()
        t.cached = load_ocr_cache(cache_path, image_path, src)
        t.import_capture = None
        print('使用识别缓存，跳过检测和 OCR' if t.cached is not None else '没有可用的识别缓存，重新检测和 OCR', flush=True)
        try:
            ctx = await t.translate(open_image(image_path), config, skip_context_save=True)
        finally:
            t.cached = None
        captured, t.import_capture = t.import_capture, None
        # 识别结果与导出稿一致时补存缓存，下次重新生成可跳过检测和 OCR（旧工程导出时没有缓存）
        if cache_path and captured and captured[0] == list(src):
            try:
                save_ocr_cache(cache_path, image_path, *captured)
            except Exception as e:  # 缓存失败不影响生成
                print(f'识别缓存保存失败（不影响生成）：{e}', flush=True)
        if ctx.result is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_page(ctx.result, output_path)
        return list(t.warnings)

    async def auto_page(self, image_path, output_path, config):
        t = self.t
        t.mode = 'auto'
        ctx = await t.translate(open_image(image_path), config)
        if ctx.result is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_page(ctx.result, output_path)


# ================================================================ worker（GUI 并行用）

WORKER_RECYCLE_PAGES = 150     # 每个 GPU 进程处理多少页后换新进程（config.json 的 worker_recycle_pages 可改，0 = 不换）
WORKER_MAX_MEMORY_GB = 5.0     # 进程内存超过这么多也立即换新（config.json 的 worker_max_memory_gb 可改，0 = 不限）
RECYCLE_LINE = '@@RECYCLE'


def cmd_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    engine = PageEngine()
    threads = int(os.environ.get('WORKER_THREADS') or 0)
    if threads:     # 界面按进程数分配的线程上限（OMP_NUM_THREADS 已由环境变量传入）
        import cv2
        import torch
        torch.set_num_threads(threads)
        cv2.setNumThreads(threads)
    # 引擎的原生代码（检测 / OCR / 擦字）每页会泄漏约 18MB 内存且不会回落，
    # 处理够一定页数后主动退出，由界面换新进程接手，内存随进程整体释放
    # 几个进程一起涨，内存提交量耗尽后整机会卡死（系统忙着换页、压缩内存），直到关掉程序才恢复
    recycle = int(_config().get('worker_recycle_pages', WORKER_RECYCLE_PAGES) or 0)
    mem_limit = float(_config().get('worker_max_memory_gb', WORKER_MAX_MEMORY_GB) or 0) * 2 ** 30
    try:
        import psutil
        me = psutil.Process()
    except ImportError:
        me = None
    done = 0
    print('@@READY', flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        done += 1
        res = {'id': job.get('id'), 'ok': True}
        try:
            if job['mode'] == 'export':
                res['texts'], res['sizes'] = loop.run_until_complete(
                    engine.export_page(job['input'], cache_path=job.get('cache')))
            elif job['mode'] == 'import':
                res['warnings'] = loop.run_until_complete(
                    engine.import_page(job['input'], job['output'], job['src'], job['trans'],
                                       base_size=job.get('base_size'), cache_path=job.get('cache')))
            else:
                raise WorkflowError(f'未知任务类型: {job["mode"]}')
        except Exception as e:
            traceback.print_exc()
            res = {'id': job.get('id'), 'ok': False, 'error': f'{e.__class__.__name__}: {e}'}
        rss = me.memory_info().rss if me else 0
        last = (recycle and done >= recycle) or (mem_limit and rss > mem_limit)
        if last:    # 先告诉界面别再派新任务（界面收到结果前就能看到），结果照常返回后退出
            print(f'{RECYCLE_LINE} {done} 页 / 内存 {rss / 2 ** 30:.1f}GB', flush=True)
        print(RESULT_PREFIX + json.dumps(res, ensure_ascii=False), flush=True)
        if last:
            break


# ================================================================ 命令行（单进程）

async def cmd_export(src, yes=False):
    project_dir = project_dir_for(src)
    if os.path.exists(os.path.join(project_dir, PROJECT_FILE)) and not yes:
        ans = input(f'工程已存在：{project_dir}\n重新导出会覆盖「{os.path.basename(export_path_for(project_dir))}」，继续吗? [y/N] ')
        if ans.strip().lower() != 'y':
            raise WorkflowError('已取消')
    os.makedirs(project_dir, exist_ok=True)
    input_dir = prepare_input(src, project_dir)
    images = list_images(input_dir)
    if not images:
        raise WorkflowError(f'没有找到图片: {input_dir}')
    engine = PageEngine()
    config = engine.export_config()
    pages, sizes = {}, {}
    for n, rel in enumerate(images, 1):
        print(f'[{n}/{len(images)}] 识别 {rel}')
        pages[rel], sizes[rel] = await engine.export_page(os.path.join(input_dir, rel), config,
                                                          cache_path_for(project_dir, rel))
    path, n_pages, total = write_export(project_dir, src, input_dir, pages, sizes)
    print(f'\n导出完成：{n_pages} 页，{total} 句\n待翻译文本：{path}')


async def cmd_import(arg, trans_file=None, out_name=OUTPUT_DIR, pdf=True):
    project_dir, txt = locate_project(arg)
    if trans_file:
        txt = trans_file if os.path.isabs(trans_file) else os.path.join(project_dir, trans_file)
    txt = txt or find_translation(project_dir) or trans_path_for(project_dir)
    if not os.path.exists(txt):
        raise WorkflowError(f'找不到译文: {txt}')
    project = load_project(project_dir)
    input_dir, pages = project['input_dir'], project['pages']
    print(f'工程：{project_dir}\n译文：{txt}')
    with open(txt, 'r', encoding='utf-8-sig') as f:
        text = f.read()
    trans, errors = parse_translation(text, pages)
    errors += check_translation(trans, pages)
    for e in errors[:30]:
        print('  - ' + e)
    keep = trans_path_for(project_dir)
    if os.path.normcase(os.path.abspath(txt)) != os.path.normcase(keep):
        with open(keep, 'w', encoding='utf-8') as f:
            f.write(text)

    engine = PageEngine()
    config = engine.import_config()
    base = book_base_size(project)
    out_dir = os.path.join(project_dir, out_name)
    warnings = []
    for n, rel in enumerate(pages, 1):
        print(f'[{n}/{len(pages)}] 生成 {rel}')
        src_path = os.path.join(input_dir, rel)
        if not os.path.exists(src_path):
            warnings.append(f'{rel}: 原图不存在，跳过')
            continue
        warnings += await engine.import_page(src_path, os.path.join(out_dir, rel), pages[rel],
                                             trans.get(rel, {}), config, base,
                                             cache_path_for(project_dir, rel))
    for w in warnings:
        print('  警告: ' + w)
    print(f'\n图片输出：{out_dir}')
    if pdf:
        path = build_pdf(out_dir, pdf_path_for(project_dir), project_name(project_dir))
        if path:
            print(f'PDF 输出：{path}')


async def cmd_mt(arg, out_file='机翻译文.txt'):
    """用 config.json 中的翻译器把原文逐页机翻，写入独立的译文文件"""
    project_dir, _ = locate_project(arg)
    pages = load_project(project_dir)['pages']
    PageEngine()  # 初始化引擎路径与日志
    from manga_translator.translators import dispatch as dispatch_translation, prepare
    tc = load_config().translator
    print(f'机翻: {tc.translator.value} -> {tc.target_lang}')
    await prepare(tc.translator_gen)
    body = []
    for n, (rel, texts) in enumerate(pages.items(), 1):
        print(f'[{n}/{len(pages)}] {rel}  {len(texts)} 句')
        body.append(f'\n### {rel}')
        if texts:
            out = await dispatch_translation(tc.translator_gen, texts, tc, device='cuda')
            body += [f'<{i}>{norm(t)}' for i, t in enumerate(out, 1)]
    header = make_header(project_dir, len(pages), sum(len(v) for v in pages.values()), tc.target_lang)
    path = os.path.join(project_dir, out_file)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n'.join(body) + '\n')
    print(f'\n机翻完成: {path}')


async def cmd_auto(src):
    project_dir = project_dir_for(src)
    os.makedirs(project_dir, exist_ok=True)
    input_dir = prepare_input(src, project_dir)
    images = list_images(input_dir)
    if not images:
        raise WorkflowError(f'没有找到图片: {input_dir}')
    engine = PageEngine()
    config = load_config()
    print(f'翻译器: {config.translator.translator.value} -> {config.translator.target_lang}')
    out_dir = os.path.join(project_dir, OUTPUT_DIR)
    for n, rel in enumerate(images, 1):
        print(f'[{n}/{len(images)}] 翻译 {rel}')
        await engine.auto_page(os.path.join(input_dir, rel), os.path.join(out_dir, rel), config)
    name = project_name(project_dir)
    path = build_pdf(out_dir, os.path.join(project_dir, f'{name}_机翻.pdf'), name)
    print(f'\n图片输出：{out_dir}\nPDF 输出：{path}')


def main():
    ap = argparse.ArgumentParser(description='漫画翻译工作流')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('export')
    p.add_argument('input', help='图片文件夹或 PDF')
    p.add_argument('--yes', action='store_true', help='工程已存在时直接覆盖')
    sub.add_parser('auto').add_argument('input', help='图片文件夹或 PDF')
    p = sub.add_parser('import')
    p.add_argument('target', help='翻译好的 txt / 工程文件夹 / 原漫画文件夹或 PDF')
    p.add_argument('--trans', help='指定译文文件')
    p.add_argument('--out', default=OUTPUT_DIR, help='输出子目录名')
    p.add_argument('--no-pdf', action='store_true')
    p = sub.add_parser('mt')
    p.add_argument('project')
    p.add_argument('--out', default='机翻译文.txt')
    sub.add_parser('worker')
    a = ap.parse_args()

    try:
        if a.cmd == 'worker':
            cmd_worker()
        elif a.cmd == 'export':
            asyncio.run(cmd_export(a.input, a.yes))
        elif a.cmd == 'import':
            asyncio.run(cmd_import(a.target, a.trans, a.out, not a.no_pdf))
        elif a.cmd == 'mt':
            asyncio.run(cmd_mt(a.project, a.out))
        else:
            asyncio.run(cmd_auto(a.input))
    except WorkflowError as e:
        sys.exit(str(e))


if __name__ == '__main__':
    main()
