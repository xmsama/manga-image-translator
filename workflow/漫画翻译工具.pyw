"""
漫画翻译工具（图形界面）

流程：添加漫画（文件夹 / PDF）-> 导出文本 -> 把「待翻译.txt」交给译者 -> 导入译文 -> 自动擦字排版并输出 PDF

并行方式：GPU 上常驻 N 个 worker 进程（每个只加载一次模型），
所有漫画的每一页都拆成独立任务放进同一个队列，空闲的 worker 立即领取。
因此同一章的多张图片会同时处理，多本漫画也会交错并行。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import count

from PySide6.QtCore import QFile, QProcess, QProcessEnvironment, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QLineEdit, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import mit_workflow as wf  # noqa: E402  只用到其中不加载引擎的工具函数
import typeset  # noqa: E402
import credits  # noqa: E402

PYTHON = os.path.join(wf.REPO, 'venv', 'Scripts', 'python.exe')
if not os.path.exists(PYTHON):  # 没用 venv 时，用启动本程序的解释器（pythonw 换成 python，便于读取输出）
    PYTHON = os.path.join(os.path.dirname(sys.executable), 'python.exe')
STATE_PATH = os.path.join(ROOT, 'gui_books.json')

LANGS = [('CHS', '简体中文'), ('CHT', '繁体中文'), ('ENG', '英文'), ('JPN', '日文'), ('KOR', '韩文')]
ANSI = re.compile(r'\x1b\[[0-9;]*m')
COLS = ['漫画', '类型', '页数', '句数', '状态', '进度']

WORKER_VRAM_MB = 4000      # 实测每个 worker（识别 + lama_large 擦字）约 3.9GB
MAX_WORKERS = 32
RECOMMENDED_WORKERS = 4    # 实测 RTX 4090：1/2/3/4 个进程加速 1.0/2.1/2.6/2.8 倍，再多 GPU 算力已饱和

IDLE_RELEASE_SECONDS = 60  # 所有任务结束后空闲这么久，自动关闭 GPU 进程释放显存（config.json 的 idle_release_seconds 可改）

NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0


def open_path(path):
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def book_name(src):
    base = os.path.basename(src.rstrip('\\/'))
    return os.path.splitext(base)[0] if base.lower().endswith('.pdf') else base


def gpu_memory():
    """返回 (已用MB, 总MB)，失败返回 None"""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW).stdout
        used, total = [int(x) for x in out.strip().splitlines()[0].split(',')]
        return used, total
    except Exception:
        return None


# ================================================================ 数据

class Book:
    def __init__(self, source):
        self.source = os.path.abspath(source)
        self.project = wf.project_dir_for(self.source)
        try:
            wf.migrate_legacy(self.project)
        except OSError:
            pass
        self.name = book_name(self.source)
        self.job = None            # '导出' / '导入' / None
        self.stage = ''            # 准备中 / 处理中 / 收尾中
        self.done = 0
        self.total = 0
        self.failed = None
        self.cancelled = False
        self.results = {}          # 导出: rel -> [原文]
        self.sizes = {}            # 导出: rel -> [原文字号]
        self.page_errors = []
        self.ctx = {}              # 当前任务的上下文
        self.log = []
        self._info = None
        self._info_mtime = None

    @property
    def kind(self):
        return 'PDF' if self.source.lower().endswith('.pdf') else '文件夹'

    @property
    def pdf(self):
        return wf.pdf_path_for(self.project)

    @property
    def export_txt(self):
        return wf.export_path_for(self.project)

    def translation_file(self):
        return wf.find_translation(self.project)

    def pending_translation(self):
        """有译文文件且比 PDF 新：说明是新放进来、还没生成过的译文"""
        t = self.translation_file()
        if not t or not self.project_info():
            return None
        try:
            if os.path.exists(self.pdf) and os.path.getmtime(self.pdf) >= os.path.getmtime(t):
                return None
        except OSError:
            return None
        return t

    def project_info(self):
        """(页数, 句数)，带文件修改时间缓存"""
        p = os.path.join(self.project, wf.PROJECT_FILE)
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            return None
        if mtime != self._info_mtime:
            try:
                pages = wf.load_project(self.project)['pages']
                self._info = (len(pages), sum(len(v) for v in pages.values()))
            except Exception:
                self._info = None
            self._info_mtime = mtime
        return self._info

    def status(self):
        if self.job:
            return f'{self.job}·{self.stage}'
        if self.failed:
            return f'失败：{self.failed}'
        if self.pending_translation():
            return '译文待加载'
        if os.path.exists(self.pdf):
            return '已完成'
        if self.project_info():
            return '待翻译'
        return '未导出'

    def add_log(self, text):
        self.log.append(text)
        if len(self.log) > 5000:
            del self.log[:2000]


class ProjectRef:
    """不在列表里的工程（只知道工程目录），提供清理需要的属性"""

    def __init__(self, project):
        self.project = os.path.abspath(project)
        self.name = wf.project_name(self.project)
        self.pdf = wf.pdf_path_for(self.project)
        self.job = None


class Worker:
    def __init__(self, wid, proc):
        self.id = wid
        self.proc = proc
        self.ready = False
        self.job = None
        self.buf = b''
        self.retiring = False


# ================================================================ 主窗口

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('漫画翻译工具')
        self.resize(1150, 740)
        self.setAcceptDrops(True)
        self.books: list[Book] = []
        self.jobs = deque()        # 待分配的单页任务
        self.workers: list[Worker] = []
        self.job_ids = count(1)
        self.worker_ids = count(1)
        self.system_log = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.futures = []          # (future, 完成回调)
        self.gpu = None
        self.closing = False
        self.last_busy = time.time()
        self._build_ui()
        self._load_state()
        self.refresh()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(500)
        self._gpu_tick = 0

    # ------------------------------------------------------------ 界面
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        row1 = QHBoxLayout()
        for text, slot in [('添加文件夹', self.add_folder), ('添加 PDF', self.add_pdf),
                           ('导出文本', self.export_selected), ('复制待翻译文本', self.copy_export),
                           ('加载译文', self.load_translations), ('从剪贴板导入译文', self.import_clipboard),
                           ('选择译文文件导入', self.import_file), ('打开中转目录', self.open_transfer),
                           ('重新生成', self.regenerate), ('打开 PDF', self.open_pdf),
                           ('打开工程目录', self.open_project), ('清理工程…', self.cleanup_selected)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            row1.addWidget(b)
        row1.addStretch()
        root.addLayout(row1)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(COLS) - 1):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(len(COLS) - 1, QHeaderView.Fixed)
        self.table.setColumnWidth(len(COLS) - 1, 220)
        self.table.itemSelectionChanged.connect(self.show_log)
        self.table.doubleClicked.connect(lambda _: self.open_project())
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.context_menu)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.log_view)
        split.setSizes([460, 240])
        root.addWidget(split)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel('GPU 并行进程：'))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, MAX_WORKERS)
        self.workers_spin.setToolTip(f'同时处理的页数。每个进程约占 {WORKER_VRAM_MB / 1000:.1f}GB 显存，'
                                     '所有漫画的所有页面共用这些进程')
        self.workers_spin.valueChanged.connect(self.on_workers_changed)
        row2.addWidget(self.workers_spin)
        free_btn = QPushButton('释放显存')
        free_btn.setToolTip('立即关闭所有空闲的 GPU 进程；有新任务时会自动重新启动（加载约 15 秒）')
        free_btn.clicked.connect(lambda: self.release_workers('手动'))
        row2.addWidget(free_btn)
        auto = QPushButton('按显存自动')
        auto.clicked.connect(lambda: self.workers_spin.setValue(self.auto_workers()))
        row2.addWidget(auto)
        self.vram_hint = QLabel()
        row2.addWidget(self.vram_hint)
        row2.addSpacing(20)
        row2.addWidget(QLabel('译文语言：'))
        self.lang = QComboBox()
        for code, label in LANGS:
            self.lang.addItem(label, code)
        codes = [c for c, _ in LANGS]
        cur = wf.target_lang()
        self.lang.setCurrentIndex(codes.index(cur) if cur in codes else 0)
        self.lang.currentIndexChanged.connect(self._save_lang)
        row2.addWidget(self.lang)
        row2.addStretch()
        self.summary = QLabel()
        row2.addWidget(self.summary)
        root.addLayout(row2)
        self._build_typeset_row(root)
        self._build_output_row(root)

        self.setCentralWidget(central)
        self.statusBar().showMessage('把漫画文件夹 / PDF 拖进窗口即可添加；把翻译好的 txt 拖进窗口即可导入')

    def _build_typeset_row(self, root):
        ts = dict(typeset.DEFAULTS)
        ts.update(wf.read_config().get('typeset', {}))
        row = QHBoxLayout()
        row.addWidget(QLabel('排版：'))
        self.ts_renderer = QComboBox()
        self.ts_renderer.addItem('新嵌字器（按气泡排版）', 'typeset')
        self.ts_renderer.addItem('引擎原渲染', 'default')
        self.ts_renderer.setCurrentIndex(0 if ts.get('renderer', 'typeset') == 'typeset' else 1)
        row.addWidget(self.ts_renderer)
        row.addWidget(QLabel('字体：'))
        self.ts_font = QComboBox()
        for f in typeset.list_fonts():
            self.ts_font.addItem(os.path.splitext(f)[0], f)
        self.ts_font.setCurrentIndex(max(0, self.ts_font.findData(ts.get('font'))))
        row.addWidget(self.ts_font)
        row.addWidget(QLabel('字号：'))
        self.ts_scale = QDoubleSpinBox()
        self.ts_scale.setRange(0.5, 2.0)
        self.ts_scale.setSingleStep(0.05)
        self.ts_scale.setSuffix(' 倍')
        self.ts_scale.setValue(float(ts.get('size_scale', 1.0)))
        self.ts_scale.setToolTip('全书统一字号 = 原文字号中位数 × 倍率；放不下的气泡会自动缩小')
        row.addWidget(self.ts_scale)
        row.addWidget(QLabel('描边：'))
        self.ts_stroke = QComboBox()
        for label, val in (('自动（气泡外才描边）', 'auto'), ('总是描边', 'always'), ('不描边', 'never')):
            self.ts_stroke.addItem(label, val)
        self.ts_stroke.setCurrentIndex(max(0, self.ts_stroke.findData(ts.get('stroke', 'auto'))))
        row.addWidget(self.ts_stroke)
        row.addWidget(QLabel('方向：'))
        self.ts_dir = QComboBox()
        for label, val in (('自动（跟随原文，放不下换方向）', 'auto'), ('只横排', 'h'), ('只竖排', 'v')):
            self.ts_dir.addItem(label, val)
        self.ts_dir.setCurrentIndex(max(0, self.ts_dir.findData(ts.get('direction', 'auto'))))
        row.addWidget(self.ts_dir)
        cr = QPushButton('汉化组设置…')
        cr.setToolTip('在 PDF 首页/末页插入汉化组信息页、给页面打水印')
        cr.clicked.connect(self.credits_dialog)
        row.addWidget(cr)
        prev = QPushButton('预览排版')
        prev.setToolTip('用当前设置只生成选中漫画的一页，确认效果后再整本生成')
        prev.clicked.connect(self.preview)
        row.addWidget(prev)
        row.addStretch()
        root.addLayout(row)
        for wdg in (self.ts_renderer, self.ts_font, self.ts_stroke, self.ts_dir):
            wdg.currentIndexChanged.connect(self._save_typeset)
        self.ts_scale.valueChanged.connect(self._save_typeset)

    def _save_typeset(self):
        cfg = wf.read_config()
        ts = cfg.setdefault('typeset', {})
        ts.update({'renderer': self.ts_renderer.currentData(), 'font': self.ts_font.currentData(),
                   'size_scale': round(self.ts_scale.value(), 2), 'stroke': self.ts_stroke.currentData(),
                   'direction': self.ts_dir.currentData()})
        with open(wf.CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self.statusBar().showMessage('排版设置已保存，之后生成的页面生效（可先点「预览排版」看效果）', 5000)

    def preview(self):
        b = self.one_book()
        if not b:
            return
        tpath = b.translation_file()
        if not (b.project_info() and tpath):
            QMessageBox.information(self, '提示', '这本还没有导入过译文，无法预览')
            return
        project = wf.load_project(b.project)
        with open(tpath, 'r', encoding='utf-8-sig') as f:
            trans, _ = wf.parse_translation(f.read(), project['pages'])
        # 选译文句子最多的一页作为预览
        rel = max(project['pages'], key=lambda r: sum(1 for t in trans.get(r, {}).values() if t))
        out = os.path.join(b.project, '_预览', rel)
        self.jobs.appendleft({'id': next(self.job_ids), 'mode': 'import', 'book': b, 'rel': rel, 'preview': True,
                              'input': os.path.join(project['input_dir'], rel), 'output': out,
                              'src': project['pages'][rel],
                              'trans': {str(k): v for k, v in trans.get(rel, {}).items()},
                              'base_size': wf.book_base_size(project),
                              'cache': wf.cache_path_for(b.project, rel)})
        self.statusBar().showMessage(f'正在生成预览：{b.name} / {rel}', 8000)
        self.adjust_pool()

    def context_menu(self, pos):
        if not self.selected_books():
            return
        menu = QMenu(self)
        for text, slot in [('导出文本', self.export_selected), ('复制待翻译文本', self.copy_export),
                           ('打开待翻译 txt', self.open_export_txt), ('加载译文', self.load_translations),
                           ('从剪贴板导入译文', self.import_clipboard),
                           ('选择译文文件导入', self.import_file), ('重新生成', self.regenerate),
                           ('预览排版', self.preview), (None, None), ('打开 PDF', self.open_pdf), ('重新合成 PDF', self.rebuild_pdfs), ('导出成品 PDF', self.export_pdfs),
                           ('打开工程目录', self.open_project),
                           (None, None), ('取消任务', self.cancel_selected), ('清理工程…', self.cleanup_selected),
                           ('从列表移除', self.remove_selected)]:
            if text is None:
                menu.addSeparator()
                continue
            act = QAction(text, menu)
            act.triggered.connect(slot)
            menu.addAction(act)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _save_lang(self):
        cfg = wf.read_config()
        cfg.setdefault('translator', {})['target_lang'] = self.lang.currentData()
        with open(wf.CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self.statusBar().showMessage(f'译文语言已设为 {self.lang.currentText()}（影响之后导出的说明与排版方式）', 5000)

    # ------------------------------------------------------------ 状态保存
    def auto_workers(self):
        g = gpu_memory()
        if not g:
            return 2
        used, total = g
        mine = len(self.workers) * WORKER_VRAM_MB  # 本工具已占用的部分可重新分配
        free = total - used + mine - 800           # 留 800MB 余量
        return max(1, min(RECOMMENDED_WORKERS, free // WORKER_VRAM_MB))

    def _load_state(self):
        data = {}
        try:
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            pass
        for src in data.get('books', []):
            if os.path.exists(src):
                self.books.append(Book(src))
        self.workers_spin.blockSignals(True)
        self.workers_spin.setValue(int(data.get('workers') or self.auto_workers()))
        self.workers_spin.blockSignals(False)

    def _save_state(self):
        with open(STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump({'books': [b.source for b in self.books],
                       'workers': self.workers_spin.value()}, f, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------ 书目
    def add_books(self, paths):
        added = 0
        for p in paths:
            p = os.path.abspath(p)
            if p.endswith(wf.PROJECT_SUFFIX) and os.path.isfile(os.path.join(p, wf.PROJECT_FILE)):
                p = wf.load_project(p).get('source', p)
            if not (os.path.isdir(p) or p.lower().endswith('.pdf')):
                continue
            if any(os.path.normcase(b.source) == os.path.normcase(p) for b in self.books):
                continue
            self.books.append(Book(p))
            if os.path.isfile(os.path.join(self.books[-1].project, wf.PROJECT_FILE)):
                try:
                    wf.register_project(self.books[-1].project)
                except OSError:
                    pass
            added += 1
        if added:
            self._save_state()
            self.refresh()
        return added

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, '选择漫画图片文件夹')
        if d:
            self.add_books([d])

    def add_pdf(self):
        files, _ = QFileDialog.getOpenFileNames(self, '选择漫画 PDF', '', 'PDF (*.pdf)')
        self.add_books(files)

    def selected_books(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.books[r] for r in rows if r < len(self.books)]

    def one_book(self):
        books = self.selected_books()
        if len(books) != 1:
            QMessageBox.information(self, '提示', '请先选中一本漫画')
            return None
        return books[0]

    def book_for_project(self, project):
        key = os.path.normcase(os.path.abspath(project))
        for b in self.books:
            if os.path.normcase(b.project) == key:
                return b
        if os.path.isfile(os.path.join(project, wf.PROJECT_FILE)):
            src = wf.load_project(project).get('source')
            if src and os.path.exists(src) and self.add_books([src]):
                return self.book_for_project(project)
        return None

    def remove_selected(self):
        for b in self.selected_books():
            if not b.job:
                self.books.remove(b)
        self._save_state()
        self.refresh()

    # ------------------------------------------------------------ 后台线程（PDF 提取、写文件、合成 PDF）
    def run_bg(self, fn, done):
        self.futures.append((self.executor.submit(fn), done))

    def poll_futures(self):
        pending = []
        for fut, done in self.futures:
            if fut.done():
                try:
                    done(fut.result(), None)
                except Exception as e:  # noqa: BLE001
                    done(None, e)
            else:
                pending.append((fut, done))
        self.futures = pending

    # ------------------------------------------------------------ worker 进程池
    def on_workers_changed(self):
        self._save_state()
        self.update_vram_hint()
        self.adjust_pool()

    def update_vram_hint(self):
        n = self.workers_spin.value()
        need = n * WORKER_VRAM_MB / 1024
        g = self.gpu
        if g:
            used, total = g
            avail = (total - used) / 1024 + len(self.workers) * WORKER_VRAM_MB / 1024
            if need > avail:
                warn = '  ⚠ 可能超出显存，会明显变慢'
            elif n > RECOMMENDED_WORKERS:
                warn = '  （超过 4 个提速很小）'
            else:
                warn = ''
            self.vram_hint.setText(f'约需 {need:.1f}GB · 显存 {used / 1024:.1f}/{total / 1024:.1f}GB{warn}')
        else:
            self.vram_hint.setText(f'约需 {need:.1f}GB 显存')

    def adjust_pool(self):
        target = self.workers_spin.value()
        alive = [w for w in self.workers if not w.retiring]
        # 进程数超过设定：先关掉空闲的，忙碌的做完当前页再关
        for w in alive[target:]:
            if w.job is None:
                self.retire(w)
        alive = [w for w in self.workers if not w.retiring]
        # 有排队任务且进程不足：按需启动
        idle = sum(1 for w in alive if w.job is None)
        need = min(target - len(alive), len(self.jobs) - idle)
        for _ in range(max(0, need)):
            self.spawn_worker()
        self.dispatch()

    def spawn_worker(self):
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        for k, v in (('PYTHONUTF8', '1'), ('PYTHONIOENCODING', 'utf-8'), ('PYTHONUNBUFFERED', '1')):
            env.insert(k, v)
        # PyTorch / OpenCV / numpy 默认各开「逻辑核数」个线程，几个 GPU 进程同时跑会互相抢核，
        # CPU 占用飙高反而更慢（实测单进程限 4 线程：CPU 消耗少 28%，速度略快）
        threads = str(max(2, min(4, (os.cpu_count() or 8) // max(1, self.workers_spin.value()))))
        for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'WORKER_THREADS'):
            env.insert(k, threads)
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(ROOT)
        w = Worker(next(self.worker_ids), proc)
        proc.readyReadStandardOutput.connect(lambda w=w: self.on_worker_output(w))
        proc.finished.connect(lambda code, _s, w=w: self.on_worker_exit(w, code))
        self.workers.append(w)
        self.sys_log(f'启动 GPU 进程 #{w.id}（加载模型约 15 秒）')
        proc.start(PYTHON, ['-u', wf.__file__, 'worker'])

    def idle_seconds_setting(self):
        try:
            return float(wf.read_config().get('idle_release_seconds', IDLE_RELEASE_SECONDS))
        except Exception:
            return IDLE_RELEASE_SECONDS

    def release_workers(self, reason):
        idle = [w for w in self.workers if not w.retiring and w.job is None]
        for w in idle:
            self.retire(w)
        if idle:
            self.sys_log(f'{reason}释放显存：关闭 {len(idle)} 个空闲 GPU 进程')
            self.statusBar().showMessage(f'已关闭 {len(idle)} 个空闲 GPU 进程，显存已释放', 5000)

    def check_idle(self):
        busy = bool(self.jobs) or any(w.job for w in self.workers) or any(b.job for b in self.books)
        now = time.time()
        if busy:
            self.last_busy = now
            return
        alive = [w for w in self.workers if not w.retiring]
        if alive and now - self.last_busy >= self.idle_seconds_setting():
            self.release_workers(f'空闲 {int(now - self.last_busy)} 秒，自动')

    def retire(self, w):
        w.retiring = True
        w.proc.closeWriteChannel()   # worker 读到 EOF 后自行退出

    def dispatch(self):
        """把排队的单页任务分给空闲进程。多本书时轮流分配，保证每本书都在同时推进"""
        self.jobs = deque(j for j in self.jobs if not j['book'].cancelled)
        for w in self.workers:
            if not self.jobs:
                return
            if not (w.ready and w.job is None and not w.retiring):
                continue
            inflight = {}
            for x in self.workers:
                if x.job:
                    k = id(x.job['book'])
                    inflight[k] = inflight.get(k, 0) + 1
            best = min(range(len(self.jobs)),
                       key=lambda i: (inflight.get(id(self.jobs[i]['book']), 0), i))
            job = self.jobs[best]
            del self.jobs[best]
            w.job = job
            payload = {k: v for k, v in job.items() if k not in ('book', 'rel', 'tries', 'preview')}
            w.proc.write((json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8'))

    def on_worker_output(self, w):
        w.buf += bytes(w.proc.readAllStandardOutput())
        *lines, w.buf = w.buf.split(b'\n')
        for raw in lines:
            line = ANSI.sub('', raw.decode('utf-8', errors='replace')).rstrip()
            if not line:
                continue
            if line.startswith('@@READY'):
                w.ready = True
                self.sys_log(f'GPU 进程 #{w.id} 就绪')
                self.dispatch()
            elif line.startswith(wf.RECYCLE_LINE):
                # 该进程做完当前页就退出（定期换新进程释放泄漏的内存）：不再派任务，提前启动接替的进程
                w.retiring = True
                info = line[len(wf.RECYCLE_LINE):].strip()
                self.sys_log(f'GPU 进程 #{w.id}（已处理 {info}）做完当前页后换新进程，释放内存')
                self.adjust_pool()
            elif line.startswith(wf.RESULT_PREFIX):
                job, w.job = w.job, None
                if job:
                    self.on_page_result(job, json.loads(line[len(wf.RESULT_PREFIX):]))
                if len([x for x in self.workers if not x.retiring]) > self.workers_spin.value():
                    self.retire(w)
                self.dispatch()
            else:
                target = w.job['book'] if w.job else None
                if target:
                    tagged = f'[#{w.id} {w.job["rel"]}] {line}'
                    target.add_log(tagged)
                    if target in self.selected_books():
                        self.log_view.appendPlainText(tagged)
                else:
                    self.sys_log(f'[#{w.id}] {line}', echo=False)

    def on_worker_exit(self, w, code):
        if self.closing:
            return
        if w in self.workers:
            self.workers.remove(w)
        job = w.job
        w.proc.deleteLater()
        if job:  # 进程异常退出：该页重试一次
            job['tries'] = job.get('tries', 0) + 1
            if job['tries'] <= 1 and not job['book'].cancelled:
                self.sys_log(f'GPU 进程 #{w.id} 意外退出（{code}），重试 {job["rel"]}')
                self.jobs.appendleft(job)
            else:
                self.on_page_result(job, {'ok': False, 'error': f'进程退出码 {code}'})
        elif not w.retiring:
            self.sys_log(f'GPU 进程 #{w.id} 退出（{code}）')
        self.adjust_pool()

    def sys_log(self, text, echo=True):
        self.system_log.append(text)
        if len(self.system_log) > 3000:
            del self.system_log[:1000]
        if echo and not self.selected_books():
            self.log_view.appendPlainText(text)

    # ------------------------------------------------------------ 单页结果汇总
    def enqueue_pages(self, book, jobs):
        book.total, book.done = len(jobs), 0
        book.stage = '处理中'
        self.jobs.extend(jobs)
        self.adjust_pool()

    def on_page_result(self, job, res):
        book = job['book']
        if job.get('preview'):
            if res.get('ok') and os.path.exists(job['output']):
                open_path(job['output'])
                self.statusBar().showMessage(f'预览已生成：{job["rel"]}', 6000)
            else:
                QMessageBox.warning(self, '预览失败', str(res.get('error')))
            return
        if book.cancelled or book.job is None:
            return
        rel = job['rel']
        book.done += 1
        if res.get('ok'):
            if job['mode'] == 'export':
                book.results[rel] = res.get('texts', [])
                book.sizes[rel] = res.get('sizes', [])
            book.page_errors.extend(res.get('warnings') or [])
        else:
            book.page_errors.append(f'{rel}: {res.get("error")}')
            book.add_log(f'!! {rel} 处理失败: {res.get("error")}')
            if job['mode'] == 'export':
                book.results[rel] = []
            else:  # 生成失败的页面用原图占位，保证 PDF 页数完整
                try:
                    os.makedirs(os.path.dirname(job['output']), exist_ok=True)
                    shutil.copyfile(job['input'], job['output'])
                except OSError:
                    pass
        if book.done >= book.total:
            self.finish_book(book)

    # ------------------------------------------------------------ 汉化组信息
    def rebuild_pdfs(self, books=None):
        """用已生成的页面图片重新合成 PDF（应用最新的汉化组设置），不需要 GPU"""
        books = [b for b in (books or self.selected_books() or self.books)
                 if not b.job and wf.list_images(os.path.join(b.project, wf.OUTPUT_DIR))]
        if not books:
            QMessageBox.information(self, '重新合成 PDF',
                                    '没有可以合成的漫画（需要已生成过页面；清理过中间文件的要先「重新生成」）')
            return
        names = [b.name for b in books]

        def work():
            for b in books:
                wf.build_pdf(os.path.join(b.project, wf.OUTPUT_DIR), b.pdf, b.name)

        def done(_, err):
            if err:
                QMessageBox.warning(self, '合成失败', str(err))
            else:
                self.statusBar().showMessage(f'已重新合成 {len(names)} 本 PDF：' + '、'.join(names[:5]), 8000)
                if self.auto_export.isChecked():
                    self.export_pdfs(books, quiet=True)
            self.refresh()

        self.statusBar().showMessage(f'正在重新合成 {len(names)} 本 PDF…', 8000)
        self.run_bg(work, done)

    def credits_dialog(self):
        cfg = wf.read_config()
        st = credits.settings(cfg)
        wm = st['watermark']
        dlg = QDialog(self)
        dlg.setWindowTitle('汉化组设置')
        dlg.resize(620, 560)
        v = QVBoxLayout(dlg)

        def combo(items, cur):
            c = QComboBox()
            for label, val in items:
                c.addItem(label, val)
            c.setCurrentIndex(max(0, c.findData(cur)))
            return c

        def path_row(value, title, filt):
            edit = QLineEdit(value)
            btn = QPushButton('选择…')
            btn.clicked.connect(lambda: (lambda p: edit.setText(p) if p else None)(
                QFileDialog.getOpenFileName(dlg, title, '', filt)[0]))
            h = QHBoxLayout()
            h.addWidget(edit, 1)
            h.addWidget(btn)
            return edit, h

        img_filter = '图片 (*.png *.jpg *.jpeg *.webp *.bmp)'
        g1 = QGroupBox('汉化组信息页')
        f1 = QFormLayout(g1)
        page_pos = combo([('不插入', 'none'), ('插在首页', 'first'), ('插在末页', 'last'), ('首页和末页', 'both')],
                         st['page_position'])
        f1.addRow('位置：', page_pos)
        page_img, row = path_row(st['page_image'], '选择信息页图片', img_filter)
        f1.addRow('自备图片：', row)
        f1.addRow('', QLabel('有图片就直接用图片（自动缩放到漫画页宽）；没有图片按下面文字自动生成'))
        group = QLineEdit(st['group'])
        group.setPlaceholderText('汉化组名称')
        f1.addRow('组名：', group)
        lines = QPlainTextEdit(st['lines'])
        lines.setPlaceholderText('每行一条，例如：\n翻译：xxx\n嵌字：xxx\n仅供学习交流，请支持正版')
        lines.setFixedHeight(90)
        f1.addRow('信息：', lines)
        v.addWidget(g1)

        g2 = QGroupBox('水印')
        f2 = QFormLayout(g2)
        mode = combo([('不加水印', 'none'), ('文字水印', 'text'), ('图片水印（logo）', 'image')], wm['mode'])
        f2.addRow('类型：', mode)
        wtext = QLineEdit(wm['text'])
        wtext.setPlaceholderText('例如：某某汉化组')
        f2.addRow('文字：', wtext)
        wimg, row = path_row(wm['image'], '选择水印图片', img_filter)
        f2.addRow('图片：', row)
        pages = combo([('全部页', 'all'), ('仅首页', 'first'), ('仅末页', 'last'), ('首页和末页', 'first_last')],
                      wm['pages'])
        f2.addRow('打在：', pages)
        pos = combo([('右下角', 'br'), ('左下角', 'bl'), ('右上角', 'tr'), ('左上角', 'tl'), ('居中', 'center')],
                    wm['position'])
        f2.addRow('位置：', pos)
        opacity = QDoubleSpinBox()
        opacity.setRange(0.05, 1.0)
        opacity.setSingleStep(0.05)
        opacity.setValue(float(wm['opacity']))
        f2.addRow('不透明度：', opacity)
        size = QDoubleSpinBox()
        size.setRange(0.02, 0.8)
        size.setSingleStep(0.01)
        size.setValue(float(wm['size']))
        size.setToolTip('图片水印：宽度占页宽的比例；文字水印：数值越大字越大')
        f2.addRow('大小：', size)
        v.addWidget(g2)

        def current():
            return {'page_position': page_pos.currentData(), 'page_image': page_img.text().strip(),
                    'group': group.text().strip(), 'lines': lines.toPlainText().strip(),
                    'watermark': {'mode': mode.currentData(), 'text': wtext.text().strip(),
                                  'image': wimg.text().strip(), 'pages': pages.currentData(),
                                  'position': pos.currentData(), 'opacity': round(opacity.value(), 2),
                                  'size': round(size.value(), 3)}}

        def preview():
            s = credits.settings({'credits': current()})
            sample = None
            sel = self.selected_books()
            for b in sel[:1]:
                for d in (os.path.join(b.project, wf.OUTPUT_DIR),):
                    imgs = wf.list_images(d) if os.path.isdir(d) else []
                    if imgs:
                        sample = os.path.join(d, imgs[0])
            out_dir = os.path.join(wf.ROOT, '_预览')
            os.makedirs(out_dir, exist_ok=True)
            from PIL import Image
            page = Image.open(sample) if sample else Image.new('RGB', (1184, 1682), (235, 235, 235))
            shown = []
            if s['watermark']['mode'] != 'none':
                p = os.path.join(out_dir, '水印预览.jpg')
                credits.watermark(page, s['watermark']).save(p, quality=92)
                shown.append(p)
            if s['page_position'] != 'none':
                p = os.path.join(out_dir, '信息页预览.png')
                credits.make_credit_page(page.size, s).save(p)
                shown.append(p)
            if not shown:
                QMessageBox.information(dlg, '预览', '信息页和水印都没有启用')
            for p in shown:
                open_path(p)

        tip = QLabel('信息页和水印只加在 PDF 里，不改动生成好的页面图片。'
                     '保存后，已完成的漫画用右键「重新合成 PDF」即可应用（几秒，不需要重新生成）。')
        tip.setWordWrap(True)
        v.addWidget(tip)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Save).setText('保存')
        bb.button(QDialogButtonBox.Cancel).setText('取消')
        pv = bb.addButton('预览', QDialogButtonBox.ActionRole)
        pv.setToolTip('选中列表里一本已生成的漫画时，用它的第一页预览；否则用空白页')
        pv.clicked.connect(preview)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        cfg = wf.read_config()
        cfg['credits'] = current()
        with open(wf.CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        done = [b for b in self.books if not b.job and os.path.exists(b.pdf)]
        if done and QMessageBox.question(self, '汉化组设置已保存',
                                         f'之后合成的 PDF 都会使用新设置。\n\n'
                                         f'要现在把已完成的 {len(done)} 本重新合成 PDF 吗？') == QMessageBox.Yes:
            self.rebuild_pdfs(done)

    def finish_book(self, book):
        book.stage = '收尾中'
        if book.job == '导出':
            c = book.ctx
            self.run_bg(lambda: wf.write_export(book.project, book.source, c['input_dir'], book.results, book.sizes),
                        lambda r, e: self.end_job(book, e, f'导出完成：{r[1]} 页，{r[2]} 句' if r else ''))
        else:
            out_dir = os.path.join(book.project, wf.OUTPUT_DIR)
            self.run_bg(lambda: wf.build_pdf(out_dir, book.pdf, book.name),
                        lambda r, e: self.end_job(book, e, f'PDF 已生成：{r}' if r else ''))

    def end_job(self, book, err, msg):
        job, book.job = book.job, None
        if err:
            book.failed = str(err)[:60]
            book.add_log(f'===== {job}失败：{err} =====')
        else:
            if book.page_errors:
                book.add_log(f'共 {len(book.page_errors)} 条警告：')
                for e in book.page_errors:
                    book.add_log('  - ' + e)
            book.add_log(f'===== {msg} =====')
            self.statusBar().showMessage(f'「{book.name}」{msg}', 8000)
            if job == '导入' and self.auto_export.isChecked():
                self.export_pdfs([book], quiet=True)
        if book in self.selected_books():
            self.show_log()
        self.refresh()

    # ------------------------------------------------------------ 导出 / 导入
    def begin(self, book, job):
        if book.job:
            self.statusBar().showMessage(f'「{book.name}」已有任务在进行', 4000)
            return False
        book.job, book.stage = job, '准备中'
        book.failed, book.cancelled = None, False
        book.results, book.sizes, book.page_errors, book.ctx = {}, {}, [], {}
        book.done = book.total = 0
        book.add_log(f'===== 开始{job} =====')
        self.refresh()
        return True

    def export_selected(self):
        books = self.selected_books() or [b for b in self.books if b.status() == '未导出']
        if not books:
            QMessageBox.information(self, '提示', '选中要导出的漫画（可多选）')
            return
        redo = [b for b in books if b.project_info() and not b.job]
        if redo:
            names = '\n'.join(b.name for b in redo)
            if QMessageBox.question(self, '重新导出', f'以下漫画已导出过，重新导出会覆盖待翻译文本：\n{names}\n继续吗？') \
                    != QMessageBox.Yes:
                books = [b for b in books if b not in redo]
        for b in books:
            self.start_export(b)

    def start_export(self, book):
        if not self.begin(book, '导出'):
            return

        def prepare():
            os.makedirs(book.project, exist_ok=True)
            input_dir = wf.prepare_input(book.source, book.project)
            return input_dir, wf.list_images(input_dir)

        def ready(r, err):
            if err or not r or not r[1]:
                return self.end_job(book, err or wf.WorkflowError('没有找到图片'), '')
            input_dir, images = r
            book.ctx['input_dir'] = input_dir
            book.add_log(f'共 {len(images)} 页，分发到 GPU 进程并行识别')
            self.enqueue_pages(book, [{'id': next(self.job_ids), 'mode': 'export', 'book': book, 'rel': rel,
                                       'input': os.path.join(input_dir, rel),
                                       'cache': wf.cache_path_for(book.project, rel)} for rel in images])

        self.run_bg(prepare, ready)

    def start_import(self, book, text, save=True):
        try:
            project = wf.load_project(book.project)
            pages, input_dir = project['pages'], project['input_dir']
            trans, errors = wf.parse_translation(text, pages)
            errors += wf.check_translation(trans, pages)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, '无法导入', str(e))
            return
        if not self.begin(book, '导入'):
            return
        if save:
            wf.ensure_transfer_dirs()
            with open(wf.trans_path_for(book.project), 'w', encoding='utf-8') as f:
                f.write(text if text.endswith('\n') else text + '\n')
        book.page_errors.extend(errors)
        out_dir = os.path.join(book.project, wf.OUTPUT_DIR)
        base = wf.book_base_size(project)
        jobs = [{'id': next(self.job_ids), 'mode': 'import', 'book': book, 'rel': rel,
                 'input': os.path.join(input_dir, rel), 'output': os.path.join(out_dir, rel),
                 'src': src, 'trans': {str(k): v for k, v in trans.get(rel, {}).items()},
                 'base_size': base, 'cache': wf.cache_path_for(book.project, rel)}
                for rel, src in pages.items()]
        book.add_log(f'共 {len(jobs)} 页，分发到 GPU 进程并行生成')
        self.enqueue_pages(book, jobs)
        self.statusBar().showMessage(f'已收到「{book.name}」的译文，开始生成', 6000)

    def import_text(self, text, source_desc):
        """导入一段译文：优先按「# 工程:」行定位，否则用当前选中的书"""
        if not re.search(r'^\s*[<＜]\d+[>＞]', text, re.M):
            QMessageBox.warning(self, '无法导入', f'{source_desc}里没有找到 <编号> 格式的译文')
            return
        project = wf.project_from_text(text)
        book = self.book_for_project(project) if project else None
        if not book:
            sel = self.selected_books()
            if len(sel) != 1:
                QMessageBox.warning(self, '无法导入', '译文里没有「# 工程:」行，请先选中对应的漫画再导入')
                return
            book = sel[0]
        if not book.project_info():
            QMessageBox.warning(self, '无法导入', f'「{book.name}」还没导出过文本')
            return
        self.start_import(book, text)

    def load_translations(self):
        """扫描中转目录/译文，列出新放进来的译文，勾选后逐本加载生成"""
        found, problems = [], []
        for path, owner in wf.scan_translations():
            name = os.path.basename(path)
            if not owner:
                problems.append(f'{name}：认不出是哪本漫画（文件头缺少「# 工程:」且没有同名的待翻译文件）')
                continue
            book = self.book_for_project(owner)
            if not book:
                problems.append(f'{name}：对应的漫画原文件已不存在')
                continue
            if book.job or book.pending_translation() != path:
                continue        # 正在处理，或这份译文已经生成过
            with open(path, 'r', encoding='utf-8-sig') as f:
                text = f.read()
            if not re.search(r'^\s*[<＜]\d+[>＞]', text, re.M):
                problems.append(f'{name}：没有 <编号> 格式的译文')
                continue
            found.append((book, path, text))
        _, done = wf.transfer_dirs()
        if not found:
            if problems:
                QMessageBox.warning(self, '译文未加载', '\n'.join(problems))
            else:
                QMessageBox.information(self, '加载译文', f'「{done}」里没有新的译文。\n'
                                        '让 AI 把「待翻译」文件夹里的 txt 翻译后，以同名文件写入「译文」文件夹，再点「加载译文」。')
            return
        picked = self.pick_translations(found, problems)
        started = []
        for book, _, text in picked:
            self.start_import(book, text, save=False)
            if book.job:
                started.append(book.name)
        self.statusBar().showMessage(
            f'开始加载 {len(started)} 本：' + '、'.join(started) if started else '没有加载任何译文', 8000)

    def pick_translations(self, found, problems):
        """列出发现的新译文，勾选要生成的；取消返回空列表"""
        dlg = QDialog(self)
        dlg.setWindowTitle('加载译文')
        dlg.resize(900, 460)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f'发现 {len(found)} 份新译文，勾选这次要生成的：'))
        table = QTableWidget(len(found), 3)
        table.setHorizontalHeaderLabels(['漫画', '译文文件', '修改时间'])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        h = table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        for r, (book, path, _) in enumerate(found):
            name = QTableWidgetItem(book.name)
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Checked)
            table.setItem(r, 0, name)
            table.setItem(r, 1, QTableWidgetItem(os.path.basename(path)))
            table.setItem(r, 2, QTableWidgetItem(time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(path)))))
        # 点整行任意位置都能切换勾选
        table.cellClicked.connect(lambda r, c: c and table.item(r, 0).setCheckState(
            Qt.Unchecked if table.item(r, 0).checkState() == Qt.Checked else Qt.Checked))
        v.addWidget(table)
        if problems:
            warn = QLabel('以下译文无法加载：\n' + '\n'.join(problems))
            warn.setWordWrap(True)
            warn.setStyleSheet('color: #c0392b')
            v.addWidget(warn)

        def check_all(state):
            for r in range(table.rowCount()):
                table.item(r, 0).setCheckState(state)

        row = QHBoxLayout()
        for text, slot in (('全选', lambda: check_all(Qt.Checked)), ('全不选', lambda: check_all(Qt.Unchecked))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('开始生成')
        bb.button(QDialogButtonBox.Cancel).setText('取消')
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return []
        return [found[r] for r in range(table.rowCount()) if table.item(r, 0).checkState() == Qt.Checked]

    def open_transfer(self):
        todo, _ = wf.ensure_transfer_dirs()
        open_path(os.path.dirname(todo))

    def import_clipboard(self):
        text = QGuiApplication.clipboard().text()
        if not text.strip():
            QMessageBox.information(self, '提示', '剪贴板是空的')
            return
        self.import_text(text, '剪贴板')

    def import_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, '选择翻译好的 txt', '', '文本 (*.txt)')
        for p in files:
            self.import_txt_file(p)

    def import_txt_file(self, path):
        with open(path, 'r', encoding='utf-8-sig') as f:
            self.import_text(f.read(), os.path.basename(path))

    def regenerate(self):
        for b in self.selected_books():
            p = b.translation_file()
            if p:
                with open(p, 'r', encoding='utf-8-sig') as f:
                    self.start_import(b, f.read(), save=False)
            else:
                self.statusBar().showMessage(f'「{b.name}」还没有导入过译文', 5000)

    def cancel_selected(self):
        for b in self.selected_books():
            if not b.job:
                continue
            b.cancelled = True
            self.jobs = deque(j for j in self.jobs if j['book'] is not b)
            b.add_log(f'===== {b.job}已取消 =====')
            b.job, b.failed = None, '已取消'
        self.refresh()

    # ------------------------------------------------------------ 其他操作
    def copy_export(self):
        b = self.one_book()
        if not b:
            return
        if not os.path.exists(b.export_txt):
            QMessageBox.information(self, '提示', '这本还没导出文本')
            return
        with open(b.export_txt, 'r', encoding='utf-8') as f:
            QGuiApplication.clipboard().setText(f.read())
        self.statusBar().showMessage(f'已复制「{b.name}」的待翻译文本，粘贴给译者即可', 6000)

    def open_export_txt(self):
        b = self.one_book()
        if b and os.path.exists(b.export_txt):
            open_path(b.export_txt)

    # ------------------------------------------------------------ 清理工程
    @staticmethod
    def _size(path):
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    @staticmethod
    def _fmt(n):
        return f'{n / 1073741824:.2f} GB' if n >= 1073741824 else f'{n / 1048576:.1f} MB'

    def _cleanup_plan(self, b):
        """返回 (中间文件, 整个工程要删的路径, 工程里非本工具生成的文件名)"""
        light = [p for p in (os.path.join(b.project, wf.OUTPUT_DIR), os.path.join(b.project, '_预览'))
                 if os.path.exists(p)]
        if os.path.isdir(b.project):
            light += [os.path.join(r, f) for r, _, fs in os.walk(b.project) for f in fs if f.endswith('.part')]
        full = [b.project] if os.path.isdir(b.project) else []
        txts = set()
        try:
            txts = {wf.export_path_for(b.project), wf.trans_path_for(b.project)}
            _, done = wf.transfer_dirs()
            me = os.path.normcase(os.path.abspath(b.project))
            if os.path.isdir(done):
                txts |= {os.path.join(done, f) for f in os.listdir(done)
                         if f.lower().endswith('.txt') and wf._owner_of(os.path.join(done, f)) == me}
        except Exception:
            pass
        full += sorted(p for p in txts if os.path.isfile(p))
        known = {wf.PROJECT_FILE, wf.PAGES_DIR, wf.OUTPUT_DIR, '_预览', wf.TRANS_FILE, wf.EXPORT_FILE, wf.CACHE_DIR,
                 os.path.basename(b.pdf), f'{wf.project_name(b.project)}_译文.txt',
                 f'{wf.project_name(b.project)}_待翻译.txt'}
        extra = sorted(f for f in os.listdir(b.project) if f not in known) if os.path.isdir(b.project) else []
        return light, full, extra

    def pick_orphan_projects(self):
        """列出不在列表里的工程（含已从列表移除的），勾选后返回。取消返回空列表"""
        in_list = {os.path.normcase(b.project) for b in self.books}
        projects = [p for p in wf.known_projects() if os.path.normcase(p) not in in_list]

        dlg = QDialog(self)
        dlg.setWindowTitle('清理不在列表中的工程')
        dlg.resize(900, 460)
        v = QVBoxLayout(dlg)
        tip = QLabel('没有选中列表里的漫画时，这里列出已从列表移除（或从未加入）的工程。勾选要清理的：')
        v.addWidget(tip)
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(['工程', '大小', 'PDF', '位置'])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        h = table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        v.addWidget(table)
        shown = []

        def add_rows(paths):
            for p in paths:
                if os.path.normcase(p) in in_list or any(os.path.normcase(p) == os.path.normcase(x) for x in shown):
                    continue
                ref = ProjectRef(p)
                shown.append(p)
                r = table.rowCount()
                table.insertRow(r)
                name = QTableWidgetItem(ref.name)
                name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
                name.setCheckState(Qt.Unchecked)
                name.setData(Qt.UserRole, p)
                table.setItem(r, 0, name)
                table.setItem(r, 1, QTableWidgetItem(self._fmt(self._size(p))))
                pdf = '已导出' if self._exported(ref) else ('有' if os.path.exists(ref.pdf) else '无')
                table.setItem(r, 2, QTableWidgetItem(pdf))
                table.setItem(r, 3, QTableWidgetItem(os.path.dirname(p)))
            tip.setText(f'找到 {table.rowCount()} 个不在列表中的工程，勾选要清理的：' if table.rowCount()
                        else '没有找到不在列表中的工程。可以点「扫描文件夹…」到漫画所在的目录里找。')

        def scan():
            d = QFileDialog.getExistingDirectory(dlg, '选择要扫描的文件夹（会查找里面所有「_翻译工程」）')
            if d:
                found = wf.find_projects_under(d)
                for p in found:
                    try:
                        wf.register_project(p)
                    except OSError:
                        pass
                add_rows(found)

        def check_all(state):
            for r in range(table.rowCount()):
                table.item(r, 0).setCheckState(state)

        add_rows(projects)
        row = QHBoxLayout()
        for text, slot in (('扫描文件夹…', scan), ('全选', lambda: check_all(Qt.Checked)),
                           ('全不选', lambda: check_all(Qt.Unchecked))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('下一步')
        bb.button(QDialogButtonBox.Cancel).setText('取消')
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return []
        picked = [table.item(r, 0).data(Qt.UserRole) for r in range(table.rowCount())
                  if table.item(r, 0).checkState() == Qt.Checked]
        if not picked:
            QMessageBox.information(self, '清理工程', '没有勾选任何工程')
        return [ProjectRef(p) for p in picked]

    def _exported(self, b):
        out = self.output_dir()
        if not out or not os.path.exists(b.pdf):
            return False
        dst = os.path.join(out, self.pdf_out_name(b))
        return os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(b.pdf)

    def cleanup_selected(self):
        books = [b for b in self.selected_books() if not b.job]
        if not books:
            books = self.pick_orphan_projects()
            if not books:
                return
        plans = {b: self._cleanup_plan(b) for b in books}
        light_size = sum(self._size(p) for l, _, _ in plans.values() for p in l)
        full_size = sum(self._size(p) for _, f, _ in plans.values() for p in f)

        dlg = QDialog(self)
        dlg.setWindowTitle(f'清理工程（{len(books)} 本）')
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel('、'.join(b.name for b in books[:6]) + (' 等' if len(books) > 6 else '')))
        r1 = QRadioButton(f'清理中间文件（释放 {self._fmt(light_size)}）\n'
                          '删除生成过程的页面图片和预览；保留原图、译文和 PDF，之后仍可「重新生成」')
        r2 = QRadioButton(f'删除整个工程（释放 {self._fmt(full_size)}）\n'
                          '工程文件夹和中转目录里这本的 txt 一起删除，并从列表移除；原漫画文件不动')
        r1.setChecked(True)
        v.addWidget(r1)
        v.addWidget(r2)
        warn = []
        no_pdf = [b.name for b in books if not os.path.exists(b.pdf)]
        not_exported = [b.name for b in books if os.path.exists(b.pdf) and not self._exported(b)]
        extras = [f'{b.name}：' + '、'.join(e[:5]) + (' …' if len(e) > 5 else '')
                  for b, (_, _, e) in plans.items() if e]
        if not_exported:
            warn.append('PDF 还没导出到成品目录（删除整个工程会连 PDF 一起删）：\n  ' + '\n  '.join(not_exported))
        if no_pdf:
            warn.append('还没生成 PDF：\n  ' + '\n  '.join(no_pdf))
        if extras:
            warn.append('工程里有不是本工具生成的文件（删除整个工程时一起删）：\n  ' + '\n  '.join(extras))
        if warn:
            lab = QLabel('\n\n'.join(warn))
            lab.setWordWrap(True)
            lab.setStyleSheet('color:#b3261e')
            v.addWidget(lab)
        v.addWidget(QLabel('删除的文件都会移到回收站，可以还原。'))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('清理')
        bb.button(QDialogButtonBox.Cancel).setText('取消')
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        full = r2.isChecked()
        targets = [p for _, (l, f, _) in plans.items() for p in (f if full else l)]
        freed = full_size if full else light_size

        def work():
            failed = []
            for p in targets:
                if not os.path.exists(p):
                    continue
                r = QFile.moveToTrash(p)
                ok = r[0] if isinstance(r, tuple) else bool(r)
                if not ok:
                    failed.append(p)
            return failed

        def done(failed, err):
            if full:
                for b in books:
                    if b in self.books and not os.path.exists(b.project):
                        self.books.remove(b)
                self._save_state()
            failed = failed or []
            msg = f'已清理 {len(books)} 本（{"删除整个工程" if full else "中间文件"}），释放约 {self._fmt(freed)}'
            if err or failed:
                QMessageBox.warning(self, '部分未清理', str(err) if err else
                                    '以下文件无法移到回收站（可能正在被占用）：\n' + '\n'.join(failed[:20]))
            self.sys_log(msg)
            self.statusBar().showMessage(msg, 8000)
            self.refresh()

        self.run_bg(work, done)

    # ------------------------------------------------------------ 成品导出
    def _build_output_row(self, root):
        cfg = wf.read_config()
        row = QHBoxLayout()
        row.addWidget(QLabel('成品目录：'))
        self.out_label = QLabel()
        self.out_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.out_label, 1)
        for text, slot in (('更改…', self.choose_output_dir), ('打开', self.open_output_dir)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        exp = QPushButton('一键导出成品 PDF')
        exp.setToolTip('把选中的（未选中则全部）已完成漫画的 PDF 复制到成品目录；已是最新的会跳过')
        exp.clicked.connect(lambda: self.export_pdfs())
        row.addWidget(exp)
        self.auto_export = QCheckBox('生成完成后自动导出')
        self.auto_export.setChecked(bool(cfg.get('auto_export_pdf', False)))
        self.auto_export.toggled.connect(self._save_output_settings)
        row.addWidget(self.auto_export)
        root.addLayout(row)
        self._update_out_label()

    def output_dir(self):
        d = wf.read_config().get('output_dir') or ''
        return d if d and os.path.isdir(d) else ''

    def _update_out_label(self):
        d = self.output_dir()
        self.out_label.setText(d or '（未设置，第一次导出时选择）')

    def _save_output_settings(self, output_dir=None):
        cfg = wf.read_config()
        if isinstance(output_dir, str):
            cfg['output_dir'] = output_dir
        cfg['auto_export_pdf'] = self.auto_export.isChecked()
        with open(wf.CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self._update_out_label()

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, '选择成品 PDF 导出目录', self.output_dir())
        if d:
            self._save_output_settings(os.path.abspath(d))
        return d

    def open_output_dir(self):
        d = self.output_dir() or self.choose_output_dir()
        if d:
            open_path(d)

    @staticmethod
    def pdf_out_name(book):
        """成品文件名：与中转目录里的 txt 同名（重名漫画已自动区分），加 _中文.pdf"""
        try:
            stem = os.path.splitext(wf.export_name_for(book.project))[0]
        except Exception:
            stem = book.name
        return f'{stem}_中文.pdf'

    def export_pdfs(self, books=None, quiet=False):
        out = self.output_dir()
        if not out:
            if quiet:
                return
            out = self.choose_output_dir()
            if not out:
                return
        books = books or self.selected_books() or self.books
        targets = [b for b in books if not b.job and os.path.exists(b.pdf)]
        if not targets:
            if not quiet:
                QMessageBox.information(self, '导出成品', '没有已完成的 PDF 可以导出')
            return
        jobs = [(b.name, b.pdf, os.path.join(out, self.pdf_out_name(b))) for b in targets]

        def copy_all():
            copied, skipped, failed = [], [], []
            for name, src, dst in jobs:
                try:
                    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) \
                            and os.path.getmtime(dst) >= os.path.getmtime(src):
                        skipped.append(name)
                        continue
                    tmp = dst + '.part'
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
                    copied.append(name)
                except OSError as e:
                    failed.append(f'{name}：{e}')
            return copied, skipped, failed

        def done(r, err):
            if err:
                QMessageBox.warning(self, '导出失败', str(err))
                return
            copied, skipped, failed = r
            msg = f'成品导出：新导出 {len(copied)} 本，已是最新跳过 {len(skipped)} 本'
            if failed:
                msg += f'，失败 {len(failed)} 本'
            self.sys_log(msg)
            self.statusBar().showMessage(msg + f'  →  {out}', 8000)
            if failed:
                QMessageBox.warning(self, '部分导出失败', '\n'.join(failed))
            elif not quiet:
                QMessageBox.information(self, '导出成品', msg + f'\n\n目录：{out}')

        self.run_bg(copy_all, done)

    def open_pdf(self):
        b = self.one_book()
        if b:
            if os.path.exists(b.pdf):
                open_path(b.pdf)
            else:
                QMessageBox.information(self, '提示', '还没有生成 PDF')

    def open_project(self):
        b = self.one_book()
        if b:
            open_path(b.project if os.path.isdir(b.project) else os.path.dirname(b.source))

    # ------------------------------------------------------------ 拖放
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        txts = [p for p in paths if p.lower().endswith('.txt')]
        for p in txts:
            self.import_txt_file(p)
        n = self.add_books([p for p in paths if p not in txts])
        if n:
            self.statusBar().showMessage(f'已添加 {n} 本漫画，选中后点「导出文本」', 6000)

    # ------------------------------------------------------------ 刷新
    def tick(self):
        self.poll_futures()
        self.check_idle()
        self._gpu_tick += 1
        if self._gpu_tick % 6 == 1:  # 每 3 秒查询一次显存
            self.run_bg(gpu_memory, lambda r, e: setattr(self, 'gpu', r) or self.update_vram_hint())
        self.refresh()

    def refresh(self):
        self.table.setRowCount(len(self.books))
        for r, b in enumerate(self.books):
            info = b.project_info()
            status = b.status()
            vals = [b.name, b.kind, str(info[0]) if info else '', str(info[1]) if info else '', status]
            for c, v in enumerate(vals):
                item = self.table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(r, c, item)
                if item.text() != v:
                    item.setText(v)
                if c == 0:
                    item.setToolTip(b.source)
                if c == 4:
                    color = {'已完成': '#1b7f3b', '待翻译': '#1f5fbf', '译文待加载': '#b26a00'}.get(v, '#b3261e' if v.startswith('失败') else None)
                    item.setForeground(QColor(color) if color else self.palette().text().color())
            bar = self.table.cellWidget(r, len(COLS) - 1)
            if not isinstance(bar, QProgressBar):
                bar = QProgressBar()
                bar.setTextVisible(True)
                self.table.setCellWidget(r, len(COLS) - 1, bar)
            if b.job and b.total:
                bar.setRange(0, b.total)
                bar.setValue(b.done)
                bar.setFormat(f'{b.job} {b.done}/{b.total} 页')
            elif b.job:
                bar.setRange(0, 0)
            else:
                done = status == '已完成'
                bar.setRange(0, 1)
                bar.setValue(1 if done else 0)
                bar.setFormat('完成' if done else '')
        alive = [w for w in self.workers if not w.retiring]
        busy = sum(1 for w in alive if w.job)
        loading = sum(1 for w in alive if not w.ready)
        extra = f'（{loading} 个加载中）' if loading else ''
        self.summary.setText(f'{len(self.books)} 本 · GPU 进程 {len(alive)}{extra} · 处理中 {busy} 页 · 排队 {len(self.jobs)} 页')

    def show_log(self):
        books = self.selected_books()
        lines = books[0].log if len(books) == 1 else self.system_log
        self.log_view.setPlainText('\n'.join(lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def closeEvent(self, e):
        busy = [b for b in self.books if b.job]
        if busy and QMessageBox.question(self, '退出', f'还有 {len(busy)} 本漫画在处理中，确定要退出吗？') \
                != QMessageBox.Yes:
            e.ignore()
            return
        self.closing = True
        for w in list(self.workers):
            w.proc.kill()
            w.proc.waitForFinished(3000)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self._save_state()
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('漫画翻译工具')
    app.setFont(QFont('Microsoft YaHei UI', 9))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
