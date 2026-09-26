"""
下载嵌字用的开源中文字体到 workflow/fonts/（均为 SIL Open Font License 1.1）。

    python install_fonts.py

不运行也能用：会退回 manga-image-translator 自带的 fonts/msyh.ttc。
"""
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')

# (下载地址, 压缩包内文件 -> 保存文件名；直接是字体文件时为 None)
SOURCES = [
    ('https://github.com/CyanoHao/Resource-Han-Rounded/releases/download/v0.990/RHR-CN-0.990.7z',
     {'ResourceHanRoundedCN-Medium.ttf': '源柔黑体-Medium.ttf',
      'ResourceHanRoundedCN-Bold.ttf': '源柔黑体-Bold.ttf'}),
    ('https://github.com/adobe-fonts/source-han-sans/releases/download/2.005R/09_SourceHanSansSC.zip',
     {'SourceHanSansSC-Medium.otf': '思源黑体-Medium.otf',
      'SourceHanSansSC-Bold.otf': '思源黑体-Bold.otf'}),
    ('https://github.com/lxgw/LxgwWenKai/releases/download/v1.522/LXGWWenKai-Medium.ttf',
     {None: '霞鹜文楷-Medium.ttf'}),
    ('https://github.com/atelier-anchor/smiley-sans/releases/download/v2.0.1/smiley-sans-v2.0.1.zip',
     {'SmileySans-Oblique.ttf': '得意黑.ttf'}),
]


def download(url, dest):
    print(f'下载 {url}')
    with urllib.request.urlopen(url) as r, open(dest, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {done / 1048576:.0f}/{total / 1048576:.0f} MB', end='', flush=True)
    print()


def extract(archive, wanted, out_dir):
    """把压缩包里按文件名匹配到的字体解出来"""
    tmp = tempfile.mkdtemp(prefix='fonts_')
    try:
        if archive.endswith('.7z'):
            import py7zr
            with py7zr.SevenZipFile(archive) as z:
                z.extractall(tmp)
        else:
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp)
        for root, _, files in os.walk(tmp):
            for f in files:
                if f in wanted:
                    shutil.copyfile(os.path.join(root, f), os.path.join(out_dir, wanted[f]))
                    print(f'  已安装 {wanted[f]}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    os.makedirs(FONT_DIR, exist_ok=True)
    for url, wanted in SOURCES:
        targets = [os.path.join(FONT_DIR, n) for n in wanted.values()]
        if all(os.path.exists(t) for t in targets):
            print(f'已存在，跳过：{"、".join(wanted.values())}')
            continue
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, url.rsplit('/', 1)[1])
            download(url, path)
            if None in wanted:
                shutil.copyfile(path, targets[0])
                print(f'  已安装 {wanted[None]}')
            else:
                extract(path, wanted, FONT_DIR)
    print('完成')


if __name__ == '__main__':
    sys.exit(main())
