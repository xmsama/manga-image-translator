# 漫画翻译工具（workflow）

基于 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 的桌面工具，
把「识别 → 翻译 → 嵌字」拆成三步，翻译这一步交给你自己或任意 AI（ChatGPT、Claude 等），
其余全部自动完成，输出成品 PDF。

```
漫画（文件夹 / PDF） ──导出──▶ 待翻译 txt ──AI 翻译──▶ 译文 txt ──加载──▶ 擦字 + 嵌字 ──▶ 成品 PDF
```

## 功能

- **批量导入**：拖入图片文件夹或 PDF（PDF 页面无损提取），可同时处理多本
- **文本导出**：整本原文导出成一个 txt，文件头自带给 AI 的完整翻译规则；所有书的 txt 集中在一个中转目录，
  让 AI 批量翻译后同名写回即可，点「加载译文」自动识别是哪本书并开始生成
- **按页并行**：GPU 上常驻多个处理进程，所有书的每一页拆成独立任务并行处理；空闲自动释放显存
- **新嵌字器**（替换原引擎渲染）
  - 识别气泡形状，文字按气泡轮廓排版并居中，不溢出
  - 全书统一字号，放不下才缩小
  - 横排 / 竖排自动选择：优先跟随原文，放不下时换方向
  - 中文断行禁则、竖排标点
  - 清理气泡内引擎漏擦的注音假名、残字和杂点
- **碎片合并**：检测模型把一句话拆成多个框时（如英文的 `I` / `SEE...`）自动合并
- **成品导出**：一键把完成的 PDF 复制到指定目录，可设为生成后自动导出
- **清理工程**：清理中间文件或删除整个工程（移到回收站），包括已从列表移除的工程

## 安装（Windows + NVIDIA 显卡）

在本仓库根目录执行：

```bat
:: Python 3.10 或 3.11
py -3.11 -m venv venv
venv\Scripts\python -m pip install --upgrade pip

:: 按显卡的 CUDA 版本安装 PyTorch，见 https://pytorch.org/get-started/locally/
venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m pip install -r workflow\requirements.txt

:: 可选：下载嵌字字体（源柔黑体、思源黑体、霞鹜文楷、得意黑，均为 OFL 开源字体）
venv\Scripts\python workflow\install_fonts.py
```

模型在第一次使用时自动下载到仓库的 `models/`。

## 使用

双击 `workflow\漫画翻译工具.bat` 打开界面。

1. **添加漫画**：把图片文件夹或 PDF 拖进窗口
2. **导出文本**：选中漫画点「导出文本」，txt 会写到 `workflow\翻译中转\待翻译\`（点「打开中转目录」可直接打开）
3. **翻译**：让 AI 翻译 `待翻译` 里的所有 txt，以同名文件写到 `翻译中转\译文\`。
   每个 txt 的文件头已经写明全部规则，目录里的 `00_给AI的说明.md` 是总任务说明。
   也可以把 txt 内容复制给 AI，再用「从剪贴板导入译文」
4. **加载译文**：点「加载译文」，有新译文的书自动开始擦字、嵌字并合成 PDF
5. **导出成品**：点「一键导出成品 PDF」复制到你的成品目录

排版不满意时，可在界面的「排版」一行调整字体、字号、描边、方向，先「预览排版」看一页，再「重新生成」。

### 译文 txt 格式

```
# 工程: ...            ← 程序用来定位漫画，需保留
### 001.jpg             ← 页标记
<1>第一个气泡的译文       ← 保留编号，只替换编号后的文字，一行一个
<2>                     ← 留空 = 保留原图（致谢、水印、拟声词等）
```

## 配置

`workflow\config.json`：

| 键 | 说明 |
|---|---|
| `translator.target_lang` | 译文语言：CHS / CHT / ENG / JPN / KOR（界面可改） |
| `typeset` | 嵌字设置：`renderer`（`typeset` 新嵌字器 / `default` 原引擎）、`font`、`size_scale`、`stroke`、`direction`、`clean`（气泡残留清理） |
| `transfer_dir` | 中转目录位置，默认 `workflow\翻译中转` |
| `idle_release_seconds` | 任务结束后空闲多少秒释放显存，默认 60 |
| `output_dir` / `auto_export_pdf` | 成品目录与是否自动导出（界面可改） |

检测、OCR、擦字等引擎参数与 manga-image-translator 的配置格式相同。

## 命令行

```bat
venv\Scripts\python workflow\mit_workflow.py export  <图片文件夹或PDF>
venv\Scripts\python workflow\mit_workflow.py import  <译文txt | 工程文件夹>
venv\Scripts\python workflow\mit_workflow.py auto    <图片文件夹或PDF>   :: 用 config.json 里的机翻引擎一键翻译
```

## 性能参考

RTX 4090，1184×1682 页面，擦字用 lama_large：

| GPU 进程数 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 每页耗时 | 2.95s | 1.41s | 1.14s | 1.07s |
| 显存 | 6.2GB | 10.1GB | 13.8GB | 17.8GB |

每个进程约占 3.9GB 显存，超过 3～4 个后提速有限。

## 说明

- 请只翻译你有权使用的漫画，不要把漫画图片、PDF 或翻译文本提交到公开仓库（`.gitignore` 已排除这些运行时文件）
- 本目录代码随上游项目采用 GPL-3.0 许可证
