"""文档加载：把 data/ 下的 PDF / Word / Markdown / TXT 统一读成纯文本。"""
from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def load_file(path: Path) -> str:
    """读取单个文件，返回纯文本。"""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError(f"暂不支持的格式 {suffix}（{path.name}）")


def load_all(data_dir: Path) -> list[dict]:
    """遍历目录（含子目录），返回 [{"text": 正文, "source": 文件名}, ...]。

    已知限制（留待后续迭代）：
    - 扫描版 PDF（图片型）提取不出文字，pypdf 返回空串，需要 OCR（PaddleOCR、marker 等）；
    - Word 中的表格不会读取（python-docx 需单独遍历 document.tables）。
    """
    docs: list[dict] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = load_file(path).strip()
            if text:
                docs.append({"text": text, "source": path.name})
            else:
                print(f"[loader] 警告：未提取到文字（可能是扫描件）: {path.name}")
    return docs
