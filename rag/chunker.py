"""切块（Chunking）：把整篇文档切成适合检索的小块 —— RAG 质量的第一道关口。

实现前建议想清楚的几个设计点：

1. 为什么不能按固定 400 字硬切？
   会把一句话、一个概念拦腰斩断，检索召回了残缺片段，
   生成时模型只能靠猜 —— 幻觉往往从切块就开始了。

2. 推荐策略（自顶向下逐级兜底）：
   a. 先按 Markdown 标题（# ## ###，正则 r"^#{1,6}\\s"）切出"章节"；
   b. 章节超过 chunk_size，再按空行切成段落合并/拆分；
   c. 段落仍超长，用滑动窗口硬切兜底（步长 = chunk_size - overlap）。

3. overlap（相邻块重叠）的作用：让关键句即使恰好跨块，也至少完整地
   存在于其中一块里。典型取 chunk_size 的 10% ~ 15%。

4. 元数据必须保留：source（哪个文件）、chunk_index（第几块），
   建议再加当前章节标题 —— 检索命中后展示"出处"全靠它们。

5. 块大小是效果/成本的权衡：
   太小 -> 单块语义不完整，召回碎；
   太大 -> 一块里混多个主题，向量表示被稀释，prompt 也更贵。
   中文场景 300~500 字是常用起点，参数已放在 .env（CHUNK_SIZE），可实验。
"""

from __future__ import annotations

import re

from .config import get_settings

_HEADING_RE = re.compile(r"^#{1,6}\s")


def chunk_text(text: str, source: str) -> list[dict]:
    """把一篇文档切成块。

    Args:
        text: 整篇文档的纯文本。
        source: 来源文件名，写入每个块的元数据。

    Returns:
        [{"text": 块内容, "source": 文件名, "chunk_index": 序号}, ...]
    """
    settings = get_settings()
    size, ov = settings.chunk_size, settings.chunk_overlap

    # 按空行分自然段（str.split 不认正则，必须用 re.split）
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    pieces: list[str] = []  # 先攒纯文本，最后统一编号、包装元数据
    buf = ""
    for p in paras:
        if len(p) > size:  # 超长段落：冲掉缓冲区，滑动窗口硬切兜底
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(window_split(p, size, ov))
        elif not buf:
            buf = p
        elif _HEADING_RE.match(p):  # Markdown 标题总是开新块（章节分界）
            pieces.append(buf)
            buf = p
        elif len(buf) + len(p) + 2 <= size:  # +2 = 连接符 "\n\n"
            buf = f"{buf}\n\n{p}"
        else:  # 装不下了：收走旧 buf，带上 overlap 尾巴开新块
            pieces.append(buf)
            tail = buf[-ov:] if ov else ""
            buf = f"{tail}\n\n{p}" if tail else p
    if buf:
        pieces.append(buf)

    # 统一编号 + 元数据，顺手过滤空白块
    return [
        {"text": piece, "source": source, "chunk_index": i}
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]


def window_split(text: str, size: int, ov: int) -> list[str]:
    """滑动窗口硬切兜底：步长 = size - ov，尽量在句末标点处断开。"""
    if len(text) <= size:
        return [text]

    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # 在块内后半段找句末标点，避免把句子拦腰斩断
            head = text[start:end]
            cut = max(
                head.rfind("。"), head.rfind("！"), head.rfind("？"),
                head.rfind("."), head.rfind("!"), head.rfind("?"),
                head.rfind("\n"),
            )
            if cut >= size * 0.6:  # 断点足够靠后才采用；找不到是 -1，自然被排除
                end = start + cut + 1
        pieces.append(text[start:end])
        if end >= len(text):
            break  # 最后一块不再回退 overlap，否则死循环
        start = max(end - ov, start + 1)  # 至少前进 1，保证循环可终止
    return pieces


def chunk_all(docs: list[dict]) -> list[dict]:
    """对 load_all() 的结果逐篇切块，汇总返回。"""
    settings = get_settings()
    chunks: list[dict] = []
    for doc in docs:
        chunks.extend(chunk_text(doc["text"], doc["source"]))
    print(
        f"[chunker] {len(docs)} 篇文档 -> {len(chunks)} 块 "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks
