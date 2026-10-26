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

from .config import get_settings


def chunk_text(text: str, source: str) -> list[dict]:
    """把一篇文档切成块。

    Args:
        text: 整篇文档的纯文本。
        source: 来源文件名，写入每个块的元数据。

    Returns:
        [{"text": 块内容, "source": 文件名, "chunk_index": 序号}, ...]
    """
    # TODO(你来实现):
    #   1. 按空行先把原文分成"自然段"，再识别 Markdown 标题行作为分界
    #   2. 用 settings.chunk_size / settings.chunk_overlap 控制块长与重叠
    #   3. 超长段落用滑动窗口硬切兜底
    #   4. 丢弃纯空白块，按顺序给每块编号 chunk_index
    raise NotImplementedError("chunker.chunk_text 待实现 —— 实现思路见本文件顶部注释")


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
