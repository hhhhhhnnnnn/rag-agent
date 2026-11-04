"""打印库内全部切块（source#chunk + 正文截断），用于人工核对评估金标标签。

用法:
    .venv\\Scripts\\python scripts\\dump_kb.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import Retriever


def main() -> None:
    r = Retriever()
    r._ensure_bm25()  # 复用懒加载：从 Chroma 拉全量文档
    for text, meta in zip(r._bm25_docs, r._bm25_metas):
        print(f"=== {meta['source']}#chunk{meta['chunk_index']} ===")
        print(text[:220])
        print()


if __name__ == "__main__":
    main()
