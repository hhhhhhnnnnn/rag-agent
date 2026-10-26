"""入库脚本：data/ 下的文档 -> 加载 -> 切块 -> 向量化 -> 写入 Chroma。

用法:
    .venv\\Scripts\\python ingest.py
"""
from rag import chunker, loader
from rag.config import get_settings
from rag.retriever import Retriever


def main() -> None:
    settings = get_settings()

    docs = loader.load_all(settings.data_dir)
    if not docs:
        print(f"[ingest] {settings.data_dir} 下没有可用文档（支持 md/txt/pdf/docx）")
        return

    chunks = chunker.chunk_all(docs)
    if not chunks:
        print("[ingest] 切块结果为空，请检查 chunker 实现")
        return

    n = Retriever().ingest(chunks)
    print(f"[ingest] 完成：写入 {n} 块 -> {settings.db_dir}（集合: {settings.collection}）")


if __name__ == "__main__":
    main()
