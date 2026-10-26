"""命令行问答入口。

用法:
    .venv\\Scripts\\python chat.py
输入问题回车提问，输入 q 退出。
"""
from rag.config import get_settings
from rag.generator import answer
from rag.retriever import Retriever


def main() -> None:
    settings = get_settings()
    retriever = Retriever()

    print("RAG 文档问答（输入 q 退出）")
    print("=" * 46)
    while True:
        try:
            question = input("\n你问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"q", "quit", "exit"}:
            break

        hits = retriever.search(question, top_k=settings.top_k)
        if not hits or hits[0]["score"] < settings.sim_threshold:
            print("助手 > 知识库中没有相关内容。")
            continue

        print(f"助手 > {answer(question, hits)}")
        print("--- 引用来源 ---")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h['source']}#chunk{h['chunk_index']}  score={h['score']:.3f}")


if __name__ == "__main__":
    main()
