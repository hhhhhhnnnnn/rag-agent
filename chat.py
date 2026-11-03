"""命令行问答入口（v0.3：多轮对话 + 查询改写）。

用法:
    .venv\\Scripts\\python chat.py
输入问题回车提问，输入 q 退出。

数据流：history 累积原始问答 -> 每轮先 rewrite() 成独立问题 -> 检索/作答
历史里存的是用户原话（不是改写后的），改写只在检索前临时发生。
"""
from rag.config import get_settings
from rag.generator import answer
from rag.retriever import Retriever
from rag.rewriter import rewrite


def main() -> None:
    settings = get_settings()
    retriever = Retriever()
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": ...}] 旧 -> 新

    print("RAG 文档问答（输入 q 退出）")
    print("=" * 46)
    while True:
        try:
            question = input("\n你问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"q", "quit", "exit"}:
            break

        # v0.3: 检索前改写。骨架未实现时自动直通，系统保持可用
        try:
            standalone = rewrite(question, history)
        except NotImplementedError:
            standalone = question
            print("（改写未实现，按原问题检索）")
        if standalone != question:
            print(f"（改写: {question} -> {standalone}）")

        history.append({"role": "user", "content": question})

        hits = retriever.search(standalone, top_k=settings.top_k)
        if not hits or hits[0]["score"] < settings.sim_threshold:
            print("助手 > 知识库中没有相关内容。")
            # 也记入历史，保持 user/assistant 交替，且让改写模型知道上一问没答案
            history.append({"role": "assistant", "content": "知识库中没有相关内容。"})
            continue

        reply = answer(standalone, hits)
        history.append({"role": "assistant", "content": reply})
        print(f"助手 > {reply}")
        print("--- 引用来源 ---")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h['source']}#chunk{h['chunk_index']}  score={h['score']:.3f}")


if __name__ == "__main__":
    main()
