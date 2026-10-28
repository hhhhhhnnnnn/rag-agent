"""单次问答（问题从命令行传入），便于脚本化测试，不用进交互模式。

用法:
    .venv\\Scripts\\python scripts\\ask_once.py "保修多久？"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import get_settings  # noqa: E402
from rag.generator import answer  # noqa: E402
from rag.retriever import Retriever  # noqa: E402


def main() -> None:
    question = " ".join(sys.argv[1:]) or "保修政策是什么？"
    settings = get_settings()
    hits = Retriever().search(question, top_k=settings.top_k)

    print(f"问题: {question}")
    if hits:
        print(f"命中 {len(hits)} 块，最高分 {hits[0]['score']:.3f}（拒答阈值 {settings.sim_threshold}）")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h['source']}#chunk{h['chunk_index']}  score={h['score']:.3f}")
    else:
        print("未命中任何块")

    if not hits or hits[0]["score"] < settings.sim_threshold:
        print("助手 > 知识库中没有相关内容。   <- 触发拒答阈值")
        return
    print(f"助手 > {answer(question, hits)}")


if __name__ == "__main__":
    main()
