"""v0.3 验收脚本：模拟多轮对话，检验查询改写能否接住省略与指代。

用法:
    .venv\\Scripts\\python scripts\\test_multiturn.py

四轮剧本设计：
    1. 独立问题（首轮，免改写直通）
    2. 省略主语："那序列号在哪里？" —— 主语在第一轮里
    3. 再次省略："工作温度范围呢？"
    4. 无关问题："火箭怎么上天？" —— 改写不许把它"变"成可检索的，
       拒答闸必须依然拦得住（防止改写模型编造上下文骗开闸门）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import get_settings
from rag.generator import answer
from rag.retriever import Retriever
from rag.rewriter import rewrite

TURNS = [
    "星尘音箱X1的保修期是多久？",
    "那序列号在哪里？",
    "工作温度范围呢？",
    "火箭怎么上天？",
]


def main() -> None:
    settings = get_settings()
    retriever = Retriever()
    history: list[dict] = []

    for q in TURNS:
        try:
            standalone = rewrite(q, history)
            tag = "" if standalone == q else f"  [改写] -> {standalone}"
        except NotImplementedError:
            standalone, tag = q, "  （改写未实现，直通）"
        history.append({"role": "user", "content": q})

        hits = retriever.search(standalone, top_k=settings.top_k)
        print("=" * 60)
        print(f"你问 > {q}{tag}")
        if not hits or hits[0]["score"] < settings.sim_threshold:
            print("助手 > 知识库中没有相关内容。   <- 拒答闸")
            history.append({"role": "assistant", "content": "知识库中没有相关内容。"})
            continue

        top = hits[0]
        print(f"最高分 {top['score']:.3f} | {top['source']}#chunk{top['chunk_index']}")
        reply = answer(standalone, hits)
        history.append({"role": "assistant", "content": reply})
        print(f"助手 > {reply}")


if __name__ == "__main__":
    main()
