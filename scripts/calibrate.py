"""阈值校准工具：只跑检索不调 LLM，看一批问题各自的最高分，据此定 SIM_THRESHOLD。

用法:
    .venv\\Scripts\\python scripts\\calibrate.py "问题1" "问题2" ...
不带参数则使用内置的一组示例问题（相关/半相关/无关）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import get_settings  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

DEFAULT_QUESTIONS = [
    "星尘音箱X1的保修期是多久？",        # 相关
    "序列号在机身哪个位置？",            # 相关
    "音箱支持连接苹果手机吗？",          # 半相关（有蓝牙，无苹果）
    "设备的工作温度范围是多少？",        # 相关
    "火箭发动机的工作原理是什么？",      # 无关
]


def main() -> None:
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    settings = get_settings()
    retriever = Retriever()
    print(f"库内总块数: {retriever.collection.count()}，当前阈值: {settings.sim_threshold}\n")
    print(f"{'问题':<24s} 最高分   命中块")
    print("-" * 60)
    for q in questions:
        hits = retriever.search(q)
        if hits:
            top = hits[0]
            print(f"{q:<24s} {top['score']:.3f}   {top['source']}#chunk{top['chunk_index']}")
        else:
            print(f"{q:<24s} （无命中）")


if __name__ == "__main__":
    main()
