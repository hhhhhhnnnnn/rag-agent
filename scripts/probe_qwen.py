"""探测专属工作空间的模型列表与连通性（不打印 key）。

用法:
    .venv\\Scripts\\python scripts\\probe_qwen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from rag.config import get_settings  # noqa: E402


def main() -> None:
    s = get_settings()
    print(f"端点: {s.base_url}")
    client = OpenAI(api_key=s.api_key, base_url=s.base_url)

    print("\n== 模型列表 ==")
    model_ids: list[str] = []
    try:
        for m in client.models.list():
            model_ids.append(m.id)
            print(f"  - {m.id}")
    except Exception as e:
        print(f"  获取失败: {type(e).__name__}: {str(e)[:150]}")

    print("\n== 向量模型实测 ==")
    try:
        r = client.embeddings.create(model=s.embed_model, input=["连通性测试"])
        print(f"  {s.embed_model} 连通，维度 = {len(r.data[0].embedding)}")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {str(e)[:150]}")

    print("\n== 对话模型实测 ==")
    # 优先用命令行指定的候选，否则自动从列表里挑（排除非对话类）
    argv_models = sys.argv[1:]
    candidates = argv_models or [s.chat_model] + [
        m for m in model_ids
        if m != s.chat_model
        and not any(k in m.lower() for k in
                    ("embed", "rerank", "asr", "tts", "realtime", "omni",
                     "image", "mt-", "-mt", "ocr", "speech", "audio", "livetranslate"))
    ]
    for name in candidates[:8]:
        try:
            r = client.chat.completions.create(
                model=name,
                messages=[{"role": "user", "content": "请只回复两个字：正常"}],
                max_tokens=2048,
            )
            print(f"  [成功] {name} -> {r.choices[0].message.content!r}")
            print(f"  >>> 建议把 .env 的 QWEN_CHAT_MODEL 改为: {name}")
            return
        except Exception as e:
            print(f"  [失败] {name}: {str(e)[:100]}")
    print("  没有找到可用的对话模型，请去控制台查看模型名")


if __name__ == "__main__":
    main()
