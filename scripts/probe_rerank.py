"""探测工作空间 rerank 模型的可用调用路径（不打印 key）。

用法:
    .venv\\Scripts\\python scripts\\probe_rerank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402  (openai 的依赖，无需单独安装)

from rag.config import get_settings  # noqa: E402

MODEL = "qwen3.7-text-rerank"


def main() -> None:
    s = get_settings()
    host = s.base_url.split("/compatible-mode")[0]
    headers = {"Authorization": f"Bearer {s.api_key}"}

    cases = [
        # DashScope 原生协议路径
        ("原生 /api/v1/services/rerank",
         f"{host}/api/v1/services/rerank/text-rerank/text-rerank",
         {"model": MODEL,
          "input": {"query": "音箱保修多久", "documents": ["保修期为12个月", "产品颜色为白色"]},
          "parameters": {"return_documents": False, "top_n": 2}}),
        # OpenAI 风格 /rerank（SiliconFlow/Jina 风格）
        ("兼容模式 /compatible-mode/v1/rerank",
         f"{host}/compatible-mode/v1/rerank",
         {"model": MODEL, "query": "音箱保修多久",
          "documents": ["保修期为12个月", "产品颜色为白色"], "top_n": 2}),
    ]

    for name, url, body in cases:
        print(f"\n== {name} ==")
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=30)
            print(f"HTTP {r.status_code}")
            print(r.text[:400])
        except Exception as e:
            print(f"异常: {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    main()
