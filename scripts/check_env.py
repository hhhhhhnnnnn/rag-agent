"""环境自检：逐项验证 Python 版本、依赖、配置与 API 连通性。

填好 .env 后运行:
    .venv\\Scripts\\python scripts\\check_env.py
四项全过即可开始实现 rag/ 里的核心逻辑。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_all_ok = True


def _report(passed: bool, msg: str) -> None:
    global _all_ok
    _all_ok = _all_ok and passed
    print(f"  [{'通过' if passed else '失败'}] {msg}")


def main() -> int:
    print("=" * 46)
    print("RAG 项目环境自检")
    print("=" * 46)

    print("\n[1/4] Python 版本")
    v = sys.version_info
    _report((v.major, v.minor) >= (3, 10), f"当前 {v.major}.{v.minor}.{v.micro}，要求 >= 3.10")

    print("\n[2/4] 依赖包导入")
    missing = []
    for mod in ("openai", "dotenv", "chromadb", "pypdf", "docx"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        _report(False, f"缺失 {missing}，请执行: .venv\\Scripts\\python -m pip install -r requirements.txt")
    else:
        _report(True, "openai / dotenv / chromadb / pypdf / docx 全部可用")

    print("\n[3/4] 配置加载 (.env)")
    settings = None
    try:
        from rag.config import get_settings

        settings = get_settings()
        _report(
            True,
            f"提供商={settings.provider} | 对话={settings.chat_model} | "
            f"向量={settings.embed_model}({settings.embed_dim}维) | top_k={settings.top_k}",
        )
    except Exception as e:
        _report(False, f"{e}")

    print("\n[4/4] API 连通性")
    if settings is not None:
        from rag import llm

        try:
            reply = llm.chat([{"role": "user", "content": "请只回复两个字：正常"}])
            _report(True, f"对话模型连通，回复：{reply.strip()[:20]}")
        except Exception as e:
            _report(False, f"对话模型调用失败：{type(e).__name__}: {e}")
        try:
            dim = len(llm.embed(["连通性测试"])[0])
            extra = "" if dim == settings.embed_dim else (
                f"；与 .env 配置的 {settings.embed_dim} 维不一致，请同步修改 EMBED_DIM"
            )
            _report(dim == settings.embed_dim, f"向量模型连通，返回 {dim} 维{extra}")
        except Exception as e:
            _report(False, f"向量模型调用失败：{type(e).__name__}: {e}")
    else:
        _report(False, "配置未就绪，跳过（填好 .env 后重跑本脚本）")

    print("\n" + "-" * 46)
    if _all_ok:
        print("底层全部就绪！接下来实现 rag/ 内带 TODO 的三个模块即可。")
        return 0
    print("有未通过项，请按上方提示修复后重跑本脚本。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
