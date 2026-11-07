"""全局配置：从 .env 读取，支持 qwen / glm 两个 OpenAI 兼容提供商。

约定：所有变量都允许在 .env 中以 {PROVIDER}_XXX 的形式覆盖内置默认值，
例如 GLM_CHAT_MODEL=glm-4-plus。换模型只改 .env，不动代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 各提供商内置默认值（.env 中同名变量可覆盖）
_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "qwen": {  # 阿里云百炼 DashScope，OpenAI 兼容模式
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-plus",
        "embed_model": "text-embedding-v3",
        "embed_dim": "1024",
    },
    "glm": {  # 智谱开放平台，OpenAI 兼容
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "chat_model": "glm-4-flash",
        "embed_model": "embedding-3",
        "embed_dim": "1024",
    },
}


@dataclass
class Settings:
    provider: str
    api_key: str
    base_url: str
    chat_model: str
    embed_model: str
    embed_dim: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    sim_threshold: float
    recall_n: int
    rerank_model: str
    rerank_candidates: int
    rrf_w_bm: float
    db_dir: Path
    data_dir: Path
    collection: str


def _get(provider: str, key: str) -> str:
    """优先读 .env 中的 {PROVIDER}_{KEY}，否则用内置默认值。"""
    return os.environ.get(f"{provider.upper()}_{key}", _PROVIDER_DEFAULTS[provider][key])


def get_settings() -> Settings:
    provider = os.environ.get("LLM_PROVIDER", "qwen").strip().lower()
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(f"未知提供商 {provider!r}（可选: {', '.join(_PROVIDER_DEFAULTS)}）")

    api_key = os.environ.get(f"{provider.upper()}_API_KEY", "").strip()
    if not api_key or "填入" in api_key:
        raise ValueError(
            f"请先在 .env 中填写 {provider.upper()}_API_KEY"
            "（当前未填写或仍是占位符，可参照 .env.example）"
        )

    return Settings(
        provider=provider,
        api_key=api_key,
        base_url=_get(provider, "base_url").rstrip("/"),
        chat_model=_get(provider, "chat_model"),
        embed_model=_get(provider, "embed_model"),
        embed_dim=int(_get(provider, "embed_dim")),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "400")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "50")),
        top_k=int(os.environ.get("TOP_K", "5")),
        sim_threshold=float(os.environ.get("SIM_THRESHOLD", "0.3")),
        recall_n=int(os.environ.get("RECALL_N", "10")),
        rerank_model=os.environ.get("RERANK_MODEL", "qwen3.7-text-rerank"),
        rerank_candidates=int(os.environ.get("RERANK_CANDIDATES", "8")),
        rrf_w_bm=float(os.environ.get("RRF_W_BM", "1.0")),
        db_dir=ROOT / os.environ.get("DB_DIR", "db/chroma"),
        data_dir=ROOT / os.environ.get("DATA_DIR", "data"),
        collection=os.environ.get("COLLECTION_NAME", "docs"),
    )
