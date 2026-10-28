"""调用工作空间的 rerank 模型做精排（cross-encoder 相关度重排序）。

实测可用路径为 DashScope 原生协议（兼容模式无 /rerank 端点）:
    POST {host}/api/v1/services/rerank/text-rerank/text-rerank
    body = {"model", "input": {"query", "documents"}, "parameters": {"top_n"}}
    响应 = {"output": {"results": [{"index", "relevance_score"}, ...]}}

设计要点：
1. rerank 与 embedding 的区别：
   - embedding 是"双塔"——query 和文档各自独立编码，文档向量可以事先算好，
     检索快，但两边互相看不到对方，精度有天花板；
   - rerank 是 cross-encoder——query + 文档拼在一起过模型，逐词交叉注意，
     精度远高于双塔，但每对 (query, doc) 都要现算，慢且贵。
   所以工程上的定式是：" embedding 粗召回一个小集合 → rerank 精排 "。
2. 降级策略：任何异常都返回 None，由调用方退回向量排序，保证服务不中断。
"""
from __future__ import annotations

import httpx

from .config import get_settings

_TIMEOUT = 30


def rerank(query: str, documents: list[str], top_n: int) -> list[dict] | None:
    """对 documents 相对 query 重排序。

    Args:
        query: 用户问题。
        documents: 候选文本列表（顺序即 index 的含义）。
        top_n: 只保留相关度最高的前 n 条。

    Returns:
        [{"index": 文档原始下标, "relevance_score": 0~1}, ...] 按分数降序；
        调用失败返回 None（调用方降级，不要让精排故障拖垮问答）。
    """
    s = get_settings()
    host = s.base_url.split("/compatible-mode")[0]  # 兼容域名 -> 裸主机名
    url = f"{host}/api/v1/services/rerank/text-rerank/text-rerank"
    body = {
        "model": s.rerank_model,
        "input": {"query": query, "documents": documents},
        "parameters": {"return_documents": False, "top_n": top_n},
    }
    try:
        r = httpx.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {s.api_key}"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json()["output"]["results"]
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    except Exception as e:
        print(f"[rerank] 调用失败，将降级: {type(e).__name__}: {str(e)[:120]}")
        return None
