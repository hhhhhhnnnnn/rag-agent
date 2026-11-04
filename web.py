"""RAG 问答 Web 服务（v0.4）：裸 ASGI + uvicorn，零新增 pip 依赖。

为什么不用 FastAPI：chromadb 1.x 不再携带 fastapi/starlette，装它们要碰
pip（本机 pip 前科累累：jieba/rank_bm25 两次卡死）。而 ASGI 应用的本质
只是一个协程:

    async def app(scope, receive, send): ...

uvicorn 已在依赖树里（chromadb 的依赖），HTTP 解析/并发它全包了，
路由和 JSON 我们手写 60 行——与手写 BM25 同一哲学。

请求流（每轮）:
    浏览器 POST /api/chat {session_id, question}
      -> rewrite()（历史在服务端内存里，按 session_id 隔离）
      -> search()（v0.2 混合检索管线）
      -> answer()（带引用回答）
    <- {answer, rewrite, refused, hits[]}

用法:
    .venv\\Scripts\\python web.py            # http://127.0.0.1:8000
    端口可用环境变量 PORT 覆盖；只绑 127.0.0.1（本地使用，不对外暴露）
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import uvicorn

from rag import chunker, loader
from rag.config import get_settings
from rag.generator import answer
from rag.retriever import Retriever
from rag.rewriter import rewrite

ROOT = Path(__file__).resolve().parent
_INDEX_HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

settings = get_settings()
retriever = Retriever()
retriever._ensure_bm25()  # 启动即建索引：消灭首问并发双建的竞态，首问不用等懒加载

# 会话历史（内存态：重启即清空，重启后等于新对话——文档里注明）
_histories: dict[str, list[dict]] = {}
_MAX_SESSIONS = 200      # 粗暴上限，防长期运行的内存泄漏
_MAX_TURNS = 40          # 单会话保留轮数（改写窗口只需要 6 条，40 绰绰有余）


# ---------- 同步业务逻辑（在线程池里跑，不阻塞事件循环） ----------

def _handle_chat(data: dict) -> dict:
    sid = str(data.get("session_id") or "")[:64]
    question = str(data.get("question") or "").strip()
    if not sid or not question:
        return {"error": "缺少 session_id 或 question"}

    if sid not in _histories and len(_histories) >= _MAX_SESSIONS:
        _histories.clear()
    history = _histories.setdefault(sid, [])

    standalone = rewrite(question, history)  # rewriter 自带降级：失败返回原问题
    hits = retriever.search(standalone, top_k=settings.top_k)
    refused = (not hits) or hits[0]["score"] < settings.sim_threshold

    history.append({"role": "user", "content": question})
    reply = "知识库中没有相关内容。" if refused else answer(standalone, hits)
    history.append({"role": "assistant", "content": reply})
    del history[:-_MAX_TURNS]

    return {
        "refused": refused,
        "rewrite": None if standalone == question else f"{question} → {standalone}",
        "answer": reply,
        "hits": [] if refused else [
            {"source": h["source"], "chunk_index": h["chunk_index"],
             "score": round(h["score"], 3)}
            for h in hits[: settings.top_k]
        ],
    }


def _handle_ingest() -> dict:
    """重跑入库并热重建 BM25 索引——常驻服务的专属能力（CLI 时代做不到）。"""
    docs = loader.load_all(settings.data_dir)
    if not docs:
        return {"error": f"{settings.data_dir} 下没有可用文档"}
    n = retriever.ingest(chunker.chunk_all(docs))
    retriever.invalidate_bm25()
    retriever._ensure_bm25()
    return {"ingested": n, "total": retriever.collection.count()}


# ---------- 裸 ASGI 应用 ----------

async def _read_body(receive) -> bytes:
    chunks = []
    while True:
        msg = await receive()
        chunks.append(msg.get("body", b""))
        if not msg.get("more_body", False):
            return b"".join(chunks)


async def _respond(send, status: int, body: bytes, content_type: str) -> None:
    await send({
        "type": "http.response.start", "status": status,
        "headers": [
            [b"content-type", content_type.encode()],
            [b"content-length", str(len(body)).encode()],
        ],
    })
    await send({"type": "http.response.body", "body": body})


async def _json(send, status: int, obj: dict) -> None:
    await _respond(send, status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")


async def app(scope, receive, send):
    """ASGI 入口：uvicorn 把每个 HTTP 请求翻译成一次本协程调用。"""
    if scope["type"] != "http":
        return
    path, method = scope["path"], scope["method"]

    try:
        if path == "/" and method == "GET":
            await _respond(send, 200, _INDEX_HTML.encode("utf-8"),
                           "text/html; charset=utf-8")

        elif path == "/api/health" and method == "GET":
            await _json(send, 200, {"ok": True, "chunks": retriever.collection.count()})

        elif path == "/api/chat" and method == "POST":
            data = json.loads(await _read_body(receive) or b"{}")
            # 同步逻辑（网络调用+检索）丢进默认线程池，事件循环保持畅通
            result = await asyncio.get_running_loop().run_in_executor(
                None, _handle_chat, data)
            await _json(send, 200, result)

        elif path == "/api/ingest" and method == "POST":
            result = await asyncio.get_running_loop().run_in_executor(
                None, _handle_ingest)
            await _json(send, 200, result)

        elif path == "/api/reset" and method == "POST":
            data = json.loads(await _read_body(receive) or b"{}")
            _histories.pop(str(data.get("session_id") or "")[:64], None)
            await _json(send, 200, {"ok": True})

        else:
            await _json(send, 404, {"error": f"no route: {method} {path}"})

    except Exception as e:
        # 最外层兜底：任何异常都变成 500 JSON，浏览器端能看到错误而不是断连
        await _json(send, 500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"RAG 文档问答服务已启动: http://127.0.0.1:{port}")
    print("（会话历史在内存中，重启服务即开新对话；Ctrl+C 停止）")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
