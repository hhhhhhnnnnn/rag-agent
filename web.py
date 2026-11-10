"""RAG 问答 Web 服务（v0.4）：裸 ASGI + uvicorn，零新增 pip 依赖。

为什么不用 FastAPI：chromadb 1.x 不再携带 fastapi/starlette，装它们要碰
pip（本机 pip 前科累累：jieba/rank_bm25 两次卡死）。而 ASGI 应用的本质
只是一个协程:

    async def app(scope, receive, send): ...

uvicorn 已在依赖树里（chromadb 的依赖），HTTP 解析/并发它全包了，
路由和 JSON 我们手写 60 行——与手写 BM25 同一哲学。

请求流（每轮）:
    浏览器 POST /api/chat/stream {session_id, question}   ← SSE 流式（主力）
      -> event:meta   改写+引用清单（检索一完成就先行到达）
      -> event:delta  回答增量逐段推（边生成边显示）
      -> event:done   收尾（历史此刻落盘 sessions/）
    POST /api/chat                                        ← 非流式（调试用）
    会话历史存 sessions/<sid>.json，服务重启后追问不丢上下文

用法:
    .venv\\Scripts\\python web.py            # http://127.0.0.1:8000
    端口可用环境变量 PORT 覆盖；只绑 127.0.0.1（本地使用，不对外暴露）
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import uvicorn

from rag import chunker, loader
from rag.agent import answer_agent
from rag.config import get_settings
from rag.generator import answer, answer_stream
from rag.retriever import Retriever
from rag.rewriter import rewrite

ROOT = Path(__file__).resolve().parent
_INDEX_HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

settings = get_settings()
retriever = Retriever()
retriever._ensure_bm25()  # 启动即建索引：消灭首问并发双建的竞态，首问不用等懒加载

# 会话历史：落盘到 sessions/<sid>.json，服务重启不丢（v0.4.1 之前是内存态）
SESSIONS_DIR = ROOT / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")   # sid 只留文件名安全字符，防路径穿越
_MAX_SESSION_FILES = 200                  # 会话文件上限，超了删最旧的
_MAX_TURNS = 40                           # 单会话保留轮数（改写窗口只需 6 条）


def _session_path(sid: str) -> Path:
    """会话文件路径（v0.6.1: 安全债已偿——sid 是浏览器来的不可信输入）。

    三层清洗: _UNSAFE 剥掉非 [A-Za-z0-9_-] 字符（"../../.env" 穿越弹头
    被拆成残渣）→ 截断 64 → 空串兜底（全符号 sid 清洗后为空，落 _anon）。
    穿越攻击的终点从"任意路径"收敛为"本目录内的安全文件名"。
    """
    clean = _UNSAFE.sub("", sid)[:64]
    return SESSIONS_DIR / f"{clean or '_anon'}.json"


def _load_history(sid: str) -> list[dict]:
    """读会话历史；新会话或文件损坏（如断电留下半个 JSON）都返回 []。

    只捕这两个异常：文件不存在是新会话的正常路径；JSON 解析失败是
    损坏数据。其余异常（权限/磁盘）照常抛出——静默吞掉会掩盖真故障。
    """
    try:
        return json.loads(_session_path(sid).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_history(sid: str, history: list[dict]) -> None:
    """写会话历史（最小版：utf-8 + ensure_ascii=False，未做数量防泄漏）。"""
    _session_path(sid).write_text(
        json.dumps(history, ensure_ascii=False), encoding="utf-8")


# ---------- 同步业务逻辑（在线程池里跑，不阻塞事件循环） ----------

def _prep_chat(data: dict):
    """问答前半程：改写 -> 检索 -> 拒答闸 -> 历史登记用户回合（流式/非流式共用）。"""
    sid = str(data.get("session_id") or "")[:64]
    question = str(data.get("question") or "").strip()
    if not sid or not question:
        raise ValueError("缺少 session_id 或 question")
    history = _load_history(sid)
    standalone = rewrite(question, history)  # rewriter 自带降级：失败返回原问题
    hits = retriever.search(standalone, top_k=settings.top_k)
    refused = (not hits) or hits[0]["score"] < settings.sim_threshold
    history.append({"role": "user", "content": question})
    return sid, question, standalone, hits, refused, history


def _finish_chat(sid: str, history: list[dict], reply: str) -> None:
    """问答后半程：登记助手回合 + 落盘（流式收完增量后也走这里）。"""
    history.append({"role": "assistant", "content": reply})
    del history[:-_MAX_TURNS]
    _save_history(sid, history)


def _hit_meta(question: str, standalone: str, hits: list[dict], refused: bool) -> dict:
    """返回给前端的元信息：拒答标志 / 改写透明化 / 引用列表。"""
    return {
        "refused": refused,
        "rewrite": None if standalone == question else f"{question} → {standalone}",
        "hits": [] if refused else [
            {"source": h["source"], "chunk_index": h["chunk_index"],
             "score": round(h["score"], 3)}
            for h in hits[: settings.top_k]
        ],
    }


def _handle_chat(data: dict) -> dict:
    """非流式端点逻辑（curl 调试 / 兼容旧前端用）。"""
    sid, question, standalone, hits, refused, history = _prep_chat(data)
    reply = "知识库中没有相关内容。" if refused else answer(standalone, hits)
    _finish_chat(sid, history, reply)
    return _hit_meta(question, standalone, hits, refused) | {"answer": reply}


def _handle_agent(data: dict) -> dict:
    """v0.6 混合路由端点: 规划 → (fact | agg+反思) → 带 Routing 元信息的完整回答。

    与 _handle_chat 的三点差异（都是有意为之）:
      1. 不做 rewrite——规划器自己做分解，两级改写叠一起反而互相干扰;
      2. 非流式——聚合路径要跑完整扫描+综合（30~60s），SSE 逐字流没有意义;
      3. 多返回 route / sub_queries / contexts——前端展示路由透明化。
    历史照常落盘，与 v0.5 路径共享同一会话，追问上下文不丢。
    """
    sid = str(data.get("session_id") or "")[:64]
    question = str(data.get("question") or "").strip()
    if not sid or not question:
        raise ValueError("缺少 session_id 或 question")
    history = _load_history(sid)
    history.append({"role": "user", "content": question})
    result = answer_agent(question)
    reply = result["answer"]
    _finish_chat(sid, history, reply)
    return {
        "answer": reply,
        "route": result["route"],
        "sub_queries": result.get("sub_queries") or [],
        "contexts": len(result.get("contexts") or []),
        "refused": "没有相关内容" in reply and len(reply) < 30,
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


async def _sse(send, event: str, payload: dict) -> None:
    """SSE 帧：event 行 + data 行 + 空行（帧分隔符是 \n\n）。"""
    frame = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    await send({"type": "http.response.body",
                "body": frame.encode("utf-8"), "more_body": True})


_DONE = object()  # 生成器耗尽哨兵（干什么用的？见 _chat_stream 坑2）


async def _chat_stream(send, data: dict) -> None:
    """SSE 端点主体：meta（改写+引用）先行 -> delta 逐段推 -> done 收尾。

    帧契约与三个坑的答案（教学版注释，对照 60dcd7a 参考）：
      坑1: prep/finish 走 run_in_executor 进线程池，事件循环不卡；
      坑2: run_in_executor(None, next, gen, _DONE) 的哨兵默认值让生成器
           耗尽时平安返回哨兵而非抛 StopIteration——它跨线程抛出会炸（PEP 479）；
      坑3: 头已发出无法改 500，降级为 error 帧（前端统一处理）。
    """
    loop = asyncio.get_running_loop()
    await send({"type": "http.response.start", "status": 200, "headers": [
        [b"content-type", b"text/event-stream; charset=utf-8"],
        [b"cache-control", b"no-cache"],
    ]})
    try:
        sid, question, standalone, hits, refused, history = await loop.run_in_executor(
            None, _prep_chat, data)
        meta = _hit_meta(question, standalone, hits, refused)
        if refused:
            await _sse(send, "meta", {**meta, "answer": "知识库中没有相关内容。"})
            await loop.run_in_executor(
                None, _finish_chat, sid, history, "知识库中没有相关内容。")
        else:
            await _sse(send, "meta", meta)
            gen, parts = answer_stream(standalone, hits), []
            while True:
                delta = await loop.run_in_executor(None, next, gen, _DONE)
                if delta is _DONE:
                    break
                parts.append(delta)
                await _sse(send, "delta", {"text": delta})
            full = "".join(parts) or "（模型未返回内容）"
            await loop.run_in_executor(None, _finish_chat, sid, history, full)
        await _sse(send, "done", {})
    except Exception as e:
        await _sse(send, "error", {"message": f"{type(e).__name__}: {str(e)[:200]}"})
    await send({"type": "http.response.body", "body": b""})  # 收官帧（more_body 缺省 False）


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

        elif path == "/api/chat/stream" and method == "POST":
            data = json.loads(await _read_body(receive) or b"{}")
            await _chat_stream(send, data)

        elif path == "/api/chat" and method == "POST":
            data = json.loads(await _read_body(receive) or b"{}")
            # 同步逻辑（网络调用+检索）丢进默认线程池，事件循环保持畅通
            result = await asyncio.get_running_loop().run_in_executor(
                None, _handle_chat, data)
            await _json(send, 200, result)

        elif path == "/api/agent" and method == "POST":
            data = json.loads(await _read_body(receive) or b"{}")
            result = await asyncio.get_running_loop().run_in_executor(
                None, _handle_agent, data)
            await _json(send, 200, result)

        elif path == "/api/ingest" and method == "POST":
            result = await asyncio.get_running_loop().run_in_executor(
                None, _handle_ingest)
            await _json(send, 200, result)

        elif path == "/api/reset" and method == "POST":
            data = json.loads(await _read_body(receive) or b"{}")
            _session_path(str(data.get("session_id") or "")[:64]).unlink(missing_ok=True)
            await _json(send, 200, {"ok": True})

        else:
            await _json(send, 404, {"error": f"no route: {method} {path}"})

    except Exception as e:
        # 最外层兜底：任何异常都变成 500 JSON，浏览器端能看到错误而不是断连
        await _json(send, 500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"RAG 文档问答服务已启动: http://127.0.0.1:{port}")
    print("（会话历史落盘 sessions/，重启服务不丢；Ctrl+C 停止）")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
