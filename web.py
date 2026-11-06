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
    """会话文件路径。（你来实现）

    三个要求，全是安全/健壮性细节：
    1. sid 来自浏览器，不可信——先用 _UNSAFE.sub('', sid) 清洗成安全字符
       （想想为什么不能用 os.path.join 时信任它：路径穿越 ../ 攻击）
    2. 清洗后可能为空串——兜底一个文件名，别让路径变成 sessions/.json
    3. 截断到 64 个字符，防怪长文件名

    验收:
        .venv\\Scripts\\python -c "import web; print(web._session_path('../../etc/passwd')); print(web._session_path(''))"
        两个都应该落在 sessions/ 目录内，且不含路径分隔符
    """
    raise NotImplementedError("_session_path 待实现 —— 要点见 docstring")


def _load_history(sid: str) -> list[dict]:
    """读会话历史；新会话或文件损坏都返回 []。（你来实现）

    想清楚要捕获哪【两个】异常：
    - 文件不存在（新会话的正常路径，不是错误）
    - JSON 解析失败（文件损坏——半个 JSON，比如上次写盘断电）
    其余异常（如权限）应该照常抛出——静默吞掉掩盖真问题。

    验收:
        .venv\\Scripts\\python -c "import web; print(web._load_history('不存在的会话'))"
        应打印 [] 且不报错
    """
    raise NotImplementedError("_load_history 待实现 —— 提示见 docstring")


def _save_history(sid: str, history: list[dict]) -> None:
    """写会话历史 + 超上限清最旧。（你来实现）

    1. 写文件：json.dumps(history, ensure_ascii=False)，utf-8 编码
    2. 防泄漏：glob 会话目录全部 *.json，按 mtime 倒序排列，
       第 _MAX_SESSION_FILES 个【之后】的 unlink(missing_ok=True)
       ——注意 missing_ok=True 是哪个 Python 版本才有的？本机 3.11+，放心用
    """
    raise NotImplementedError("_save_history 待实现 —— 要点见 docstring")


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
    """SSE 端点主体：本轮核心练习。（你来实现）

    目标帧序列（对照前端 web/index.html 的解析代码——那就是契约）：
        1) http.response.start：200 + content-type: text/event-stream; charset=utf-8
           + cache-control: no-cache。注意【不写 content-length】，
           且这条消息本身不带 body
        2) meta 帧（_sse 已提供）：拒答时 payload 里直接带 answer 拒答句，
           并 _finish_chat 落盘后直接 done；非拒答时 meta 只带改写+引用
        3) delta 帧若干：每拿到一段增量推一帧 {"text": 段}
        4) done 帧收尾，最后再发一个空 body 的收官帧（more_body 不传即 False）

    三个坑（先自己想，卡住看参考答案 git show 60dcd7a -- web.py）：
        坑1: _prep_chat/_finish_chat 是同步阻塞函数（网络+检索），
             直接在 async 里调用会卡死事件循环——老规矩，怎么办？
        坑2: answer_stream 是【同步生成器】。逐段取要 run_in_executor(None, next, gen, X)——
             第二个位置参数 X 是什么？为什么必须传？
             （提示：StopIteration 从线程池里抛出来会发生什么？查查 PEP 479）
        坑3: 流到一半抛异常时，response 头早发出去了，没法改 500——
             降级方案是什么？（前端已经会处理一种特殊帧）

    验收:
        重启服务后:
        $r = Invoke-WebRequest -UseBasicParsing -Method Post -Uri http://127.0.0.1:8000/api/chat/stream `
            -Headers @{"Content-Type"="application/json; charset=utf-8"} `
            -Body ([Text.Encoding]::UTF8.GetBytes('{"session_id":"mytest","question":"保修多久"}'))
        $r.Content 里应有 event: meta、多个 event: delta、一个 event: done
        再开浏览器 http://127.0.0.1:8000 看打字机效果
    """
    raise NotImplementedError("_chat_stream 待实现 —— 帧契约和三个坑见 docstring")


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
