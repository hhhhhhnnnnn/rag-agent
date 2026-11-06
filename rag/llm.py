"""模型调用层：统一的 chat / embed 接口，底层走 OpenAI 兼容协议。

千问与智谱都兼容 OpenAI 协议，所以只需要 openai 一个 SDK。
换模型 = 改 .env，代码零改动。
"""
from __future__ import annotations

from kubernetes.stream import stream
from openai import OpenAI
from openai.resources.beta.threads.runs import StepsWithStreamingResponse

from .config import Settings, get_settings

# embedding 单次请求批量上限：千问 text-embedding-v3 为 10 条，智谱为 64 条，取保守值
_EMBED_BATCH = 10


def _client(s: Settings) -> OpenAI:
    return OpenAI(api_key=s.api_key, base_url=s.base_url)


def chat(
    messages: list[dict],
    *,
    settings: Settings | None = None,
    temperature: float = 0.2,
) -> str:
    """调用对话模型，返回回复文本。

    messages 形如:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    s = settings or get_settings()
    resp = _client(s).chat.completions.create(
        model=s.chat_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def chat_stream(
    messages: list[dict],
    *,
    settings: Settings | None = None,
    temperature: float = 0.2,
):
    """流式版 chat：逐段 yield 增量文本，调用方拼接（核心由学习者实现）。

    与 chat() 的唯一结构差别：create(..., stream=True) 不等生成完毕，
    立即返回事件迭代器。守门条件跳过不带正文的事件（choices 为空 /
    content 为 None，如首事件只带 role、结束事件）。
    """
    s = settings or get_settings()
    stream = _client(s).chat.completions.create(
        model=s.chat_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        stream=True
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def embed(texts: list[str], *, settings: Settings | None = None) -> list[list[float]]:
    """把一批文本向量化，返回与输入顺序一致的向量列表（超长自动分批）。"""
    s = settings or get_settings()
    client = _client(s)
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        resp = client.embeddings.create(model=s.embed_model, input=batch)
        # 按 index 排序，保证与输入顺序一一对应
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out
