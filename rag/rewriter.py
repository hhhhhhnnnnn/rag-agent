"""查询改写器 v0.3：把多轮对话里"依赖上下文的问题"改写成独立问题。

为什么需要：
    检索是无记忆的。向量库和 BM25 只认"独立成篇"的问题——
    "那序列号在哪里？"单独拿去检索，主语是谁没人知道，两个通道都会失灵。

位置（v0.3 数据流）：
    用户问题 ──(history 非空时)──> rewrite() ──> 独立问题 ──> search() ──> answer()
              └─(首轮，history 空)──> 原样返回，零成本

设计要点：
1. 首轮免改写：history 为空直接返回原问题。省一次 LLM 调用，
   也避免改写模型对已完整的问题画蛇添足。
2. 改写用 chat 模型 + temperature=0：改写要的是确定性映射，不是创意。
3. prompt 三条铁律（写进 system 消息，缺一条都会翻车）：
   a. 只输出改写后的问题本身——不要解释、不要引号、不要"改写后："前缀
   b. 问题已经独立完整时，原样返回
   c. 只能使用对话历史里出现的信息，不许编造
      （历史没提"防水"，就不能凭空把"防水"加进去）
4. 历史截断：只带最近 HISTORY_WINDOW 条消息。指代消解通常只看上一两轮，
   全量历史又贵又引入干扰。
5. 降级契约（比 rerank 的 None 还简单）：任何异常返回原问题。
   改写是增强不是依赖——坏了就当没有，绝不拖垮问答。
   回忆 v0.2 的三条件：预期失败 + 留痕 + 有退路，这里全占。
6. 模型输出要清洗：LLM 常自作主张加引号、句号或"问题："前缀，
   strip 掉这些再返回；清洗完是空串就退回原问题。

自测：
    # 离线（零成本）：首轮直通，应原样打印
    .venv\\Scripts\\python -c "from rag.rewriter import rewrite; print(rewrite('保修多久', []))"
    # 在线：指代消解，期望独立问题里出现"星尘音箱X1"（或等价主语）
    .venv\\Scripts\\python -c "from rag.rewriter import rewrite; h=[{'role':'user','content':'星尘音箱X1的保修期是多久？'},{'role':'assistant','content':'享受12个月免费保修。'}]; print(rewrite('那序列号在哪里？', h))"
"""
from __future__ import annotations

from .llm import chat

HISTORY_WINDOW = 6  # 带给改写模型的最近消息条数（防 token 膨胀 + 减少干扰）


def rewrite(question: str, history: list[dict]) -> str:
    """把 question 结合 history 改写成不依赖上下文的独立问题。

    Args:
        question: 用户最新一问（可能含指代/省略）。
        history: 之前的对话 [{"role": "user"|"assistant", "content": ...}, ...]，旧 -> 新。

    Returns:
        独立问题；首轮或任何失败返回原 question（降级契约）。
    """
    if not history:
        return question  # 已实现：首轮免改写，零成本

    # TODO(你来实现):
    #   1. 构造 messages: system(三条铁律) + history 最近 HISTORY_WINDOW 条 + user(question)
    #   2. out = chat(messages, temperature=0)
    #   3. 清洗输出（见设计要点 6）；清洗后为空就用原问题
    #   4. try/except 兜底：打印一行原因后 return question（三条件里的"留痕"）
    raise NotImplementedError("rewrite 待实现 —— 思路见 docstring")
