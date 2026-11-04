"""评估器 v0.5：LLM-as-judge，给生成的回答打分（忠实度 / 答案相关性）。

为什么不装 RAGAS：依赖链重（langchain/datasets 一大家子），这台机器 pip
必卡死（jieba / rank_bm25 两次实测教训）。judge 的本质是"固定提示词 +
解析输出"，手写 60 行即可——与手写 BM25 同一哲学：维度定义抄业界，
实现零依赖。

打分维度（对应 RAGAS 四件套，我们实现两个 judge + 两个免费指标）：
- faithfulness 忠实度（judge）    回答是否句句有出处 → 防幻觉的核心指标
- relevance    答案相关性（judge） 回答是否切题回答了问题
- context precision（免费）       已由 eval.py 的 hit@k / MRR 离线覆盖
- context recall（免费）          已由 expect_source 金标覆盖

设计要点：
1. judge 也用 temperature=0：评估要可复现，同一输入两次跑分差很大
   就说明量表定义不清，而不是模型有个性。
2. 输出必须是 JSON（{"score": 1-5, "reason": "一句话"}）：
   结构化输出让 eval.py 能聚合；reason 让分数可解释、可 debug。
3. 量表要有锚点定义（写进提示词，模型才不会乱打）：
   faithfulness: 5=全部有出处  3=大部分有出处、个别无据  1=基本编造
   relevance:    5=完全切题    3=沾边但没答到点上        1=答非所问
4. 解析要防御：模型可能加 ```json 围栏、前后客套话——用正则抠出
   第一个 {...} 段再 json.loads，失败返回 None（降级契约同 rerank）。
5. 一次 judge 失败 = 这一格记 None 并计数，不拖垮整个评估批次。

自测（骨架实现后）:
    .venv\\Scripts\\python -c "from rag.evaluator import judge_faithfulness as f; print(f('保修多久', '保修12个月。', ['本产品保修12个月。']))"
    期望: {"score": 5, "reason": "..."}（回答完全来自资料）
"""
from __future__ import annotations

import json
import re

from .llm import chat


def _parse_json_block(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象，失败返回 None。

    已实现（管道层）：模型输出常见形态是
        好的，评估结果如下：\\n```json\\n{"score": 5, ...}\\n```
    正则 r"\\{.*?\\}" 配 DOTALL 抠出花括号段再 loads。
    """
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _valid_score(obj: dict) -> int | None:
    """从 judge 输出里提取合法的 1~5 整数分；非法返回 None。

    防御点：json.loads 后 true/false 会变成 bool——而 bool 是 int 的子类，
    isinstance(True, int) 为 True！所以要先排除 bool，再验范围。
    """
    raw = obj.get("score")
    if isinstance(raw, bool):
        return None
    try:
        s = int(raw)
    except (TypeError, ValueError):
        return None
    return s if 1 <= s <= 5 else None


_FAITH_SYSTEM = (
    "你是RAG系统的质量评估员。判断\"回答\"是否严格基于给定的参考资料，"
    "逐句对照：回答中的每个事实是否都能在参考资料中找到依据。\n"
    "量表（score 取 1~5 的整数）：\n"
    "5 = 全部有出处，没有任何编造或外部信息\n"
    "4 = 基本有出处，仅措辞性引申\n"
    "3 = 大部分有出处，个别事实无依据\n"
    "2 = 少部分有出处，多数内容无依据\n"
    "1 = 基本编造，与资料无关\n"
    '只输出 JSON：{"score": 整数, "reason": "一句话依据"}，不要输出其他任何内容'
)

_RELEV_SYSTEM = (
    "你是问答系统的质量评估员。判断\"回答\"是否切题地回答了\"问题\""
    "（只看问答对本身，不引入外部知识）。\n"
    "量表（score 取 1~5 的整数）：\n"
    "5 = 完全切题，直接回答了问题\n"
    "4 = 切题，但略欠完整\n"
    "3 = 沾边，没有答到问题核心\n"
    "2 = 大部分跑题\n"
    "1 = 答非所问\n"
    '只输出 JSON：{"score": 整数, "reason": "一句话依据"}，不要输出其他任何内容'
)


def _judge(system: str, user: str) -> dict | None:
    """两个 judge 的公共骨架：调用 -> 解析 -> 校验，任何失败返回 None。"""
    try:
        out = chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0,  # 评估要可复现
        )
    except Exception as e:
        print(f"[evaluator] judge 调用失败: {type(e).__name__}")
        return None
    obj = _parse_json_block(out)
    if obj is None:
        return None
    score = _valid_score(obj)
    if score is None:
        return None
    return {"score": score, "reason": str(obj.get("reason", ""))[:100]}


def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> dict | None:
    """忠实度：回答是否句句有出处（对照检索片段判断，防幻觉）。"""
    ctx = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    user = f"问题：{question}\n\n参考资料：\n{ctx}\n\n回答：{answer}\n\n请评估回答的忠实度。"
    return _judge(_FAITH_SYSTEM, user)


def judge_relevance(question: str, answer: str) -> dict | None:
    """答案相关性：回答是否切题回答了问题（不看检索片段，只看问答对）。"""
    user = f"问题：{question}\n\n回答：{answer}\n\n请评估回答的相关性。"
    return _judge(_RELEV_SYSTEM, user)
