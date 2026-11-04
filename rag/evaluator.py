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


def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> dict | None:
    """忠实度：回答是否句句有出处（对照检索片段判断，防幻觉）。

    Args:
        question: 用户问题（给 judge 背景用）。
        answer: 待评估的回答。
        contexts: 检索到的参考片段列表。

    Returns:
        {"score": 1~5 的整数, "reason": "一句话依据"}；
        解析失败或调用异常返回 None（runner 记 judge_failed）。
    """
    # TODO(你来实现):
    #   1. system: 评估员人设 + 量表锚点（见设计要点 3）+ "只输出 JSON" 铁律
    #   2. user:   问题描述 + 参考资料全文 + 回答全文，要求逐句对照打分
    #   3. out = chat(messages, temperature=0)
    #   4. _parse_json_block(out) 解析；score 不在 1~5 整数范围内视为 None
    #   5. try/except 兜底返回 None（三条件：预期失败+留痕+有退路）
    raise NotImplementedError("judge_faithfulness 待实现 —— 思路见 docstring")


def judge_relevance(question: str, answer: str) -> dict | None:
    """答案相关性：回答是否切题回答了问题（不看检索片段，只看问答对）。

    Returns: 同 judge_faithfulness。
    """
    # TODO(你来实现):
    #   与 faithfulness 同构，区别：user 消息里没有参考资料，
    #   问的是"回答有没有答到问题的点上"
    raise NotImplementedError("judge_relevance 待实现 —— 思路见 docstring")
