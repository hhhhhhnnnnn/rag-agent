"""生成器：把「问题 + 检索片段」组装成 Prompt，让模型严格基于片段作答。

实现前建议想清楚的几个设计点：

1. 系统提示词要约束三件事（防幻觉的核心）：
   a. 只依据提供的片段回答，不要用自带知识补充；
   b. 回答中用 [1][2] 标注引用了哪个片段，方便人工核对；
   c. 片段不足以回答时，明确回复"知识库中没有相关内容"。

2. Prompt 结构（简单有效）：系统规则 + 编号片段 + 问题。
   片段编号 = 引用编号，让模型"抄近道"标注来源。

3. temperature 调低（约 0.2）：问答任务要稳定、忠实，不要发散创作。

4. 后续迭代方向：
   - stream=True 流式输出，改善体感；
   - 多轮对话时先用 LLM 做指代消解/查询改写，
     把"它保修多久"改写成"星尘音箱 X1 保修多久"再去检索。
"""

from __future__ import annotations

from .llm import chat

SYSTEM_PROMPT = """\
你是一个严谨的文档问答助手。请严格遵守：
1. 只依据下方提供的【参考片段】回答问题，不要使用片段之外的知识；
2. 回答中用 [1]、[2] 标注信息来自哪个片段；
3. 如果参考片段不足以回答问题，只回复："知识库中没有相关内容。"
"""


def build_prompt(question: str, hits: list[dict]) -> list[dict]:
    """组装 messages。hits 是 retriever.search() 的返回值。

    建议的 user 消息格式：
        【参考片段】
        [1] (来源: xxx.pdf)
        片段内容……
        [2] (来源: xxx.md)
        片段内容……

        【问题】
        用户的问题
    """
    # TODO(你来实现):
    #   1. 用 enumerate(hits, 1) 给片段编号并格式化
    #   2. 拼成一条 user 消息（片段 + 问题）
    #   3. 返回 [{"role": "system", "content": SYSTEM_PROMPT},
    #            {"role": "user", "content": ...}]
    raise NotImplementedError("generator.build_prompt 待实现 —— 格式建议见 docstring")


def answer(question: str, hits: list[dict]) -> str:
    """端到端：组装 prompt -> 调用 LLM -> 返回回答文本。"""
    return chat(build_prompt(question, hits), temperature=0.2)
