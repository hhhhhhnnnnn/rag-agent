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

from .llm import chat, chat_stream

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
    #raise NotImplementedError("generator.build_prompt 待实现 —— 格式建议见 docstring")

    blocks=[]
    for i , h in enumerate(hits,1):
        blocks.append(f"[{i}](来源:{h['source']}第{h['chunk_index']+1}块)\n{h['text']}")
    context="\n\n".join(blocks)

    user=f"用户的问题{question},参考资料：{context}"

    return [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": user}]



def answer(question: str, hits: list[dict]) -> str:
    """端到端：组装 prompt -> 调用 LLM -> 返回回答文本。"""
    return chat(build_prompt(question, hits), temperature=0.2)


def answer_stream(question: str, hits: list[dict]):
    """流式版 answer：逐段 yield 增量文本（SSE 用）。（你来实现）

    两行的事：prompt 组装复用 build_prompt()，底层换 chat_stream。
    Python 里"把一个生成器的产出原样转手"有个专用语法，想想是哪个。
    （提示：不是 for 循环里 yield——那也行，但啰嗦）

    验收:
        .venv\\Scripts\\python -c "from rag.retriever import Retriever; from rag.generator import answer_stream; r = Retriever(); hits = r.search('蓝牙版本', top_k=3); [print(p, end='', flush=True) for p in answer_stream('蓝牙版本', hits)]"
    """
    raise NotImplementedError("answer_stream 待实现 —— 提示见 docstring")
