"""v0.6 混合路由 Agentic 检索（设计文档: docs/v0.6-设计.md）。

架构:
    question → 规划器(1次LLM, 路由+分解一体)
                 ├─ fact → search() 单发 → answer()          【v0.5 快路径】
                 └─ agg  → scan() 扫描执行器 → synthesize()   【慢路径, M2/M3】

分工（设计文档 §6）:
    规划器/接线 = 助手；扫描执行器 = 学习者（M2）；综合器 prompt = 学习者（M3）。
"""
from __future__ import annotations

import json

from web import retriever
from .config import Settings, get_settings
from .generator import answer
from .llm import chat
from .retriever import Retriever ,_rrf_fuse

MAX_SUBS = 8  # 设计 §3.2: 子查询数上限（预算）

# 类别/文档关键词 → 文件名：子查询命中关键词时整架搬回（浏览优先于搜索，设计 §3.2 v3）
CATEGORY_FILES = {
    "音箱": "星尘智能家居-智能音箱.md",     "灯具": "星尘智能家居-智能灯具.md",
    "插座": "星尘智能家居-智能插座.md",     "传感器": "星尘智能家居-智能传感器.md",
    "安防": "星尘智能家居-智能安防.md",     "影音": "星尘智能家居-智能影音.md",
    "对比": "星尘家族产品对比.md",          "家族": "星尘家族产品对比.md",
    "App": "星尘之声App常见问题.md",        "配网": "星尘之声App常见问题.md",
    "手册": "sample.md",                    "办公": "智能办公助手使用与运维知识库.docx",
}

# ---------------------------------------------------------------------------
# 规划器（助手实现, M1）
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
你是一个检索路由器。判断用户问题该走哪条检索路径，只输出严格 JSON，不要任何其他文字。

输出格式:
{"type": "fact", "sub_queries": []}
{"type": "agg", "sub_queries": ["子查询1", "子查询2", ...]}

判定规则:
- fact: 问题指向明确实体或单一事实（某产品的参数/保修/操作步骤/某文档内容）
- agg: 答案需要对多个产品/多个块做汇总、比较、计数、求极值
  （特征词: 最/最便宜/最重/最长、几款/多少款/共有、哪些、分别、之间）

agg 的 sub_queries 拆法: 按产品类别或文档逐个覆盖，确保并集能圈住候选全集。
可用类别: 智能音箱 / 智能灯具 / 智能插座 / 智能传感器 / 智能安防 / 智能影音，
另有文档: 星尘家族产品对比 / 星尘之声App常见问题 / 星尘音箱X1手册 / 智能办公助手知识库。
例: "支持NFC的产品有几款" →
    ["支持NFC一碰配对的智能音箱", "支持NFC一碰配对的智能安防", "支持NFC一碰配对的智能影音"]
"""


def plan(question: str, *, settings: Settings | None = None) -> "Plan":
    """路由+分解。健壮性铁律（设计 §3.1）: 任何失败 → 降级 fact，永不劣化。"""
    s = settings or get_settings()
    try:
        raw = chat(
            [{"role": "system", "content": _PLAN_SYSTEM},
             {"role": "user", "content": question}],
            settings=s, temperature=0.0,
        )
        # 容错: 模型可能包 ```json 围栏或夹带文字——取首个 { 到末个 } 之间
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        t = data.get("type")
        if t not in ("fact", "agg"):
            raise ValueError(f"type 非法: {t!r}")
        subs = [str(x) for x in (data.get("sub_queries") or [])][:MAX_SUBS]
        if t == "agg" and not subs:
            subs = [question]  # 拆不出来就退化为原问题单发扫
        print(f"[agent] 规划: {t}  子查询{len(subs)}条")
        return Plan(type=t, sub_queries=subs, raw=raw)
    except Exception as e:  # noqa: BLE001 —— 规划器死则退 v0.5
        print(f"[agent] 规划失败({type(e).__name__})，降级 fact")
        return Plan(type="fact", sub_queries=[])


class Plan:
    __slots__ = ("type", "sub_queries", "raw")

    def __init__(self, type: str, sub_queries: list[str], raw: str = ""):
        self.type = type
        self.sub_queries = sub_queries
        self.raw = raw


# ---------------------------------------------------------------------------
# 扫描执行器（学习者实现区, M2 —— 设计 §3.2）
# ---------------------------------------------------------------------------

def scan(sub_queries: list[str], *, settings: Settings | None = None) -> list[dict]:
    """逐子查询检索，累积候选池（Agentic 循环控制的本体）。

    输入: plan() 给出的子查询列表
    输出: 去重后的候选池 list[dict]，与 search() 的 hit 结构一致，按得分降序

    """
    s=settings or get_settings()
    retriever=_get_retriever(s)

    pool: dict[tuple[str,int],dict]={}
    barren=0
    for sq in sub_queries:
        matched = [fn for kw, fn in CATEGORY_FILES.items() if kw in sq]
        if matched:
            hits = retriever.get_by_source(matched[0])   # 浏览: 整架搬回（构造性全召回）
        else:
            vec = retriever._vector_search(sq, 16)       # 搜索: 语义路径兜底
            bm = retriever._bm25_search(sq, 16)
            hits = _rrf_fuse(vec, bm)[:20]
        added=0
        for h in hits:
            key=(h['source'],h['chunk_index'])
            if key not in pool:
                added+=1
                pool[key]=h
            else:
                pool[key]= h if h['score']>pool[key]['score'] else pool[key]
        barren = barren + 1 if added == 0 else 0
        if barren>=2 or len(pool)>=96:
            break

    return sorted(pool.values(),key=lambda h: h['score'],reverse=True)




# ---------------------------------------------------------------------------
# 综合器（学习者实现区, M3 —— 设计 §3.3）
# ---------------------------------------------------------------------------
AGG_SYSTEM_PROMPT = """\
你是一个严谨的文档统计助手。候选片段可能来自多个产品目录，请严格遵守：
1. 只依据下方提供的【候选片段】，不要使用片段之外的知识；
2. 先逐条列出满足条件的产品清单（型号 + 关键数字 + [n] 引用），一个不漏；
3. 清单列完之后，才基于完整清单做比较/计数，最后一句给出最终答案；
4. 片段不足以覆盖全集时，明确说明覆盖范围，不要用局部结果冒充全集结论；
5. 如果片段完全无法回答，只回复："知识库中没有相关内容。"
"""

def synthesize(question: str, pool: list[dict], *, settings: Settings | None = None) -> str:
    """聚合题专用生成: 先逐块列出满足条件的产品清单，再比较/计数，最后给结论。

    Prompt 骨架（在 generator.SYSTEM_PROMPT 基础上换 user 部分）:
        【候选片段】…同 build_prompt 的编号格式…
        【任务】先从片段中逐条列出满足条件的产品（型号+关键数字+引用），
        再基于完整清单比较/计数，最后一句给出最终答案。
    """
    blocks=[f"[{i}](来源：{h['source']}第{h['chunk_index']+1}块)\n{h['text']}" for i ,h in enumerate(pool,1)]

    context="\n\n".join(blocks)
    user= (
        "【候选片段】\n"
        f"{context}\n\n"
        "【任务】\n"
        f"{question}\n\n"
        "回答格式要求（两节，缺一不可）:\n"
        "一、满足条件的产品清单（逐条: 型号 + 关键数字 + 引用编号）\n"
        "二、最终答案（一句话，基于第一节清单得出）"
    )

    message=[
        {"role": "system", "content": AGG_SYSTEM_PROMPT},
        {"role": "user","content": user}
    ]
    return chat(message,temperature=0.0)



# ---------------------------------------------------------------------------
# 实体一致性闸门（助手实现, M4 —— 设计 §3.4）
# ---------------------------------------------------------------------------

_ENTITY_SYSTEM = """\
你是检索结果的把关员。判断给出的检索片段【能否用来直接回答】用户问题。
只输出"能"或"不能"两个字，不要任何其他文字。

判定标准（两条都满足才算"能"）:
1. 片段讨论的主体与问题所问的主体是同一产品/文档（配套关系也算:
   问音箱的手机支持，配套 App 文档 = 能）;
2. 片段里含有与问题直接相关的信息（参数/步骤/政策），不是仅仅话题沾边。

反例（要记住）: 问"音箱可以贴墙安装吗"，检索到影音产品（幕光/声场）的
参数表——主体不同（影音≠音箱）且没有音箱贴墙信息，答"不能"，
哪怕字面都提到"安装/墙/音响设备"也不行。

同产品家族不同型号算一致（问"音箱"，检索到"星尘音箱 X1"=能）。
"""

REFUSAL_ANSWER = "知识库中没有相关内容。"


def _entity_gate(question: str, hits: list[dict], *, settings: Settings | None = None) -> bool:
    """充分性闸门（M4, 由实体闸门升级）: True=放行生成，False=拦截拒答。

    陷阱预警（写给自己和后来人）: 判断返回值时不能用 `"能" in reply`——
    "不**能**"里也包含"能"，子串会误判。先判"不能"，才是唯一安全写法。
    """
    s = settings or get_settings()
    sources = "\n".join(
        f"[{i}] {h['source']} 第{h['chunk_index']+1}块: {h['text'][:150].strip()}"
        for i, h in enumerate(hits[:3], 1)
    )
    user = f"【问题】\n{question}\n\n【检索到的片段】\n{sources}"
    reply = chat(
        [{"role": "system", "content": _ENTITY_SYSTEM},
         {"role": "user", "content": user}],
        settings=s, temperature=0.0,
    )
    refuse = "不能" in reply            # 先判"不能"——见 docstring 陷阱
    print(f"[agent] 闸门: {'拦截' if refuse else '放行'}")
    return not refuse


# ---------------------------------------------------------------------------
# 主入口（助手实现）
# ---------------------------------------------------------------------------

_retriever: Retriever | None = None


def _get_retriever(settings: Settings) -> Retriever:
    """模块级懒加载单例: BM25 构建有成本，一次建好反复用。

    注意: Retriever() 自己读环境配置（不收参数），settings 仅用于判空触发懒加载。
    """
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _fact_path(question: str, r: Retriever, s: Settings, route: str = "fact") -> dict:
    """单发快路径: 检索 → 分数闸 → 充分性闸 → 生成。agg 兜底时复用（route 打标区分）。"""
    hits = r.search(question)
    top = hits[0]["score"] if hits else 0.0
    if top < s.sim_threshold:          # 闸门一: 分数（v0.5 遗产）
        return {"answer": REFUSAL_ANSWER, "route": "fact-拒(分数)",
                "sub_queries": [], "contexts": [h["text"] for h in hits]}
    if not _entity_gate(question, hits, settings=s):   # 闸门二: 充分性（M4）
        return {"answer": REFUSAL_ANSWER, "route": "fact-拒(闸门)",
                "sub_queries": [], "contexts": [h["text"] for h in hits]}
    return {"answer": answer(question, hits), "route": route,
            "sub_queries": [], "contexts": [h["text"] for h in hits]}


def answer_agent(question: str, *, settings: Settings | None = None,
                 retriever: Retriever | None = None) -> dict:
    """混合路由主入口。返回 {answer, route, sub_queries, contexts}。

    contexts 一并返回是为了 eval 的 judge 能拿到与生成时一致的片段。
    聚合路径拒答时自动退化单发兜底（belt and suspenders）——
    子查询落不进类别货架的漏网之鱼，由 fact 路径接住。
    """
    s = settings or get_settings()
    p = plan(question, settings=s)
    r = retriever or _get_retriever(s)

    if p.type == "fact":
        return _fact_path(question, r, s)

    pool = scan(p.sub_queries, settings=s)
    ans = synthesize(question, pool, settings=s)
    pool_keys = [f"{h['source']}#chunk{h['chunk_index']}" for h in pool]
    if "没有相关内容" in ans and len(ans) < 30:      # 聚合路径答不动 → 单发兜底
        print("[agent] 聚合路径拒答，退化单发兜底")
        result = _fact_path(question, r, s, route="fact(agg兜底)")
        result["sub_queries"] = p.sub_queries
        result["pool_keys"] = pool_keys
        return result
    return {"answer": ans, "route": "agg",
            "sub_queries": p.sub_queries,
            "contexts": [h["text"] for h in pool],
            "pool_keys": pool_keys}
