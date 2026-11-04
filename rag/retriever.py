"""检索器：v0.1 纯向量检索（当前可用） + v0.2 混合检索（待实现）。

【当前状态】v0.2 已完成：分词 / BM25 / RRF / 混合编排 + 云端精排全部就位。
（阶段 1-3 由学习者独立实现并经 review 修正；阶段 4 的修复对照见 git 历史。）

【v0.2 目标流水线】
    query
      ├─ 向量通道: embed → Chroma(cosine) → top RECALL_N      懂"同义改写"
      └─ BM25 通道: 分词 → 词频打分 → top RECALL_N             懂"序列号/型号/低频词"
           ↓ RRF 融合（只用排名，免疫两路分数量纲差异）
    前 RERANK_CANDIDATES 名 → rag/rerank.py 精排（已实现，cross-encoder）
           ↓
    最终 top_k，score = 相关度(0~1)；rerank 失败降级为纯向量结果

【设计要点】（实现前想清楚）
1. 为什么要混合：向量检索对专有名词、编号、低频词弱（embedding 把它们
   碾成一团语义），BM25 恰好相反；两者互补，漏召回显著减少。
2. 为什么用 RRF 而不是分数相加：cosine ∈ [0,1]，BM25 ∈ [0,+∞)，量纲不同，
   直接相加会被 BM25 的大分数主导；RRF 只看排名，天然免疫。
3. rerank 只精排融合后前几名：cross-encoder 每对 (query, doc) 都要现算，
   全库精排又慢又贵；"粗召回小集合 → 精排"是工程定式。rag/rerank.py
   已实现（含失败降级），你只需要在 search() 里正确地调用它。
4. 分词不要用 jieba：PyPI 上无 wheel（只有源码包），这台机器 pip 编译
   必卡死（已实测两次）。零依赖方案：CJK 字符二元组（Lucene CJKAnalyzer
   同款思路）+ ASCII 连续段整词（"X1""12V" 天然成词）。
5. BM25 语料从哪来：不要维护第二份文档副本——首次检索时从 Chroma 把
   全部文档拉出来在内存里建索引（单一数据源），进程内缓存。
6. 降级路径：rerank 调用失败（返回 None）时，退回纯向量 top_k，
   千万别让精排故障拖垮整个问答。

参考实现在 git 历史：git show bcf53e9:rag/retriever.py
（建议先自己写，卡住了再对照。）
"""

from __future__ import annotations

import math
import re
from collections import Counter

import chromadb

from .config import get_settings
from .llm import embed
from .rerank import rerank  # 阶段 4 才会用到；已实现，直接调用即可


CJK = r'\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af'
PATTERN = re.compile(rf'([{CJK}]+)|([A-Za-z0-9]+)')

# ============================================================
# 阶段 1: 中文分词（零依赖）
# ============================================================

def _tokenize(text: str) -> list[str]:
    """把文本切成词项列表（BM25 的文档和查询必须共用同一套分词器）。

    建议方案（别装 jieba，见文件头设计点 4）：
      - text.lower() 归一化后，用两个正则各取所需：
          ASCII 连续段  r"[a-z0-9]+"          -> 整词（型号/编号天然成词）
          CJK 连续段    r"[\u4e00-\u9fff]+"   -> 段内切字符二元组
      - 例: "音箱X1的输入功率12V"
            -> ['音箱', 'x1', '的输', '输入', '入功', '功率', '12v']
      - 注意 CJK 段长度为 1 时取不出二元组，自然跳过即可

    自测（纯函数，不花 API 钱）:
        .venv\\Scripts\\python -c "from rag.retriever import _tokenize; print(_tokenize('音箱X1的输入功率12V'))"
    """
    text = text.lower()
    tokens = []
    for m in PATTERN.finditer(text):
        cjk_seg,ascii_seg = m.group(1) , m.group(2)
        if ascii_seg:
            tokens.append(ascii_seg)
        elif cjk_seg:
            if len(cjk_seg)==1:
                tokens.append(cjk_seg)
            else:
                tokens.extend(cjk_seg[i:i+2]for i in range(len(cjk_seg)-1))

    return tokens


# ============================================================
# 阶段 2: BM25 打分（教科书公式手写，约 30 行）
# ============================================================

class _BM25:
    """Okapi BM25。语料是"已分词的文档列表的列表"。

    打分公式（对 query 里每个词 t 累加到文档 d 上）:
        idf(t)  = ln( (N - df + 0.5) / (df + 0.5) + 1 )
        score   = Σ idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1-b+b·|d|/avgdl))

    参数直觉:
        k1 = 1.5    tf 饱和速度：词出现 10 次不比 3 次相关多少
        b  = 0.75   长度惩罚：长文档词多，但未必更相关
        N            文档总数；df(t) 含 t 的文档数；tf(t,d) t 在 d 中出现次数
        |d|          文档 d 的词数；avgdl 全库平均文档长度

    构建索引只需存三样：每篇文档的词频表（collections.Counter 很顺手）、
    每篇长度、每个词的 df。查询时对 query 的每个词逐篇累加分数。

    自测（迷你语料）:
        .venv\\Scripts\\python -c "from rag.retriever import _BM25; b = _BM25([['保修','保修','序列号'], ['颜色','重量']]); print(b.get_scores(['保修']))"
        期望: 第 0 篇为正分，第 1 篇为 0.0
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.tfs = [Counter(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in corpus]
        self.N = len(corpus)
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        self.b = b
        self.k1 = k1
        self.df:dict[str,int]={}
        for tf in self.tfs:
            for term in tf:
                self.df[term]=self.df.get(term,0)+1

        self.idf: dict[str,float]={}
        for term,freq in self.df.items():
            self.idf[term]=math.log(
                1+(self.N-freq+0.5)/(freq+0.5)
            )




    def get_scores(self, query: list[str]) -> list[float]:
        """返回 query 对语料中每篇文档的 BM25 分数（列表与 corpus 等长）。"""
        k1, b, avgdl = self.k1, self.b, self.avgdl

        # 不去重：查询词重复出现时贡献按次数计（教科书 / rank_bm25 的行为）
        terms = []
        for t in query:
            idf = self.idf.get(t)
            if idf is not None:
                terms.append((t, idf))

        scores=[]
        for i in range(self.N):
            tf = self.tfs[i]
            dl =self.doc_len[i]
            norm =k1*(1-b+b*dl/avgdl)
            s=0.0
            for term,idf in terms:
                f=tf.get(term,0)
                if f:
                    s+=idf*(f*(k1+1))/(f+norm)
            scores.append(s)
        return scores




# ============================================================
# 阶段 3: RRF 融合
# ============================================================

def _rrf_fuse(vec_hits: list[dict], bm_hits: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion：把两路命中列表融成一个有序列表。

    公式: 每个文档 score = Σ 1/(k + rank_i)，k=60 是社区常用值。
    文档唯一键建议用 (source, chunk_index) 元组。
    双通道都命中的文档两项相加，自然排最前——这就是 RRF 的全部。

    自测:
        .venv\\Scripts\\python -c "from rag.retriever import _rrf_fuse; a=[{'source':'x','chunk_index':0,'score':0.9},{'source':'x','chunk_index':1,'score':0.8}]; b=[{'source':'x','chunk_index':2,'score':5},{'source':'x','chunk_index':0,'score':3}]; print([h['chunk_index'] for h in _rrf_fuse(a,b)])"
        期望: [0, 2, 1]（chunk0 双通道命中排第一；融合结果不依赖原始分数量纲）
    """
    fused = {}
    pool={}
    for hits in (vec_hits,bm_hits):
        for rank , h in enumerate(hits,start=1):
            key=(h["source"],h["chunk_index"])
            fused[key]=fused.get(key,0)+1.0/(k+rank)
            pool[key]=h

    ranked=sorted(fused.items(),key=lambda kv:kv[1],reverse=True)

    result=[]
    for key ,score in ranked:
        hit =dict(pool[key])
        hit["rrf"]=score
        result.append(hit)
    return result


# ============================================================
# 检索器本体
# ============================================================


class Retriever:
    def __init__(self) -> None:
        s = get_settings()
        s.db_dir.mkdir(parents=True, exist_ok=True)
        self.settings = s
        self.client = chromadb.PersistentClient(path=str(s.db_dir))
        self.collection = self.client.get_or_create_collection(
            name=s.collection, metadata={"hnsw:space": "cosine"})
        # BM25 懒加载缓存（阶段 4 的 _ensure_bm25 会用到）
        self._bm25: _BM25 | None = None
        self._bm25_loaded = False
        self._bm25_docs: list[str] = []
        self._bm25_metas: list[dict] = []

    # ---------- 对外接口 ----------

    def ingest(self, chunks: list[dict]) -> int:
        """把切块向量化后写入向量库，返回写入条数。"""
        ids = [f"{c['source']}#{c['chunk_index']}" for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
        vectors = embed(docs)
        self.collection.upsert(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
        print(f"[retriever] 已写入 {len(chunks)} 块（库内共 {self.collection.count()} 块）")
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """混合检索 + 精排，返回 [{"text","source","chunk_index","score"}, ...] 按分数降序。

        漏斗: 向量 recall_n + BM25 recall_n → RRF 融合 → 前 rerank_candidates 名
              → rerank 精排 → top_k（score = 相关度 0~1）
        rerank 失败（None）时降级为纯向量结果，score 仍是 cosine 量纲，
        与 SIM_THRESHOLD 的旧校准兼容，拒答闸门照常工作。
        """
        top_k = top_k or self.settings.top_k

        # ① 双通道粗召回（宽进：给精排留候选，用 recall_n 不是 top_k）
        vec_hits = self._vector_search(query, self.settings.recall_n)
        bm_hits = self._bm25_search(query, self.settings.recall_n)

        # ② RRF 融合，取前 rerank_candidates 名作为精排候选
        fused = _rrf_fuse(vec_hits, bm_hits)[: self.settings.rerank_candidates]

        # ③ 云端精排；ranked 为 None 说明 rerank 失败 -> 降级纯向量
        ranked = rerank(query, [h["text"] for h in fused], top_n=top_k)
        if ranked is None:
            return vec_hits[:top_k]

        # ④ r["index"] 指向传入的 fused 列表；展开保留元数据，score 换成精排相关度
        return [{**fused[r["index"]], "score": r["relevance_score"]} for r in ranked[:top_k]]

    # ---------- 向量通道（已完成） ----------

    def _vector_search(self, query: str, n: int) -> list[dict]:
        """向量通道：语义召回（cosine 相似度 = 1 - distance）。"""
        qvec = embed([query])[0]
        res = self.collection.query(
            query_embeddings=[qvec], n_results=n,
            include=["documents", "metadatas", "distances"])
        return [
            {"text": doc, "score": 1 - dist, **meta}
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    # ---------- BM25 通道 ----------

    def _ensure_bm25(self) -> None:
        """首次检索时从 Chroma 读取全部文档，在内存中构建 BM25 索引。

        - collection.get() 返回 {键: 平铺列表}（注意与 query() 的嵌套结构不同），
          documents[i] 与 metadatas[i] 一一对应
        - 空库时值可能是 None，用 or [] 兜底
        - _bm25_loaded 的语义是"建过索引（或确认过空库）"：无论空不空都置 True，
          否则空库时每次检索都白拉一遍 Chroma
        - 本地 Chroma 的 get() 基本不会失败；真失败就让它响亮地炸出来——
          静默吞掉会让 BM25 通道无声失效，比崩溃更难排查
        """
        if self._bm25_loaded:
            return
        data = self.collection.get(include=["documents", "metadatas"])
        documents = list(data.get("documents") or [])
        metadatas = list(data.get("metadatas") or [])
        # 槽位对号入座：docs 装正文，metas 装元数据（平行数组，靠下标关联）
        self._bm25_docs = documents
        self._bm25_metas = metadatas
        self._bm25 = _BM25([_tokenize(d) for d in documents]) if documents else None
        self._bm25_loaded = True

    def invalidate_bm25(self) -> None:
        """v0.4：入库新文档后调用，让下次检索重建 BM25 索引。

        CLI 场景（v0.1~v0.3）每次运行都是新进程，不存在此问题；
        Web 常驻服务里 ingest 之后必须调这个，否则 BM25 通道永远看不到新文档。
        """
        self._bm25 = None
        self._bm25_loaded = False
        self._bm25_docs = []
        self._bm25_metas = []


    def _bm25_search(self, query: str, n: int) -> list[dict]:
        """BM25 通道：精确词频召回（对编号/型号/专有名词友好）。

        返回结构与 _vector_search 一致。score 是 BM25 原始分（无上界），
        量纲和 cosine 不同没关系——后面 RRF 只用排名。
        只保留 score > 0 的文档：0 分文档不含任何查询词，没资格上榜
        （RRF 按位次投票，塞 0 分文档进去等于凭空送票，会污染融合）。
        """
        self._ensure_bm25()
        if self._bm25 is None:   # 检查索引本身；_bm25_loaded 只说明"建过"
            return []

        scores = self._bm25.get_scores(_tokenize(query))  # 查询必须过同一个分词器
        ranked = sorted(
            (i for i, s in enumerate(scores) if s > 0),
            key=lambda i: scores[i],
            reverse=True,
        )[:n]

        return [
            {"text": self._bm25_docs[i], "score": float(scores[i]), **self._bm25_metas[i]}
            for i in ranked
        ]