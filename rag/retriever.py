"""检索器：v0.1 纯向量检索（当前可用） + v0.2 混合检索（待实现）。

【当前状态】
- 已完成：ingest / _vector_search / search(纯向量版)，系统端到端可用
- 待实现（文件内搜 TODO）：_tokenize / _BM25 / _rrf_fuse / _ensure_bm25 /
  _bm25_search，以及 search() 的混合编排改造

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

import chromadb

from .config import get_settings
from .llm import embed
from .rerank import rerank  # 阶段 4 才会用到；已实现，直接调用即可


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
    # TODO(你来实现):
    #   1. 转小写
    #   2. 两个 re.findall 各自收集
    #   3. CJK 段内滑窗取二元组
    raise NotImplementedError("_tokenize 待实现 —— 思路见 docstring")


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
        # TODO(你来实现):
        #   1. self.tfs = [Counter(doc) for doc in corpus]
        #   2. self.doc_len / self.avgdl 长度统计（空语料除零要防）
        #   3. self.df: 遍历所有 tf 表，统计每个词出现在几篇文档
        raise NotImplementedError("_BM25.__init__ 待实现")

    def get_scores(self, query: list[str]) -> list[float]:
        """返回 query 对语料中每篇文档的 BM25 分数（列表与 corpus 等长）。"""
        # TODO(你来实现):
        #   1. scores = [0.0] * 文档数
        #   2. 对 query 每个词: 查 df，全库都没有就 continue
        #   3. 算 idf，遍历含该词的文档按公式累加（不含该词的跳过）
        raise NotImplementedError("_BM25.get_scores 待实现")


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
    # TODO(你来实现):
    #   1. 两个字典: {键: 融合分} 与 {键: 原始 hit}（保留元数据用）
    #   2. 遍历两路 hits，rank 从 1 开始，累加 1/(k+rank)
    #   3. 按融合分降序输出 hit 列表
    raise NotImplementedError("_rrf_fuse 待实现")


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
        """检索，返回 [{"text","source","chunk_index","score"}, ...] 按分数降序。

        当前是 v0.1 纯向量版，系统保持可用。阶段 4 改造成目标结构:

            # ① 双通道粗召回（注意用 recall_n 不是 top_k）
            vec_hits = self._vector_search(query, self.settings.recall_n)
            bm_hits  = self._bm25_search(query, self.settings.recall_n)
            # ② RRF 融合，取前 rerank_candidates 名作为精排候选
            fused = _rrf_fuse(vec_hits, bm_hits)[: self.settings.rerank_candidates]
            # ③ 云端精排；ranked 为 None 说明 rerank 失败 -> 降级纯向量
            ranked = rerank(query, [h["text"] for h in fused], top_n=top_k)
            if ranked is None:
                return vec_hits[:top_k]
            # ④ 用 fused[r["index"]] 取回原 hit（元数据都在），
            #    score 换成 relevance_score，截断 top_k 返回
        """
        top_k = top_k or self.settings.top_k
        return self._vector_search(query, top_k)[:top_k]

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

    # ---------- BM25 通道（阶段 4） ----------

    def _ensure_bm25(self) -> None:
        """首次检索时从 Chroma 读取全部文档，在内存中构建 BM25 索引。

        提示:
          - self.collection.get(include=["documents", "metadatas"]) 拉全量
          - 返回 {键: 列表} 字典；空库时值可能是 None，注意兜底
          - 建好索引后置 self._bm25_loaded = True，之后直接复用（懒加载）
          - 空语料时让 self._bm25 保持 None（调用方会处理）
        """
        # TODO(你来实现)
        raise NotImplementedError("_ensure_bm25 待实现")

    def _bm25_search(self, query: str, n: int) -> list[dict]:
        """BM25 通道：精确词频召回（对编号/型号/专有名词友好）。

        返回结构与 _vector_search 一致。score 是 BM25 原始分（无上界），
        量纲和 cosine 不同没关系——后面 RRF 只用排名。
        只保留 score > 0 的文档。
        """
        # TODO(你来实现):
        #   1. 先 self._ensure_bm25()，索引为 None 就返回 []
        #   2. self._bm25.get_scores(_tokenize(query)) -> 按分数排序取前 n
        #   3. 用 self._bm25_docs[i] / self._bm25_metas[i] 组装 hit 字典
        raise NotImplementedError("_bm25_search 待实现")
