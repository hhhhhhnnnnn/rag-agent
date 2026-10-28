"""检索器 v0.2：混合检索（向量 + BM25）→ RRF 融合 → 云端 rerank 精排。

流水线：
    query
      ├─ 向量通道: embed → Chroma(cosine) → top RECALL_N    懂"同义改写"
      └─ BM25 通道: 分词 → 词频统计 → top RECALL_N           懂"序列号/型号/低频词"
           ↓
    RRF 融合: score(d) = Σ 1/(60 + rank_i(d))    只用排名，免疫两路分数量纲差异
           ↓
    融合前 RERANK_CANDIDATES 名 → rerank 精排(cross-encoder)
           ↓
    最终 top_k，score = 相关度(0~1)；rerank 失败自动降级为纯向量结果

设计要点：
1. 为什么要混合：向量检索对专有名词、编号、低频词弱（embedding 把它们
   碾成一团语义），BM25 恰好相反；两者互补，漏召回显著减少。
2. 为什么用 RRF 而不是分数相加：cosine ∈ [0,1]，BM25 ∈ [0,+∞)，
   量纲不同，直接相加会被 BM25 的大分数主导；RRF 只看排名，天然免疫。
3. rerank 只精排融合后前几名：cross-encoder 每对 (query, doc) 都要
   现算，全库精排又慢又贵；粗召回小集合再精排是工程定式。
4. 降级设计：rerank 失败 → 返回纯向量 top_k，服务不中断。
5. BM25 索引在首次检索时从 Chroma 读取全部文档懒加载构建：
   Chroma 是唯一数据源，不维护第二份语料副本；缺点是入库后
   需要重启进程才能让 BM25 看到新文档（CLI 场景每次运行都是新进程，无影响）。
6. 中文分词不依赖 jieba（PyPI 上无 wheel，装一次卡一次）：
   CJK 走字符二元组（Lucene CJKAnalyzer 同款思路），ASCII 连续段
   整词处理（"X1""12V" 天然成词）——检索效果不差，零依赖。
"""

from __future__ import annotations

import math
import re
from collections import Counter

import chromadb

from .config import get_settings
from .llm import embed
from .rerank import rerank

# ---------- 中文检索分词（零依赖） ----------

_ASCII_RUN = re.compile(r"[a-z0-9]+")           # 字母/数字连续段（型号、编号）
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")      # 汉字连续段


def _tokenize(text: str) -> list[str]:
    """CJK 字符二元组 + ASCII 整词（文档与查询必须共用同一套分词器）。

    例: "音箱X1的输入功率12V"
        -> ['音箱', 'x1', '的输', '输入', '入功', '功率', '12v']
    二元组对"词语边界模糊"天然鲁棒，代价是索引略膨胀——检索场景划算。
    """
    text = text.lower()
    toks = _ASCII_RUN.findall(text)
    for seg in _CJK_RUN.findall(text):
        toks.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return toks


# ---------- BM25（教科书公式手写实现，替代 rank_bm25 库） ----------


class _BM25:
    """Okapi BM25。打分三要素：idf（词稀有度）× tf 饱和 × 文档长度归一。

    score(q, d) = Σ idf(t) · tf · (k1+1) / (tf + k1·(1-b+b·|d|/avgdl))
    - k1=1.5: tf 饱和速度（词出现 10 次不比 3 次相关多少）
    - b=0.75: 长度惩罚力度（长文档词多但未必更相关）
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        self.tfs: list[dict[str, int]] = [Counter(d) for d in corpus]
        self.df: dict[str, int] = {}
        for tf in self.tfs:
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for t in query:
            df = self.df.get(t)
            if not df:
                continue
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for i in range(self.n):
                tf = self.tfs[i].get(t)
                if not tf:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


# ---------- 检索器 ----------


class Retriever:
    def __init__(self) -> None:
        s = get_settings()
        s.db_dir.mkdir(parents=True, exist_ok=True)
        self.settings = s
        self.client = chromadb.PersistentClient(path=str(s.db_dir))
        self.collection = self.client.get_or_create_collection(
            name=s.collection, metadata={"hnsw:space": "cosine"})
        # BM25 懒加载缓存（首次检索时才构建）
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
        """混合检索 + 精排，返回 [{"text","source","chunk_index","score"}, ...] 按分数降序。"""
        top_k = top_k or self.settings.top_k

        # ① 双通道粗召回
        vec_hits = self._vector_search(query, self.settings.recall_n)
        bm_hits = self._bm25_search(query, self.settings.recall_n)

        # ② RRF 融合，取前 RERANK_CANDIDATES 名作为精排候选
        fused = self._rrf_fuse(vec_hits, bm_hits)[: self.settings.rerank_candidates]

        # ③ 云端精排；失败则降级为纯向量结果
        ranked = rerank(query, [h["text"] for h in fused], top_n=top_k)
        if ranked is None:
            return vec_hits[:top_k]
        return [{**fused[r["index"]], "score": r["relevance_score"]} for r in ranked[:top_k]]

    # ---------- 三条通道 ----------

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

    def _bm25_search(self, query: str, n: int) -> list[dict]:
        """BM25 通道：精确词频召回（对编号/型号/专有名词友好）。"""
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            {"text": self._bm25_docs[i], "score": float(scores[i]), **self._bm25_metas[i]}
            for i in order if scores[i] > 0
        ]

    def _rrf_fuse(self, vec_hits: list[dict], bm_hits: list[dict], k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion：score = Σ 1/(k + rank)，k=60 是社区常用值。"""
        scores: dict[tuple, float] = {}
        pool: dict[tuple, dict] = {}
        for hits in (vec_hits, bm_hits):
            for rank, h in enumerate(hits, start=1):
                key = (h["source"], h["chunk_index"])  # 文档唯一键
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                pool[key] = h
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [{**pool[key], "rrf": sc} for key, sc in ranked]

    # ---------- BM25 索引管理 ----------

    def _ensure_bm25(self) -> None:
        """首次检索时从 Chroma 读取全部文档，在内存中构建 BM25 索引。"""
        if self._bm25_loaded:
            return
        data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_docs = list(data.get("documents") or [])
        self._bm25_metas = list(data.get("metadatas") or [])
        self._bm25 = (
            _BM25([_tokenize(d) for d in self._bm25_docs])
            if self._bm25_docs else None
        )
        self._bm25_loaded = True
