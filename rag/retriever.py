"""检索器：向量库的写入与查询。MVP 用 Chroma（本地持久化、零配置）。

实现前建议想清楚的几个设计点：

1. 距离度量：Chroma 默认用 L2，建议建集合时显式指定 cosine
   （metadata={"hnsw:space": "cosine"}），语义检索的分数更直观。

2. 写入要带齐元数据（source / chunk_index），查询时原样返回，
   "引用来源"功能的数据源头就在这里。ids 要唯一，建议 f"{source}#{chunk_index}"。

3. 重复入库问题：ingest 可重复运行，用 upsert 或每次先清空集合，
   否则同一段文字会存多份、检索结果重复。

4. 拒答阈值（settings.sim_threshold）：最高分都低于阈值，大概率知识库
   里没有答案，宁可说"没有相关内容"也不要让模型硬编。
   注意阈值必须拿真实问题实测校准，0.3 只是个起点。

5. top_k 不是越大越好：塞太多弱相关片段会稀释模型注意力、拉高成本。

6. v0.2 升级方向：BM25 + 向量混合召回，再接 bge-reranker 精排。
"""

from __future__ import annotations

import chromadb

from .config import get_settings
from .llm import embed


class Retriever:
    def __init__(self) -> None:
        s = get_settings()
        s.db_dir.mkdir(parents=True, exist_ok=True)
        self.settings = s
        self.client = chromadb.PersistentClient(path=str(s.db_dir))
        # TODO(建议): 创建集合时指定 cosine 距离，例如
        #   self.client.get_or_create_collection(
        #       name=s.collection, metadata={"hnsw:space": "cosine"})
        self.collection = self.client.get_or_create_collection(name=s.collection,metadata={"hnsw:space":"cosine"})

    def ingest(self, chunks: list[dict]) -> int:
        """把切块向量化后写入向量库，返回写入条数。"""
        # TODO:
        #   1. ids    = [f"{c['source']}#{c['chunk_index']}" for c in chunks]
        #   2. docs   = [c["text"] for c in chunks]
        #   3. metas  = [{"source": c["source"], "chunk_index": c["chunk_index"]} ...]
        #   4. vectors = embed(docs)
        #   5. self.collection.upsert(ids=..., documents=..., embeddings=..., metadatas=...)
        ids=[f"{c['source']}#{c['chunk_index']}"for c in chunks]
        docs=[c['text']for c in chunks]
        metas= [ {"source": c["source"],"chunk_index":c["chunk_index"]} for c in chunks]
        vectors=embed(docs)
        self.collection.upsert(ids=ids,documents=docs,embeddings=vectors,metadatas=metas)
        print(f"已添加{self.collection.count()}条文档到数据库")
        return len(chunks)

        # raise NotImplementedError("retriever.Retriever.ingest 待实现")

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """向量检索，返回 [{"text", "source", "chunk_index", "score"}, ...] 按分数降序。"""
        # TODO:
        #   1. qvec = embed([query])[0]
        #   2. res = self.collection.query(
        #          query_embeddings=[qvec], n_results=top_k or self.settings.top_k,
        #          include=["documents", "metadatas", "distances"])
        #   3. cosine 空间下 distance = 1 - similarity，
        #      换算 score = 1 - distance 后按 score 降序返回
        qvec=embed([query])[0]
        result=self.collection.query(
            query_embeddings=[qvec],
            n_results=top_k or self.settings.top_k,
            include=["documents","metadatas","distances"]
        )
        hits=[
            {"text":doc,"score":1-dist,**meta}
            for doc,meta,dist in zip(
                result["documents"][0],result["metadatas"][0],result["distances"][0]
            )
        ]

        return sorted(hits,key=lambda h : h["score"],reverse=True)
