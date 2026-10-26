"""RAG 问答智能体核心包。

模块分工:
- config.py    配置加载与双提供商切换（已实现）
- llm.py       chat / embed 统一调用层（已实现）
- loader.py    PDF/Word/MD/TXT 文档加载（已实现）
- chunker.py   切块（待实现，文件顶部有设计要点）
- retriever.py 向量库写入与检索（待实现，文件顶部有设计要点）
- generator.py Prompt 组装与回答（待实现，文件顶部有设计要点）
"""
