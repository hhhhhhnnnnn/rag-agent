# RAG 问答智能体（MVP）

文档问答最小可用版本：千问 / 智谱 GLM 双提供商（均为 OpenAI 兼容接口，改 `.env` 即可切换），
本地 Chroma 向量库，Python 3.12。

## 项目结构

```
rag-agent/
├── data/                # 文档放这里（md/txt/pdf/docx，含子目录）
│   └── sample.md        # 示例文档，正式使用时替换成你自己的
├── db/chroma/           # 向量库持久化（ingest 自动创建，可随时删除重建）
├── rag/
│   ├── config.py        # [已实现] .env 配置加载、双提供商切换
│   ├── llm.py           # [已实现] chat / embed 统一调用层
│   ├── loader.py        # [已实现] PDF/Word/MD/TXT 加载
│   ├── chunker.py       # [待实现] 切块 —— 文件顶部有完整设计要点
│   ├── retriever.py     # [待实现] 向量库写入与检索 —— 同上
│   └── generator.py     # [待实现] Prompt 组装 —— 同上
├── scripts/
│   └── check_env.py     # 环境自检（版本/依赖/配置/API 连通性）
├── ingest.py            # 入库入口：加载 -> 切块 -> 向量化 -> 入库
├── chat.py              # 问答入口：命令行多轮问答
├── .env                 # 你的密钥与参数（已被 .gitignore 排除）
└── .env.example         # 配置模板
```

## 已完成的环境初始化

- [x] 虚拟环境 `.venv`（Python 3.12.3）
- [x] 依赖安装（openai / python-dotenv / chromadb / pypdf / python-docx）
- [x] 双提供商底层打通（千问 DashScope、智谱 bigmodel，均走 openai SDK）

## 使用步骤

```powershell
# 1. 填 key：编辑 .env，把 QWEN_API_KEY / GLM_API_KEY 换成真实 key
#    （用哪家填哪家；想切到智谱把 LLM_PROVIDER=qwen 改成 glm）

# 2. 环境自检（四项全过说明底层就绪）
.venv\Scripts\python scripts\check_env.py

# 3. 实现 rag/ 三个核心模块：在 IDE 里全局搜索 TODO，
#    每个文件顶部都写好了设计要点和实现步骤提示

# 4. 入库（把 data/ 下的文档向量化）
.venv\Scripts\python ingest.py

# 5. 问答
.venv\Scripts\python chat.py
```

## 注意事项

- **换提供商/换 embedding 模型后必须重新 ingest**：不同模型的向量空间不兼容，
  删除 `db/chroma/` 再跑一次 `ingest.py` 即可。
- **SIM_THRESHOLD 需要实测校准**：`.env` 里的 0.3 只是起点，拿 10 个真实问题
  看检索分数分布再定。
- **扫描版 PDF 提不出文字**：`loader.py` 会警告，这类文档需要 OCR，属于后续迭代。
- 向量库数据可随时删除重建：`Remove-Item -Recurse db/chroma`。

## 迭代路线

| 版本 | 内容 | 备注 |
|------|------|------|
| v0.1 MVP | 切块 + 向量检索 + 带引用回答 + 拒答 | 当前 |
| v0.2 | BM25 + 向量混合召回，bge-reranker 精排 | 检索质量最大提升点 |
| v0.3 | 查询改写 + 多轮对话（指代消解） | Agentic RAG 起点 |
| v0.4 | Web UI（Streamlit 或 FastAPI + 前端） | |
| v0.5 | 评估集 + RAGAS 指标（忠实度/相关性/上下文精度） | 让迭代可量化 |
