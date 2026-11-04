# RAG 问答智能体

文档问答系统（v0.3）：多轮对话 + 查询改写（指代消解），底层为混合检索 + 云端精排，接阿里云百炼**业务空间专属端点**（OpenAI 兼容协议）。

## 多轮对话（v0.3）

```
用户问题 ──> rewrite()：结合最近 6 条历史，把"那序列号在哪里？"改写成
             "星尘音箱X1的序列号在哪里？"（首轮免改写，temperature=0，
             失败自动退回原问题——改写是增强不是依赖）
                ↓ 独立问题
           v0.2 检索管线（混合召回 → RRF → 精排）
                ↓
           带引用回答，原始问答计入历史
```

改写三铁律：只输出问题本身 / 已独立完整则原样返回 / 只用历史里出现过的信息（不许编造——保证拒答闸不被无关问题的"改写"骗开）。

## 检索架构（v0.2）

```
query
  ├─ 向量通道: embed → Chroma(cosine) → top 10     懂"同义改写"
  └─ BM25 通道: 二元组分词 → 词频打分 → top 10      懂"序列号/型号/低频词"
       ↓ RRF 融合（只用排名，免疫量纲差异）
  前 8 名 → qwen3.7-text-rerank 精排（cross-encoder）
       ↓
  最终 top 5 → score = 相关度(0~1) → 阈值拦截 → LLM 生成带引用回答
```

两道防幻觉闸门：① 检索分低于 `SIM_THRESHOLD` 直接拒答；② 系统提示词要求模型在片段不足时明确说"没有相关内容"。rerank 不可用时自动降级为纯向量检索，服务不中断。

实测分数分布（校准工具 `scripts/calibrate.py`）：相关问题 0.82~0.91，半相关 0.67，无关 ≤ 0.02——分离度约 40 倍，阈值 0.3 落在真空带内。

## 项目结构

```
rag-agent/
├── data/                # 文档放这里（md/txt/pdf/docx，含子目录）
├── db/chroma/           # 向量库持久化（ingest 自动创建，可随时删除重建）
├── rag/
│   ├── config.py        # .env 配置加载、双提供商切换
│   ├── llm.py           # chat / embed 统一调用层（openai SDK）
│   ├── loader.py        # PDF/Word/MD/TXT 加载
│   ├── chunker.py       # 切块：标题分界 + 段落合并 + 滑窗兜底
│   ├── retriever.py     # 混合检索：向量+BM25+RRF+精排（含手写 BM25、零依赖分词）
│   ├── rerank.py        # 云端 rerank 精排（DashScope 原生协议，失败自动降级）
│   ├── rewriter.py      # 查询改写：多轮指代消解（失败退回原问题）
│   └── generator.py     # Prompt 组装与回答
├── scripts/
│   ├── check_env.py     # 环境自检（版本/依赖/配置/API 连通性）
│   ├── probe_qwen.py    # 探测工作空间可用模型（对话模型候选实测）
│   ├── probe_rerank.py  # 探测 rerank 接口路径
│   ├── ask_once.py      # 单发一问（显示命中块与分数，便于调试）
│   ├── calibrate.py     # 阈值校准：批量问题只看检索分，不调 LLM
│   └── test_multiturn.py # v0.3 验收：四轮剧本测改写、指代消解与拒答闸
├── ingest.py            # 入库：加载 -> 切块 -> 向量化 -> 入库
├── chat.py              # 交互式问答
├── .env                 # 密钥与参数（已被 .gitignore 排除）
└── .env.example         # 配置模板
```

## 使用

```powershell
# 配置：编辑 .env 填 API key（专属端点 key 必须配专属域名，见下方注意事项）
.venv\Scripts\python scripts\check_env.py     # 自检，四项全过
.venv\Scripts\python ingest.py                # 入库（data/ 加了新文档后重跑）
.venv\Scripts\python chat.py                  # 交互问答（多轮，自动改写）
.venv\Scripts\python scripts\ask_once.py "保修多久"   # 单发一问
.venv\Scripts\python scripts\test_multiturn.py        # 多轮改写验收剧本
```

## 注意事项

- **key 与域名必须配套**：业务空间专属 key（`ws-xxx.maas.aliyuncs.com`）调公共域名会 401。
  专属域名的 OpenAI 兼容地址是 `https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1`，
  rerank 走原生路径 `/api/v1/services/rerank/text-rerank/text-rerank`。
- **换提供商/换 embedding 模型后必须重新 ingest**：向量空间不兼容，删 `db/chroma/` 重跑 `ingest.py`。
- **对话/向量模型名以工作空间实际部署为准**：用 `scripts/probe_qwen.py` 实测（当前 qwen3.7-flash / qwen3.7-text-embedding-flash）。
- **阈值已按实测校准**（0.3，落在无关/相关的分数真空带），换了文档规模后可用 `scripts/calibrate.py` 复查。
- **扫描版 PDF 提不出文字**：`loader.py` 会警告，需要 OCR，属于后续迭代。

## 迭代路线

| 版本 | 内容 | 状态 |
|------|------|------|
| v0.1 | 切块 + 向量检索 + 带引用回答 + 拒答 | ✅ 完成 |
| v0.2 | BM25+向量混合召回、RRF 融合、云端 rerank 精排 | ✅ 完成 |
| v0.3 | 查询改写 + 多轮对话（指代消解） | ✅ 完成 |
| v0.4 | Web UI（Streamlit 或 FastAPI + 前端） | |
| v0.5 | 评估集 + RAGAS 指标（忠实度/相关性/上下文精度） | |
