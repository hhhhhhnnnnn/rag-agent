# RAG 问答智能体

文档问答系统（v0.6）：Agentic 混合路由 + 评估体系 + Web 服务，底层为混合检索 + 云端精排，接阿里云百炼**业务空间专属端点**（OpenAI 兼容协议）。

## Agentic 混合路由（v0.6）

```
question → 规划器（1次LLM：路由+分解一体，失败降级fact）
             ├─ fact → search() 单发 → 分数闸 → 充分性闸 → 生成     快路径
             └─ agg  → scan() 扫描执行器 → synthesize() 先列全集再答 慢路径
```

- **为什么需要**：121 块语料上，聚合题（"最便宜的双频设备""支持NFC的有几款"）单发全线翻车——top5 装不下候选全集、显式否定标签（"NFC：不支持"）反成最佳词汇匹配
- **浏览优先于搜索**：类别子查询按文件整取（`get_by_source`，构造性完美召回、零 embed），语义路径仅兜底——"读完整个货架"不该问搜索引擎
- **两道新闸门**：分数闸（v0.5 遗产补进 agent）+ 充分性闸（"片段能否直接回答"，治跨实体污染——音箱问题被影音产品的说明书骗走）
- **专属指标**：`--agent` 模式下聚合题带**池召回率**（候选池对金标集合的覆盖率）——单发 0/13 → v0.6 13/13
- **降级链**：规划失败 → fact 路径；聚合路径拒答 → 单发兜底。Agentic 层任何失败都退回 v0.5 行为，永不劣化
- 设计文档：`docs/v0.6-设计.md`（含三次规格修订的失败存档）

## Web 服务（v0.4）

```powershell
.venv\Scripts\python web.py     # http://127.0.0.1:8000（PORT 环境变量可改端口）
```

- **零新增依赖**：裸 ASGI + uvicorn（chromadb 依赖树自带），路由/JSON 手写——不赌本机的 pip
- 功能：多轮对话（历史落盘 `sessions/`，**服务重启不丢**）、**SSE 流式回答**（引用清单先行、逐字上屏）、改写透明展示、引用来源 chips、拒答样式、一键"重建索引"（入库后热重建 BM25，常驻进程专属能力）
- 同步业务逻辑走线程池（`run_in_executor`），事件循环不被检索/LLM 调用阻塞；流式增量同样在线程池逐段搬运
- 限制：仅绑 127.0.0.1 本地使用、无鉴权、流式中途断线该轮作废（不续传）

## 评估体系（v0.5）

零 RAGAS 依赖（pip 装不动大依赖链，judge 本质是固定提示词+解析输出，手写实现）：

| 指标 | 类型 | 说明 |
|------|------|------|
| 拒答准确率 | 免费 | 该拒的拒了 / 该答的答了 |
| hit@5 / MRR | 免费 | 金标块是否命中、排第几 |
| 关键词命中 | --full | 回答含期望关键词 |
| 忠实度 / 相关性 | --full | LLM-as-judge 打分 1~5（量表锚点 + 只输出 JSON） |

用法：`eval.py`（仅检索，零 LLM 成本）/ `eval.py --full`（全量）/ `eval.py --full --agent`（走 v0.6 混合路由，聚合题附池召回率）；报告落 `eval/reports/`（含设置快照 + git 版本）。**改参数前后各跑一次对比聚合分，从此优化从"感觉"变"数据"。**

当前基线（24 题 / 121 块语料）：单发 关键词 0.79（聚合题 0/5）→ v0.6 混合路由 关键词 0.89（聚合题 4/5，q24 池召回 13/13）。

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
│   ├── evaluator.py     # LLM-as-judge：忠实度/相关性打分（零 RAGAS 依赖）
│   ├── generator.py     # Prompt 组装与回答
│   └── agent.py         # v0.6 混合路由：规划器/扫描执行器/综合器/充分性闸门
├── scripts/
│   ├── check_env.py     # 环境自检（版本/依赖/配置/API 连通性）
│   ├── probe_qwen.py    # 探测工作空间可用模型（对话模型候选实测）
│   ├── probe_rerank.py  # 探测 rerank 接口路径
│   ├── ask_once.py      # 单发一问（显示命中块与分数，便于调试）
│   ├── calibrate.py     # 阈值校准：批量问题只看检索分，不调 LLM
│   ├── test_multiturn.py # v0.3 验收：四轮剧本测改写、指代消解与拒答闸
│   ├── eval.py          # v0.5 评估：问题集跑批，检索指标+judge，报告落盘（--agent 走 v0.6）
│   ├── gen_corpus.py    # 合成规模语料生成器（种子固定可复现，90款/6类）
│   └── dump_kb.py       # 打印库内全部切块（人工核对金标用）
├── eval/
│   ├── questions.json   # 评估问题集（金标：expect_source/keywords/should_refuse）
│   └── reports/         # 评估报告产物（gitignore）
├── ingest.py            # 入库：加载 -> 切块 -> 向量化 -> 入库
├── chat.py              # 交互式问答（CLI）
├── web.py               # Web 服务入口（裸 ASGI + uvicorn，零新增依赖）
├── web/
│   └── index.html       # 前端单页（原生 JS，零构建零 CDN）
├── .env                 # 密钥与参数（已被 .gitignore 排除）
└── .env.example         # 配置模板
```

## 使用

```powershell
# 配置：编辑 .env 填 API key（专属端点 key 必须配专属域名，见下方注意事项）
.venv\Scripts\python scripts\check_env.py     # 自检，四项全过
.venv\Scripts\python ingest.py                # 入库（data/ 加了新文档后重跑）
.venv\Scripts\python chat.py                  # 交互问答（多轮，自动改写）
.venv\Scripts\python web.py                   # Web 界面: http://127.0.0.1:8000
.venv\Scripts\python scripts\ask_once.py "保修多久"   # 单发一问
.venv\Scripts\python scripts\test_multiturn.py        # 多轮改写验收剧本
.venv\Scripts\python scripts\eval.py                  # 评估：仅检索指标（零成本）
.venv\Scripts\python scripts\eval.py --full           # 评估：加生成+judge打分
.venv\Scripts\python scripts\eval.py --full --agent   # 评估：v0.6 混合路由（聚合题附池召回率）
```

## 注意事项（续）

- **重建规模语料**：改 `scripts/gen_corpus.py` 参数后需重跑生成器 + `ingest.py`，且**聚合题金标要对照 `eval/scale_truth.json` 重新核对**——语料一变，"最/唯一"类题目前提会腐烂。

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
| v0.4 | Web UI（裸 ASGI + uvicorn，零新增依赖；热重建索引） | ✅ 完成 |
| v0.5 | 评估集 + 指标（检索 hit/MRR + LLM judge 忠实度/相关性，零 RAGAS 依赖） | ✅ 完成 |
| v0.6 | Agentic 混合路由：规划器 + 扫描执行器（浏览优先）+ 综合器 + 充分性闸门；规模语料实验链 | ✅ 完成 |
