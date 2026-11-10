"""v0.5 评估入口：跑问题集 -> 检索指标(免费) + 生成指标(judge) -> 报告。

用法:
    .venv\\Scripts\\python scripts\\eval.py           # 检索指标（不调 LLM，零成本）
    .venv\\Scripts\\python scripts\\eval.py --full    # 加生成+judge（每题约3次LLM调用）

指标说明:
    hit       金标块是否出现在 top_k 命中里（0/1，expect_source 为空则不计）
    mrr       金标块排名的倒数（第1名=1.0，第3名=0.33，未命中=0）
    拒答      拒答行为是否符合预期（该拒的拒了 / 该答的答了）
    关键词    回答是否包含期望关键词（--full）
    忠实/相关 judge 打分 1~5（--full，evaluator 未实现时显示"待实现"）

报告写入 eval/reports/（已 gitignore），含设置快照 + git 版本，
改参数前后各跑一次即可 A/B 对比。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import get_settings
from rag.generator import answer
from rag.retriever import Retriever


def load_questions() -> list[dict]:
    data = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    qs = [q for q in data if not q["question"].startswith("TODO")]
    skipped = len(data) - len(qs)
    if skipped:
        print(f"（跳过 {skipped} 道待填题）")
    return qs


def retrieval_row(q: dict, hits: list[dict], threshold: float, top_k: int) -> dict:
    """免费指标：拒答正确性 / hit / mrr / 最高分。"""
    top_score = hits[0]["score"] if hits else 0.0
    refused = (not hits) or top_score < threshold
    row = {
        "id": q["id"],
        "question": q["question"],
        "refused": refused,
        "refusal_ok": refused == q["should_refuse"],
        "top_score": round(top_score, 3),
        "hit": None,
        "mrr": None,

    }
    gold = q.get("expect_source")
    if gold:
        # 多金标支持：expect_source 可以是 "file#chunkN" 或其数组（跨块题）。
        # hit 语义升级为金标召回率（1.0=全中 0.5=中一半）；mrr 取最好命中的排名。
        golds = gold if isinstance(gold, list) else [gold]
        keys = [f"{h['source']}#chunk{h['chunk_index']}" for h in hits[:top_k]]
        ranks = [keys.index(g) + 1 for g in golds if g in keys]
        match=q.get("match","all")
        if match == "any":
            row["hit"] = 1.0 if len(ranks)>0 else 0.0
        else:
            row["hit"] = round(len(ranks) / len(golds), 2)


        row["mrr"] = round(1.0 / min(ranks), 2) if ranks else 0.0
    return row


def gen_row(q: dict, hits: list[dict], row: dict) -> dict:
    """--full 阶段：生成回答 + 关键词命中 + judge 打分（未实现自动降级）。

    拒答式回答（"没有相关内容"）只判忠实度：faith judge 手里有资料，
    能核实拒答对不对（资料确实无答案=5 / 有答案却拒=3）；
    relev judge 看不到资料，无从核实拒答正确性，记"拒答"哨兵值不进均值。
    """
    from rag import evaluator  # 延迟导入：骨架态也能跑检索阶段

    reply = answer(q["question"], hits)
    row["answer"] = reply
    row["keyword_ok"] = all(k in reply for k in q["expect_keywords"]) if q["expect_keywords"] else None

    is_refusal = "没有相关内容" in reply and len(reply) < 30
    for name, fn, args in (
        ("faith", evaluator.judge_faithfulness, (q["question"], reply, [h["text"] for h in hits])),
        ("relev", evaluator.judge_relevance, (q["question"], reply)),
    ):
        if name == "relev" and is_refusal:
            row[name] = "拒答"
            continue
        try:
            judged = fn(*args)
        except NotImplementedError:
            row[name] = "待实现"
            continue
        except Exception as e:
            print(f"[eval] judge {name} 失败: {type(e).__name__}")
            row[name] = None
            continue
        row[name] = judged["score"] if judged else None
    return row


def agent_gen_row(q: dict, retriever, row: dict) -> dict:
    """--agent --full 阶段：走 v0.6 混合路由。agg 路径墓碑未拔时降级标注，不崩整批。"""
    from rag.agent import answer_agent

    try:
        result = answer_agent(q["question"], retriever=retriever)
        reply = result["answer"]
        row["route"] = result["route"]
        row["sub_queries"] = result["sub_queries"]
        contexts = result["contexts"]
        if result["route"] == "agg":                 # v0.6 池召回率: 聚合路径的专属镜头
            gold = q.get("expect_source")
            if gold:
                golds = gold if isinstance(gold, list) else [gold]
                keys = set(result.get("pool_keys") or [])
                row["pool_recall"] = round(sum(1 for g in golds if g in keys) / len(golds), 2)
    except NotImplementedError as e:
        reply = f"（AGG 路径未实现: {e}）"
        row["route"] = "agg(墓碑)"
        contexts = []
    row["answer"] = reply
    row["keyword_ok"] = all(k in reply for k in q["expect_keywords"]) if q["expect_keywords"] else None

    from rag import evaluator
    for name, fn, args in (
        ("faith", evaluator.judge_faithfulness, (q["question"], reply, contexts)),
        ("relev", evaluator.judge_relevance, (q["question"], reply)),
    ):
        if not contexts and name == "faith":
            row[name] = None
            continue
        try:
            judged = fn(*args)
        except Exception:
            row[name] = None
            continue
        row[name] = judged["score"] if judged else None
    return row


def mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def main() -> None:
    full = "--full" in sys.argv
    agent_mode = "--agent" in sys.argv
    s = get_settings()
    retriever = Retriever()
    rows: list[dict] = []

    mode = '全量(检索+生成+judge)' if full else '仅检索(零LLM成本)'
    if full and agent_mode:
        mode += ' + v0.6混合路由'
    print(f"模式: {mode}  |  阈值 {s.sim_threshold}  top_k {s.top_k}")
    print("-" * 76)
    for q in load_questions():
        hits = retriever.search(q["question"], top_k=s.top_k)
        row = retrieval_row(q, hits, s.sim_threshold, s.top_k)
        if full and not row["refused"]:
            row = agent_gen_row(q, retriever, row) if agent_mode else gen_row(q, hits, row)
        rows.append(row)
        hit_s = "-" if row["hit"] is None else str(row["hit"])
        mrr_s = "-" if row["mrr"] is None else f"{row['mrr']:.2f}"
        extra = ""
        if full and not row["refused"]:
            route_s = f"  路:{row['route']}" if agent_mode else ""
            if agent_mode and "pool_recall" in row:
                route_s += f"  池召回:{row['pool_recall']}"
            extra = f"  词:{row.get('keyword_ok','-')}  忠实:{row.get('faith','-')}  相关:{row.get('relev','-')}{route_s}"
        print(f"{row['id']}  {'拒' if row['refused'] else '答'}{'✓' if row['refusal_ok'] else '✗'}"
              f"  hit:{hit_s}  mrr:{mrr_s}  top:{row['top_score']:.3f}{extra}")

    # refusal_ok 是 bool（int 子类），mean 直接可用
    agg = {
        "refusal_acc": mean(rows, "refusal_ok"),
        "hit_rate": mean(rows, "hit"),
        "mrr": mean(rows, "mrr"),
        "keyword_rate": mean(rows, "keyword_ok") if full else None,
        "faith": mean(rows, "faith") if full else None,
        "relev": mean(rows, "relev") if full else None,
    }
    print("-" * 76)
    print(f"聚合: 拒答准确 {agg['refusal_acc']}  hit率 {agg['hit_rate']}  MRR {agg['mrr']}"
          + (f"  关键词 {agg['keyword_rate']}  忠实 {agg['faith']}  相关 {agg['relev']}" if full else ""))

    try:
        git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        git_sha = "?"
    report = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_sha, "full": full,
              "agent": agent_mode,
              "settings": {"chunk_size": s.chunk_size, "top_k": s.top_k,
                           "sim_threshold": s.sim_threshold, "recall_n": s.recall_n,
                           "rerank_candidates": s.rerank_candidates,
                           "rrf_w_bm": s.rrf_w_bm},
              "aggregate": agg, "rows": rows}
    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入: {out}")


if __name__ == "__main__":
    main()
