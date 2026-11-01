"""E6：Telco-RAG* 基线（PLAN.md E6）。每题 3 次 deepseek-flash 调用，只在空闲时段运行。

用法：
  python scripts/run_telco_rag.py --split val            # 先跑 val 500 题
  python scripts/run_telco_rag.py --split val --limit 20 # 小样本试跑
  python scripts/run_telco_rag.py --build-index-only     # 只建索引（本地 GPU，不花钱）
"""
import argparse
import json

import numpy as np

from _common import ROOT  # noqa: F401


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--build-index-only", action="store_true")
    p.add_argument("--budget-cny", type=float, default=3.0, help="超过这个估算花费就停止")
    p.add_argument("--no-route", action="store_true", help="PLAN_round2.md R2-A：不按系列过滤，在全部规范中检索")
    args = p.parse_args()

    from telerag.baselines.telco_rag import TelcoIndex, answer_one, load_vocabulary
    from telerag.config import load_config, resolve
    from telerag.data import load_questions
    from telerag.explog import Experiment
    from telerag.llm import LLMClient, cost_cny, is_peak

    cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    index = TelcoIndex(cfg)
    print(f"[Telco-RAG*] {len(index.chunks)} 个块")
    if args.build_index_only:
        return
    split = "dev" if args.split == "val" else "test"
    items = [it for it in load_questions(cfg) if it["split"] == split][: args.limit]
    terms, abbrs = load_vocabulary(resolve("data/raw/telco_rag/3GPP_vocabulary.docx"))
    print(f"[Telco-RAG*] 词表：术语 {len(terms)}，缩写 {len(abbrs)}；{args.split} {len(items)} 题")
    llm = LLMClient(cfg["llm"])
    tag = "telco_rag_noroute" if args.no_route else "telco_rag_star"
    plan = "PLAN_round2.md R2-A" if args.no_route else "PLAN.md E6"
    exp_name = f"R2A_telco_noroute_{args.split}" if args.no_route else f"E6_telco_rag_{args.split}"
    exp = Experiment(exp_name, cfg, {"plan": plan, "n": len(items), "route": not args.no_route})
    out_path = resolve(f"outputs/runs/{tag}_{args.split}.jsonl")
    done = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["qid"]] = r
    spent = 0.0
    with out_path.open("a", encoding="utf-8") as f:
        for it in items:
            if it["qid"] in done:
                continue
            r = answer_one(llm, index, terms, abbrs, it, route=not args.no_route)
            spent += cost_cny(cfg["llm"]["model"], r["prompt_tokens"], r["completion_tokens"],
                              r["cache_hit_tokens"], is_peak())
            done[it["qid"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if spent > args.budget_cny:
                print(f"[Telco-RAG*] 估算花费 {spent:.2f} 元超过上限 {args.budget_cny}，停止")
                break
    recs = [done[it["qid"]] for it in items if it["qid"] in done]
    acc = float(np.mean([r["pred"] == r["answer"] for r in recs]))
    by_cat = {}
    for it in items:
        if it["qid"] in done:
            by_cat.setdefault(it["category"], []).append(done[it["qid"]]["pred"] == it["answer"])
    summ = {"n": len(recs), "accuracy": acc, **{f"acc[{k}]": float(np.mean(v)) for k, v in by_cat.items()},
            "invalid": sum(r["pred"] is None for r in recs), "new_calls": llm.usage["calls"],
            "cost_cny_this_run": spent}
    exp.save_json("metrics.json", summ)
    exp.save_records("records.jsonl", recs)
    exp.finish(summ, cost_cny=spent)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
