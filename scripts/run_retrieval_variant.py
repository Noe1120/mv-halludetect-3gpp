"""E7（bge-m3）/ E8（固定长度切块）：只改一个检索开关，在 val 上重新答题并做判定（PLAN.md E7、E8）。

生成提示词、temperature、top_k、数据划分全部沿用冻结值，只改：
  E8：chunking.strategy = fixed
  E7：retrieval.embed_model = BAAI/bge-m3（稠密视角 dense 和混合视角 hybrid_rerank 都要重答）

判定（预注册，PLAN.md 第 2 节和第 6 节）：
  E8 成立：val 上 section 比 fixed 高 ≥ 1 个百分点
  E7 成立：val 上 dense-m3 比 dense-base 高 ≥ 1 个百分点，或 hybrid-m3 比 hybrid-base 高 ≥ 1 个百分点；成立才跑 test
另外报告配对 McNemar 检验（p < 0.05 记为"显著"）。

用法（需要 API，只在空闲时段运行）：
  python scripts/run_retrieval_variant.py --exp E8
  python scripts/run_retrieval_variant.py --exp E7
  python scripts/run_retrieval_variant.py --exp E7 --split test   # 只有 val 判定成立时才运行
"""
import argparse
import json

import numpy as np

from _common import ROOT  # noqa: F401

THRESH = 0.01
VARIANTS = {
    "E8": {"fixed_chunk": ["chunking.strategy=fixed"]},
    # PLAN_round2.md R2-B：章节切块取 top-7，使上下文长度（约 2,517 token）与固定切块 top-5（2,577 token）相当
    "R2B": {"section_top7": ["retrieval.top_k=7"]},
    "E7": {"dense_m3": ["retrieval.method=dense", "retrieval.rerank=false", "retrieval.embed_model=BAAI/bge-m3",
                        "retrieval.query_prefix="],
           "hybrid_m3": ["retrieval.embed_model=BAAI/bge-m3", "retrieval.query_prefix="]},
}
# 每个变体对照的视角（已有缓存，不花钱）
BASELINE = {"fixed_chunk": "hybrid_rerank", "dense_m3": "dense", "hybrid_m3": "hybrid_rerank",
            "section_top7": "r1_fixed_chunk"}


def mcnemar(a, b, gold):
    from scipy.stats import binomtest

    g = sum(x == y_ and z != y_ for x, z, y_ in zip(a, b, gold))
    l = sum(x != y_ and z == y_ for x, z, y_ in zip(a, b, gold))
    return {"gain": int(g), "loss": int(l), "p": float(binomtest(g, g + l).pvalue) if g + l else 1.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True, choices=list(VARIANTS))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--budget-cny", type=float, default=4.0, help="本次运行的估算花费上限")
    args = p.parse_args()

    from telerag.config import load_config
    from telerag.data import load_questions
    from telerag.explog import Experiment
    from telerag.io import load_records, save_records
    from telerag.llm import LLMClient
    from telerag.pipeline import get_chunks, get_retrieval, run_qa

    overrides0 = ["data.seed=3407", "data.dev_size=500"]
    base_cfg = load_config(overrides=overrides0)
    split = "dev" if args.split == "val" else "test"
    items = [it for it in load_questions(base_cfg) if it["split"] == split]
    gold = [it["answer"] for it in items]

    def baseline_preds(name):
        out = {}
        # 旧划分下的结果叫 <名>_dev / <名>_test；第 1 轮变体实验的结果叫 r1_<名>_val / r1_<名>_test
        for suffix in ("dev", "test", "val"):
            try:
                out.update({r["qid"]: r["pred"] for r in load_records(f"{name}_{suffix}")})
            except FileNotFoundError:
                pass
        missing = sum(it["qid"] not in out for it in items)
        if missing:
            raise RuntimeError(f"对照组 {name} 缺少 {missing} 题的结果")
        return [out[it["qid"]] for it in items]

    exp = Experiment(f"{args.exp}_{args.split}", base_cfg, {"plan": f"PLAN.md {args.exp}", "split": args.split})
    res, spent = {}, 0.0
    for name, sets in VARIANTS[args.exp].items():
        cfg = load_config(overrides=overrides0 + sets)
        chunks, path = get_chunks(cfg)
        retrieval = get_retrieval(cfg, items, chunks, path)
        llm = LLMClient(cfg["llm"])
        records = run_qa(cfg, items, llm, retrieval, chunks)
        spent += llm.usage.get("cost_cny", 0.0)
        save_records(f"r1_{name}_{args.split}", records)
        exp.save_records(f"records_{name}.jsonl", records)
        pv = [r["pred"] for r in records]
        pb = baseline_preds(BASELINE[name])
        acc_v = float(np.mean([a == g for a, g in zip(pv, gold)]))
        acc_b = float(np.mean([a == g for a, g in zip(pb, gold)]))
        res[name] = {"acc_variant": acc_v, "acc_baseline": acc_b, "baseline": BASELINE[name],
                     "delta": acc_v - acc_b, "mcnemar": mcnemar(pv, pb, gold),
                     "invalid": sum(x is None for x in pv), "new_calls": llm.usage["calls"],
                     "cost_cny": llm.usage.get("cost_cny", 0.0)}
        print(f"[{args.exp}] {name}: {acc_v:.3f} vs {BASELINE[name]} {acc_b:.3f} (Δ {acc_v - acc_b:+.3f}), "
              f"McNemar {res[name]['mcnemar']}, 本次花费约 {res[name]['cost_cny']:.2f} 元", flush=True)
        if spent > args.budget_cny:
            print(f"[{args.exp}] 累计估算花费 {spent:.2f} 元超过上限 {args.budget_cny}，停止")
            break

    if args.exp == "R2B":
        r = res.get("section_top7")
        if not r:
            verdict = "未完成"
        elif r["delta"] >= THRESH:
            verdict = "章节切块更好（等预算下 H8 成立）"
        elif r["delta"] <= -THRESH:
            verdict = "固定切块更好"
        else:
            verdict = "两种切法相当（E8 的差异来自上下文长度）"
        if r:
            verdict += "，显著" if r["mcnemar"]["p"] < 0.05 else "，未达显著"
    elif args.exp == "E8":
        r = res.get("fixed_chunk")
        # 成立条件是 section（宿主）比 fixed 高 ≥ 1 个百分点，即 delta ≤ −0.01
        verdict = ("成立" if r and -r["delta"] >= THRESH else "不成立") if r else "未完成"
        if r and verdict == "成立":
            verdict += "（显著）" if r["mcnemar"]["p"] < 0.05 else "（未达显著）"
    else:
        ok = [n for n in ("dense_m3", "hybrid_m3") if n in res and res[n]["delta"] >= THRESH]
        verdict = "成立" if ok else ("不成立" if len(res) == 2 else "未完成")
        if ok:
            verdict += "（显著）" if any(res[n]["mcnemar"]["p"] < 0.05 for n in ok) else "（未达显著）"
    summ = {"split": args.split, "threshold": THRESH, "variants": res, "verdict": verdict, "cost_cny": spent}
    exp.save_json("metrics.json", summ)
    exp.finish({"verdict": verdict, **{k: round(v["delta"], 4) for k, v in res.items()}}, cost_cny=spent)
    print(json.dumps({"verdict": verdict, "cost_cny": round(spent, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
