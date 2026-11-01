"""E6 判定：Telco-RAG* 与宿主及各视角在同一批 val 题上的配对比较（PLAN.md E6）。不调用 API。

PLAN.md E6：不设阈值，作为领域基线报告；如果显著高于宿主（McNemar p < 0.05），后续轮次考虑把它作为新的宿主。

用法：
  python scripts/compare_telco_rag.py --split val
"""
import argparse
import json

import numpy as np

from _common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--run", default="telco_rag_star", help="telco_rag_star（E6）或 telco_rag_noroute（R2-A）")
    args = p.parse_args()

    from scipy.stats import binomtest

    from telerag.config import load_config
    from telerag.data import load_questions
    from telerag.explog import Experiment
    from telerag.io import load_records

    cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    sp = "dev" if args.split == "val" else "test"
    items = {it["qid"]: it for it in load_questions(cfg) if it["split"] == sp}
    tr = {json.loads(l)["qid"]: json.loads(l)
          for l in (ROOT / f"outputs/runs/{args.run}_{args.split}.jsonl").open(encoding="utf-8")}
    qids = sorted(q for q in tr if q in items)

    def preds(name):
        out = {}
        for s in ("dev", "test"):
            try:
                out.update({r["qid"]: r["pred"] for r in load_records(f"{name}_{s}")})
            except FileNotFoundError:
                pass
        return out

    gold = [items[q]["answer"] for q in qids]
    t = [tr[q]["pred"] for q in qids]
    res = {"n": len(qids), "telco_rag_star": float(np.mean([a == g for a, g in zip(t, gold)])), "vs": {}}
    names = ["closed", "bm25", "dense", "hybrid_rerank"] + (["telco_rag_star"] if args.run != "telco_rag_star" else [])
    for name in names:
        pv = preds(name) if name != "telco_rag_star" else {json.loads(l)["qid"]: json.loads(l)["pred"] for l in
                                                            (ROOT / f"outputs/runs/telco_rag_star_{args.split}.jsonl").open(encoding="utf-8")}
        b = [pv.get(q) for q in qids]
        g = sum(x == y and z != y for x, z, y in zip(t, b, gold))
        l = sum(x != y and z == y for x, z, y in zip(t, b, gold))
        res["vs"][name] = {"acc": float(np.mean([a == y for a, y in zip(b, gold)])), "telco_better": g,
                           "telco_worse": l, "p": float(binomtest(g, g + l).pvalue) if g + l else 1.0}
    by_cat = {}
    for q in qids:
        by_cat.setdefault(items[q]["category"], []).append(tr[q]["pred"] == items[q]["answer"])
    res["by_category"] = {k: float(np.mean(v)) for k, v in by_cat.items()}
    res["invalid"] = sum(x is None for x in t)
    res["calls_per_q"] = float(np.mean([tr[q]["calls"] for q in qids]))
    res["tokens_per_q"] = float(np.mean([tr[q]["prompt_tokens"] + tr[q]["completion_tokens"] for q in qids]))
    h = res["vs"]["hybrid_rerank"]
    res["verdict"] = ("显著高于宿主" if h["p"] < 0.05 and res["telco_rag_star"] > h["acc"]
                      else "显著低于宿主" if h["p"] < 0.05 else "与宿主无显著差异")
    name = "E6_compare" if args.run == "telco_rag_star" else "R2A_compare"
    exp = Experiment(f"{name}_{args.split}", cfg, {"plan": "PLAN.md E6" if name == "E6_compare" else "PLAN_round2.md R2-A",
                                                   "run": args.run})
    exp.save_json("metrics.json", res)
    exp.finish({"verdict": res["verdict"], "acc": round(res["telco_rag_star"], 4)})
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
