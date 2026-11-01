"""RAG 答题实验（需要 API 密钥）。

用法：
  python scripts/run_qa.py --name closed --no-rag --split dev        # 闭卷基线
  python scripts/run_qa.py --name hybrid_rerank --split dev          # 默认配置（混合检索 + 重排）
  python scripts/run_qa.py --name bm25 --split dev --set retrieval.method=bm25 retrieval.rerank=false
  python scripts/run_qa.py --name hybrid_rerank --split test         # 正式测试

逐题结果保存在 outputs/runs/<name>_<split>.jsonl，汇总追加到 outputs/qa_results.jsonl。
"""
import json
import time

from _common import base_parser, load


def main():
    p = base_parser(__doc__)
    p.add_argument("--name", required=True, help="实验名，用于结果文件命名")
    p.add_argument("--no-rag", action="store_true", help="不检索，直接回答（闭卷基线）")
    args = p.parse_args()
    cfg, items = load(args)

    from telerag.config import resolve
    from telerag.io import save_records
    from telerag.llm import LLMClient
    from telerag.metrics import accuracy
    from telerag.pipeline import get_chunks, get_retrieval, run_qa

    llm = LLMClient(cfg["llm"])
    chunks = retrieval = None
    if not args.no_rag:
        chunks, path = get_chunks(cfg)
        retrieval = get_retrieval(cfg, items, chunks, path)

    t0 = time.time()
    records = run_qa(cfg, items, llm, retrieval, chunks)
    name = f"{args.name}_{args.split}"
    save_records(name, records)

    pt = sum(r["prompt_tokens"] for r in records)
    ct = sum(r["completion_tokens"] for r in records)
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    summary = {
        "name": name, "model": cfg["llm"]["model"], "rag": not args.no_rag, "n": len(records),
        "accuracy": round(accuracy(records), 4),
        **{f"acc[{k}]": round(accuracy(v), 4) for k, v in sorted(by_cat.items())},
        "invalid_answers": sum(r["pred"] is None for r in records),
        "prompt_tokens": pt, "completion_tokens": ct,
        "cache_hit_tokens": sum(r.get("cache_hit_tokens", 0) for r in records),
        # 成本按配置里的单价估算；换模型时用 --set llm.price_input_per_m=... 指定该模型的价格
        "cost_usd": round(llm.cost(pt, ct, sum(r.get("cache_hit_tokens", 0) for r in records)), 4),
        "new_api_calls": llm.usage["calls"], "cache_hits": llm.usage["cached"],
        "wall_seconds": round(time.time() - t0, 1),
    }
    if not args.no_rag:
        summary["retrieval"] = {k: cfg["retrieval"][k] for k in ("method", "rerank", "top_k", "query_mode")}
        summary["chunking"] = cfg["chunking"]["strategy"]
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    with resolve("outputs/qa_results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
