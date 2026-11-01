"""检索质量评测（不需要 API）。

TeleQnA 没有标注"金标准文档"，这里用两个代理指标：
- AnsHit@k：前 k 个片段中，是否有片段包含正确选项的大部分关键词（覆盖率 >= 0.8）。
- AnsGap@k：正确选项的关键词覆盖率减去干扰选项的最高覆盖率，越大说明证据越能区分选项。

用法：
  python scripts/eval_retrieval.py --split dev
  python scripts/eval_retrieval.py --split dev --set retrieval.method=bm25 retrieval.rerank=false
"""
import json
import re

from _common import base_parser, load

STOP = set("""a an the of to in for on and or by with is are be as at from that this which it its
its into via using use used when where not no any all can may shall should will""".split())


def keywords(s):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", s.lower()) if w not in STOP and len(w) > 1}


def coverage(option, text_words):
    kw = keywords(option)
    return len(kw & text_words) / len(kw) if kw else 0.0


def main():
    p = base_parser(__doc__)
    p.add_argument("--k", type=int, nargs="*", default=[1, 3, 5])
    args = p.parse_args()
    cfg, items = load(args)

    from telerag.pipeline import get_chunks, get_retrieval, retrieval_tag

    chunks, path = get_chunks(cfg)
    retrieval = get_retrieval(cfg, items, chunks, path)

    res = {k: {"hit": 0, "gap": 0.0} for k in args.k}
    for it in items:
        ret = retrieval[it["qid"]]
        for k in args.k:
            words = set()
            for cid, _ in ret[:k]:
                words |= keywords(chunks[cid]["text"])
            cov = {o: coverage(t, words) for o, t in it["options"].items()}
            gold = cov.pop(it["answer"])
            res[k]["hit"] += gold >= 0.8
            res[k]["gap"] += gold - max(cov.values())
    n = len(items)
    summary = {
        "config": {"chunking": cfg["chunking"]["strategy"], **{k: cfg["retrieval"][k] for k in
                   ("method", "rerank", "query_mode", "top_k", "embed_model")}},
        "tag": retrieval_tag(cfg), "n": n,
        **{f"AnsHit@{k}": round(v["hit"] / n, 4) for k, v in res.items()},
        **{f"AnsGap@{k}": round(v["gap"] / n, 4) for k, v in res.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    from telerag.config import resolve
    with resolve("outputs/retrieval_results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"split": args.split, **summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
