"""把实验结果整理成论文表格（Markdown，可直接粘进 Word 再调格式）。

用法：
  python scripts/make_tables.py
输出到 outputs/tables.md。
"""
import json

from _common import ROOT

LABELS = {"closed": "Closed-book (no retrieval)", "bm25": "BM25", "dense": "Dense (bge-base)",
          "no_rerank": "Hybrid (BM25+Dense, RRF)", "hybrid_rerank": "Hybrid + rerank (full)",
          "fixed_chunk": "Full, fixed-length chunks", "top3": "Full, top-3", "top10": "Full, top-10"}


def latest(path, key):
    rows = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            rows[key(r)] = r
    return rows


def qa_table(prefix=""):
    qa = latest(ROOT / "outputs" / "qa_results.jsonl", lambda r: r["name"])
    ret = latest(ROOT / "outputs" / "retrieval_results.jsonl", lambda r: (r["split"], r["tag"]))
    ret_by_cfg = {}
    for (split, _), r in ret.items():
        c = r["config"]
        ret_by_cfg[(split, c["chunking"], c["method"], c["rerank"], c["top_k"])] = r
    lines = ["| Setting | AnsHit@5 | Acc. (all) | Acc. (Spec.) | Acc. (Overview) | Cost (USD) |",
             "|---|---|---|---|---|---|"]
    for name, label in LABELS.items():
        r = qa.get(f"{prefix}{name}_test")
        if not r:
            continue
        hit = "–"
        if r.get("rag") and not prefix:
            k = ("test", r["chunking"], r["retrieval"]["method"], r["retrieval"]["rerank"], r["retrieval"]["top_k"])
            if k in ret_by_cfg:
                hit = f"{ret_by_cfg[k]['AnsHit@5'] * 100:.1f}"
        lines.append(f"| {label} | {hit} | {r['accuracy'] * 100:.1f} | "
                     f"{r['acc[Standards specifications]'] * 100:.1f} | {r['acc[Standards overview]'] * 100:.1f} | "
                     f"{r['cost_usd']:.2f} |")
    return "\n".join(lines)


def detect_table(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    sec = d["local_seconds_per_q"]
    nli, slm = sec.get("nli", 0), sec.get("slm", 0)

    def extra(m, r):
        api = (f"{r['api_calls_per_q']} API call(s), ${r['cost_usd_per_q'] * 1000:.2f}/1k q"
               if "api_calls_per_q" in r else "")
        if m in ("LLM-Judge",) or m.startswith("Self-Consist") or m == "MV-Consist" or m.startswith("MV ("):
            return api
        t = (nli if "NLI" in m or m.startswith("Ours") else 0) + (slm if "SLM" in m or m.startswith("Ours") else 0)
        if m.startswith("Ours") and "w/o NLI" in m:
            t -= nli
        if m.startswith("Ours") and "w/o SLM" in m:
            t -= slm
        local_s = f"{t:.2f}s local GPU" if t else ""
        return " + ".join(x for x in (api, local_s) if x) or "free"

    lines = [f"答题模型准确率 {d['answer_accuracy'] * 100:.1f}%，幻觉率 {d['hallucination_rate'] * 100:.1f}%，"
             f"n = {d.get('n', '?')}；核验小模型 {d.get('slm_model')}；视角 {d.get('views')}；{d.get('protocol', '')}", "",
             "| Detector | AUROC [95% CI] | ΔAUROC vs Judge [95% CI] | AUPRC | F1 | AURC↓ | Extra cost/q |",
             "|---|---|---|---|---|---|---|"]
    for m, r in d["methods"].items():
        ci = r.get("auroc_ci95")
        dj = r.get("auroc_minus_judge_ci95")
        ci_s = f"{r['auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{r['auroc']:.3f}"
        dj_s = f"[{dj[0]:+.3f}, {dj[1]:+.3f}]" if dj else "–"
        lines.append(f"| {m} | {ci_s} | {dj_s} | {r['auprc']:.3f} | {r['f1@tau']:.3f} | {r['aurc']:.3f} | "
                     f"{extra(m, r)} |")
    if d.get("coefficients"):
        lines += ["", "Ours 的标准化逻辑回归系数（正值 = 越大越可能是幻觉）：", ""]
        for m, co in d["coefficients"].items():
            lines.append(", ".join(f"{k} {v:+.2f}" for k, v in sorted(co.items(), key=lambda kv: -abs(kv[1]))))
    return "\n".join(lines)


def main():
    parts = ["## 表 1  检索配置对答题准确率的影响（测试集 2,800 题，deepseek-flash）", "", qa_table(), "",
             "## 表 1b  第二个答题模型（deepseek-v4-pro，测试集前 1,000 题）", "", qa_table("pro__"), ""]
    for path in sorted((ROOT / "outputs").glob("detect_*.json")):
        parts += [f"## 幻觉检测（{path.stem}）", "", detect_table(path), ""]
    out = ROOT / "outputs" / "tables.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
