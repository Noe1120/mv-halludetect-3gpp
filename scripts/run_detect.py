"""幻觉检测实验：在 run_qa.py 的输出上计算各检测器的风险分数并评测。

前提：
  - 已对 test 划分跑过 run_qa.py（--name 指定的主系统）；
  - 多视角一致性需要 closed、bm25、dense 三个视角在同一划分上的答题结果（run_qa.py / run_ablation.py 的副产品）。

评测协议：测试集上 5 折分层交叉验证。需要训练的方法（逻辑回归融合）每折只用其余 4 折训练（out-of-fold）；
F1 的阈值同样在其余折上选。开发集只有约 40 个幻觉样本，只用于调提示词。

方法分三组：
  免费信号（不需要额外模型或 API）
    Retrieval      重排分数
    Self-cite      大模型是否声明引用了证据
    MV-Consist     多视角一致性：闭卷、BM25、稠密检索三个视角的答案与最终答案是否一致（本文）
  本地核验（RTX 4060 上运行）
    NLI-own / NLI-contrast    DeBERTa NLI，后者加选项对比特征
    SLM-own / SLM-contrast    Qwen3 Yes/No 核验，后者加选项对比特征
  付费基线（需要额外 API 调用）
    LLM-Judge      大模型给出 0–100 置信度（--judge）
    Self-Consist   同一提示词采样 K 次的一致率（--sc K）
  本文完整方法
    Ours           MV-Consist + SLM-contrast + NLI-contrast + 免费信号，逻辑回归融合
    Ours-lite      只用闭卷一个视角 + 本地核验 + 免费信号

成本说明：多视角需要额外的答题调用（闭卷 1 次、BM25 和稠密检索各 1 次）。
如果系统本来就要做检索消融，这些调用是副产品；否则应按实际成本计算，结果里的 view_cost_usd_per_q 就是这个成本。

用法：
  python scripts/run_detect.py --name hybrid_rerank --judge --sc 5
"""
import json
import time

import numpy as np

from _common import base_parser

FREE = ["rerank_top", "cites_evidence"]
NLI_OWN = ["nli_own", "nli_contra"]
NLI_CON = ["nli_own", "nli_rival", "nli_margin", "nli_contra"]
SLM_OWN = ["slm_own"]
SLM_CON = ["slm_own", "slm_rival", "slm_margin", "slm_argmax_agree"]
MV = ["agree_closed", "agree_bm25", "agree_dense", "view_vote"]

METHODS = {
    "Retrieval": ["rerank_top"],
    "Self-cite": ["cites_evidence"],
    "MV-Consist": MV,
    "NLI-own": NLI_OWN,
    "NLI-contrast": NLI_CON,
    "SLM-own": SLM_OWN,
    "SLM-contrast": SLM_CON,
    "Ours": FREE + MV + NLI_CON + SLM_CON,
    # 轻量版：只加一个闭卷视角（闭卷提示词很短，额外成本约为 LLM-Judge 的 1/8）
    "Ours-lite": FREE + ["agree_closed"] + NLI_CON + SLM_CON,
}

# 消融：从完整方法中每次去掉一组特征
ABLATIONS = {
    "Ours w/o MV": FREE + NLI_CON + SLM_CON,
    "Ours w/o SLM": FREE + MV + NLI_CON,
    "Ours w/o NLI": FREE + MV + SLM_CON,
    "Ours w/o contrast": FREE + MV + NLI_OWN + SLM_OWN,
    "Ours w/o free": MV + NLI_CON + SLM_CON,
    "MV (closed only)": ["agree_closed"],
    "MV + SLM-contrast": MV + SLM_CON,
}


def attach_scores(cfg, records, items_by_qid, chunks, use_slm):
    from telerag.pipeline import get_scores

    retrieval = {r["qid"]: [tuple(x) for x in r["retrieved"]] for r in records}
    items = [items_by_qid[r["qid"]] for r in records]
    timing = {}
    for which in (["nli", "slm"] if use_slm else ["nli"]):
        scores = get_scores(cfg, items, retrieval, chunks, which)
        for r in records:
            r[which] = scores[r["qid"]]["scores"]
        timing[which] = float(np.mean([scores[r["qid"]]["seconds"] for r in records]))
    return timing


def run_judge(cfg, records, items_by_qid, chunks):
    from telerag.detect import llm_judge_messages, parse_judge
    from telerag.llm import LLMClient

    llm = LLMClient(cfg["llm"])

    def one(r):
        msgs = llm_judge_messages(items_by_qid[r["qid"]], r["pred"],
                                  [tuple(x) for x in r["retrieved"]], chunks)
        resp = llm.chat(msgs, max_tokens=10)
        return 1 - parse_judge(resp["text"]), resp

    outs = llm.map(one, records)
    tokens = sum(o[1]["prompt_tokens"] + o[1]["completion_tokens"] for o in outs)
    pt = sum(o[1]["prompt_tokens"] for o in outs)
    ct = sum(o[1]["completion_tokens"] for o in outs)
    return (np.array([o[0] for o in outs]),
            {"api_calls_per_q": 1, "tokens_per_q": tokens / len(records),
             "cost_usd_per_q": llm.cost(pt, ct) / len(records),
             "seconds_per_q": float(np.mean([o[1]["latency"] for o in outs]))})


def run_sc(cfg, records, items_by_qid, chunks, k):
    from telerag.consistency import sc_agreement, self_consistency
    from telerag.llm import LLMClient

    llm = LLMClient(cfg["llm"])
    items = [items_by_qid[r["qid"]] for r in records]
    retrieval = {r["qid"]: [tuple(x) for x in r["retrieved"]] for r in records}
    t0 = time.time()
    res = self_consistency(llm, items, retrieval, chunks, k=k)
    risk = np.array([1 - sc_agreement(r["pred"], res[r["qid"]]["preds"]) for r in records])
    tokens = sum(v["tokens"] for v in res.values())
    u = llm.usage
    return risk, {"api_calls_per_q": k, "tokens_per_q": tokens / len(records),
                  "cost_usd_per_q": llm.cost(u["prompt_tokens"], u["completion_tokens"]) / len(records),
                  "wall_seconds": time.time() - t0}


def main():
    p = base_parser(__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--judge", action="store_true", help="LLM-as-a-Judge 基线（花钱）")
    p.add_argument("--sc", type=int, default=0, help="自洽性采样次数 K，0 表示不跑（花钱，约 K 倍答题成本）")
    p.add_argument("--no-slm", action="store_true", help="跳过 Qwen3 核验器（显存或内存不足时）")
    p.add_argument("--folds", type=int, default=5, help="测试集上交叉验证的折数")
    p.add_argument("--tag", default="", help="结果文件名后缀，例如 --tag slm1.7b，避免覆盖默认结果")
    p.add_argument("--views-prefix", default="", help="视角结果的文件名前缀（换答题模型时用，如 pro__）")
    args = p.parse_args()

    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from telerag.config import load_config, resolve
    from telerag.consistency import load_views, view_features
    from telerag.data import load_questions
    from telerag.detect import RiskModel
    from telerag.io import load_records, save_records
    from telerag.metrics import accuracy, aurc, best_f1_threshold, bootstrap_ci, detection_metrics, risk_coverage
    from telerag.pipeline import get_chunks

    cfg = load_config(args.config, args.set)
    items_by_qid = {it["qid"]: it for it in load_questions(cfg)}
    chunks, _ = get_chunks(cfg)
    test = load_records(f"{args.name}_test")
    if args.limit:
        test = test[:args.limit]

    timing = attach_scores(cfg, test, items_by_qid, chunks, not args.no_slm)
    views = load_views("test", [f"{args.views_prefix}{v}" for v in ("closed", "bm25", "dense")])
    views = {k[len(args.views_prefix):]: v for k, v in views.items()}
    for r in test:
        r["views"] = view_features(r, views)
    methods = {**METHODS, **ABLATIONS}
    if args.no_slm:
        methods = {k: v for k, v in methods.items() if not any(f.startswith("slm") for f in v)}
    available = set(test[0]["views"]) | {"rerank_top", "cites_evidence", "nli_own", "nli_rival", "nli_margin",
                                         "nli_contra", "slm_own", "slm_rival", "slm_margin", "slm_argmax_agree"}
    methods = {k: v for k, v in methods.items() if set(v) <= available}

    y = np.array([r["hallucinated"] for r in test])
    correct = 1 - y
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=0).split(y, y))

    risks, coefs = {}, {}
    for m, feats in methods.items():
        risk = np.zeros(len(test))
        for tr, te in folds:
            risk[te] = RiskModel(feats).fit([test[i] for i in tr]).predict([test[i] for i in te])
        risks[m] = risk
        if m == "Ours":
            coefs[m] = RiskModel(feats).fit(test).coefficients()

    paid = {}
    if args.judge:
        risks["LLM-Judge"], paid["LLM-Judge"] = run_judge(cfg, test, items_by_qid, chunks)
    if args.sc:
        name = f"Self-Consist (K={args.sc})"
        risks[name], paid[name] = run_sc(cfg, test, items_by_qid, chunks, args.sc)

    def boot_auroc(score):
        return lambda idx: roc_auc_score(y[idx], score[idx]) if 0 < y[idx].sum() < len(idx) else 0.5

    ref = "LLM-Judge" if "LLM-Judge" in risks else None
    results, curves = {}, {}
    for m, risk in risks.items():
        pred_pos = np.zeros(len(test), bool)
        for tr, te in folds:
            pred_pos[te] = risk[te] >= best_f1_threshold(y[tr], risk[tr])
        met = detection_metrics(y, risk)
        met["f1@tau"] = float(f1_score(y, pred_pos, zero_division=0))
        met["auroc_ci95"] = bootstrap_ci(boot_auroc(risk), len(y))
        met["aurc"] = aurc(correct, risk)
        if ref and m != ref:
            # 配对 bootstrap：AUROC(m) − AUROC(LLM-Judge) 的 95% 置信区间，区间不含 0 即差异显著
            met["auroc_minus_judge_ci95"] = bootstrap_ci(
                lambda idx, a=risk, b=risks[ref]: boot_auroc(a)(idx) - boot_auroc(b)(idx), len(y))
        met.update(paid.get(m, {}))
        results[m] = met
        curves[m] = risk_coverage(correct, risk)
        for r, s in zip(test, risk):
            r.setdefault("risk", {})[m] = float(s)

    # 视角的实际额外答题成本（按 run_qa.py 记录的花费折算到每题）
    qa_rows = {}
    qa_path = resolve("outputs/qa_results.jsonl")
    for line in qa_path.open(encoding="utf-8"):
        row = json.loads(line)
        qa_rows[row["name"]] = row
    view_cost = {v: qa_rows[f"{args.views_prefix}{v}_test"]["cost_usd"] / qa_rows[f"{args.views_prefix}{v}_test"]["n"]
                 for v in views if f"{args.views_prefix}{v}_test" in qa_rows}
    for m, feats in methods.items():
        used = [v for v in views if f"agree_{v}" in feats or "view_vote" in feats or "view_agree" in feats]
        if used and m in results:
            results[m]["api_calls_per_q"] = len(used)
            results[m]["cost_usd_per_q"] = sum(view_cost.get(v, 0.0) for v in used)

    out = {"name": args.name, "view_cost_usd_per_q": view_cost, "answer_accuracy": accuracy(test), "hallucination_rate": float(y.mean()),
           "n": len(test), "protocol": f"{args.folds}-fold stratified CV on test (out-of-fold)",
           "views": list(views), "slm_model": None if args.no_slm else cfg["detection"]["slm_model"],
           "local_seconds_per_q": timing, "coefficients": coefs,
           "methods": results, "risk_coverage": curves}
    suffix = f"_{args.tag}" if args.tag else ""
    path = resolve(f"outputs/detect_{args.name}{suffix}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in test:
        r.pop("views", None)
    save_records(f"{args.name}_test_scored{suffix}", test)

    print(f"答题准确率 {out['answer_accuracy']:.3f}，幻觉率 {out['hallucination_rate']:.3f}，"
          f"本地核验耗时（秒/题） {timing}")
    print(f"{'方法':<22}{'AUROC':>8}{'AUPRC':>8}{'F1':>8}{'AURC↓':>8}   ΔAUROC vs Judge")
    for m, r in results.items():
        d = r.get("auroc_minus_judge_ci95")
        ds = f"[{d[0]:+.3f}, {d[1]:+.3f}]" if d else ""
        print(f"{m:<22}{r['auroc']:>8.3f}{r['auprc']:>8.3f}{r['f1@tau']:>8.3f}{r['aurc']:>8.3f}   {ds}")
    print(f"详细结果：{path}")


if __name__ == "__main__":
    main()
