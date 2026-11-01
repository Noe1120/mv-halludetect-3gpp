"""第 1 轮预注册实验（PLAN.md）的主评测脚本：val 上训练和选阈值，test 上只评测一次。

数据划分固定为 seed=3407、val 500 题，其余 2,500 题为 test（PLAN.md 第 0 节）。
答题结果来自已有的 run_qa.py 输出（outputs/runs/<视角>_{dev,test}.jsonl）。旧划分下的 dev 和 test 合起来
覆盖全部 3,000 题，这里按 qid 合并后重新分成 val 和 test，不需要重新答题。

用法（全部只读缓存，不花钱）：
  python scripts/run_eval.py --exp E0
  python scripts/run_eval.py --exp E1
  python scripts/run_eval.py --exp E2      # 首次运行要算原子事实核查（本地 GPU）
  python scripts/run_eval.py --exp E4
  python scripts/run_eval.py --exp all

  python scripts/run_eval.py --exp E5 --sc 5   # 需要 API（PLAN.md E5），只在空闲时段运行
"""
import argparse
import json
import os

import numpy as np

from _common import ROOT  # noqa: F401  (设置 sys.path 和 HF 环境变量)

FREE = ["rerank_top", "cites_evidence"]
NLI_OWN = ["nli_own", "nli_contra"]
NLI_CON = ["nli_own", "nli_rival", "nli_margin", "nli_contra"]
SLM_OWN = ["slm_own"]
SLM_CON = ["slm_own", "slm_rival", "slm_margin", "slm_argmax_agree"]
SLM_CON_LOGIT = ["slm_own_logit", "slm_rival_logit", "slm_margin_logit", "slm_argmax_agree"]
MV = ["agree_closed", "agree_bm25", "agree_dense", "view_vote"]
ATOMIC = ["atomic_n", "atomic_min", "atomic_mean", "faithfulness"]

# E0 的方法集合与旧实验相同，只是换了评测协议
E0_METHODS = {
    "Retrieval": ["rerank_top"],
    "Self-cite": ["cites_evidence"],
    "MV-Consist": MV,
    "NLI-own": NLI_OWN,
    "NLI-contrast": NLI_CON,
    "SLM-own": SLM_OWN,
    "SLM-contrast": SLM_CON,
    "Ours": FREE + MV + NLI_CON + SLM_CON,
    "Ours-lite": FREE + ["agree_closed"] + NLI_CON + SLM_CON,
    "Ours w/o MV": FREE + NLI_CON + SLM_CON,
    "Ours w/o SLM": FREE + MV + NLI_CON,
    "Ours w/o NLI": FREE + MV + SLM_CON,
    "Ours w/o contrast": FREE + MV + NLI_OWN + SLM_OWN,
}
# E1：概率尺度与 logit 尺度的对照
E1_PAIRS = {
    "SLM-contrast": (SLM_CON, SLM_CON_LOGIT),
    "Ours": (FREE + MV + NLI_CON + SLM_CON, FREE + MV + NLI_CON + SLM_CON_LOGIT),
    "Ours-lite": (FREE + ["agree_closed"] + NLI_CON + SLM_CON, FREE + ["agree_closed"] + NLI_CON + SLM_CON_LOGIT),
}
# 预注册阈值（PLAN.md 第 2 节、第 6 节）
THRESH_AUROC = 0.01
THRESH_ACC_E4 = 0.005

VIEW_RUNS = {"closed": "closed", "bm25": "bm25", "dense": "dense"}
HOST = "hybrid_rerank"


def load_split(cfg):
    from telerag.data import load_questions

    items = {it["qid"]: it for it in load_questions(cfg)}
    val = sorted(q for q, it in items.items() if it["split"] == "dev")
    test = sorted(q for q, it in items.items() if it["split"] == "test")
    return items, val, test


def union_runs(name, prefix=""):
    from telerag.io import load_records

    out = {}
    for sp in ("dev", "test"):
        try:
            for r in load_records(f"{prefix}{name}_{sp}"):
                out[r["qid"]] = r
        except FileNotFoundError:
            pass
    return out


def build_records(cfg, items, qids, prefix=""):
    """宿主系统的逐题记录 + 本地核验打分 + 多视角特征。"""
    from telerag.consistency import view_features
    from telerag.pipeline import get_chunks, get_scores

    host = union_runs(HOST, prefix)
    missing = [q for q in qids if q not in host]
    if missing:
        raise RuntimeError(f"宿主系统缺少 {len(missing)} 题的答题结果，例如 {missing[:5]}")
    recs = [dict(host[q]) for q in qids]
    chunks, _ = get_chunks(cfg)
    retrieval = {r["qid"]: [tuple(x) for x in r["retrieved"]] for r in recs}
    its = [items[q] for q in qids]
    timing = {}
    for which in ("nli", "slm"):
        sc = get_scores(cfg, its, retrieval, chunks, which)
        for r in recs:
            r[which] = sc[r["qid"]]["scores"]
        timing[which] = float(np.mean([sc[q]["seconds"] for q in qids]))
    views = {v: {q: r["pred"] for q, r in union_runs(run, prefix).items()} for v, run in VIEW_RUNS.items()}
    for r in recs:
        r["views"] = view_features(r, views)
    return recs, chunks, timing, views


def boot_auroc_fn(y, s):
    from sklearn.metrics import roc_auc_score

    return lambda idx: roc_auc_score(y[idx], s[idx]) if 0 < y[idx].sum() < len(idx) else 0.5


def evaluate(name, feats, val, test, yv, yt):
    """val 上训练融合模型、选阈值，test 上只评测一次。"""
    from sklearn.metrics import f1_score

    from telerag.detect import RiskModel
    from telerag.metrics import aurc, best_f1_threshold, bootstrap_ci, detection_metrics

    model = RiskModel(feats).fit(val)
    rv, rt = model.predict(val), model.predict(test)
    tau = best_f1_threshold(yv, rv)
    out = {}
    for split, y, r in (("val", yv, rv), ("test", yt, rt)):
        m = detection_metrics(y, r)
        m["f1"] = float(f1_score(y, r >= tau, zero_division=0))
        m["auroc_ci95"] = bootstrap_ci(boot_auroc_fn(y, r), len(y))
        m["aurc"] = aurc(1 - y, r)
        out[split] = m
    out["tau"] = tau
    out["features"] = feats
    out["coefficients"] = model.coefficients()
    return out, rv, rt


def paired_diff(y, a, b):
    from telerag.metrics import bootstrap_ci

    fa, fb = boot_auroc_fn(y, a), boot_auroc_fn(y, b)
    return bootstrap_ci(lambda idx: fa(idx) - fb(idx), len(y))


def verdict(dv, dt, ci_v, thr):
    """PLAN.md 第 6 节的判定规则。dv/dt：val/test 上的提升；ci_v：val 上配对差值的 95% CI。"""
    if dv >= thr and dt > 0:
        return "成立（显著）" if ci_v[0] > 0 else "成立（未达显著）"
    return "不成立"


def judge_risk(cfg, items, recs, chunks):
    from telerag.detect import llm_judge_messages, parse_judge
    from telerag.llm import LLMClient

    llm = LLMClient(cfg["llm"])

    def one(r):
        msgs = llm_judge_messages(items[r["qid"]], r["pred"], [tuple(x) for x in r["retrieved"]], chunks)
        return 1 - parse_judge(llm.chat(msgs, max_tokens=10)["text"])

    return np.array(llm.map(one, recs)), llm.usage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", default="all", help="E0 / E1 / E2 / E4 / E5 / all")
    p.add_argument("--sc", type=int, default=5, help="E5 自洽性采样次数")
    p.add_argument("--sc-split", default="val", choices=["val", "test", "both"], help="E5 在哪个划分上跑")
    args = p.parse_args()

    from telerag.config import load_config
    from telerag.explog import Experiment

    cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    items, val_q, test_q = load_split(cfg)
    assert not set(val_q) & set(test_q) and len(val_q) + len(test_q) == len(items)
    val, chunks, timing, views = build_records(cfg, items, val_q)
    test, _, _, _ = build_records(cfg, items, test_q)
    yv = np.array([r["hallucinated"] for r in val])
    yt = np.array([r["hallucinated"] for r in test])
    base_info = {"n_val": len(val), "n_test": len(test), "halluc_val": int(yv.sum()), "halluc_test": int(yt.sum()),
                 "acc_val": float(1 - yv.mean()), "acc_test": float(1 - yt.mean()), "local_seconds_per_q": timing}
    run_all = args.exp == "all"
    os.environ.setdefault("TELERAG_NO_API", "1")   # E0–E4 只读缓存；E5 显式解除
    results = {}

    # LLM-Judge：val 和 test 的缓存都已存在（旧实验跑过全部 3,000 题），只读
    jv, _ = judge_risk(cfg, items, val, chunks)
    jt, _ = judge_risk(cfg, items, test, chunks)
    from telerag.metrics import aurc, best_f1_threshold, bootstrap_ci, detection_metrics
    from sklearn.metrics import f1_score

    def fixed_score_metrics(sv, st):
        """不需要训练的分数（Judge、SC、无训练 MV）：阈值同样只在 val 上选。"""
        tau = best_f1_threshold(yv, sv)
        out = {"tau": tau}
        for split, y, s in (("val", yv, sv), ("test", yt, st)):
            m = detection_metrics(y, s)
            m["f1"] = float(f1_score(y, s >= tau, zero_division=0))
            m["auroc_ci95"] = bootstrap_ci(boot_auroc_fn(y, s), len(y))
            m["aurc"] = aurc(1 - y, s)
            out[split] = m
        return out

    judge = fixed_score_metrics(jv, jt)

    if run_all or args.exp == "E0":
        exp = Experiment("E0_val2test", cfg, {"plan": "PLAN.md E0"})
        res, risks = {"LLM-Judge": judge}, {"LLM-Judge": (jv, jt)}
        mv_free_v = np.array([1 - r["views"]["view_vote"] for r in val])
        mv_free_t = np.array([1 - r["views"]["view_vote"] for r in test])
        res["MV (training-free)"] = fixed_score_metrics(mv_free_v, mv_free_t)
        risks["MV (training-free)"] = (mv_free_v, mv_free_t)
        for m, feats in E0_METHODS.items():
            res[m], rv, rt = evaluate(m, feats, val, test, yv, yt)
            risks[m] = (rv, rt)
        for m in res:
            if m != "LLM-Judge":
                res[m]["test"]["auroc_minus_judge_ci95"] = paired_diff(yt, risks[m][1], jt)
                res[m]["val"]["auroc_minus_judge_ci95"] = paired_diff(yv, risks[m][0], jv)
        from telerag.metrics import risk_coverage
        curves = {m: risk_coverage(1 - yt, risks[m][1]) for m in res}
        exp.save_json("metrics.json", {**base_info, "methods": res, "risk_coverage_test": curves})
        exp.save_array("risk_test.npy", np.stack([risks[m][1] for m in res]), columns=list(res))
        exp.save_array("risk_val.npy", np.stack([risks[m][0] for m in res]), columns=list(res))
        exp.finish({m: {"val_auroc": round(r["val"]["auroc"], 4), "test_auroc": round(r["test"]["auroc"], 4)}
                    for m, r in res.items()})
        results["E0"] = {"dir": str(exp.dir), "methods": res}
        print(f"[E0] {exp.dir}")
        for m, r in res.items():
            d = r["test"].get("auroc_minus_judge_ci95")
            print(f"  {m:<20} val {r['val']['auroc']:.3f}  test {r['test']['auroc']:.3f} "
                  f"[{r['test']['auroc_ci95'][0]:.3f},{r['test']['auroc_ci95'][1]:.3f}]"
                  + (f"  Δjudge(test) [{d[0]:+.3f},{d[1]:+.3f}]" if d else ""))

    if run_all or args.exp == "E1":
        exp = Experiment("E1_margin_logit", cfg, {"plan": "PLAN.md E1", "switch": "detection.margin_scale"})
        res = {}
        for m, (fp, fl) in E1_PAIRS.items():
            a, av, at = evaluate(m, fp, val, test, yv, yt)
            b, bv, bt = evaluate(m, fl, val, test, yv, yt)
            dv, dt = b["val"]["auroc"] - a["val"]["auroc"], b["test"]["auroc"] - a["test"]["auroc"]
            ci_v, ci_t = paired_diff(yv, bv, av), paired_diff(yt, bt, at)
            res[m] = {"prob": a, "logit": b, "delta_val": dv, "delta_test": dt, "delta_val_ci95": ci_v,
                      "delta_test_ci95": ci_t, "verdict": verdict(dv, dt, ci_v, THRESH_AUROC)}
        exp.save_json("metrics.json", {**base_info, "threshold_auroc": THRESH_AUROC, "pairs": res})
        exp.finish({m: r["verdict"] for m, r in res.items()})
        results["E1"] = {"dir": str(exp.dir), "pairs": res}
        print(f"[E1] {exp.dir}")
        for m, r in res.items():
            print(f"  {m:<14} prob→logit  val {r['prob']['val']['auroc']:.3f}→{r['logit']['val']['auroc']:.3f} "
                  f"(Δ {r['delta_val']:+.3f}, CI [{r['delta_val_ci95'][0]:+.3f},{r['delta_val_ci95'][1]:+.3f}])  "
                  f"test Δ {r['delta_test']:+.3f}  → {r['verdict']}")

    if run_all or args.exp in ("E2", "E3"):
        from telerag.pipeline import get_atomic

        exp = Experiment("E2E3_atomic_faithfulness", cfg, {"plan": "PLAN.md E2/E3", "switch": "detection.atomic_facts"})
        for recs in (val, test):
            host_name = HOST
            at = get_atomic(cfg, host_name, recs, items, chunks)
            for r in recs:
                r["atomic"] = at[r["qid"]]["features"]
        res = {}
        for m, base in {"Ours": FREE + MV + NLI_CON + SLM_CON,
                        "Ours-lite": FREE + ["agree_closed"] + NLI_CON + SLM_CON,
                        "NLI-contrast": NLI_CON}.items():
            a, av, at_ = evaluate(m, base, val, test, yv, yt)
            b, bv, bt = evaluate(m, base + ATOMIC, val, test, yv, yt)
            dv, dt = b["val"]["auroc"] - a["val"]["auroc"], b["test"]["auroc"] - a["test"]["auroc"]
            ci_v = paired_diff(yv, bv, av)
            res[m] = {"off": a, "on": b, "delta_val": dv, "delta_test": dt, "delta_val_ci95": ci_v,
                      "delta_test_ci95": paired_diff(yt, bt, at_), "verdict": verdict(dv, dt, ci_v, THRESH_AUROC)}
        alone, _, _ = evaluate("Atomic-only", ATOMIC, val, test, yv, yt)
        faith_v = np.array([1 - r["atomic"]["faithfulness"] for r in val])
        faith_t = np.array([1 - r["atomic"]["faithfulness"] for r in test])
        faith = fixed_score_metrics(faith_v, faith_t)
        n_claims = np.array([r["atomic"]["atomic_n"] for r in val + test])
        diag = {"claims_per_answer_median": float(np.median(n_claims)),
                "claims_le2_ratio": float(np.mean(n_claims <= 2)),
                "faithfulness_mean_correct": float(np.mean([r["atomic"]["faithfulness"] for r in val + test
                                                            if not r["hallucinated"]])),
                "faithfulness_mean_halluc": float(np.mean([r["atomic"]["faithfulness"] for r in val + test
                                                           if r["hallucinated"]]))}
        exp.save_json("metrics.json", {**base_info, "threshold_auroc": THRESH_AUROC, "pairs": res,
                                       "atomic_only": alone, "faithfulness_as_detector": faith, "diagnostics": diag})
        exp.finish({m: r["verdict"] for m, r in res.items()})
        results["E2"] = {"dir": str(exp.dir), "pairs": res, "atomic_only": alone, "faith": faith, "diag": diag}
        print(f"[E2/E3] {exp.dir}")
        for m, r in res.items():
            print(f"  {m:<14} +atomic  val {r['off']['val']['auroc']:.3f}→{r['on']['val']['auroc']:.3f} "
                  f"(Δ {r['delta_val']:+.3f}, CI [{r['delta_val_ci95'][0]:+.3f},{r['delta_val_ci95'][1]:+.3f}])  "
                  f"test Δ {r['delta_test']:+.3f}  → {r['verdict']}")
        print(f"  Atomic-only val {alone['val']['auroc']:.3f} test {alone['test']['auroc']:.3f}; "
              f"1−Faithfulness val {faith['val']['auroc']:.3f} test {faith['test']['auroc']:.3f}; 诊断 {diag}")

    if run_all or args.exp == "E4":
        from telerag.correction import accuracy_of, apply_correction

        exp = Experiment("E4_correction", cfg, {"plan": "PLAN.md E4", "switch": "correction.mode"})
        feats = FREE + MV + NLI_CON + SLM_CON
        _, rv, rt = evaluate("Ours", feats, val, test, yv, yt)
        alt = views["bm25"]
        acc_none_v, acc_none_t = accuracy_of(val, [r["pred"] for r in val]), accuracy_of(test, [r["pred"] for r in test])
        acc_all_v = accuracy_of(val, [alt.get(r["qid"], r["pred"]) for r in val])
        acc_all_t = accuracy_of(test, [alt.get(r["qid"], r["pred"]) for r in test])
        grid = {t: accuracy_of(val, apply_correction(val, rv, alt, "switch_view", t)) for t in (0.1, 0.2, 0.3)}
        tau = max(grid, key=grid.get)                       # 只在 val 上选
        acc_v = grid[tau]
        acc_t = accuracy_of(test, apply_correction(test, rt, alt, "switch_view", tau))
        bar = max(acc_none_v, acc_all_v)
        dv, dt = acc_v - bar, acc_t - max(acc_none_t, acc_all_t)
        from scipy.stats import binomtest

        def mcnemar(recs, p1, p2):
            g = sum((a == r["answer"]) and (b != r["answer"]) for r, a, b in zip(recs, p1, p2))
            l = sum((a != r["answer"]) and (b == r["answer"]) for r, a, b in zip(recs, p1, p2))
            return {"gain": g, "loss": l, "p": binomtest(g, g + l).pvalue if g + l else 1.0}

        corr_v = apply_correction(val, rv, alt, "switch_view", tau)
        best_base_v = [r["pred"] for r in val] if acc_none_v >= acc_all_v else [alt.get(r["qid"], r["pred"]) for r in val]
        sig = mcnemar(val, corr_v, best_base_v)
        v = ("成立（显著）" if sig["p"] < 0.05 else "成立（未达显著）") if dv >= THRESH_ACC_E4 and dt > 0 else "不成立"
        res = {"tau_grid_val": grid, "tau": tau, "acc_none": [acc_none_v, acc_none_t], "acc_all_bm25": [acc_all_v, acc_all_t],
               "acc_corrected": [acc_v, acc_t], "delta_val": dv, "delta_test": dt, "mcnemar_val_vs_best_base": sig,
               "verdict": v}
        exp.save_json("metrics.json", {**base_info, "threshold_acc": THRESH_ACC_E4, **res})
        exp.finish({"verdict": v, "tau": tau})
        results["E4"] = {"dir": str(exp.dir), **res}
        print(f"[E4] {exp.dir}\n  val: 不纠正 {acc_none_v:.3f} / 全用BM25 {acc_all_v:.3f} / 纠正(τ={tau}) {acc_v:.3f}；"
              f"test: {acc_none_t:.3f} / {acc_all_t:.3f} / {acc_t:.3f} → {v}")

    if args.exp == "E5":
        os.environ.pop("TELERAG_NO_API", None)
        from telerag.consistency import sc_agreement, self_consistency
        from telerag.llm import LLMClient

        exp = Experiment("E5_self_consistency", cfg, {"plan": "PLAN.md E5", "K": args.sc})
        llm = LLMClient(cfg["llm"])
        out = {}
        splits = {"val": val, "test": test}
        todo = ["val", "test"] if args.sc_split == "both" else [args.sc_split]
        for sp in todo:
            recs = splits[sp]
            retrieval = {r["qid"]: [tuple(x) for x in r["retrieved"]] for r in recs}
            res = self_consistency(llm, [items[r["qid"]] for r in recs], retrieval, chunks, k=args.sc)
            out[sp] = np.array([1 - sc_agreement(r["pred"], res[r["qid"]]["preds"]) for r in recs])
            exp.save_records(f"sc_samples_{sp}.jsonl", [{"qid": q, **v} for q, v in res.items()])
        summ = {"usage": llm.usage, "cost_cny_estimate": llm.usage.get("cost_cny", 0.0)}
        if "val" in out:
            from sklearn.metrics import roc_auc_score
            summ["val_auroc"] = float(roc_auc_score(yv, out["val"]))
            _, ov, ot = evaluate("Ours", FREE + MV + NLI_CON + SLM_CON, val, test, yv, yt)
            summ["ours_val_auroc"] = float(roc_auc_score(yv, ov))
            summ["ours_minus_sc_val_ci95"] = paired_diff(yv, ov, out["val"])
        if "val" in out and "test" in out:
            summ["sc"] = fixed_score_metrics(out["val"], out["test"])
            summ["ours_minus_sc_test_ci95"] = paired_diff(yt, ot, out["test"])
        exp.save_json("metrics.json", {**base_info, **summ})
        exp.finish({k: v for k, v in summ.items() if k != "usage"}, cost_cny=summ["cost_cny_estimate"])
        print(f"[E5] {exp.dir}\n  {json.dumps({k: v for k, v in summ.items() if k != 'usage'}, ensure_ascii=False, default=float)}")
        results["E5"] = {"dir": str(exp.dir), **summ}

    from telerag.config import resolve
    (resolve("outputs") / "round1_latest.json").write_text(
        json.dumps({"base": base_info, **{k: {"dir": v["dir"]} for k, v in results.items()}}, ensure_ascii=False,
                   indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
