"""审稿式修改（2026-11-01）需要的补充统计。全部只读缓存和已有输出，不调用 API。

  python scripts/run_revision_stats.py

输出 outputs/exp/<时间>_REV_stats/metrics.json，内容：
  1. 只用闭卷一致性 a_closed（不加核验器）的检测结果，以及 free + a_closed；
  2. 采样与改证据两种扰动在 val 上被标记题目中真正答错的比例（精确率）；
  3. Flash 在 Pro 答过的 840 道 test 题上的准确率，以及 judge、SC 在这 840 题上对 Flash 回答的 AUROC；
  4. BM25 与 dense 视角各自答错的题目重叠情况；
  5. 置换重要性（test 上逐特征打乱，AUROC 下降量，重复 20 次）；
  6. 含 "all/none/both of the above" 一类选项的题目比例；
  7. 3 次额外答题调用的实测 API 延迟（来自缓存中记录的 latency）。
"""
import json
import re

import numpy as np

from _common import ROOT  # noqa: F401

import run_eval as RE


def main():
    from sklearn.metrics import roc_auc_score

    from telerag.config import load_config
    from telerag.explog import Experiment
    from telerag.metrics import bootstrap_ci

    import os
    os.environ["TELERAG_NO_API"] = "1"
    cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    items, val_q, test_q = RE.load_split(cfg)
    val, chunks, _, views = RE.build_records(cfg, items, val_q)
    test, _, _, _ = RE.build_records(cfg, items, test_q)
    yv = np.array([r["hallucinated"] for r in val])
    yt = np.array([r["hallucinated"] for r in test])
    out = {}

    # 1. a_closed 单独、free + a_closed
    for name, feats in {"a_closed only": ["agree_closed"], "free + a_closed": RE.FREE + ["agree_closed"],
                        "a_closed + SLM-contrast": ["agree_closed"] + RE.SLM_CON}.items():
        res, _, _ = RE.evaluate(name, feats, val, test, yv, yt)
        out[name] = {"val": round(res["val"]["auroc"], 4), "test": round(res["test"]["auroc"], 4),
                     "ci": [round(x, 3) for x in res["test"]["auroc_ci95"]]}

    # 2. 扰动的标记精确率（val）：SC 中任何一次采样与最终答案不同即"标记"；MV 中任何视角不同即"标记"
    sc = {}
    exp_dir = sorted((ROOT / "outputs" / "exp").glob("*_E5_self_consistency"))[-1]
    for sp in ("val", "test"):
        for line in open(exp_dir / f"sc_samples_{sp}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            sc[r["qid"]] = r["preds"]
    flag_sc = np.array([any(p != r["pred"] for p in sc[r["qid"]]) for r in val])
    flag_mv = np.array([r["views"]["view_vote"] < 1 for r in val])
    for name, f in (("sampling", flag_sc), ("evidence", flag_mv)):
        out[f"flag_{name}_val"] = {"flag_rate": round(f.mean(), 4), "recall": round(f[yv == 1].mean(), 4),
                                   "precision": round(yv[f].mean(), 4), "n_flag": int(f.sum())}

    # 3. Pro 的 840 题子集
    pro = RE.union_runs(RE.HOST, "pro__")
    tq = set(test_q)
    sub = [i for i, r in enumerate(test) if r["qid"] in pro]
    sub_set = {test[i]["qid"] for i in sub}
    assert sub_set <= tq
    out["pro_subset"] = {"n": len(sub), "n_pro_rows": len(pro),
                         "flash_acc": round(1 - yt[sub].mean(), 4),
                         "pro_acc": round(1 - np.mean([pro[q]["hallucinated"] for q in sub_set]), 4)}
    jt, _ = RE.judge_risk(cfg, items, test, chunks)
    sct = np.array([1 - sum(p == r["pred"] for p in sc[r["qid"]]) / len(sc[r["qid"]]) for r in test])
    _, _, ours_t = RE.evaluate("Ours", RE.FREE + RE.MV + RE.NLI_CON + RE.SLM_CON, val, test, yv, yt)
    ys = yt[sub]
    for name, s in (("judge_flash", jt), ("sc_flash", sct), ("ours_flash", ours_t)):
        ss = s[sub]
        out["pro_subset"][name] = {"auroc": round(roc_auc_score(ys, ss), 4),
                                   "ci": [round(x, 3) for x in bootstrap_ci(RE.boot_auroc_fn(ys, ss), len(ys))]}

    # 4. BM25 与 dense 各自答错题目的重叠（test）
    acc = {}
    for v in ("bm25", "dense"):
        recs = RE.union_runs(v)
        acc[v] = {q for q in test_q if recs[q]["hallucinated"]}
    b, d = acc["bm25"], acc["dense"]
    out["bm25_dense_errors_test"] = {"bm25_wrong": len(b), "dense_wrong": len(d), "both": len(b & d),
                                     "only_bm25": len(b - d), "only_dense": len(d - b),
                                     "jaccard": round(len(b & d) / len(b | d), 4)}

    # 5. 置换重要性
    from telerag.detect import RiskModel
    feats = RE.FREE + RE.MV + RE.NLI_CON + RE.SLM_CON
    model = RiskModel(feats).fit(val)
    Xt = model._X(test)
    base = roc_auc_score(yt, model.predict(test))
    rng = np.random.default_rng(0)
    imp = {}
    for j, f in enumerate(feats):
        drops = []
        for _ in range(20):
            X = Xt.copy()
            X[:, j] = X[rng.permutation(len(X)), j]
            drops.append(base - roc_auc_score(yt, model.clf.predict_proba(X)[:, 1]))
        imp[f] = round(float(np.mean(drops)), 4)
    out["perm_importance_test"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    out["perm_base_auroc"] = round(base, 4)
    # 共线性：vote share 与 a_v 之和
    vv = np.array([r["views"]["view_vote"] for r in val])
    sa = np.array([sum(r["views"][f"agree_{v}"] for v in ("closed", "bm25", "dense")) for r in val])
    out["vote_is_linear_in_agreements"] = bool(np.allclose(vv, (1 + sa) / 4))

    # 6. 集合类选项
    pat = re.compile(r"\b(all|none|both|neither)\b.*\b(above|options|answers|these)\b", re.I)
    def has_set(q):
        return any(pat.search(o) for o in items[q]["options"].values())
    n_set_test = sum(has_set(q) for q in test_q)
    miss_flag = [has_set(r["qid"]) for r in test]
    out["set_options"] = {"test_questions_with_set_option": n_set_test, "share": round(n_set_test / len(test_q), 4),
                          "selected_is_set_option_test": int(sum(bool(pat.search(items[r["qid"]]["options"].get(r["pred"], "")))
                                                                 for r in test if r["pred"]))}

    # 7. API 延迟
    lat = {}
    for v in ("closed", "bm25", "dense", "hybrid_rerank"):
        recs = RE.union_runs(v)
        l = [recs[q]["latency"] for q in test_q if recs[q].get("latency")]
        lat[v] = {"median_s": round(float(np.median(l)), 2), "mean_s": round(float(np.mean(l)), 2), "n": len(l)}
    out["api_latency_test"] = lat

    exp = Experiment("REV_stats", cfg, {"plan": "修改建议.md（2026-11-01）补充统计，只读缓存"})
    exp.save_json("metrics.json", out)
    exp.finish({"done": True})
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
