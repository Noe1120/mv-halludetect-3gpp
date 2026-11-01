"""泛化性实验（不调用 API）：

1. 跨模型迁移：检测器在一个答题模型（deepseek-flash）的输出上训练，直接用于另一个答题模型（deepseek-v4-pro），
   反过来也做一次。用来说明检测器学到的不是某个模型特有的模式。
2. 无训练版本：MV-Consist 直接用 1 − 视角得票率作为风险分数，不需要任何标注数据。

前提：已运行 run_detect.py --name hybrid_rerank 和 run_detect.py --name pro__hybrid_rerank --views-prefix pro__。
跨模型比较只用两个模型都回答过的题（pro 跑了测试集前 1,000 题）。

用法：
  python scripts/run_transfer.py
"""
import json

import numpy as np

from _common import ROOT

import run_detect as rd


def load(name, views_prefix):
    from telerag.consistency import load_views, view_features
    from telerag.io import load_records

    recs = load_records(f"{name}_test_scored")
    views = load_views("test", [f"{views_prefix}{v}" for v in ("closed", "bm25", "dense")])
    views = {k[len(views_prefix):]: v for k, v in views.items()}
    for r in recs:
        r["views"] = view_features(r, views)
    return recs


def main():
    from sklearn.metrics import roc_auc_score

    from telerag.detect import RiskModel
    from telerag.metrics import bootstrap_ci

    models = {"flash": load("hybrid_rerank", ""), "pro": load("pro__hybrid_rerank", "pro__")}
    common = set(r["qid"] for r in models["flash"]) & set(r["qid"] for r in models["pro"])
    sub = {k: [r for r in v if r["qid"] in common] for k, v in models.items()}

    def auroc_ci(y, s):
        y, s = np.asarray(y), np.asarray(s)
        return roc_auc_score(y, s), bootstrap_ci(
            lambda i: roc_auc_score(y[i], s[i]) if 0 < y[i].sum() < len(i) else 0.5, len(y))

    out = {"n_common": len(common), "transfer": {}, "training_free": {}}
    for src, tgt in [("flash", "pro"), ("pro", "flash")]:
        for m in ["MV-Consist", "SLM-contrast", "Ours"]:
            model = RiskModel(rd.METHODS[m]).fit(models[src])
            if src == "flash":   # 训练集不能包含评测的题；flash 的 2,800 题包含这 1,000 题，所以排除掉
                model = RiskModel(rd.METHODS[m]).fit([r for r in models["flash"] if r["qid"] not in common])
            y = [r["hallucinated"] for r in sub[tgt]]
            a, ci = auroc_ci(y, model.predict(sub[tgt]))
            out["transfer"][f"{src}→{tgt} {m}"] = {"auroc": a, "ci95": ci}
            print(f"{src}→{tgt}  {m:<13} AUROC {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    for name, recs in models.items():
        y = [r["hallucinated"] for r in recs]
        a, ci = auroc_ci(y, [1 - r["views"]["view_vote"] for r in recs])
        out["training_free"][name] = {"auroc": a, "ci95": ci, "n": len(recs)}
        print(f"{name} 无训练 MV（1 − 得票率） AUROC {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  n={len(recs)}")
    (ROOT / "outputs" / "transfer.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
