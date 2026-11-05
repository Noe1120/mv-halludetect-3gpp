"""为什么改变证据比重复采样更能暴露错误：用已有缓存做机制分析（IJMLC 拒稿意见第 2 条，2026-11-05）。

全部只读缓存（TELERAG_NO_API=1），不调用 API。
  python scripts/run_disagreement_analysis.py

分析思路：把每种扰动看成从同一道题生成"另一个答案"。一致性检测器的风险分是 1 − 一致比例，它的期望由
"复现概率" c(q) = P(扰动后的答案 = 宿主答案 | q) 决定。扰动有没有用，取决于 c(q) 在答错题和答对题上差多少。
输出 outputs/exp/<时间>_REV2_disagreement/metrics.json，内容：
  1. 复现概率：每种扰动（采样、闭卷、BM25、稠密）在答错 / 答对题上的平均复现概率，以及两者之差（配对 bootstrap CI）；
  2. 同等扰动次数的对照：3 次采样 vs 3 个证据视角（AUROC 及配对差值 CI），排除"视角多所以更好"的解释；
  3. 错误相关性：宿主错误与每个扰动答案错误之间的 phi 系数和 Yule's Q（集成学习里的多样性度量）；
  4. 错误分类：按闭卷答案是否正确、检索视角是否一致，把宿主错误分成三类，分别给出两种扰动的检出率；
  5. 两种扰动检出的错误是否重叠，以及把两种一致比例相加后的无训练 AUROC（看两者是否互补）。
"""
import json
import os

import numpy as np

from _common import ROOT  # noqa: F401

import run_eval as RE

VIEWS = ("closed", "bm25", "dense")


def phi_and_q(a, b):
    """两个 0/1 错误指示向量的 phi 系数和 Yule's Q（Kuncheva and Whitaker 2003）。"""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n11, n00 = np.sum(a & b), np.sum(~a & ~b)
    n10, n01 = np.sum(a & ~b), np.sum(~a & b)
    den = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    phi = (n11 * n00 - n10 * n01) / den if den else float("nan")
    qd = n11 * n00 + n10 * n01
    q = (n11 * n00 - n10 * n01) / qd if qd else float("nan")
    return round(float(phi), 4), round(float(q), 4)


def main():
    from sklearn.metrics import roc_auc_score

    from telerag.config import load_config
    from telerag.explog import Experiment
    from telerag.metrics import bootstrap_ci

    os.environ["TELERAG_NO_API"] = "1"
    cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    items, val_q, test_q = RE.load_split(cfg)
    view_runs = {v: RE.union_runs(v) for v in VIEWS}
    host = RE.union_runs(RE.HOST)
    sc = {}
    e5 = sorted((ROOT / "outputs" / "exp").glob("*_E5_self_consistency"))[-1]
    for sp in ("val", "test"):
        for line in open(e5 / f"sc_samples_{sp}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            sc[r["qid"]] = r["preds"]

    out = {"note": "cache only; c = replication probability of the host answer under a perturbation"}
    for split, qids in (("val", val_q), ("test", test_q)):
        y = np.array([host[q]["hallucinated"] for q in qids], bool)
        pred = [host[q]["pred"] for q in qids]
        # 每道题、每种扰动下"复现宿主答案"的比例
        c_s5 = np.array([np.mean([p == h for p in sc[q]]) for q, h in zip(qids, pred)])
        c_s3 = np.array([np.mean([p == h for p in sc[q][:3]]) for q, h in zip(qids, pred)])
        rep = {v: np.array([view_runs[v][q]["pred"] == h for q, h in zip(qids, pred)], float) for v in VIEWS}
        c_v3 = np.mean([rep[v] for v in VIEWS], axis=0)
        rep_sample1 = np.array([sc[q][0] == h for q, h in zip(qids, pred)], float)

        res = {"n": len(qids), "n_wrong": int(y.sum())}
        # 1. 复现概率在答错 / 答对题上的差
        def gap(c):
            f = lambda idx: float(c[idx][~y[idx]].mean() - c[idx][y[idx]].mean())
            return {"wrong": round(float(c[y].mean()), 4), "correct": round(float(c[~y].mean()), 4),
                    "gap": round(f(np.arange(len(y))), 4), "gap_ci": [round(x, 4) for x in bootstrap_ci(f, len(y))]}
        res["replication"] = {"sample (one of 5)": gap(c_s5), "closed": gap(rep["closed"]), "bm25": gap(rep["bm25"]),
                              "dense": gap(rep["dense"]), "three views": gap(c_v3), "three samples": gap(c_s3)}
        # 差值的差：证据视角的复现概率差 − 采样的复现概率差（同为 3 个扰动答案）
        dd = lambda idx: float((c_v3[idx][~y[idx]].mean() - c_v3[idx][y[idx]].mean())
                               - (c_s3[idx][~y[idx]].mean() - c_s3[idx][y[idx]].mean()))
        res["gap_views_minus_samples"] = {"value": round(dd(np.arange(len(y))), 4),
                                          "ci": [round(x, 4) for x in bootstrap_ci(dd, len(y))]}

        # 2. AUROC：同为 3 个扰动答案
        def auc(s):
            return round(float(roc_auc_score(y, s)), 4)
        a_v3, a_s3, a_s5 = 1 - c_v3, 1 - c_s3, 1 - c_s5
        res["auroc"] = {"three views (MV-vote)": auc(a_v3), "three samples": auc(a_s3), "five samples (SC)": auc(a_s5),
                        "one sample": auc(1 - rep_sample1), **{f"{v} alone": auc(1 - rep[v]) for v in VIEWS},
                        "views + five samples (sum)": auc(a_v3 + a_s5)}
        fa, fb = RE.boot_auroc_fn(y.astype(int), a_v3), RE.boot_auroc_fn(y.astype(int), a_s3)
        res["auroc_views_minus_three_samples_ci"] = [round(x, 4) for x in
                                                      bootstrap_ci(lambda idx: fa(idx) - fb(idx), len(y))]
        fc = RE.boot_auroc_fn(y.astype(int), a_v3 + a_s5)
        res["auroc_sum_minus_views_ci"] = [round(x, 4) for x in bootstrap_ci(lambda idx: fc(idx) - fa(idx), len(y))]

        # 二值标记（任一扰动答案与宿主不同）：召回、误报率、Youden J
        def flag(f):
            return {"recall": round(float(f[y].mean()), 4), "fpr": round(float(f[~y].mean()), 4),
                    "youden": round(float(f[y].mean() - f[~y].mean()), 4),
                    "precision": round(float(y[f].mean()), 4) if f.any() else None}
        res["flag_any_disagreement"] = {"three views": flag(c_v3 < 1), "three samples": flag(c_s3 < 1),
                                        "five samples": flag(c_s5 < 1)}

        # 3. 错误相关性：宿主错误 vs 每个扰动答案错误
        err = {v: np.array([view_runs[v][q]["hallucinated"] for q in qids], bool) for v in VIEWS}
        gold = [items[q]["answer"] for q in qids]
        err["sample 1"] = np.array([sc[q][0] != g for q, g in zip(qids, gold)], bool)
        res["error_correlation_with_host"] = {k: dict(zip(("phi", "yule_q"), phi_and_q(y, e))) for k, e in err.items()}
        # 宿主答错时，扰动答案恰好是同一个错误选项的比例
        res["same_wrong_option_given_host_wrong"] = {
            **{v: round(float(rep[v][y].mean()), 4) for v in VIEWS},
            "sample 1": round(float(rep_sample1[y].mean()), 4)}

        # 4. 错误分类
        closed_right = ~err["closed"]
        retr_agree = (rep["bm25"] == 1) & (rep["dense"] == 1)
        cls = {"A: closed-book correct (evidence misled the model)": y & closed_right,
               "B: closed-book wrong, retrievers disagree": y & ~closed_right & ~retr_agree,
               "C: closed-book wrong, retrievers agree with host": y & ~closed_right & retr_agree}
        res["error_classes"] = {}
        for k, m in cls.items():
            res["error_classes"][k] = {
                "n": int(m.sum()), "share_of_errors": round(float(m.sum() / y.sum()), 4),
                "flagged_by_views": round(float((c_v3[m] < 1).mean()), 4) if m.any() else None,
                "flagged_by_five_samples": round(float((c_s5[m] < 1).mean()), 4) if m.any() else None}
        # 第 C 类里闭卷答案是否与宿主相同（即四个视角完全一致）
        c_cls = cls["C: closed-book wrong, retrievers agree with host"]
        res["error_classes"]["C: of which closed-book gives the same wrong option"] = round(
            float(rep["closed"][c_cls].mean()), 4) if c_cls.any() else None

        # 5. 两种扰动检出的错误是否重叠
        fv, fs = (c_v3 < 1) & y, (c_s5 < 1) & y
        res["overlap_of_flagged_errors"] = {
            "flagged by views": int(fv.sum()), "flagged by five samples": int(fs.sum()), "both": int((fv & fs).sum()),
            "samples only": int((fs & ~fv).sum()), "views only": int((fv & ~fs).sum()),
            "share of sample-flagged errors also flagged by views": round(float((fv & fs).sum() / fs.sum()), 4)}
        out[split] = res

    cfg_exp = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    exp = Experiment("REV2_disagreement", cfg_exp, {"plan": "IJMLC 拒稿意见第 2 条：机制分析，只读缓存"})
    exp.save_json("metrics.json", out)
    exp.finish({"done": True})
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("[REV2]", exp.dir)


if __name__ == "__main__":
    main()
