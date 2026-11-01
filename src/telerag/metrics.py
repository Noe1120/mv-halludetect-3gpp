"""评测指标：答题准确率、检测 AUROC/AUPRC/F1、风险-覆盖率曲线、bootstrap 置信区间。"""
import numpy as np


def accuracy(records):
    return float(np.mean([r["pred"] == r["answer"] for r in records])) if records else 0.0


def detection_metrics(y_true, risk, tau=0.5):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_true, risk = np.asarray(y_true, int), np.asarray(risk, float)
    out = {"n": len(y_true), "pos_rate": float(y_true.mean())}
    if 0 < y_true.sum() < len(y_true):
        out["auroc"] = float(roc_auc_score(y_true, risk))
        out["auprc"] = float(average_precision_score(y_true, risk))
    out["f1@tau"] = float(f1_score(y_true, risk >= tau, zero_division=0))
    return out


def best_f1_threshold(y_true, risk):
    """在开发集上选使 F1 最大的阈值。"""
    from sklearn.metrics import f1_score

    cands = np.unique(np.round(risk, 4))
    best = max(cands, key=lambda t: f1_score(y_true, risk >= t, zero_division=0))
    return float(best)


def risk_coverage(correct, risk, points=20):
    """按风险从低到高只回答前 x% 的题，返回 (覆盖率, 已回答部分的准确率)。用于画选择性回答曲线。"""
    correct, risk = np.asarray(correct, float), np.asarray(risk, float)
    order = np.argsort(risk)
    out = []
    for cov in np.linspace(1.0 / points, 1.0, points):
        k = max(1, int(round(cov * len(order))))
        out.append((float(cov), float(correct[order[:k]].mean())))
    return out


def aurc(correct, risk):
    """Area Under Risk-Coverage curve（越低越好）。"""
    correct, risk = np.asarray(correct, float), np.asarray(risk, float)
    order = np.argsort(risk)
    errors = 1 - correct[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def bootstrap_ci(values_fn, n_items, n_boot=1000, seed=0):
    """values_fn(idx) -> 标量；返回 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    stats = [values_fn(rng.integers(0, n_items, n_items)) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
