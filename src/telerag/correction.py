"""E4 纠正：对风险分最高的 τ 比例题目，把最终答案换成另一个视角的答案。

mode = "none"         不纠正（默认，与宿主基线一致）
mode = "switch_view"  风险最高的 τ 比例题改用 alt_view 视角的答案（视角答案已有缓存，不调用 API）
τ 只在 val 上选（候选值 0.1 / 0.2 / 0.3），然后固定，用于 test。
"""
import numpy as np


def apply_correction(records, risk, alt_preds, mode="none", tau=0.2):
    """返回纠正后的预测列表（不修改 records）。alt_preds: {qid: 另一视角的预测}。"""
    preds = [r["pred"] for r in records]
    if mode == "none" or tau <= 0:
        return preds
    if mode != "switch_view":
        raise ValueError(f"未知的纠正模式 {mode}")
    risk = np.asarray(risk, float)
    k = int(round(tau * len(records)))
    if k == 0:
        return preds
    # 稳定排序，风险分并列时按原顺序，保证结果可复现
    top = np.argsort(-risk, kind="stable")[:k]
    out = list(preds)
    for i in top:
        alt = alt_preds.get(records[i]["qid"])
        if alt is not None:
            out[i] = alt
    return out


def accuracy_of(records, preds):
    return float(np.mean([p == r["answer"] for r, p in zip(records, preds)])) if records else 0.0
