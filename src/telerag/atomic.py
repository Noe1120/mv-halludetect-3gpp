"""E2 原子事实拆分 + 本地 NLI 核查；E3 RAGAS 式忠实度（本地实现）。

拆分规则（预注册，不再调整）：
  1. 去掉解释中的证据编号，如 "[2]"、"(Retrieval 3)"；
  2. 按句号、分号、逗号以及连接词 and / while / whereas / but 切分，去掉子句开头残留的连接词；
  3. 只保留至少 3 个词的子句；
  4. 一个都没有时，退回到"题目 + 所选选项"这一条主张。

忠实度沿用 RAGAS 的定义：被检索上下文蕴含的陈述数 / 陈述总数。
陈述来自上面的拆分，蕴含判定用 DeBERTa-NLI（蕴含概率 > 0.5）。
官方 RAGAS 用 LLM 拆分陈述并判定蕴含；这里换成规则拆分加本地 NLI，不调用 API。
"""
import re

import numpy as np

SPLIT_RE = re.compile(r"[.;,](?:\s|$)|\s(?:and|while|whereas|but)\s", re.I)
CITE_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]|\(\s*retrieval\s*\d+\s*\)", re.I)
# 逗号切分后，子句开头可能残留连接词（如 ", and the gNB ..."），去掉
LEAD_CONJ_RE = re.compile(r"^(?:and|while|whereas|but)\s+", re.I)
MIN_WORDS = 3


def split_claims(explanation, fallback):
    text = CITE_RE.sub(" ", explanation or "")
    parts = [LEAD_CONJ_RE.sub("", p.strip(" -:")).strip() for p in SPLIT_RE.split(text)]
    claims = [p for p in parts if len(p.split()) >= MIN_WORDS]
    return claims if claims else [fallback]


def fallback_claim(item, pred):
    if pred is None or pred not in item["options"]:
        return item["question"]
    return f"{item['question']} {item['options'][pred]}"


def atomic_features(verifier, item, rec, chunks, threshold=0.5):
    """返回 {atomic_n, atomic_min, atomic_mean, faithfulness}，以及逐条主张的打分（便于诊断）。

    verifier 是 detect.NLIVerifier；每条主张对每个证据块计算蕴含概率，取最大值。
    """
    claims = split_claims(rec.get("explanation", ""), fallback_claim(item, rec["pred"]))
    ent = verifier.entail_matrix([chunks[cid]["text"] for cid, _ in rec["retrieved"]], claims) \
        if rec.get("retrieved") else np.zeros((len(claims), 1))
    best = ent.max(axis=1) if ent.size else np.zeros(len(claims))
    feats = {
        "atomic_n": float(len(claims)),
        "atomic_min": float(best.min()),
        "atomic_mean": float(best.mean()),
        "faithfulness": float((best > threshold).mean()),
    }
    return feats, [{"claim": c, "entail": float(b)} for c, b in zip(claims, best)]
