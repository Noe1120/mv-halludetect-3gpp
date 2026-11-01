"""多视角一致性信号。

思路：同一道题从不同"视角"各回答一次。视角可以是闭卷（不检索），也可以是不同的检索器（BM25、稠密向量）。
如果各视角的答案和最终的 RAG 答案不一致，说明答案依赖于具体看到了哪些证据，出错的可能性更大。

这些视角的答案本来就是消融实验的副产品，所以这个信号不需要额外调用 API。
对比基线是自洽性采样（Self-Consistency，Wang et al. 2023）：同一提示词采样 K 次，看答案是否一致，需要额外 K 次 API 调用。
"""
from collections import Counter

import numpy as np

from .generate import build_messages, parse_response
from .io import load_records

DEFAULT_VIEWS = ["closed", "bm25", "dense"]


def load_views(split, views=DEFAULT_VIEWS):
    """返回 {视角名: {qid: 预测选项}}；某个视角的结果文件不存在时跳过并提示。"""
    out = {}
    for v in views:
        try:
            out[v] = {r["qid"]: r["pred"] for r in load_records(f"{v}_{split}")}
        except FileNotFoundError:
            print(f"[consistency] 缺少 {v}_{split}，跳过该视角")
    return out


def view_features(rec, views):
    """agree_<v>：视角 v 与最终答案是否一致；view_agree：所有视角的一致比例；view_vote：最终答案的得票率。"""
    pred = rec["pred"]
    f = {}
    votes = [pred]
    for v, preds in views.items():
        p = preds.get(rec["qid"])
        f[f"agree_{v}"] = float(p == pred)
        votes.append(p)
    f["view_agree"] = float(np.mean([f[f"agree_{v}"] for v in views])) if views else 1.0
    f["view_vote"] = Counter(votes)[pred] / len(votes)
    return f


def self_consistency(llm, items, retrieval, chunks, k=5, temperature=0.7):
    """同一 RAG 提示词随机采样 k 次。返回 {qid: {'preds': [...], 'tokens': int}}。"""

    def one(it):
        msgs = build_messages(it, retrieval[it["qid"]], chunks)
        preds, tokens = [], 0
        for i in range(1, k + 1):
            r = llm.chat(msgs, temperature=temperature, sample=i)
            preds.append(parse_response(r["text"], it["options"])["pred"])
            tokens += r["prompt_tokens"] + r["completion_tokens"]
        return it["qid"], {"preds": preds, "tokens": tokens}

    return dict(llm.map(one, items))


def sc_agreement(pred, samples):
    """自洽性：k 次采样中与最终答案相同的比例。"""
    return sum(p == pred for p in samples) / len(samples) if samples else 0.0
