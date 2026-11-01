"""E10：RAGTruth 迁移（PLAN.md E10）。不调用 API。

只迁移本地核验器部分：RAGTruth 的回答是给定的，没有闭卷、BM25 等视角，多视角一致性无法迁移。
任务：回答级幻觉检测（回答中有任一标注片段即为幻觉），与 RAGTruth 论文和 LettuceDetect 的样本级设置相同。

检测器（全部在本地运行）：
  NLI-claim     原子拆分（atomic.split_claims，规则与 3GPP 实验相同）后，每条陈述对上下文各段取最大蕴含概率，
                风险 = 1 − 最小蕴含概率（与 3GPP 实验里的 atomic_min 相同）
  SLM-verify    Qwen3-0.6B 判断"回答是否完全被上下文支持"，风险 = 1 − P(Yes)
  Fusion        两者特征（atomic_n/min/mean、faithfulness、slm_yes）做逻辑回归

协议：融合模型和阈值在 RAGTruth train 划分中按 seed=3407 抽取的 1,000 条上训练；在官方 test 划分（2,700 条）上只评测一次。
上下文切分：按段落切分，每段截断到 512 个 token 以内（DeBERTa 的上限）。

用法：
  python scripts/run_ragtruth.py
  python scripts/run_ragtruth.py --limit 200    # 小样本试跑
"""
import argparse
import json
import random
import time

import numpy as np

from _common import ROOT  # noqa: F401

SLM_PROMPT = """Context:
{context}

Response:
{response}

Is every statement in the response supported by the context? Reply with Yes or No only."""


def context_text(src):
    info = src["source_info"]
    if isinstance(info, str):
        return info
    if src["task_type"] == "QA":
        return f"Question: {info.get('question', '')}\n{info.get('passages', '')}"
    return json.dumps(info, ensure_ascii=False)


def segments(text, max_words=300):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out = []
    for p in paras:
        w = p.split()
        for i in range(0, len(w), max_words):
            out.append(" ".join(w[i:i + max_words]))
    return out or [text[:2000]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-train", type=int, default=1000)
    args = p.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from telerag.atomic import split_claims
    from telerag.config import load_config, resolve
    from telerag.detect import NLIVerifier
    from telerag.explog import Experiment
    from telerag.metrics import best_f1_threshold, bootstrap_ci

    cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    d = resolve("data/raw/ragtruth")
    src = {json.loads(l)["source_id"]: json.loads(l) for l in open(d / "source_info.jsonl", encoding="utf-8")}
    R = [json.loads(l) for l in open(d / "response.jsonl", encoding="utf-8")]
    train = [r for r in R if r["split"] == "train"]
    random.Random(3407).shuffle(train)
    train = train[: args.n_train]
    test = [r for r in R if r["split"] == "test"][: args.limit]
    exp = Experiment("E10_ragtruth", cfg, {"plan": "PLAN.md E10", "n_train": len(train), "n_test": len(test)})

    cache_p = resolve("data/processed/ragtruth_scores.jsonl")
    cache = {}
    if cache_p.exists():
        for line in cache_p.open(encoding="utf-8"):
            r = json.loads(line)
            cache[r["id"]] = r
    todo = [r for r in train + test if r["id"] not in cache]
    if todo:
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer

        nli = NLIVerifier(cfg["detection"])
        tok = AutoTokenizer.from_pretrained(cfg["detection"]["slm_model"])
        slm = AutoModelForCausalLM.from_pretrained(cfg["detection"]["slm_model"], dtype="auto",
                                                   device_map=cfg["detection"]["device"]).eval()
        yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
        no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
        with cache_p.open("a", encoding="utf-8") as f:
            for r in tqdm(todo, desc="RAGTruth 打分"):
                t0 = time.time()
                ctx = context_text(src[r["source_id"]])
                claims = split_claims(r["response"], r["response"][:500])
                ent = nli.entail_matrix(segments(ctx), claims).max(axis=1)
                ctx_ids = tok.encode(ctx, add_special_tokens=False)[:1500]
                resp_ids = tok.encode(r["response"], add_special_tokens=False)[:500]
                msg = SLM_PROMPT.format(context=tok.decode(ctx_ids), response=tok.decode(resp_ids))
                text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
                enc = tok(text, return_tensors="pt").to(slm.device)
                with torch.no_grad():
                    h = slm.model(**enc).last_hidden_state[:, -1, :]
                    lp = torch.log_softmax(slm.lm_head(h).float(), -1)[0]
                p_yes = float(torch.sigmoid(torch.logsumexp(lp[yes], -1) - torch.logsumexp(lp[no], -1)))
                row = {"id": r["id"], "atomic_n": float(len(claims)), "atomic_min": float(ent.min()),
                       "atomic_mean": float(ent.mean()), "faithfulness": float((ent > 0.5).mean()),
                       "slm_yes": p_yes, "seconds": time.time() - t0}
                cache[r["id"]] = row
                f.write(json.dumps(row) + "\n")
                f.flush()

    feats = ["atomic_n", "atomic_min", "atomic_mean", "faithfulness", "slm_yes"]
    task = {r["id"]: src[r["source_id"]]["task_type"] for r in R}
    X = lambda rs: np.array([[cache[r["id"]][k] for k in feats] for r in rs])
    y = lambda rs: np.array([int(bool(r["labels"])) for r in rs])
    Xtr, ytr, Xte, yte = X(train), y(train), X(test), y(test)
    scores_tr = {"NLI-claim": 1 - Xtr[:, 1], "SLM-verify": 1 - Xtr[:, 4]}
    scores_te = {"NLI-claim": 1 - Xte[:, 1], "SLM-verify": 1 - Xte[:, 4]}
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(Xtr, ytr)
    scores_tr["Fusion"], scores_te["Fusion"] = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xte)[:, 1]
    res = {}
    for m in scores_te:
        tau = best_f1_threshold(ytr, scores_tr[m])
        s = scores_te[m]
        row = {"auroc": float(roc_auc_score(yte, s)),
               "auroc_ci95": bootstrap_ci(lambda i: roc_auc_score(yte[i], s[i]) if 0 < yte[i].sum() < len(i) else 0.5,
                                          len(yte)),
               "f1": float(f1_score(yte, s >= tau, zero_division=0)), "tau": tau}
        for t in ("QA", "Summary", "Data2txt"):
            idx = np.array([task[r["id"]] == t for r in test])
            if idx.any() and 0 < yte[idx].sum() < idx.sum():
                row[f"auroc[{t}]"] = float(roc_auc_score(yte[idx], s[idx]))
                row[f"f1[{t}]"] = float(f1_score(yte[idx], s[idx] >= tau, zero_division=0))
        res[m] = row
    ref = {"RAGTruth paper / LettuceDetect（引用，回答级 F1）": "LettuceDetect-large 79.22%（摘要已核实）；其余数字需查原文"}
    summ = {"n_train": len(train), "n_test": len(test), "halluc_rate_test": float(yte.mean()), "methods": res,
            "seconds_per_response": float(np.mean([cache[r["id"]]["seconds"] for r in test])), "reference": ref,
            "note": "只迁移本地核验器；多视角一致性需要多个检索视角，RAGTruth 不提供，无法迁移。"}
    exp.save_json("metrics.json", summ)
    exp.finish({m: round(v["auroc"], 4) for m, v in res.items()})
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
