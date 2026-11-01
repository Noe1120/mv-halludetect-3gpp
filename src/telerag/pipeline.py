"""把各模块串起来：检索结果缓存、RAG 答题、证据打分缓存。"""
import json
import time

from .config import config_hash, resolve
from .corpus import build_chunks, load_chunks
from .generate import build_messages, parse_response
from .retrieval import Retriever, build_query


def retrieval_tag(cfg):
    return config_hash({"chunking": cfg["chunking"], "retrieval": cfg["retrieval"]})


def get_chunks(cfg):
    path = build_chunks(cfg)
    return load_chunks(path), path


def get_retrieval(cfg, items, chunks=None, chunk_path=None):
    """返回 {qid: [(cid, score), ...]}，结果按检索配置缓存到磁盘。"""
    out_path = resolve(f"data/processed/retrieval_{retrieval_tag(cfg)}.jsonl")
    cached = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cached[r["qid"]] = [tuple(x) for x in r["retrieved"]]
    todo = [it for it in items if it["qid"] not in cached]
    if todo:
        if chunks is None:
            chunks, chunk_path = get_chunks(cfg)
        retriever = Retriever(cfg, chunks, chunk_path)
        mode = cfg["retrieval"]["query_mode"]
        t0 = time.time()
        with out_path.open("a", encoding="utf-8") as f:
            for i in range(0, len(todo), 64):
                batch = todo[i:i + 64]
                res = retriever.search([build_query(it, mode) for it in batch])
                for it, r in zip(batch, res):
                    cached[it["qid"]] = r
                    f.write(json.dumps({"qid": it["qid"], "retrieved": r}) + "\n")
        print(f"[retrieval] {len(todo)} 题，平均 {(time.time() - t0) / len(todo) * 1000:.0f} ms/题")
    return {it["qid"]: cached[it["qid"]] for it in items}


def run_qa(cfg, items, llm, retrieval=None, chunks=None):
    """retrieval 为 None 时不检索（闭卷基线）。返回逐题记录。"""

    def one(it):
        retrieved = retrieval[it["qid"]] if retrieval is not None else None
        resp = llm.chat(build_messages(it, retrieved, chunks))
        parsed = parse_response(resp["text"], it["options"])
        return {
            "qid": it["qid"], "split": it["split"], "category": it["category"],
            "release": it["release"], "answer": it["answer"], **parsed,
            "hallucinated": int(parsed["pred"] != it["answer"]),
            "retrieved": retrieved, "raw": resp["text"],
            "prompt_tokens": resp["prompt_tokens"], "completion_tokens": resp["completion_tokens"],
            "cache_hit_tokens": resp.get("cache_hit_tokens", 0),
            "latency": resp["latency"],
        }

    return llm.map(one, items)


def get_scores(cfg, items, retrieval, chunks, which):
    """计算或读取证据打分。which: 'nli' 或 'slm'。

    打分只依赖题目和检索结果，与哪个大模型答题无关，所以按检索配置缓存，所有大模型共用。
    """
    from .detect import NLIVerifier, SLMVerifier

    model = cfg["detection"][f"{which}_model"]
    out_path = resolve(f"data/processed/scores_{which}_{retrieval_tag(cfg)}_{config_hash(model)}.jsonl")
    cached = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cached[r["qid"]] = r
    todo = [it for it in items if it["qid"] not in cached]
    if todo:
        from tqdm import tqdm

        verifier = (NLIVerifier if which == "nli" else SLMVerifier)(cfg["detection"])
        with out_path.open("a", encoding="utf-8") as f:
            for it in tqdm(todo, desc=f"{which} 打分"):
                t0 = time.time()
                s = verifier.score(it, retrieval[it["qid"]], chunks)
                r = {"qid": it["qid"], "scores": s, "seconds": time.time() - t0}
                cached[it["qid"]] = r
                f.write(json.dumps(r) + "\n")
    return {it["qid"]: cached[it["qid"]] for it in items}


def get_atomic(cfg, run_name, records, items_by_qid, chunks):
    """E2/E3：原子事实核查和忠实度。依赖答题输出里的解释句，所以缓存键里带上答题实验名。"""
    from .atomic import atomic_features
    from .detect import NLIVerifier

    model = cfg["detection"]["nli_model"]
    out_path = resolve(f"data/processed/atomic_{run_name}_{config_hash(model)}.jsonl")
    cached = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cached[r["qid"]] = r
    todo = [r for r in records if r["qid"] not in cached]
    if todo:
        from tqdm import tqdm

        verifier = NLIVerifier(cfg["detection"])
        with out_path.open("a", encoding="utf-8") as f:
            for rec in tqdm(todo, desc="原子事实核查"):
                t0 = time.time()
                feats, claims = atomic_features(verifier, items_by_qid[rec["qid"]], rec, chunks)
                r = {"qid": rec["qid"], "features": feats, "claims": claims, "seconds": time.time() - t0}
                cached[rec["qid"]] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {r["qid"]: cached[r["qid"]] for r in records}
