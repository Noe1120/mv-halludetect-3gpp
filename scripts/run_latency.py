"""E9：本地模块的 CPU 与 GPU 延迟（PLAN.md E9）。不调用 API。

取 val（seed=3407）的前 N 题，分别在 cuda 和 cpu 上测量每题耗时（ms/题）：
  检索（BM25 + 稠密 + RRF + 重排）、NLI 核验、Qwen3-0.6B 核验。
第 1 题作为预热，不计入统计。

CPU 推理很慢，内存占用也大（第一次 100 题的 CPU 测量因系统内存不足被中止），所以：
  - CPU 默认只测 20 题（--n-cpu），GPU 仍测 100 题（--n）；
  - 每测完一题就把耗时追加写入 timings.jsonl，每测完一个模块就重写一次 metrics.json，
    中途被中止也能保留已测到的数据；
  - 每个模块测完立即释放模型。

用法：
  python scripts/run_latency.py                 # GPU 100 题 + CPU 20 题
  python scripts/run_latency.py --devices cpu   # 只测 CPU
"""
import argparse
import gc
import json
import time

import numpy as np

from _common import ROOT  # noqa: F401

PARAMS = {"bge-base-en-v1.5": "109M", "bge-reranker-base": "278M",
          "DeBERTa-v3-base-mnli-fever-anli": "184M", "Qwen3-0.6B": "0.6B"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100, help="GPU 上测的题数")
    p.add_argument("--n-cpu", type=int, default=20, help="CPU 上测的题数")
    p.add_argument("--devices", nargs="*", default=["cuda", "cpu"])
    args = p.parse_args()

    import torch

    from telerag.config import load_config
    from telerag.data import load_questions
    from telerag.detect import NLIVerifier, SLMVerifier
    from telerag.explog import Experiment
    from telerag.pipeline import get_chunks
    from telerag.retrieval import Retriever, build_query

    base = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    n_max = max(args.n, args.n_cpu) + 1
    all_items = [it for it in load_questions(base) if it["split"] == "dev"][:n_max]
    chunks, path = get_chunks(base)
    exp = Experiment("E9_latency", base, {"plan": "PLAN.md E9", "n_gpu": args.n, "n_cpu": args.n_cpu,
                                          "torch_threads": torch.get_num_threads()})
    out = {"params": PARAMS, "n": {}, "torch_threads": torch.get_num_threads()}
    tf = (exp.dir / "timings.jsonl").open("a", encoding="utf-8")

    def record(dev, module, qid, sec):
        tf.write(json.dumps({"device": dev, "module": module, "qid": qid, "ms": sec * 1000}) + "\n")
        tf.flush()

    def release(dev):
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    try:
        for dev in args.devices:
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            n = args.n if dev == "cuda" else args.n_cpu
            items = all_items[: n + 1]
            out["n"][dev] = n
            cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500", f"retrieval.device={dev}",
                                         f"detection.device={dev}"])
            res = out.setdefault(dev, {})

            retr = Retriever(cfg, chunks, path)
            q = [build_query(it, cfg["retrieval"]["query_mode"]) for it in items]
            ts, rets = [], []
            for it, x in zip(items, q):
                t0 = time.perf_counter()
                rets.append(retr.search([x])[0])
                ts.append(time.perf_counter() - t0)
                record(dev, "retrieval", it["qid"], ts[-1])
            res["retrieval_ms"] = float(np.mean(ts[1:]) * 1000)
            exp.save_json("metrics.json", out)
            del retr
            release(dev)

            for name, cls in (("nli", NLIVerifier), ("slm", SLMVerifier)):
                v = cls(cfg["detection"])
                ts = []
                for it, r in zip(items, rets):
                    t0 = time.perf_counter()
                    v.score(it, r, chunks)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                    record(dev, name, it["qid"], ts[-1])
                res[f"{name}_ms"] = float(np.mean(ts[1:]) * 1000)
                exp.save_json("metrics.json", out)
                del v
                release(dev)
            res["total_local_ms"] = res["retrieval_ms"] + res["nli_ms"] + res["slm_ms"]
            exp.save_json("metrics.json", out)
            print(dev, f"n={n}", {k: round(v, 1) for k, v in res.items()}, flush=True)
    finally:
        tf.close()
        exp.save_json("metrics.json", out)
        exp.finish({d: out[d] for d in out["n"] if d in out})
        print(exp.dir)


if __name__ == "__main__":
    main()
