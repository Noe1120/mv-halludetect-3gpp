"""换一个答题大模型，把多视角一致性需要的全部视角跑一遍（闭卷、BM25、稠密、混合+重排），然后跑幻觉检测。

结果名带模型前缀，例如 deepseek-v4-pro 的主系统结果为 outputs/runs/pro__hybrid_rerank_test.jsonl。
检索结果和本地核验打分与答题模型无关，会直接复用缓存，不重复计算。

用法：
  python scripts/run_model.py --model deepseek-v4-pro --prefix pro --limit 1000
"""
import argparse
import subprocess
import sys

from _common import ROOT

VIEWS = {
    "closed": ["--no-rag"],
    "bm25": ["--set", "retrieval.method=bm25", "retrieval.rerank=false"],
    "dense": ["--set", "retrieval.method=dense", "retrieval.rerank=false"],
    "hybrid_rerank": [],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prefix", required=True, help="结果文件名前缀")
    p.add_argument("--limit", type=int, default=None, help="只用测试集前 N 题（控制成本）")
    p.add_argument("--judge", action="store_true")
    args = p.parse_args()
    py = sys.executable
    lim = ["--limit", str(args.limit)] if args.limit else []
    for view, opts in VIEWS.items():
        print(f"===== {args.prefix}: {view} =====", flush=True)
        sets = opts[1:] if opts[:1] == ["--set"] else []
        flags = [o for o in opts if o == "--no-rag"]
        cmd = [py, str(ROOT / "scripts" / "run_qa.py"), "--split", "test", "--name", f"{args.prefix}__{view}",
               *lim, *flags, "--set", f"llm.model={args.model}", *sets]
        subprocess.run(cmd, check=True)
    cmd = [py, str(ROOT / "scripts" / "run_detect.py"), "--name", f"{args.prefix}__hybrid_rerank",
           "--views-prefix", f"{args.prefix}__", *lim,
           "--set", f"llm.model={args.model}"]
    if args.judge:
        cmd.append("--judge")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
