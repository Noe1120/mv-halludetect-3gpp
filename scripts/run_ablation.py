"""检索消融实验：依次改变一个检索组件，记录检索代理指标和答题准确率。

用法：
  python scripts/run_ablation.py --split dev          # 先在开发集上快速看趋势
  python scripts/run_ablation.py --split test         # 论文表格用测试集
  python scripts/run_ablation.py --split test --only bm25 dense
"""
import subprocess
import sys

from _common import ROOT

VARIANTS = {
    "hybrid_rerank": [],                                                   # 完整方法
    "no_rerank": ["retrieval.rerank=false"],                               # 去掉重排
    "bm25": ["retrieval.method=bm25", "retrieval.rerank=false"],           # 只用 BM25
    "dense": ["retrieval.method=dense", "retrieval.rerank=false"],         # 只用稠密向量
    "fixed_chunk": ["chunking.strategy=fixed"],                            # 固定长度切块
    "top3": ["retrieval.top_k=3"],
    "top10": ["retrieval.top_k=10"],
}


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    py = sys.executable
    for name, sets in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        print(f"===== {name} =====", flush=True)
        extra = ["--set", *sets] if sets else []
        for script, opts in [("eval_retrieval.py", []), ("run_qa.py", ["--name", name])]:
            cmd = [py, str(ROOT / "scripts" / script), "--split", args.split, *opts, *extra]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
