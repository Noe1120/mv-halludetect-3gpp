"""两次答题实验的配对比较（McNemar 精确检验），论文里报告显著性用。

用法：
  python scripts/compare_runs.py closed_test hybrid_rerank_test
"""
import sys

from _common import ROOT  # noqa: F401


def main():
    from scipy.stats import binomtest

    from telerag.io import load_records

    a_name, b_name = sys.argv[1], sys.argv[2]
    a = {r["qid"]: r["pred"] == r["answer"] for r in load_records(a_name)}
    b = {r["qid"]: r["pred"] == r["answer"] for r in load_records(b_name)}
    common = a.keys() & b.keys()
    gain = sum(b[q] and not a[q] for q in common)
    loss = sum(a[q] and not b[q] for q in common)
    acc_a = sum(a[q] for q in common) / len(common)
    acc_b = sum(b[q] for q in common) / len(common)
    p = binomtest(gain, gain + loss).pvalue if gain + loss else 1.0
    print(f"{a_name}: {acc_a:.4f}  {b_name}: {acc_b:.4f}  (n={len(common)})")
    print(f"{b_name} 答对而 {a_name} 答错: {gain}；反之: {loss}；McNemar 精确检验 p = {p:.3g}")


if __name__ == "__main__":
    main()
