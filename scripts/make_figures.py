"""论文图表：读取 outputs/ 下的实验结果，输出 PDF（投稿用矢量图）和 PNG（预览）。

用法：
  python scripts/make_figures.py --name hybrid_rerank
图片保存到 outputs/figures/。
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import ROOT  # noqa: E402

# 分类色按固定顺序分配（已用 dataviz 校验器验证色盲可分）。每条曲线另有线型和直接标注，不只靠颜色区分。
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
LINES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
    # IJMLC 投稿指南：图中文字用 Helvetica 或 Arial，印刷后 8–12 pt。图按 0.7 倍正文宽（约 3.6 英寸）排版，与画布宽度接近 1:1
   
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# 方法在图中的固定颜色（颜色跟着方法走，不随排名或数量变化）
# 图中只画主要方法；颜色按固定顺序分配给方法，不随排名变化
METHOD_ORDER = ["Ours", "MV (training-free)", "SLM-contrast", "LLM-Judge", "Ours-lite", "Self-Consist (K=5)", "Retrieval"]
# 正文和表 3 用的名称。E0 的结果文件里仍是旧键名（Ours、LLM-Judge 等），只在画图时换成显示名
DISPLAY = {"Ours": "MVLV", "Ours-lite": "MVLV-lite", "MV (training-free)": "MV-vote", "LLM-Judge": "LLM-as-a-judge",
           "Self-Consist (K=5)": "Self-consistency (K = 5)", "Retrieval": "Retrieval score", "SLM-contrast": "SLM-contrast"}


def style(method):
    i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER) - 1
    return SERIES[i], LINES[i]


def save(fig, name):
    out = ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("saved", out / f"{name}.pdf")


def fig_risk_coverage(det):
    """选择性回答：只回答风险最低的 x% 题目时的准确率。越靠右上越好。"""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    curves = det["risk_coverage"]
    for m in [m for m in METHOD_ORDER if m in curves]:
        cov, acc = zip(*curves[m])
        c, ls = style(m)
        lw = 2.0 if m == "Ours" else 1.2
        ax.plot([x * 100 for x in cov], [y * 100 for y in acc], color=c, ls=ls, lw=lw, label=DISPLAY.get(m, m))
    ax.axhline(det["answer_accuracy"] * 100, color=INK2, lw=0.8, ls=":")
    ax.text(101, det["answer_accuracy"] * 100, " no abstention", color=INK2, va="center", fontsize=8)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy on answered (%)")
    # 图例放在坐标轴下方，避免压住曲线
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, handlelength=2.2,
              columnspacing=0.8)
    save(fig, "risk_coverage")


LABEL_OFFSETS = {"LLM-Judge": (-6, -11), "MV (training-free)": (-6, 3), "Retrieval": (5, 6), "NLI-contrast": (5, -2),
                 "Self-Consist (K=5)": (-4, 7)}


def fig_cost_auroc(det):
    """检测效果与额外 API 成本：横轴为每 1000 题的额外 API 花费（元，对数轴），纵轴为 AUROC。

    不花 API 的方法（只在本地运行）没有对数坐标，画在左侧单独的 "0" 位置，两段横轴之间画断轴符号。
    """
    methods = det["methods"]
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(3.5, 2.6), sharey=True,
                                  gridspec_kw={"width_ratios": [1, 5], "wspace": 0.06})
    # 左段在上层且背景透明，标注可以伸进右段而不被右段的背景盖住
    ax0.set_zorder(ax.get_zorder() + 1)
    ax0.patch.set_visible(False)
    pts = [(m, methods[m].get("cost_usd_per_q", 0.0) * 1000, methods[m]["auroc"]) for m in METHOD_ORDER if m in methods]
    for m, x, yv in pts:
        c, _ = style(m)
        target = ax0 if x == 0 else ax
        xx = 0 if x == 0 else x
        target.scatter(xx, yv, s=40, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy = LABEL_OFFSETS.get(m, (5, 3))
        if x == 0:
            # 检索分数随宿主输出一起得到（free）；SLM 等要在本地 GPU 上多算一遍（local）
            free = m in ("Retrieval", "Self-cite")
            dx, dy = (6, 4) if free else (6, -3)
            label = DISPLAY.get(m, m) + (" (free)" if free else " (local)")
        else:
            label = DISPLAY.get(m, m)
        target.annotate(label, (xx, yv), textcoords="offset points", xytext=(dx, dy), fontsize=8,
                        color=INK, ha="right" if dx < 0 else "left", annotation_clip=False, zorder=4)
    ax.set_xscale("log")
    ax0.set_xlim(-1, 1)
    ax0.set_xticks([0], ["0"])
    ax0.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax0.set_ylabel("AUROC")
    # 断轴符号：两段横轴相接处各画一条短斜线
    kw = dict(color=INK2, clip_on=False, lw=0.8)
    ax0.plot([0.92, 1.08], [-0.03, 0.03], transform=ax0.transAxes, **kw)
    ax.plot([-0.015, 0.015], [-0.03, 0.03], transform=ax.transAxes, **kw)
    unit = det.get("cost_unit", "USD")
    ax.set_xlabel(f"Extra API cost per 1k questions ({unit}, log scale)")
    save(fig, "cost_auroc")


def fig_ablation():
    """各证据设置在 seed=3407 test（2,500 题）上的答题准确率，与正文表 I 相同（数据来自 paper/facts.json）。

    条形从 0 开始：差异只有几个百分点，截断坐标轴会夸大差异。混合检索加重排是宿主系统，不是本文的贡献，
    所以不叫 "ours"。
    """
    rag = json.loads((ROOT / "paper" / "facts.json").read_text(encoding="utf-8"))["rag_test_seed3407"]
    order = ["closed", "dense", "no_rerank", "hybrid_rerank", "bm25"]
    labels = {"closed": "Closed-book", "bm25": "BM25-RAG", "dense": "Dense-RAG",
              "no_rerank": "Hybrid (RRF)", "hybrid_rerank": "Hybrid (RRF) + rerank"}
    names = [n for n in order if n in rag]
    acc = [rag[n]["acc"] for n in names]
    fig, ax = plt.subplots(figsize=(3.5, 0.3 * len(names) + 0.7))
    colors = [GRID if n == "closed" else SERIES[0] for n in names]
    ax.barh(range(len(names)), acc, color=colors, height=0.6)
    for i, v in enumerate(acc):
        ax.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), [labels[n] for n in names])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Answer accuracy on test (%)")
    ax.grid(axis="y", visible=False)
    save(fig, "ablation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="hybrid_rerank")
    p.add_argument("--protocol", default="val2test", choices=["val2test", "testcv"],
                   help="val2test：新协议（E0）；testcv：旧归档结果")
    args = p.parse_args()
    if args.protocol == "val2test":
        det = load_val2test()
        if det is None:
            print("没有找到 E0 的结果，请先运行 scripts/run_eval.py --exp E0")
            return
    else:
        det_path = ROOT / "outputs" / "archive_testcv_20261030" / f"detect_{args.name}.json"
        det = json.loads(det_path.read_text(encoding="utf-8"))
    fig_risk_coverage(det)
    fig_cost_auroc(det)
    fig_ablation()


def load_val2test():
    """把 E0（val 训练、test 评测）的 metrics.json 转成画图函数需要的结构。"""
    ds = sorted(d for d in (ROOT / "outputs" / "exp").glob("*_E0_val2test") if (d / "metrics.json").exists())
    if not ds:
        return None
    m = json.loads((ds[-1] / "metrics.json").read_text(encoding="utf-8"))
    # 每千题额外 API 成本（元）：官方空闲价 × seed=3407 test 上实测的平均 token 数，由论文统计脚本写入 paper/facts.json，
    # 与正文 VI-G 节的数字一致。本地方法记为 0。
    c = json.loads((ROOT / "paper" / "facts.json").read_text(encoding="utf-8"))["cost_per_1k_cny"]
    api = {"MV (training-free)": c["three_views"], "MV-Consist": c["three_views"], "Ours": c["three_views"],
           "Ours-lite": c["closed"], "LLM-Judge": c["judge"]}
    methods = {}
    for k, r in m["methods"].items():
        methods[k] = {"auroc": r["test"]["auroc"], "cost_usd_per_q": api.get(k, 0.0) / 1000}
    e5 = sorted(d for d in (ROOT / "outputs" / "exp").glob("*_E5_self_consistency") if (d / "metrics.json").exists())
    if e5:
        sc = json.loads((e5[-1] / "metrics.json").read_text(encoding="utf-8")).get("sc")
        if sc:   # SC 每题 5 次答题调用，按宿主每次调用的成本计
            methods["Self-Consist (K=5)"] = {"auroc": sc["test"]["auroc"], "cost_usd_per_q": 5 * c["hybrid_rerank"] / 1000}
    return {"answer_accuracy": m["acc_test"], "methods": methods, "risk_coverage": m["risk_coverage_test"],
            "cost_unit": "CNY"}


if __name__ == "__main__":
    main()
