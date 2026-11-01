"""论文示意图（Fig. 1 到 Fig. 3）：用 matplotlib 按真实流程和真实数据绘制，替代早期用生图模型做的草图。

Springer 不接受生成式 AI 制作的图片，所以示意图全部由这个脚本用代码画出；版式参照 picture/4.jpg 到 6.jpg。
Fig. 2 和 Fig. 3 的选项、各视角答案和 NLI 分数都直接读取实验结果，没有手填。

图按印刷尺寸绘制：Springer 模板正文宽 31 pc（约 5.15 英寸），Fig. 1 占满正文宽，其余占 0.68 倍（3.5 英寸）。
这样脚本里的字号就是印刷后的字号（7 到 8 磅，符合投稿指南"8–12 pt"附近的要求）。

用法：
  python scripts/make_schematics.py
输出到 outputs/figures/：framework、mv_example、contrast_example（各有 PDF 和 PNG）。
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

from _common import ROOT  # noqa: E402
from make_figures import INK, INK2, save  # noqa: E402  （沿用同一套字体、颜色和保存方式）

# 对和错用绿色和红色表示，并且总是配合 ✓/✗ 或文字说明，不只靠颜色区分
OK, OK_FILL = "#1b7f4b", "#d6eedf"
BAD, BAD_FILL = "#c23b3b", "#f6d9d9"
FILL = {"input": "#ececea", "view": "#fbf1cf", "mv": "#fbe0d2", "ver": "#d9efe9", "fuse": "#dde7f7"}
EVID = "#8a8984"          # 证据（语料到检索视角）用浅灰虚线，和问题的实线区分开
PT = 1 / 72               # 1 磅（英寸）；Fig. 1 的坐标单位就是英寸
TS, SS = 8, 7             # 框内标题和副标题的字号（磅）


def box(ax, x, y, w, h, title, sub=None, fill="#ffffff", edge=INK2, lw=0.8):
    """画圆角框，标题和副标题作为一个整体在框内垂直居中。坐标单位为英寸。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fill, ec=edge, lw=lw))
    ht = (title.count("\n") + 1) * TS * 1.2 * PT
    hs = (sub.count("\n") + 1) * SS * 1.2 * PT if sub else 0
    gap = 2 * PT if sub else 0
    top = y + h / 2 + (ht + gap + hs) / 2
    ax.text(x + w / 2, top - ht / 2, title, ha="center", va="center", fontsize=TS, color=INK, linespacing=1.2)
    if sub:
        ax.text(x + w / 2, top - ht - gap - hs / 2, sub, ha="center", va="center", fontsize=SS, color=INK2,
                linespacing=1.2)


def seg(ax, a, b, color=INK2, ls="-"):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.8, ls=ls, solid_capstyle="butt")


def arrow(ax, a, b, color=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=0.8, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def path(ax, pts, color=INK2, ls="-"):
    """折线，最后一段带箭头。"""
    for a, b in zip(pts[:-2], pts[1:-1]):
        seg(ax, a, b, color, ls)
    arrow(ax, pts[-2], pts[-1], color, ls)


def fig_framework():
    W, H = 5.1, 2.4
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    for x, t in ((0.42, "Input"), (1.47, "Multi-view answering"), (2.8, "Hallucination signals"), (3.86, "Fusion"),
                 (4.76, "Decision")):
        ax.text(x, 2.29, t, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    ax.text(1.47, 2.13, "same LLM, four evidence settings", ha="center", va="center", fontsize=6.5, style="italic",
            color=INK2)

    # 输入
    box(ax, 0.0, 1.56, 0.84, 0.44, "Question", "4–5 options", fill=FILL["input"])
    box(ax, 0.0, 0.12, 0.84, 0.52, "3GPP corpus", "Release 18\n554 specs", fill=FILL["input"])

    # 四个视角（最终答案来自 Hybrid-RAG，用粗框标出）
    vx, vw, vh = 1.06, 0.82, 0.36
    ys = {"closed": 1.62, "bm25": 1.18, "dense": 0.74, "hybrid": 0.30}
    box(ax, vx, ys["closed"], vw, vh, "Closed-book", "no retrieval", fill=FILL["view"])
    box(ax, vx, ys["bm25"], vw, vh, "BM25-RAG", "lexical, top-5", fill=FILL["view"])
    box(ax, vx, ys["dense"], vw, vh, "Dense-RAG", "bge-base, top-5", fill=FILL["view"])
    box(ax, vx, ys["hybrid"], vw, vh, "Hybrid-RAG", "RRF + rerank", fill=FILL["view"], edge=INK, lw=1.6)

    # 问题 → 四个视角：实线，主干在 x=0.92，接在框的偏上位置（闭卷接在正中）
    qx = 0.92
    q_ports = [ys["closed"] + vh / 2] + [ys[k] + 0.62 * vh for k in ("bm25", "dense", "hybrid")]
    seg(ax, (0.84, q_ports[0]), (qx, q_ports[0]))
    seg(ax, (qx, min(q_ports)), (qx, max(q_ports)))
    for y in q_ports:
        arrow(ax, (qx, y), (vx, y))
    # 语料 → 三个检索视角：虚线，主干在 x=0.975，接在框的偏下位置；闭卷不连语料
    cx = 0.975
    c_ports = [ys[k] + 0.3 * vh for k in ("bm25", "dense", "hybrid")]
    seg(ax, (0.84, min(c_ports)), (cx, min(c_ports)), EVID, "--")
    seg(ax, (cx, min(c_ports)), (cx, max(c_ports)), EVID, "--")
    for y in c_ports:
        arrow(ax, (cx, y), (vx, y), EVID, "--")

    # 四个答案 → 多视角一致性
    sx, sw = 2.3, 1.0
    box(ax, sx, 1.18, sw, 0.62, "Multi-view\nconsistency", "agreement of\nthe 4 answers", fill=FILL["mv"])
    bus = 2.08
    centers = [ys[k] + vh / 2 for k in ys]
    for y in centers:
        seg(ax, (vx + vw, y), (bus, y))
    seg(ax, (bus, min(centers)), (bus, max(centers)))
    arrow(ax, (bus, 1.49), (sx, 1.49))

    # 最终答案和证据 → 选项级核验（这条线在汇总线下方，不和它相交）
    box(ax, sx, 0.10, sw, 0.68, "Option-level\nverification", "NLI 0.2B, SLM 0.6B\nlocal GPU", fill=FILL["ver"])
    yv = 0.36
    arrow(ax, (vx + vw, yv), (sx, yv))
    ax.text((vx + vw + sx) / 2, yv - 0.035, "answer +\nevidence", ha="center", va="top", fontsize=6.5, color=INK2,
            linespacing=1.1)

    # 融合 → 决策
    fx, fw, fy, fh = 3.5, 0.72, 0.67, 0.46
    box(ax, fx, fy, fw, fh, "Risk fusion", "logistic\nregression", fill=FILL["fuse"])
    fc = fx + fw / 2
    path(ax, [(sx + sw, 1.49), (fc, 1.49), (fc, fy + fh)])
    path(ax, [(sx + sw, 0.44), (fc, 0.44), (fc, fy)])
    dx, dw, dh = 4.42, 0.68, 0.34
    lo_y, hi_y = 1.02, 0.44
    box(ax, dx, lo_y, dw, dh, "Low risk", "answer", fill=OK_FILL, edge=OK)
    box(ax, dx, hi_y, dw, dh, "High risk", "abstain", fill=BAD_FILL, edge=BAD)
    split, ym = 4.32, fy + fh / 2
    seg(ax, (fx + fw, ym), (split, ym))
    seg(ax, (split, hi_y + dh / 2), (split, lo_y + dh / 2))
    arrow(ax, (split, lo_y + dh / 2), (dx, lo_y + dh / 2))
    arrow(ax, (split, hi_y + dh / 2), (dx, hi_y + dh / 2))
    save(fig, "framework")


def _wrap(t, n):
    words, lines, cur = t.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def fig_mv_example(item, views, final):
    order = [("closed", "Closed-book"), ("bm25", "BM25-RAG"), ("dense", "Dense-RAG"), ("hybrid_rerank", "Hybrid-RAG\n(final)")]
    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.2)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2, 7.7), 9.6, 4.3, boxstyle="round,pad=0.02,rounding_size=0.25", fc="#f2f2f0",
                                ec=INK2, lw=0.8))
    q = _wrap("Q: " + item["question"], 46)
    ax.text(0.55, 11.6, "\n".join(q), fontsize=8, fontweight="bold", va="top", color=INK, linespacing=1.2)
    y0 = 11.6 - 0.42 * len(q) - 0.35
    for k, (o, txt) in enumerate(item["options"].items()):
        right = o == item["answer"]
        ax.text(0.75, y0 - 0.66 * k, f"({o}) {txt}" + ("  (correct)" if right else ""), fontsize=7.5, va="top",
                color=OK if right else INK, fontweight="bold" if right else "normal")
    xs = [1.35, 3.8, 6.25, 8.7]
    for x in xs:
        arrow(ax, (5, 7.62), (x, 6.3))
    ax.text(5, 7.05, "same LLM, four evidence settings", ha="center", va="center", fontsize=7, style="italic",
            color=INK2, bbox=dict(fc="white", ec="none", pad=1.2))
    for x, (k, label) in zip(xs, order):
        p = views[k]
        right = p == item["answer"]
        ax.add_patch(Circle((x, 5.3), 0.95, fc=OK_FILL if right else BAD_FILL, ec=OK if right else BAD,
                            lw=2.4 if k == "hybrid_rerank" else 1.2))
        ax.text(x, 5.3, p, ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(x + 0.82, 6.12, "✓" if right else "✗", ha="center", va="center", fontsize=8.5,
                color=OK if right else BAD, fontfamily="DejaVu Sans")
        ax.text(x, 3.95, label, ha="center", va="top", fontsize=7.5, color=INK, linespacing=1.1)
    agree = sum(views[k] == final for k, _ in order)
    ax.add_patch(FancyBboxPatch((0.2, 0.25), 9.6, 2.1, boxstyle="round,pad=0.02,rounding_size=0.25", fc="#fbe0d2",
                                ec=INK2, lw=0.8))
    ax.text(5, 1.75, f"Final answer ({final}): vote share {agree}/4  →  high risk", ha="center", fontsize=8, color=INK)
    ax.text(5, 0.9, f"The final answer is wrong; the correct answer is ({item['answer']})", ha="center",
            fontsize=8, color=INK)
    save(fig, "mv_example")


def fig_contrast_example(item, support, pred):
    opts = list(item["options"])
    labels = [f"({o}) {item['options'][o]}" for o in opts]
    vals = [support[o] for o in opts]
    rival = max((o for o in opts if o != pred), key=lambda o: support[o])
    colors = [OK if o == item["answer"] else BAD if o == pred else "#b9b8b3" for o in opts]
    fig, ax = plt.subplots(figsize=(3.5, 2.3), layout="constrained")
    y = list(range(len(opts)))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v, o in zip(y, vals, opts):
        if o == pred:
            tag = "  selected by LLM (wrong)"
        elif o == rival:
            tag = "  strongest rival" + (" (correct)" if o == item["answer"] else "")
        else:
            tag = ""
        if v > 0.6:   # 长条把标注写在条内（白字），避免伸出坐标区
            ax.text(v - 0.02, yi, f"{v:.2f}{tag}", ha="right", va="center", fontsize=7, color="white")
        else:
            ax.text(v + 0.015, yi, f"{v:.2f}{tag}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, ["\n".join(_wrap(t, 24)) for t in labels], fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("NLI entailment probability (max over top-5 chunks)", fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    num = lambda v: f"{v:.2f}".replace("-", "−")
    margin = support[pred] - support[rival]
    row2 = y[opts.index(min((o for o in opts if o not in (pred, rival)), key=lambda o: support[o]))]
    ax.text(0.99, row2, f"margin = {num(support[pred])} − {num(support[rival])} = {num(margin)}", ha="right",
            va="center", fontsize=7, color=INK)
    save(fig, "contrast_example")


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from telerag.config import load_config
    from telerag.data import load_questions
    from telerag.io import load_records

    cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    items = {it["qid"]: it for it in load_questions(cfg)}

    def pred_of(name, qid):
        for sp in ("dev", "test"):
            for r in load_records(f"{name}_{sp}"):
                if r["qid"] == qid:
                    return r["pred"]
        raise KeyError((name, qid))

    q_mv = 383          # 验证集中的一道题
    views = {n: pred_of(n, q_mv) for n in ("closed", "bm25", "dense", "hybrid_rerank")}
    q_c = 5458          # 测试集中的一道题
    scored = next(json.loads(ln) for ln in (ROOT / "outputs" / "runs" / "hybrid_rerank_test_scored.jsonl").open(encoding="utf-8")
                  if json.loads(ln)["qid"] == q_c)
    fig_framework()
    fig_mv_example(items[q_mv], views, views["hybrid_rerank"])
    fig_contrast_example(items[q_c], {k: v[0] for k, v in scored["nli"].items()}, scored["pred"])


if __name__ == "__main__":
    main()
