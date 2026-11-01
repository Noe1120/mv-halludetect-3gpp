"""自动汇总第 1 轮预注册实验，生成 REPORT.md（要求.md 第三节第 4、6 条）。

读取每个实验最新一次运行的 outputs/exp/<时间>_<实验名>/metrics.json，以及 outputs/log.jsonl 中的花费。
不调用 API，也不重新计算任何指标。

用法：
  python scripts/make_report.py
"""
import json
from pathlib import Path

from _common import ROOT

EXP = ROOT / "outputs" / "exp"


def latest(prefix):
    ds = sorted(d for d in EXP.glob(f"*_{prefix}*") if (d / "metrics.json").exists())
    return (json.loads((ds[-1] / "metrics.json").read_text(encoding="utf-8")), ds[-1]) if ds else (None, None)


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]" if c else "–"


def dci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c else "–"


def main():
    L = ["# REPORT.md：第 1 轮预注册实验结果", "",
         "> 自动生成：`scripts/make_report.py`。预注册见 `PLAN.md`。",
         "> **本实验为单种子（seed=3407），存在随机噪声风险。**",
         "> 协议：在 val（500 题）上训练融合模型、选阈值；在 test（2,500 题）上只评测一次。", ""]

    e0, d0 = latest("E0_val2test")
    if e0:
        b = e0
        L += ["## 宿主基线", "",
              f"- 答题模型 deepseek-flash，混合检索加重排：val 准确率 {b['acc_val'] * 100:.1f}%（幻觉 {b['halluc_val']} 题），"
              f"test 准确率 {b['acc_test'] * 100:.1f}%（幻觉 {b['halluc_test']} 题）",
              f"- 本地核验耗时（GPU，秒/题）：{ {k: round(v, 3) for k, v in b['local_seconds_per_q'].items()} }", "",
              f"## E0 按新协议重跑检测（`{d0.name}`）", "",
              "| 方法 | val AUROC | test AUROC [95% CI] | test ΔAUROC vs Judge [95% CI] | test AUPRC | test F1 | test AURC↓ |",
              "|---|---|---|---|---|---|---|"]
        for m, r in b["methods"].items():
            t = r["test"]
            L.append(f"| {m} | {r['val']['auroc']:.3f} | {t['auroc']:.3f} {ci(t['auroc_ci95'])} | "
                     f"{dci(t.get('auroc_minus_judge_ci95'))} | {t['auprc']:.3f} | {t['f1']:.3f} | {t['aurc']:.3f} |")
        o = b["methods"]["Ours"]["test"]
        j = b["methods"]["LLM-Judge"]["test"]
        L += ["", f"结论：test 上 Ours 的 AUROC 为 {o['auroc']:.3f}，LLM-Judge 为 {j['auroc']:.3f}，"
                  f"差值 95% CI 为 {dci(o.get('auroc_minus_judge_ci95'))}。",
              "旧协议（test 上做交叉验证）的结果在 `outputs/archive_testcv_20261030/`，"
              "其中 Ours 为 0.807。新协议下 test 结果更低，说明旧协议的数字略偏乐观。", ""]

    e1, d1 = latest("E1_margin_logit")
    if e1:
        L += [f"## E1 SLM margin 改用 logit 尺度（`{d1.name}`）", "",
              f"阈值：val ΔAUROC ≥ {e1['threshold_auroc']}，并且 test 同向。", "",
              "| 方法 | val prob → logit | val Δ [95% CI] | test Δ [95% CI] | 判定 |", "|---|---|---|---|---|"]
        for m, r in e1["pairs"].items():
            L.append(f"| {m} | {r['prob']['val']['auroc']:.3f} → {r['logit']['val']['auroc']:.3f} | "
                     f"{r['delta_val']:+.3f} {dci(r['delta_val_ci95'])} | {r['delta_test']:+.3f} "
                     f"{dci(r['delta_test_ci95'])} | {r['verdict']} |")
        L += ["", "**根因诊断（不成立，探针证据见 `docs/04_根因诊断.md`）**：尺度问题确实存在。单独使用时，"
                  "margin 在概率尺度下 val AUROC 为 0.681，在 logit 尺度下为 0.705；"
                  "own、rival、margin 三个特征一起用时，分别为 0.686 和 0.714。"
                  "但宿主的 SLM-contrast 特征组里已经有 `slm_argmax_agree`（小模型首选是否与最终答案一致）。"
                  "它与 logit margin 的 Spearman 相关系数为 0.86，99% 的题两者的正负号一致，也就是说它已经包含了 logit margin 的方向信息。"
                  "加上 argmax_agree 之后，两种尺度的 val AUROC 分别是 0.715 和 0.714，没有差别。"
                  "根因归类：**特征冗余**（不是\"尺度未归一化\"）。", ""]

    e2, d2 = latest("E2E3_atomic_faithfulness")
    if e2:
        dg = e2["diagnostics"]
        L += [f"## E2 原子事实拆分加核查 / E3 忠实度（`{d2.name}`）", "",
              "| 方法 | val off → on | val Δ [95% CI] | test Δ [95% CI] | 判定 |", "|---|---|---|---|---|"]
        for m, r in e2["pairs"].items():
            L.append(f"| {m} | {r['off']['val']['auroc']:.3f} → {r['on']['val']['auroc']:.3f} | "
                     f"{r['delta_val']:+.3f} {dci(r['delta_val_ci95'])} | {r['delta_test']:+.3f} "
                     f"{dci(r['delta_test_ci95'])} | {r['verdict']} |")
        a, f = e2["atomic_only"], e2["faithfulness_as_detector"]
        L += ["",
              f"- 只用原子特征：val {a['val']['auroc']:.3f}，test {a['test']['auroc']:.3f}",
              f"- 1 − 忠实度（RAGAS 式，本地 NLI 实现）作为检测器：val {f['val']['auroc']:.3f}，test {f['test']['auroc']:.3f}",
              f"- 忠实度均值：答对 {dg['faithfulness_mean_correct']:.3f}，答错 {dg['faithfulness_mean_halluc']:.3f}",
              f"- 每个回答的陈述数中位数 {dg['claims_per_answer_median']:.0f}，不超过 2 条的占 {dg['claims_le2_ratio'] * 100:.0f}%",
              "",
              "**根因诊断（探针证据见 `docs/04_根因诊断.md`）**：加到完整方法（Ours）上不成立："
              "val 提升 0.008，test 下降 0.003。只加到弱基线 NLI-contrast 上时，成立但未达显著。"
              "① 原子特征 atomic_min 与已有的 nli_own 的 Spearman 相关系数为 0.64，两者大部分信息重叠；"
              "② 在 Ours 中用原子特征替换 NLI 特征，val/test 为 0.859/0.789，与原来的 0.852/0.790 相当，说明两者可以互相替代，但叠加没有好处；"
              "③ 解释句的中位数只有 2 条陈述，可核查的内容很少。"
              "PLAN 中预先登记的另一个怀疑\"拆分失败、退回到'题目+选项'\"被**否定**：退回的比例为 0%。"
              "根因归类：**拆分粒度失配加上与已有特征冗余**。", ""]

    e4, d4 = latest("E4_correction")
    if e4:
        L += [f"## E4 纠正：高风险题改用 BM25 视角的答案（`{d4.name}`）", "",
              "| | 不纠正 | 全部用 BM25 | 按风险纠正 |", "|---|---|---|---|",
              f"| val | {e4['acc_none'][0] * 100:.1f}% | {e4['acc_all_bm25'][0] * 100:.1f}% | "
              f"{e4['acc_corrected'][0] * 100:.1f}%（τ={e4['tau']}） |",
              f"| test | {e4['acc_none'][1] * 100:.1f}% | {e4['acc_all_bm25'][1] * 100:.1f}% | "
              f"{e4['acc_corrected'][1] * 100:.1f}% |", "",
              f"val 上τ的候选结果：{ {k: round(v, 4) for k, v in e4['tau_grid_val'].items()} }。"
              f"val 上与更强基线的 McNemar 检验：多对 {e4['mcnemar_val_vs_best_base']['gain']} 题、"
              f"多错 {e4['mcnemar_val_vs_best_base']['loss']} 题，p = {e4['mcnemar_val_vs_best_base']['p']:.3f}。"
              f"判定：**{e4['verdict']}**。", ""]

    e9, d9 = latest("E9_latency")
    if e9:
        L += [f"## E9 本地模块延迟（`{d9.name}`）", "", "| 设备 | 检索 | NLI 核验 | SLM 核验 | 合计 |",
              "|---|---|---|---|---|"]
        for dev in ("cuda", "cpu"):
            if dev in e9:
                r = e9[dev]
                n = e9.get("n", {}).get(dev, "?")
                L.append(f"| {dev}（{n} 题） | {r['retrieval_ms']:.0f} ms | {r['nli_ms']:.0f} ms | {r['slm_ms']:.0f} ms | "
                         f"{r['total_local_ms']:.0f} ms |")
        L += ["", f"参数量：{e9['params']}；CPU 线程数 {e9.get('torch_threads', '?')}。"]
        if e9.get("provenance"):
            L += [f"数据来源：{e9['provenance']}"]
        L += ["CPU 上整套本地核验每题约 30 秒，比 GPU 慢约 29 倍。纯 CPU 部署只适合离线批量核验，不适合实时问答。", ""]

    e10, d10 = latest("E10_ragtruth")
    if e10 and e10.get("n_test", 0) >= 2000:
        L += [f"## E10 RAGTruth 迁移（`{d10.name}`）", "",
              f"在 RAGTruth train 划分中按 seed=3407 抽 {e10['n_train']} 条，用于训练融合模型、选阈值；"
              f"在官方 test 划分的 {e10['n_test']} 条上只评测一次。test 幻觉率 {e10['halluc_rate_test'] * 100:.1f}%，"
              f"本地核验 {e10['seconds_per_response']:.2f} 秒/条（GPU）。",
              "**只迁移本地核验器**：RAGTruth 的回答是现成的，没有闭卷和不同检索器的视角，多视角一致性无法迁移。", "",
              "| 检测器 | AUROC [95% CI] | F1 | AUROC QA / Summary / Data2txt | F1 QA / Summary / Data2txt |",
              "|---|---|---|---|---|"]
        for m, r in e10["methods"].items():
            a = " / ".join(f"{r.get(f'auroc[{t}]', float('nan')):.3f}" for t in ("QA", "Summary", "Data2txt"))
            f1 = " / ".join(f"{r.get(f'f1[{t}]', float('nan')):.3f}" for t in ("QA", "Summary", "Data2txt"))
            L.append(f"| {m} | {r['auroc']:.3f} {ci(r['auroc_ci95'])} | {r['f1']:.3f} | {a} | {f1} |")
        L += ["",
              "- 原子事实拆分加 NLI（NLI-claim）在 RAGTruth 上有效，AUROC 为 0.744。这和 3GPP 上的结果（E2 不成立）方向相反，"
              "支持 E2 的根因诊断：原子拆分适合长回答（RAGTruth 回答中位数约 100 词），不适合选择题的短解释（中位数 26 词）。",
              "- Qwen3-0.6B 在 RAGTruth 上几乎失效（AUROC 0.570）。探针显示根因是**打分饱和**："
              "test 上 93% 的回答 P(Yes) > 0.99，中位数 0.9975，几乎对所有回答都答 Yes。"
              "截断不是原因：上下文被截断的只有 2%，截断样本与未截断样本的 AUROC 分别为 0.598 和 0.569。"
              "在 3GPP 上，小模型是逐个选项打分，靠选项之间的相对高低（argmax 是否一致）起作用；"
              "RAGTruth 每条只有一个回答，没有可比较的对象，饱和的绝对分数就失去了区分能力。",
              "- Data2txt 子任务上各方法的 AUROC 都接近 0.5。它的上下文是 JSON 格式的结构化记录（如餐厅的名称、地址、类别），"
              "幻觉率高达 64.3%，NLI 模型没有针对这类输入训练。",
              "- 参考数字：LettuceDetect（ModernBERT，在 RAGTruth 上训练）的样本级 F1 为 79.22%（摘要已核实）。"
              "本文的检测器没有在 RAGTruth 上训练核验模型，只用 1,000 条训练逻辑回归融合，F1 为 "
              f"{e10['methods']['Fusion']['f1'] * 100:.1f}%，两者设置不同，不能直接比较。",
              ""]

    e5, d5 = latest("E5_self_consistency")
    if e5 and "sc" in e5:
        sc = e5["sc"]
        L += [f"## E5 Self-Consistency 基线（`{d5.name}`）", "",
              "设置：沿用宿主的 RAG 提示词，temperature = 0.7，采样 K = 5 次；风险分 = 1 − 与最终答案相同的采样所占比例；阈值在 val 上选。", "",
              "| 方法 | val AUROC | test AUROC [95% CI] | test AUPRC | test F1 | 每题额外答题调用 |",
              "|---|---|---|---|---|---|",
              f"| Self-Consistency（K=5） | {sc['val']['auroc']:.3f} | {sc['test']['auroc']:.3f} {ci(sc['test']['auroc_ci95'])} | "
              f"{sc['test']['auprc']:.3f} | {sc['test']['f1']:.3f} | 5 |"]
        if e0:
            o = e0["methods"]["Ours"]
            L.append(f"| Ours | {o['val']['auroc']:.3f} | {o['test']['auroc']:.3f} {ci(o['test']['auroc_ci95'])} | "
                     f"{o['test']['auprc']:.3f} | {o['test']['f1']:.3f} | 3 |")
        L += ["", f"Ours − SC 的配对差值 95% CI：val {dci(e5['ours_minus_sc_val_ci95'])}，test {dci(e5['ours_minus_sc_test_ci95'])}。"
                  "**判定：本文方法优于 SC**，并且每题的额外答题调用更少（3 次对 5 次）。",
              f"估算花费 {e5['cost_cny_estimate']:.2f} 元（按官方空闲价和实际 token 数，含前缀缓存折扣）；"
              "按余额变化计算的实际扣费见 `outputs/balance_log.txt`。",
              "根因（SC 为什么弱）：63% 的幻觉在 5 次采样中给出同一个错误答案，SC 只能发现 37% 的幻觉，"
              "多视角能发现 64%（探针见 `docs/04_根因诊断.md`）。", ""]

    e6, d6 = latest("E6_compare_val")
    if e6:
        L += [f"## E6 Telco-RAG* 领域基线（val，`{d6.name}`）", "",
              "按上游论文版流程重新实现（改写问题、术语表增强、按系列路由、用候选答案二次检索），"
              "嵌入模型换成 bge-m3，答题模型换成 deepseek-flash（见 `docs/02` 第 2.6 节）。每题调用 "
              f"{e6['calls_per_q']:.0f} 次 LLM，平均 {e6['tokens_per_q']:.0f} token。", "",
              "| 系统 | val 准确率 | Telco-RAG* 多对 / 多错 | McNemar p |", "|---|---|---|---|",
              f"| **Telco-RAG\\*** | **{e6['telco_rag_star'] * 100:.1f}%** | – | – |"]
        names = {"closed": "闭卷", "bm25": "BM25-RAG", "dense": "Dense-RAG", "hybrid_rerank": "混合检索 + 重排（宿主）"}
        for k, v in e6["vs"].items():
            L.append(f"| {names[k]} | {v['acc'] * 100:.1f}% | {v['telco_better']} / {v['telco_worse']} | {v['p']:.3g} |")
        L += ["", f"分类准确率：{ {k: round(v * 100, 1) for k, v in e6['by_category'].items()} }；无效答案 {e6['invalid']} 个。"
                  f"判定：**{e6['verdict']}**。", ""]

    e8, d8 = latest("E8_val")
    if e8:
        r = e8["variants"]["fixed_chunk"]
        L += [f"## E8 固定长度切块对照（val，`{d8.name}`）", "",
              "预注册假设：章节切块（宿主）比固定 300 词切块的准确率高 ≥ 1 个百分点。", "",
              "| 切块方式 | val 准确率 |", "|---|---|",
              f"| 章节切块 + 上下文前缀（宿主） | {r['acc_baseline'] * 100:.1f}% |",
              f"| 固定 300 词切块 | {r['acc_variant'] * 100:.1f}% |", "",
              f"McNemar：固定切块多对 {r['mcnemar']['gain']} 题、多错 {r['mcnemar']['loss']} 题，p = {r['mcnemar']['p']:.2f}。"
              f"判定：**{e8['verdict']}**（方向与假设相反，差异不显著）。",
              "根因：top_k 固定为 5 时，固定切块每块更长，每题送给模型的上下文多约 40%（2,577 对 1,832 token），"
              "AnsHit@5 也更高（47.8% 对 43.8%）；这个对照没有控制上下文长度（探针见 `docs/04_根因诊断.md`）。", ""]

    r2b, dr2b = latest("R2B_val")
    r2a, dr2a = latest("R2A_compare_val")
    if r2b or r2a:
        L += ["## 第 2 轮修正实验（`PLAN_round2.md`，只跑 val）", ""]
    if r2b:
        v = r2b["variants"]["section_top7"]
        L += [f"**R2-B 等 token 预算下比较切块方式**（`{dr2b.name}`）：章节切块 top-7 {v['acc_variant'] * 100:.1f}%，"
              f"固定切块 top-5 {v['acc_baseline'] * 100:.1f}%，McNemar {v['mcnemar']['gain']} 对 {v['mcnemar']['loss']}，"
              f"p = {v['mcnemar']['p']:.2f}。判定：**{r2b['verdict']}**。", ""]
    if r2a:
        e6v = r2a["vs"].get("telco_rag_star", {})
        L += [f"**R2-A Telco-RAG\\* 去掉系列路由**（`{dr2a.name}`）：val 准确率 {r2a['telco_rag_star'] * 100:.1f}%"
              f"（有路由的 E6 为 {e6v.get('acc', float('nan')) * 100:.1f}%，McNemar {e6v.get('telco_better')} 对 {e6v.get('telco_worse')}，"
              f"p = {e6v.get('p', float('nan')):.2f}）；证据覆盖率 26.4% → 38.4%。"
              f"与宿主比较：{r2a['verdict']}（p = {r2a['vs']['hybrid_rerank']['p']:.3f}）。"
              "判定：机制成立，效果达到阈值但未达显著（见 `docs/04_根因诊断.md`）。", ""]

    e7, d7 = latest("E7_val")
    if e7:
        L += [f"## E7 bge-m3（val，`{d7.name}`）", "", "```json",
              json.dumps(e7, ensure_ascii=False, indent=1, default=float)[:3000], "```", ""]
    else:
        L += ["## E7 bge-m3：未完成", "",
              "建 bge-m3 稠密索引时（完成约 63%），进程被 Claude Code 以系统内存不足为由中止。按规则没有自动重启，"
              "val 问答还没开始，没有产生 API 费用（见 `data/processed/index/dense_831e5f2eee.STATUS.txt`）。", ""]

    for tag, title in (("E6_telco_rag_test", "E6 Telco-RAG*（test）"),):
        m, d = latest(tag)
        if m:
            L += [f"## {title}（`{d.name}`）", "", "```json",
                  json.dumps({k: v for k, v in m.items() if k not in ("usage",)}, ensure_ascii=False, indent=1,
                             default=float)[:4000], "```", ""]

    costs = [json.loads(line) for line in (ROOT / "outputs" / "log.jsonl").open(encoding="utf-8")]
    paid = [c for c in costs if c.get("cost_cny", 0) > 0]
    total = sum(c["cost_cny"] for c in paid)
    L += ["## 本轮 API 花费", "",
          "| 实验 | 开始时间 | 估算花费（元，按官方空闲价和实际 token 数） |", "|---|---|---|"]
    for c in paid:
        L.append(f"| {c['exp']} | {c['start']} | {c['cost_cny']:.2f} |")
    L += ["", "- E5 有两条记录：第一条只跑 val；第二条跑 val + test，其中 val 部分读的是第一条留下的缓存，"
              "没有重复调用，5.42 元基本都是 test 的费用。E6 的第一条是 20 题试跑，第二条是剩下的 480 题。",
          f"- 估算合计 {total:.2f} 元。另有 E5 前 100 题的探测运行（估算 0.22 元），没有单独写进 log.jsonl。",
          "- 实际扣费以余额变化为准：2026-10-30 17:19 为 51.96 元，22:46 为 41.92 元，"
          "第 1 轮实际扣费 **10.04 元**；第 2 轮（R2-B、R2-A）22:55 为 41.92 元，23:40 为 39.43 元，实际扣费 **2.49 元**。"
          "两轮合计 **12.53 元**（总上限 30 元）。DeepSeek 扣费有延迟，单个实验结束后立刻查询的余额不完全准确。",
          "- 余额明细见 `outputs/balance_log.txt`。", ""]
    (ROOT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
