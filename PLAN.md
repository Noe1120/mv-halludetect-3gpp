# PLAN.md：第 1 轮预注册（2026-10-30）

> 依据 `要求.md` 第三节第 4 条。**本计划确认之前，不启动任何正式实验，也不调用 API。**
> 前置文档：`docs/01_文献调研总结.md`、`docs/02_模块插入位置论证.md`。
> 本实验为**单种子**（seed = 3407），结论存在随机噪声风险。

## 0. 冻结项（本轮及以后各轮都不改）

| 项 | 冻结值 | 位置 |
|---|---|---|
| 数据 | TeleQnA 中与标准相关的 3,000 题（Standards specifications 2,000 题，Standards overview 1,000 题） | `configs/default.yaml` `data.categories` |
| 划分 | **seed = 3407**，打乱后取前 500 题为 val，其余 2,500 题为 test | `data.seed=3407`、`data.dev_size=500` |
| 语料 | 3GPP Release 18，`Documents.db`，554 份规范 | — |
| 生成提示词 | `src/telerag/generate.py` 中的 `RAG_TEMPLATE` 和 `CLOSED_TEMPLATE`，逐字不改 | — |
| 生成参数 | `deepseek-flash`，temperature = 0，max_tokens = 300，关闭思考模式 | `llm.*` |
| 检索参数 | 每路召回 30 个候选，RRF k = 60，top_k = 5，查询只用题干 | `retrieval.*` |
| 检测器评测 | 在 val 上训练逻辑回归、选阈值（使 F1 最大）；在 test 上**只做一次**前向评测 | `scripts/run_detect.py --protocol val2test` |
| 统计方法 | AUROC 的 bootstrap 95% CI（1000 次，seed = 0）；与基线的配对 bootstrap 差值 CI；准确率用 McNemar 精确检验 | `src/telerag/metrics.py` |
| 日志 | 统一追加写入 `outputs/log.jsonl`；每个实验一个独立的 `outputs/exp/<实验名>/` 目录，不覆盖旧目录 | — |
| 缓存 | 所有 API 响应都缓存在 `data/processed/llm_cache.sqlite`；做离线分析时设置 `TELERAG_NO_API=1` | — |

**宿主基线（host baseline）**：混合检索（BM25 + bge-base，RRF）加 bge-reranker，配合 deepseek-flash 答题（`hybrid_rerank`）。它在 seed=3407 划分下的答题结果已经有缓存，**不需要再花钱**。

## 1. 本轮实验总览

| 编号 | 实验 | 改变的开关 | 是否用 API | 预估费用（元，空闲时段） |
|---|---|---|---|---|
| E0 | 按新协议重跑检测（只读缓存） | 无（换评测协议） | 否 | 0 |
| E1 | SLM 的 margin 改用 logit 尺度 | `detection.margin_scale: prob → logit` | 否 | 0 |
| E2 | 原子事实拆分加核查 | `detection.atomic_facts: false → true` | 否（规则拆分） | 0 |
| E3 | RAGAS 式忠实度（本地 NLI 实现） | 新增评测指标，不影响检测 | 否 | 0 |
| E4 | 纠正：高风险题改用 BM25 视角的答案 | `correction.mode: none → switch_view` | 否（BM25 视角已有缓存） | 0 |
| E5 | Self-Consistency 基线（K = 5，temperature = 0.7） | 新基线 | **是** | val 约 1.0；test 约 4.8。缓存未命中时上限分别为 5.1 和 25.6 |
| E6 | Telco-RAG* 风格复现基线 | 新基线 | **是** | val 约 1.6；test 约 8.2 |
| E7 | 稠密检索 bge-base 换 bge-m3 | `retrieval.embed_model` | **是**（稠密和混合两个视角要重新答题） | 先只跑 val：约 2.0；通过判定后再跑 test：约 9.7 |
| E8 | 固定长度切块对照（val） | `chunking.strategy: section → fixed` | **是** | val 约 1.0 |
| E9 | CPU 延迟测量（100 题） | 无 | 否 | 0 |
| E10 | RAGTruth 迁移（只迁移核验器部分） | 新数据集 | 否 | 0 |

**费用上限**：本轮合计**不超过 30 元**。当前余额 51.99 元。
- 所有 API 实验只在北京时间的**空闲时段**跑：工作日 12:00 到 14:00、18:00 以后，以及周末全天。脚本会检查当前时间，在高峰时段拒绝启动。
- 每个 API 实验先跑 val。val 判定不成立的，就**不跑 test**。

**价格来源**：DeepSeek 官方定价页（2026-10-30 抓取）。deepseek-flash 空闲时段价格为：输入缓存命中 0.02 元/百万 token，未命中 1 元/百万 token，输出 4 元/百万 token。高峰时段是北京时间工作日 9:00 到 12:00、14:00 到 18:00，价格翻倍。

## 2. 各实验的假设、对照与判定阈值

阈值都是**预先设定**的，跑出结果后不改。判定以 val 为主，test 只用来确认方向一致。

### E0：按新协议重跑检测
- **目的**：在合规的协议下重新得到所有已有方法的指标。这是后续所有对照的基线。
- **对照组**：Retrieval、Self-cite、NLI-own、SLM-own、MV-Consist、Ours、Ours-lite、LLM-Judge。
- **判定**：不设阈值，只记录结果。如果 test 上 Ours 与 LLM-Judge 的 AUROC 差值 CI 跨过 0，就把"显著优于 Judge"这个结论降级为"与 Judge 相当"。

### E1：SLM margin 改用 logit 尺度
- **假设 H1**：Qwen3-0.6B 的打分饱和（val 上 61% 的题所选选项和最强竞争选项相差不到 0.01），在概率空间计算 margin 会丢失区分度；换成 logit 尺度能恢复。
- **对照**：SLM-contrast（概率尺度）与 SLM-contrast（logit 尺度）；Ours（概率尺度）与 Ours（logit 尺度）。
- **成立条件**：val 上 SLM-contrast 的 AUROC 提升 ≥ 0.01，**并且** test 上同向。
- **预期**：val 探针显示单特征从 0.681 提到 0.705，预计成立。

### E2：原子事实拆分加核查
- **假设 H2**：把解释句拆成子句，逐条用 NLI 判断是否被证据蕴含，"被支持的子句比例"可以作为额外的风险特征。
- **实现**：按分号、逗号、"and/while/whereas"做规则拆分，只保留至少 3 个词的子句。没有子句时，退回"题目 + 所选选项"这条主张。
- **对照**：Ours 与 Ours + atomic。
- **成立条件**：val 上 AUROC 提升 ≥ 0.01，并且 test 上同向。
- **预期**：**很可能不成立**。解释句中位数只有 26 个词，56% 不超过 2 个子句（探针 P6）。如果不成立，按"拆分粒度失配"做根因诊断。

### E3：RAGAS 式忠实度（本地 NLI 实现）
- **定义**：沿用 RAGAS 的 faithfulness 定义：解释中被检索上下文蕴含的陈述数，除以陈述总数。陈述来自 E2 的规则拆分；蕴含判定用 DeBERTa-NLI（蕴含概率 > 0.5）。
- **用途**：作为各检索配置的评测指标（表 1 加一列），以及作为检测器的一个基线（Faithfulness 越低，风险越高）。
- **判定**：不设阈值，只记录结果。论文里注明这是本地替代实现，不是官方的 RAGAS（官方版本需要调用 LLM）。

### E4：纠正（高风险题换答案）
- **假设 H4**：对风险分最高的 τ 比例的题，把最终答案换成 BM25 视角的答案，可以提高整体准确率。依据：val 上 BM25 视角 85.2%，高于混合检索的 83.2%。
- **超参**：τ ∈ {10%, 20%, 30%}，只在 val 上选。
- **对照**：不纠正（83.2%）、全部换成 BM25 视角（85.2%）、按风险换答案。
- **成立条件**：val 上准确率 > max(不纠正, 全部用 BM25) + 0.5 个百分点，并且 test 上同向。
- **风险**：如果"全部用 BM25"就是最好的，说明这个结果不是纠正机制带来的，而是 BM25 检索本身更好。这种情况如实报告。

### E5：Self-Consistency 基线
- **设定**：用宿主基线的 RAG 提示词，temperature = 0.7，采样 K = 5 次；风险分 = 1 − 与最终答案相同的采样所占比例。
- **对照**：MV-Consist（三个视角）、Ours-lite、Ours。
- **判定（对本文方法而言）**：如果 test 上 Ours 的 AUROC ≥ SC 的 AUROC，并且每题额外成本更低，就判定本文方法占优。如果 SC 更好，如实报告，把论文结论改成"本文方法成本更低，但效果不如 SC"。
- **中止条件**：如果 val 上前 100 题的实际花费超过 0.5 元，就停下来，重新估算费用。

### E6：Telco-RAG* 风格复现
- **设定**：见 `docs/02` 第 2.6 节。保留四个核心设计：问题改写、术语表增强、按系列路由（bge-m3 计算问题与 18 个系列描述的相似度，取前 5 个）、用候选答案二次检索（top-10）。答题模型统一用 deepseek-flash。
- **对照**：闭卷、BM25、混合检索加重排。
- **判定**：不设阈值，作为领域基线报告。如果它的准确率显著高于宿主基线（McNemar p < 0.05），后续轮次考虑把它作为新的宿主。
- **工具**：术语表需要读取 `3GPP_vocabulary.docx`，要安装 python-docx。Telco-RAG 的仓库代码**只读，不执行**；改写后的实现放在 `src/telerag/baselines/telco_rag.py`。

### E7：bge-m3
- **假设 H7**：在三种检索中，稠密检索最弱（val 79.4%）。bge-m3 支持更长的输入，能提升稠密检索，进而提升混合检索。
- **成立条件**：val 上 dense-m3 的准确率比 dense-base 高 ≥ 1 个百分点，**或者** hybrid-m3 比 hybrid-base 高 ≥ 1 个百分点。只有成立时，才跑 test。
- **显存风险**：bge-m3 有 568M 参数，fp16 编码 11.7 万个块时 batch 设为 16。如果显存不足，降到 batch 8。

### E8：固定长度切块对照
- **假设 H8**：按章节切块并附上下文前缀，比固定 300 词切块的准确率更高。
- **成立条件**：val 上 section 比 fixed 高 ≥ 1 个百分点。只用来回答 `要求.md` 中 RQ1 的"切块方式贡献多少"，不影响宿主基线。

### E9：CPU 延迟
- 抽 100 题，强制 `device=cpu`，测 NLI 核验、SLM 核验和检索各自的毫秒数/题。报告 GPU 和 CPU 两种延迟。

### E10：RAGTruth 迁移
- **限制**：RAGTruth 是给定的回答，没有闭卷、BM25 等多个视角，**多视角一致性迁移不了**。只评测本地核验器：NLI 和 Qwen3-0.6B 对"上下文 + 回答"的支持度打分，以及原子事实拆分。
- **对照**：RAGTruth 论文和 LettuceDetect 中报告的数字（引用，不复现）。
- **数据**：从 hf-mirror 下载，放到单独目录，作为不可信数据处理。
- **判定**：不设阈值。只回答"本地核验器离开 3GPP 后还有没有区分能力"（AUROC 显著高于 0.5）。

## 3. 需要实现的代码（全部是可开关的设计，关闭时与宿主基线逐位一致）

| 模块 | 文件 | 开关 | 默认值 |
|---|---|---|---|
| 评测协议 | `scripts/run_detect.py` | `--protocol {testcv, val2test}` | `val2test`（旧的 `testcv` 保留，用于复现归档结果） |
| margin 尺度 | `src/telerag/detect.py` | `detection.margin_scale` | `prob` |
| 原子事实 | `src/telerag/atomic.py`（新建） | `detection.atomic_facts` | `false` |
| 忠实度 | `src/telerag/faithfulness.py`（新建） | 评测指标 | 只在指定时计算 |
| 纠正 | `src/telerag/correction.py`（新建） | `correction.mode` | `none` |
| SC | `src/telerag/consistency.py`（已有 `self_consistency`） | `--sc K` | 0 |
| Telco-RAG* | `src/telerag/baselines/telco_rag.py`（新建） | 单独脚本 | — |
| 高峰期保护 | `src/telerag/llm.py` | 高峰时段拒绝新的 API 请求；设置 `TELERAG_ALLOW_PEAK=1` 可以放行 | 开启 |
| 实验目录与日志 | `src/telerag/explog.py`（新建） | 每个脚本启动时写入 | — |

**单元测试**（`tests/`，用 pytest）：
- 每个开关关闭时，输出与宿主基线逐位一致（逐题比较风险分和特征矩阵）；
- 缓存命中时不发起网络请求（mock 掉 OpenAI 客户端）；`TELERAG_NO_API=1` 时缓存未命中会直接报错；
- 边界情况：检索结果为空；选项解析异常（答案为 None）；NLI 分数并列；解释为空（原子拆分要退回题目加选项）；高峰期判断（时区和周末）；val 与 test 不重叠，且合起来覆盖全部 3,000 题。

## 4. 产出

- 每个实验写到 `outputs/exp/<日期>_<编号>_<名称>/`，内容包括：`config.yaml`（完整配置快照）、`metrics.json`、逐题记录 `records.jsonl`、`features.npy`、成本统计。
- `outputs/log.jsonl`：每次运行追加一行，记录时间、实验、git 风格的配置哈希、命令行、耗时、花费。
- `REPORT.md`：自动汇总 val 和 test 的全部指标，包括准确率、忠实度、AUROC、F1、各类别和各视角的细分、模型参数量、GPU/CPU 延迟、API 花费（元），以及每个假设的判定结果。不成立的，附根因诊断和具体错题样本。

## 5. 执行顺序

1. 先写代码和单元测试（不花钱）。
2. 跑 E0 到 E4、E9（不花钱），生成第一版 `REPORT.md`。
3. 在空闲时段按 E5 → E6 → E8 → E7 的顺序跑 val。每跑完一个，就记一次余额。
4. 只有 val 判定成立的实验才跑 test。
5. E10（RAGTruth）放在最后。

## 6. 确认记录（2026-10-30 13:55，已锁定，之后不改）

- 用户确认了费用上限（30 元）和执行顺序。判定阈值交由我决定，最终如下：
  - **沿用第 2 节的效应量阈值**：AUROC 提升 ≥ 0.01，准确率提升 ≥ 1 个百分点（E4 为 0.5 个百分点）。
  - **另加一层显著性标注**：val 只有约 84 个幻觉样本，0.01 的 AUROC 差值落在噪声范围内。所以每个"成立"的判定都要额外报告配对 bootstrap 差值的 95% CI：
    - CI 不含 0，记为"成立（显著）"；
    - CI 含 0，但 val 达到阈值、test 方向一致，记为"成立（未达显著）"。论文里只把"成立（显著）"写成方法贡献。
  - 阈值在看到任何新结果之前写下，跑完之后不调整。
