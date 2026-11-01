## 表 1  检索配置对答题准确率的影响（测试集 2,800 题，deepseek-flash）

| Setting | AnsHit@5 | Acc. (all) | Acc. (Spec.) | Acc. (Overview) | Cost (USD) |
|---|---|---|---|---|---|
| Closed-book (no retrieval) | – | 75.6 | 72.5 | 81.9 | 0.15 |
| BM25 | 43.0 | 81.8 | 80.4 | 84.5 | 1.29 |
| Dense (bge-base) | 39.2 | 79.4 | 77.9 | 82.3 | 1.11 |
| Hybrid (BM25+Dense, RRF) | 43.3 | 81.6 | 80.5 | 83.7 | 1.21 |
| Hybrid + rerank (full) | 46.6 | 81.2 | 79.8 | 84.0 | 1.23 |

## 表 1b  第二个答题模型（deepseek-v4-pro，测试集前 1,000 题）

| Setting | AnsHit@5 | Acc. (all) | Acc. (Spec.) | Acc. (Overview) | Cost (USD) |
|---|---|---|---|---|---|
| Closed-book (no retrieval) | – | 75.5 | 72.0 | 81.7 | 0.05 |
| BM25 | – | 79.3 | 78.6 | 80.6 | 0.46 |
| Dense (bge-base) | – | 76.4 | 74.8 | 79.2 | 0.39 |
| Hybrid + rerank (full) | – | 77.7 | 76.5 | 79.8 | 0.42 |

## 幻觉检测（detect_hybrid_rerank）

答题模型准确率 81.2%，幻觉率 18.8%，n = 2800；核验小模型 Qwen/Qwen3-0.6B；视角 ['closed', 'bm25', 'dense']；5-fold stratified CV on test (out-of-fold)

| Detector | AUROC [95% CI] | ΔAUROC vs Judge [95% CI] | AUPRC | F1 | AURC↓ | Extra cost/q |
|---|---|---|---|---|---|---|
| Retrieval | 0.593 [0.564, 0.618] | [-0.135, -0.078] | 0.240 | 0.341 | 0.151 | free |
| Self-cite | 0.589 [0.564, 0.614] | [-0.137, -0.081] | 0.233 | 0.338 | 0.151 | free |
| MV-Consist | 0.734 [0.708, 0.760] | [+0.001, +0.069] | 0.464 | 0.505 | 0.111 | 3 API call(s), $0.91/1k q |
| NLI-own | 0.606 [0.577, 0.633] | [-0.122, -0.063] | 0.262 | 0.341 | 0.140 | 0.36s local GPU |
| NLI-contrast | 0.616 [0.589, 0.642] | [-0.113, -0.053] | 0.247 | 0.345 | 0.126 | 0.36s local GPU |
| SLM-own | 0.697 [0.672, 0.720] | [-0.029, +0.026] | 0.353 | 0.385 | 0.103 | 0.54s local GPU |
| SLM-contrast | 0.689 [0.664, 0.715] | [-0.040, +0.020] | 0.358 | 0.406 | 0.110 | 0.54s local GPU |
| Ours | 0.807 [0.785, 0.828] | [+0.081, +0.135] | 0.579 | 0.546 | 0.072 | 3 API call(s), $0.91/1k q + 0.89s local GPU |
| Ours-lite | 0.769 [0.747, 0.792] | [+0.042, +0.099] | 0.511 | 0.493 | 0.083 | 1 API call(s), $0.05/1k q + 0.89s local GPU |
| Ours w/o MV | 0.696 [0.671, 0.720] | [-0.031, +0.024] | 0.354 | 0.404 | 0.103 | 0.89s local GPU |
| Ours w/o SLM | 0.794 [0.772, 0.815] | [+0.068, +0.122] | 0.546 | 0.542 | 0.075 | 3 API call(s), $0.91/1k q + 0.36s local GPU |
| Ours w/o NLI | 0.800 [0.778, 0.821] | [+0.073, +0.129] | 0.578 | 0.536 | 0.076 | 3 API call(s), $0.91/1k q + 0.54s local GPU |
| Ours w/o contrast | 0.798 [0.775, 0.819] | [+0.071, +0.126] | 0.566 | 0.542 | 0.077 | 3 API call(s), $0.91/1k q + 0.89s local GPU |
| Ours w/o free | 0.805 [0.782, 0.826] | [+0.078, +0.133] | 0.570 | 0.542 | 0.073 | 3 API call(s), $0.91/1k q + 0.89s local GPU |
| MV (closed only) | 0.658 [0.628, 0.685] | [-0.077, -0.004] | 0.315 | 0.408 | 0.141 | 1 API call(s), $0.05/1k q |
| MV + SLM-contrast | 0.792 [0.769, 0.815] | [+0.062, +0.123] | 0.563 | 0.530 | 0.082 | 3 API call(s), $0.91/1k q + 0.54s local GPU |
| LLM-Judge | 0.700 [0.676, 0.720] | – | 0.314 | 0.481 | 0.117 | 1 API call(s), $0.40/1k q |

Ours 的标准化逻辑回归系数（正值 = 越大越可能是幻觉）：

view_vote -0.36, agree_closed -0.34, agree_bm25 -0.32, slm_argmax_agree -0.28, nli_margin -0.20, cites_evidence -0.19, slm_margin -0.17, slm_own -0.14, agree_dense -0.14, nli_own -0.13, rerank_top -0.10, nli_rival +0.03, nli_contra -0.01, slm_rival +0.00

## 幻觉检测（detect_pro__hybrid_rerank）

答题模型准确率 77.7%，幻觉率 22.3%，n = 1000；核验小模型 Qwen/Qwen3-0.6B；视角 ['closed', 'bm25', 'dense']；5-fold stratified CV on test (out-of-fold)

| Detector | AUROC [95% CI] | ΔAUROC vs Judge [95% CI] | AUPRC | F1 | AURC↓ | Extra cost/q |
|---|---|---|---|---|---|---|
| Retrieval | 0.647 [0.606, 0.689] | – | 0.339 | 0.418 | 0.161 | free |
| Self-cite | 0.623 [0.580, 0.666] | – | 0.302 | 0.398 | 0.180 | free |
| MV-Consist | 0.750 [0.707, 0.793] | – | 0.552 | 0.577 | 0.135 | 3 API call(s), $0.90/1k q |
| NLI-own | 0.665 [0.624, 0.705] | – | 0.382 | 0.413 | 0.154 | 0.33s local GPU |
| NLI-contrast | 0.666 [0.627, 0.705] | – | 0.360 | 0.422 | 0.142 | 0.33s local GPU |
| SLM-own | 0.728 [0.689, 0.764] | – | 0.472 | 0.474 | 0.113 | 0.54s local GPU |
| SLM-contrast | 0.724 [0.684, 0.762] | – | 0.479 | 0.451 | 0.123 | 0.54s local GPU |
| Ours | 0.828 [0.797, 0.859] | – | 0.658 | 0.578 | 0.080 | 3 API call(s), $0.90/1k q + 0.87s local GPU |
| Ours-lite | 0.803 [0.770, 0.837] | – | 0.630 | 0.564 | 0.092 | 1 API call(s), $0.05/1k q + 0.87s local GPU |
| Ours w/o MV | 0.729 [0.693, 0.765] | – | 0.474 | 0.461 | 0.118 | 0.87s local GPU |
| Ours w/o SLM | 0.820 [0.788, 0.852] | – | 0.645 | 0.613 | 0.085 | 3 API call(s), $0.90/1k q + 0.33s local GPU |
| Ours w/o NLI | 0.826 [0.793, 0.856] | – | 0.660 | 0.598 | 0.082 | 3 API call(s), $0.90/1k q + 0.54s local GPU |
| Ours w/o contrast | 0.821 [0.789, 0.852] | – | 0.660 | 0.613 | 0.092 | 3 API call(s), $0.90/1k q + 0.87s local GPU |
| Ours w/o free | 0.828 [0.796, 0.859] | – | 0.654 | 0.582 | 0.080 | 3 API call(s), $0.90/1k q + 0.87s local GPU |
| MV (closed only) | 0.692 [0.648, 0.735] | – | 0.401 | 0.477 | 0.153 | 1 API call(s), $0.05/1k q |
| MV + SLM-contrast | 0.825 [0.790, 0.855] | – | 0.645 | 0.577 | 0.083 | 3 API call(s), $0.90/1k q + 0.54s local GPU |

Ours 的标准化逻辑回归系数（正值 = 越大越可能是幻觉）：

agree_closed -0.49, view_vote -0.39, agree_bm25 -0.32, slm_argmax_agree -0.28, slm_margin -0.26, slm_own -0.24, nli_margin -0.19, rerank_top -0.17, cites_evidence -0.16, nli_own -0.15, nli_contra +0.10, agree_dense -0.03, slm_rival -0.01, nli_rival +0.00
