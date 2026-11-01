# Multi-View Hallucination Detection for 3GPP RAG

Code, prompts, pre-registration documents and per-question outputs for the paper

> **Perturbing the Evidence, Not the Sampler: Multi-View Consistency and Local Verification for Hallucination Detection in Retrieval-Augmented 3GPP Question Answering**
> Shengming Ni, Nanjing University of Posts and Telecommunications.
> Submitted to the *International Journal of Machine Learning and Cybernetics* (2026).

The same LLM answers each 3GPP standards question four times (closed-book, BM25-RAG, dense-RAG and hybrid-RAG). Disagreement among these answers, fused with option-level evidence checks from two small local models (a 0.2B NLI model and Qwen3-0.6B), flags likely wrong answers. On 2,500 held-out TeleQnA standards questions the detector reaches AUROC 0.790, against 0.699 for LLM-as-a-judge and 0.700 for five-sample self-consistency.

## Main results (test split, n = 2,500)

| Detector | Extra cost per question | AUROC [95% CI] |
|---|---|---|
| LLM-as-a-judge | 1 API call | 0.699 [0.674, 0.720] |
| Self-consistency (K = 5) | 5 API calls | 0.700 [0.674, 0.724] |
| SLM verifier only (Qwen3-0.6B) | 0.52 s on a laptop GPU | 0.723 [0.700, 0.745] |
| Multi-view agreement, no training | 3 API calls | 0.737 [0.712, 0.761] |
| **Ours-lite** (closed-book view + local verifiers) | 1 short API call + 0.85 s | 0.762 [0.739, 0.786] |
| **Ours** (all views + local verifiers) | 3 API calls + 0.85 s | **0.790** [0.767, 0.814] |

Abstaining on the riskiest 20% of questions raises accuracy on the remaining ones from 80.8% to 89.2%. Full results, including negative results and root-cause analyses, are in [`REPORT.md`](REPORT.md) and [`docs/04_根因诊断.md`](docs/04_根因诊断.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/telerag/` | Library code: corpus parsing and chunking, retrieval (BM25, dense, RRF, reranking), prompts, LLM client with on-disk cache, NLI/SLM verifiers, multi-view features, risk fusion, metrics |
| `src/telerag/baselines/telco_rag.py` | Re-implementation of the paper version of Telco-RAG (Telco-RAG\*) |
| `scripts/` | Experiment, figure, table and manuscript build scripts (see below) |
| `configs/default.yaml` | All experiment settings |
| `tests/` | Unit tests (switch-off equivalence, API cache, edge cases, data split) |
| `PLAN.md`, `PLAN_round2.md` | Pre-registered hypotheses, thresholds and budgets, written before each experiment |
| `REPORT.md` | Auto-generated results summary |
| `outputs/exp/` | One directory per experiment run: config snapshot, metrics, per-question records |
| `outputs/runs/` | Per-question answers of every answering setting |
| `outputs/figures/` | All figures (PDF and PNG) |
| `outputs/log.jsonl` | Append-only log of every run, including API cost |
| `paper/facts.json` | Every number reported in the paper, extracted from the experiment outputs |

The planning and diagnosis notes (`PLAN*.md`, `REPORT.md`, `docs/`) are written in Chinese.

## Setup

Tested with Python 3.11.18 on Windows 11, an NVIDIA RTX 4060 Laptop GPU (8 GB) and 16 GB RAM. The local models also run on CPU, about 30 times slower.

```bash
# 1. Install PyTorch for your CUDA version (see https://pytorch.org), then the rest
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. Download the public data into data/raw/ (each file is checked against a pinned SHA-256)
python scripts/download_data.py
#    With a slow connection to huggingface.co, use a mirror:
#    HF_ENDPOINT=https://hf-mirror.com python scripts/download_data.py

# 3. API key (only needed for new LLM calls)
cp .env.example .env        # then put your DeepSeek key in .env
```

The Hugging Face models (bge-base-en-v1.5, bge-reranker-base, DeBERTa-v3-base-mnli-fever-anli, Qwen3-0.6B and bge-m3) are downloaded automatically on first use; their exact revisions are listed in Table 1 of the paper.

## Reproducing the paper without new API calls

Every LLM response used in the paper is stored in an SQLite cache. Download `llm_cache.sqlite` from the [GitHub release](https://github.com/Noe1120/mv-halludetect-3gpp/releases) and put it at `data/processed/llm_cache.sqlite`. With `TELERAG_NO_API=1`, any request missing from the cache raises an error instead of calling the API, so the commands below reproduce the reported numbers at no cost.

```bash
export TELERAG_NO_API=1          # Windows PowerShell: $env:TELERAG_NO_API=1

# Detection (Tables 3–4, Figs. 4–5): fit on val (500 q), evaluate once on test (2,500 q)
python scripts/run_eval.py --exp E0
python scripts/run_eval.py --exp E1
python scripts/run_eval.py --exp E2
python scripts/run_eval.py --exp E4

# Retrieval comparison (Table 2)
python scripts/eval_retrieval.py --split test --set data.seed=3407 data.dev_size=500

# Latency and RAGTruth transfer
python scripts/run_latency.py
python scripts/run_ragtruth.py

# Figures and report
python scripts/make_figures.py
python scripts/make_schematics.py
python scripts/make_report.py

# Unit tests
pytest tests -q
```

## Re-running the paid experiments

Without the cache, the answering runs call the DeepSeek API (`deepseek-flash`). All paid experiments in the paper cost 12.53 CNY in total at off-peak prices. The client refuses new calls during DeepSeek's peak hours (09:00–12:00 and 14:00–18:00 Beijing time on weekdays) unless `TELERAG_ALLOW_PEAK=1` is set.

```bash
python scripts/run_qa.py --name closed --no-rag
python scripts/run_qa.py --name hybrid_rerank
python scripts/run_ablation.py --only bm25 dense no_rerank
python scripts/run_eval.py --exp E5 --sc 5 --sc-split both            # self-consistency baseline
python scripts/run_telco_rag.py --split val                            # Telco-RAG* baseline
python scripts/run_telco_rag.py --split val --no-route                 # Telco-RAG* without routing
python scripts/run_retrieval_variant.py --exp E8                       # fixed-length chunks
python scripts/run_retrieval_variant.py --exp R2B                      # equal context budget
```

API models can change behind the same name over time, so new calls may give slightly different results from those in the cache.

## Data and licenses

The code in this repository is released under the [MIT License](LICENSE). The datasets are **not** redistributed here; `scripts/download_data.py` fetches them from their original sources:

| Data | Source | License |
|---|---|---|
| TeleQnA (10,000 questions) | Hugging Face `ketchup123/TeleQnA_full`, a public copy of the official release (Maatouk et al. 2023) | MIT |
| 3GPP Release 18 corpus | Hugging Face `netop/3GPP-R18` (Telco-RAG project) | MIT |
| Telco-RAG glossary and series descriptions | GitHub `netop-team/Telco-RAG` | MIT |
| RAGTruth | GitHub `ParticleMedia/RAGTruth` | MIT |

## Use of AI tools

The experiment code in this repository was written with the assistance of an AI coding assistant and reviewed by the author.

## Citation

```bibtex
@unpublished{ni2026perturbing,
  author = {Ni, Shengming},
  title  = {Perturbing the Evidence, Not the Sampler: Multi-View Consistency and Local Verification
            for Hallucination Detection in Retrieval-Augmented {3GPP} Question Answering},
  note   = {Submitted to the International Journal of Machine Learning and Cybernetics},
  year   = {2026}
}
```
