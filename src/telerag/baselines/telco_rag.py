"""Telco-RAG*：按 Telco-RAG 论文版流程重新实现（PLAN.md E6，docs/02 第 2.6 节）。

上游：github.com/netop-team/Telco-RAG（MIT），Telco-RAG_paper_version/pipeline_offline.py。
保留的四个核心设计（与上游一致）：
  1. LLM 改写问题（提示词逐字照搬上游）；
  2. 词表增强：在 3GPP 词表中查找问题里出现的术语和缩写，把定义拼接到查询后（规则照搬上游 get_definitions.py）；
  3. 按系列路由：只在预测出的前 5 个 3GPP 系列里检索，摘要类文档（rel_xx）始终保留；
  4. 候选答案二次检索：LLM 先列出候选答案，拼接到查询里，重新路由并检索 top-10。
  最后用上游 check_question 的选择题提示词作答。块长 500 字符，重叠 25 字符（与上游相同）。

与上游不同的地方（论文里需要写明）：
  - 嵌入模型：OpenAI text-embedding-3-large（1024 维）换成本地 bge-m3（1024 维）；
  - 路由器：上游的 NNRouter 依赖 OpenAI embedding，不能复用，改成"bge-m3 计算查询与 18 个系列描述的余弦相似度，取前 5 个"；
  - 答题模型：gpt-4o-mini 换成 deepseek-flash（与本文其他方法相同，temperature=0，关闭思考模式）；
  - 不用上游的在线检索（Google）和 validator（论文版 validate_flag=False，与上游一致）。
"""
import json
import re

import numpy as np

from ..config import config_hash, resolve

CHUNK_CHARS, OVERLAP_CHARS, TOP_K, N_SERIES = 500, 25, 10, 5
SEP_RE = re.compile(r'[\s,.\-!?\[\]\(\){}":;<>]+')


# ---------- 上游规则：切块 ----------

def custom_text_splitter(text, chunk_size=CHUNK_CHARS, chunk_overlap=OVERLAP_CHARS):
    """逐行对应上游 src/chunking.py 的 custom_text_splitter(word_split=True)。"""
    chunks, start = [], 0
    while start < len(text) - chunk_overlap:
        end = min(start + chunk_size, len(text))
        m = SEP_RE.search(text, end)
        if m:
            end = m.end()
        if end == start:
            end = start + 1
        chunks.append(text[start:end])
        start = end - chunk_overlap
        m = SEP_RE.search(text, start - 1)
        if m:
            start = m.start() + 1
        if start < 0:
            start = 0
    return chunks


# ---------- 上游规则：词表 ----------

def _pre(text, lowercase=True):
    if lowercase:
        text = text.lower()
    for ch in '''!()-[]{};:'"\\,<>./?@#$%^&*_~''':
        text = text.replace(ch, "")
    return text


def load_vocabulary(path):
    """对应上游 get_definitions.read_docx：第二次出现 'References' 之后，分别读术语和缩写。"""
    from docx import Document

    terms, abbrs = {}, {}
    start, in_terms, in_abbr = 0, False, False
    for para in Document(str(path)).paragraphs:
        t = para.text.strip()
        if "References" in t:
            start += 1
        if start < 2:
            continue
        if "Terms and definitions" in t:
            in_terms, in_abbr = True, False
        elif "Abbreviations" in t:
            in_terms, in_abbr = False, True
        elif in_terms and ":" in t:
            k, v = t.split(":", 1)
            terms[k.strip()] = v.strip().rstrip(".")
        elif in_abbr and "\t" in t:
            k, v = t.split("\t", 1)
            if len(k) > 1:
                abbrs[k.strip()] = v.strip()
    return terms, abbrs


def define_question(sentence, terms, abbrs):
    """对应上游 define_TA_question：术语按小写子串匹配，缩写按大小写敏感的整词匹配，去掉被更长匹配包含的项。"""
    low = _pre(sentence)
    mt = {k: v for k, v in terms.items() if _pre(k) in low}
    mt = {k: v for k, v in mt.items() if not any(k in o and k != o for o in mt)}
    words = _pre(sentence, lowercase=False).split()
    ma = {w: abbrs[w] for w in words if w in abbrs}
    srt = sorted(ma, key=len, reverse=True)
    ma = {a: ma[a] for a in srt if not any(a in o and a != o for o in srt)}
    t = "\n".join(f"{k}: {v}" for k, v in mt.items())
    a = "\n".join(f"{k}: {v}" for k, v in ma.items())
    return f"{sentence}\n\nTerms and Definitions:\n\n{t}\n\n\nAbbreviations:\n\n{a}\n\n"


# ---------- 上游提示词（逐字照搬） ----------

def concise_prompt(question):
    return f"""Rephrase the question to be clear and concise:

        {question}"""


def candidate_prompt(query, context, question):
    return f"""
            Provide all the possible answers to the following question considering your knowledge and the text provided.
            Question: {query}
            Considering the following context:
            {context}
            Provide all the possible answers to the following question considering your knowledge and the text provided.
            Question: {question}
            Ensure none of the answers provided contradicts your knowledge and each answer has at most 100 characters.
            """


def mcq_prompt(query, content, question, options_text):
    return f"""
        Please provide the answers to the following multiple choice question.
        {query}

        Considering the following context:
        {content}

        Please provide the answers to the following multiple choice question.
        {question}

        Options:
        Write only the option number corresponding to the correct answer:\n{options_text}

        Answer format should be: Answer option <option_id>
        """


def find_option_number(text):
    return re.findall(r"option\D*(\d+)", (text or "").lower())


# ---------- 索引 ----------

def series_of(doc_id):
    return int(doc_id[:2]) if doc_id[:2].isdigit() else None


class TelcoIndex:
    """按上游规则切块，用 bge-m3 编码，按系列分组。编码结果缓存到 data/processed/telco_rag/。"""

    def __init__(self, cfg, embed_model="BAAI/bge-m3", device="cuda", batch_size=16):
        from sentence_transformers import SentenceTransformer

        from ..corpus import iter_documents

        import torch

        self.model = SentenceTransformer(embed_model, device=device,
                                         model_kwargs={"dtype": torch.float16} if device == "cuda" else {})
        self.model.max_seq_length = 512
        d = resolve(f"data/processed/telco_rag/{config_hash({'m': embed_model, 'c': CHUNK_CHARS, 'o': OVERLAP_CHARS})}")
        d.mkdir(parents=True, exist_ok=True)
        meta_p, emb_p = d / "chunks.jsonl", d / "emb.npy"
        if meta_p.exists() and emb_p.exists():
            self.chunks = [json.loads(line) for line in meta_p.open(encoding="utf-8")]
            self.emb = np.load(emb_p)
        else:
            self.chunks = []
            for doc_id, text in iter_documents(resolve(cfg["data"]["corpus_db"])):
                for c in custom_text_splitter(text):
                    self.chunks.append({"source": doc_id, "text": c})
            # 分段编码并直接写入 fp16 的内存映射文件，避免一次性在内存中持有 27 万 × 1024 的 fp32 矩阵
            # （第一次尝试时这样做使空闲内存降到 1.7GB）。分段方式不影响编码结果。
            from tqdm import tqdm

            part_p = d / "emb.partial.npy"
            dim = self.model.get_sentence_embedding_dimension()
            mm = np.lib.format.open_memmap(part_p, mode="w+", dtype=np.float16, shape=(len(self.chunks), dim))
            step = 4096
            for s in tqdm(range(0, len(self.chunks), step), desc="Telco-RAG* 编码"):
                mm[s:s + step] = self.model.encode([c["text"] for c in self.chunks[s:s + step]],
                                                   batch_size=batch_size, normalize_embeddings=True,
                                                   show_progress_bar=False, convert_to_numpy=True)
            mm.flush()
            del mm
            part_p.replace(emb_p)
            with meta_p.open("w", encoding="utf-8") as f:
                for c in self.chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            self.emb = np.load(emb_p)
        self.series = np.array([series_of(c["source"]) or -1 for c in self.chunks])
        desc = json.load(open(resolve("data/raw/telco_rag/series_description.json"), encoding="utf-8"))
        self.series_ids = [int(k.split()[0]) for k in desc]
        self.series_emb = self.model.encode([v["description"] for v in desc.values()], normalize_embeddings=True)

    def encode(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def route(self, qvec, n=N_SERIES):
        sims = self.series_emb @ qvec
        return [self.series_ids[i] for i in np.argsort(-sims)[:n]]

    def search(self, qvec, series, k=TOP_K):
        mask = np.isin(self.series, series) | (self.series == -1)    # 摘要类文档（rel_xx）始终保留
        idx = np.where(mask)[0]
        sims = self.emb[idx].astype(np.float32) @ qvec
        top = idx[np.argsort(-sims)[:k]]
        return [int(i) for i in top]

    def context(self, ids):
        return [f"\nRetrieval {i + 1}:\n...{self.chunks[c]['text']}...\nThis retrieval is performed from the document "
                f"3GPP {self.chunks[c]['source']}.\n" for i, c in enumerate(ids)]


def answer_one(llm, index, terms, abbrs, item, route=True):
    """对一道题跑完整的 Telco-RAG* 流程。返回预测、中间结果和用量。

    route=False（PLAN_round2.md R2-A）：检索时不按系列过滤，在全部规范中检索；其余步骤不变。
    """
    question = item["question"]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0, "calls": 0}

    def call(prompt):
        r = llm.chat([{"role": "user", "content": prompt}])
        for k in ("prompt_tokens", "completion_tokens", "cache_hit_tokens"):
            usage[k] += r.get(k, 0)
        usage["calls"] += 1
        return r["text"]

    all_series = sorted(set(index.series_ids))
    concise = call(concise_prompt(question)).rstrip('"')
    query = define_question(concise, terms, abbrs)
    series1 = index.route(index.encode(query)) if route else all_series
    ids1 = index.search(index.encode(query), series1)
    ctx1 = index.context(ids1)
    cand = call(candidate_prompt(query, ctx1, question))
    enhanced = query + "\n" + cand if cand != "NO" else query
    qv2 = index.encode(enhanced)
    series2 = index.route(qv2) if route else all_series
    # 上游：第二次检索时查询文本拼上候选答案（use_context=True）
    ids2 = index.search(index.encode(query + "\n" + cand), series2)
    content = "\n".join(index.context(ids2))
    options_text = "\n".join(f"option {k}: {v}" for k, v in item["options"].items())
    raw = call(mcq_prompt(query, content, question, options_text))
    found = [o for o in find_option_number(raw) if o in item["options"]]
    pred = found[0] if found else None
    return {"qid": item["qid"], "pred": pred, "answer": item["answer"], "hallucinated": int(pred != item["answer"]),
            "series_first": series1, "series_second": series2, "retrieved_chunks": ids2, "raw": raw,
            "concise": concise, "candidates": cand, **usage}
