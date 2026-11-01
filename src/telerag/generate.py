"""RAG 答题：拼提示词、调用大模型、解析选项。"""
import re

from .corpus import chunk_text
from .data import format_options

SYSTEM = "You are an expert in 3GPP telecommunications standards."

RAG_TEMPLATE = """Answer the multiple-choice question using the retrieved 3GPP specification excerpts. If the excerpts are insufficient, rely on your own knowledge.

Excerpts:
{context}

Question: {question}
Options:
{options}

Reply in exactly this format:
Answer: <option number>
Evidence: <excerpt number(s) that support the answer, e.g. [2], or none>
Explanation: <one sentence>"""

CLOSED_TEMPLATE = """Answer the multiple-choice question about 3GPP telecommunications standards.

Question: {question}
Options:
{options}

Reply in exactly this format:
Answer: <option number>
Explanation: <one sentence>"""

_ANS_RE = re.compile(r"Answer\s*[:：]\s*(?:option\s*)?\(?(\d+)", re.I)
_EVI_RE = re.compile(r"Evidence\s*[:：]\s*(.*)", re.I)
_EXP_RE = re.compile(r"Explanation\s*[:：]\s*(.*)", re.I | re.S)


def format_context(retrieved, chunks):
    return "\n\n".join(f"[{i + 1}] {chunk_text(chunks[cid])}" for i, (cid, _) in enumerate(retrieved))


def build_messages(item, retrieved=None, chunks=None):
    opts = format_options(item["options"])
    if retrieved:
        user = RAG_TEMPLATE.format(context=format_context(retrieved, chunks),
                                   question=item["question"], options=opts)
    else:
        user = CLOSED_TEMPLATE.format(question=item["question"], options=opts)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_response(text, options):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)   # 兼容会输出思考过程的模型
    m = _ANS_RE.search(text)
    pred = m.group(1) if m and m.group(1) in options else None
    if pred is None:                          # 兜底：取第一个合法的选项编号
        for d in re.findall(r"\b(\d)\b", text):
            if d in options:
                pred = d
                break
    ev = _EVI_RE.search(text)
    evidence = [int(x) for x in re.findall(r"\d+", ev.group(1))] if ev else []
    ex = _EXP_RE.search(text)
    explanation = ex.group(1).strip().split("\n")[0] if ex else ""
    return {"pred": pred, "evidence": evidence, "explanation": explanation}
