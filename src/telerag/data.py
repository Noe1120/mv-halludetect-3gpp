"""TeleQnA 题目加载与解析。"""
import random
import re

import pandas as pd

from .config import resolve

_Q_RE = re.compile(r"^Question:\s*(.*?)\s*\nOptions:\s*\n(.*)$", re.S)
_OPT_RE = re.compile(r"^option (\d+):", re.I)
_REL_RE = re.compile(r"\s*\[3GPP Release (\d+)\]\s*")


def parse_row(idx, row):
    m = _Q_RE.match(row["Question"].strip() + "\n")
    if not m:
        raise ValueError(f"无法解析第 {idx} 题: {row['Question'][:80]!r}")
    question, opt_block = m.group(1), m.group(2)
    rel = _REL_RE.search(question)
    question = _REL_RE.sub(" ", question).strip()

    options, cur = {}, None
    for line in opt_block.splitlines():
        om = re.match(r"^(\d+)\.\s*(.*)$", line)
        if om:
            cur = om.group(1)
            options[cur] = om.group(2).strip()
        elif cur and line.strip():          # 选项文本跨行时拼接到上一个选项
            options[cur] += " " + line.strip()

    am = _OPT_RE.match(row["Answer"].strip())
    if not am or am.group(1) not in options:
        raise ValueError(f"第 {idx} 题答案无法解析: {row['Answer'][:80]!r}")
    return {
        "qid": int(idx),
        "category": row["Category"],
        "release": int(rel.group(1)) if rel else None,
        "question": question,
        "options": options,
        "answer": am.group(1),
    }


def load_questions(cfg):
    dcfg = cfg["data"]
    df = pd.read_parquet(resolve(dcfg["teleqna"]))
    df = df[df["Category"].isin(dcfg["categories"])]
    items = [parse_row(i, r) for i, r in df.iterrows()]
    rng = random.Random(dcfg["seed"])
    order = list(range(len(items)))
    rng.shuffle(order)
    dev_ids = set(order[: dcfg["dev_size"]])
    for i, it in enumerate(items):
        it["split"] = "dev" if i in dev_ids else "test"
    return items


def select(items, split, limit=None):
    out = [it for it in items if split == "all" or it["split"] == split]
    return out[:limit] if limit else out


def format_options(options):
    return "\n".join(f"{k}. {v}" for k, v in options.items())
