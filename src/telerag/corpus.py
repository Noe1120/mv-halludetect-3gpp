"""3GPP 规范文本解析与切块。

Documents.db 中每份规范是一段纯文本：开头是封面和目录（目录行以页码结尾），
正文章节标题形如 "6.11.1\tServices and Functions"。
"""
import json
import re
import sqlite3

from .config import config_hash, resolve

HEADING_RE = re.compile(r"^((?:\d+|[A-Z])(?:\.\d+)*[A-Za-z]?)\t(\S.{0,180})$")
ANNEX_RE = re.compile(r"^(Annex [A-Z]+)\b[^\t]*:?\t?(.*)$")
TOC_TAIL_RE = re.compile(r"\t\d+$")
COVER_RE = re.compile(r"^3GPP (TS|TR) ([\d.\-]+) V([\d.]+)")


def _spec_number(doc_id):
    """'38101-2-i40.docx' -> '38.101-2'；'rel_17.docx' -> 'rel_17'。"""
    stem = doc_id.rsplit(".", 1)[0]
    m = re.match(r"^(\d{2})(\d{3})((?:-\d+)*)-[0-9a-z]{3}$", stem)
    return f"{m.group(1)}.{m.group(2)}{m.group(3)}" if m else stem


def _doc_title(lines):
    """封面前几行：第 1 行是编号版本，随后几行是工作组和标题，直到 '(Release xx)'。"""
    kind, title, seen_cover = "TS", [], False
    for line in lines[:12]:
        m = COVER_RE.match(line)
        if m:
            kind, seen_cover = m.group(1), True
            continue
        if not seen_cover:              # 部分文档没有封面，直接从目录开始
            continue
        if line.startswith("(Release"):
            break
        if line and not line.startswith(("Technical", "3rd Generation")):
            title.append(line.strip().rstrip(";"))
    return kind, "; ".join(t for t in title if t)


def _is_heading(line):
    if TOC_TAIL_RE.search(line):         # 目录行（以页码结尾）
        return None
    m = HEADING_RE.match(line)
    if m and m.group(2)[0].isalpha() and not m.group(2).endswith((".", ";", ",")):
        return m.group(1), m.group(2).strip()
    m = ANNEX_RE.match(line)
    if m and len(line) < 200:
        return m.group(1), m.group(2).strip()
    return None


def parse_sections(text):
    """返回 [(章节号, 章节标题, 正文行列表)]，跳过封面和目录。"""
    lines = text.split("\n")
    sections, cur = [], None
    for line in lines:
        h = _is_heading(line)
        if h:
            cur = [h[0], h[1], []]
            sections.append(cur)
        elif cur is not None:
            s = line.strip()
            if s:
                cur[2].append(s)
    # 封面后的第一个 "1 Scope" 之前通常只有 Foreword/Introduction，保留由 drop_sections 过滤
    return [(n, t, body) for n, t, body in sections]


def _windows(words, size, overlap):
    if len(words) <= size:
        yield words
        return
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        yield words[start:start + size]
        if start + size >= len(words):
            break


def iter_documents(db_path):
    con = sqlite3.connect(db_path)
    try:
        for doc_id, data in con.execute("SELECT id, data FROM Standard ORDER BY id"):
            yield doc_id, json.loads(data)["text"]
    finally:
        con.close()


def chunk_document(doc_id, text, ccfg):
    lines = text.split("\n")
    kind, title = _doc_title(lines)
    spec = _spec_number(doc_id)
    size, overlap = ccfg["max_words"], ccfg["overlap_words"]
    drop = [d.lower() for d in ccfg.get("drop_sections", [])]
    chunks = []

    def emit(section_no, section_title, words):
        parts = [f"3GPP {kind} {spec}", title, f"{section_no} {section_title}".strip()]
        header = "[" + " | ".join(p for p in parts if p) + "]"
        chunks.append({
            "doc_id": doc_id, "spec": spec, "section": section_no,
            "section_title": section_title, "header": header,
            "text": " ".join(words),
        })

    if ccfg["strategy"] == "fixed":
        # 基线：去掉目录后整篇按固定词数切，不看章节边界
        body = [ln for ln in lines if ln.strip() and not TOC_TAIL_RE.search(ln)]
        for w in _windows(" ".join(body).split(), size, overlap):
            emit("", "", w)
        return chunks

    pending_no, pending_title, pending = None, None, []
    for no, sec_title, body in parse_sections(text):
        if any(d in sec_title.lower() for d in drop):
            continue
        words = " ".join(body).split()
        if pending:                                  # 把太短的上一节并入当前节
            words = pending + words
            no, sec_title = pending_no, f"{pending_title} / {no} {sec_title}"
            pending = []
        if len(words) < ccfg["min_words"]:
            pending_no, pending_title, pending = no, sec_title, words
            continue
        for w in _windows(words, size, overlap):
            emit(no, sec_title, w)
    if pending:
        emit(pending_no, pending_title, pending)
    return chunks


def build_chunks(cfg, force=False):
    ccfg = cfg["chunking"]
    out = resolve(f"data/processed/chunks_{ccfg['strategy']}_{config_hash(ccfg)}.jsonl")
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for doc_id, text in iter_documents(resolve(cfg["data"]["corpus_db"])):
            for c in chunk_document(doc_id, text, ccfg):
                c["cid"] = n
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
    tmp.replace(out)
    return out


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_text(c):
    """送进检索模型和大模型的文本：带上规范编号和章节标题（上下文化切块）。"""
    return f"{c['header']}\n{c['text']}" if c["header"] else c["text"]
