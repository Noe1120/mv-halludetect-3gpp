"""Telco-RAG* 移植规则的单元测试（不加载模型、不访问网络）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telerag.baselines import telco_rag as tr  # noqa: E402


def test_splitter_covers_text_with_overlap():
    text = ("The UE shall send a MeasurementReport message to the gNB. " * 40).strip()
    chunks = tr.custom_text_splitter(text, 100, 10)
    assert len(chunks) > 5
    assert all(len(c) <= 100 + 60 for c in chunks)        # 只会延伸到下一个分隔符
    assert text.startswith(chunks[0])
    assert chunks[-1].rstrip() and text.rstrip().endswith(chunks[-1].rstrip()[-20:])


def test_splitter_short_text():
    # 上游循环条件是 start < len(text) - overlap：短于重叠长度（25 字符）的文本不产生任何块。
    # 这是上游行为，照搬不改；3GPP 规范都远长于 25 字符，不受影响。
    assert tr.custom_text_splitter("short text", 500, 25) == []
    assert tr.custom_text_splitter("a slightly longer text that exceeds overlap", 500, 25) == [
        "a slightly longer text that exceeds overlap"]
    assert tr.custom_text_splitter("", 500, 25) == []


def test_define_question_terms_and_abbrs():
    terms = {"User Equipment": "a device", "Equipment": "thing"}
    abbrs = {"UE": "User Equipment", "RRC": "Radio Resource Control"}
    q = tr.define_question("How does the UE use User Equipment in RRC?", terms, abbrs)
    assert "User Equipment: a device" in q
    assert "Equipment: thing" not in q                     # 被更长的术语包含，去掉
    assert "UE: User Equipment" in q and "RRC: Radio Resource Control" in q
    assert q.startswith("How does the UE")


def test_find_option_number():
    assert tr.find_option_number("Answer option 3") == ["3"]
    assert tr.find_option_number("Answer: Option <2>") == ["2"]
    assert tr.find_option_number("") == []
    assert tr.find_option_number(None) == []


def test_series_of():
    assert tr.series_of("38300-i00.docx") == 38
    assert tr.series_of("rel_17.docx") is None


class _FakeIndex:
    """不加载模型的假索引：记录每次检索用了哪些系列。"""
    series_ids = [21, 23, 29, 38]

    def __init__(self):
        self.calls = []

    def encode(self, text):
        return text

    def route(self, qvec, n=tr.N_SERIES):
        return [29]

    def search(self, qvec, series, k=tr.TOP_K):
        self.calls.append(list(series))
        return [0]

    def context(self, ids):
        return ["ctx"]


class _FakeLLM:
    def chat(self, messages):
        return {"text": "Answer option 2", "prompt_tokens": 1, "completion_tokens": 1, "cache_hit_tokens": 0}


def test_route_switch():
    item = {"qid": 1, "question": "Q?", "options": {"1": "a", "2": "b"}, "answer": "2"}
    idx = _FakeIndex()
    r = tr.answer_one(_FakeLLM(), idx, {}, {}, item)                  # 默认 route=True，与 E6 相同
    assert idx.calls == [[29], [29]] and r["series_second"] == [29] and r["pred"] == "2"
    idx = _FakeIndex()
    r = tr.answer_one(_FakeLLM(), idx, {}, {}, item, route=False)     # R2-A：两次检索都覆盖全部系列
    assert idx.calls == [[21, 23, 29, 38], [21, 23, 29, 38]] and r["calls"] == 3


def test_vocabulary_loads():
    p = ROOT / "data/raw/telco_rag/3GPP_vocabulary.docx"
    if not p.exists():
        pytest.skip("缺少词表")
    terms, abbrs = tr.load_vocabulary(p)
    assert len(terms) > 100 and len(abbrs) > 100
    assert "UE" in abbrs
