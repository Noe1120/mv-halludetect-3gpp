"""要求.md 第三节第 3 条：开关关闭时与基线逐位一致、缓存读取、边界情况。

运行：.conda\\python.exe -m pytest tests -q
这些测试不加载任何模型，也不访问网络。
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from telerag import atomic, correction, detect, generate, llm  # noqa: E402
from telerag.consistency import sc_agreement, view_features  # noqa: E402

BJ = dt.timezone(dt.timedelta(hours=8))


def rec(pred="2", answer="2", nli=None, slm=None, retrieved=((5, 0.9), (7, 0.1)), evidence=(1,), expl="x y z"):
    r = {"qid": 1, "pred": pred, "answer": answer, "hallucinated": int(pred != answer),
         "retrieved": [list(x) for x in retrieved], "evidence": list(evidence), "explanation": expl}
    if nli is not None:
        r["nli"] = nli
    if slm is not None:
        r["slm"] = slm
    return r


# ---------- 开关关闭 = 基线 ----------

def test_feature_dict_base_keys_unchanged():
    """新增 logit 和原子特征后，原有特征的值必须逐位不变。"""
    r = rec(nli={"1": (0.2, 0.1), "2": (0.7, 0.05), "3": (0.4, 0.3)}, slm={"1": 0.9, "2": 0.99, "3": 0.95})
    f = detect.feature_dict(r)
    assert f["nli_own"] == 0.7 and f["nli_rival"] == 0.4 and f["nli_margin"] == pytest.approx(0.3)
    assert f["nli_contra"] == 0.05
    assert f["slm_own"] == 0.99 and f["slm_rival"] == 0.95 and f["slm_margin"] == pytest.approx(0.04)
    assert f["rerank_top"] == 0.9 and f["cites_evidence"] == 1.0
    assert "atomic_n" not in f            # 原子开关关闭时不出现原子特征


def test_risk_model_ignores_extra_features():
    """只用基线特征训练时，多出来的 logit/原子特征不影响预测（逐位一致）。"""
    rng = np.random.default_rng(0)
    recs = []
    for i in range(80):
        s = {str(o): float(rng.uniform(0.5, 1)) for o in range(1, 5)}
        n = {str(o): (float(rng.uniform()), float(rng.uniform())) for o in range(1, 5)}
        r = rec(pred=str(1 + i % 4), answer=str(1 + (i * 7) % 4), nli=n, slm=s)
        recs.append(r)
    feats = ["nli_own", "nli_margin", "slm_own", "slm_margin"]
    a = detect.RiskModel(feats).fit(recs).predict(recs)
    for r in recs:
        r["atomic"] = {"atomic_n": 3.0, "atomic_min": 0.1, "atomic_mean": 0.2, "faithfulness": 0.5}
    b = detect.RiskModel(feats).fit(recs).predict(recs)
    assert np.array_equal(a, b)


def test_correction_off_is_identity():
    recs = [rec(pred=p) for p in ["1", "2", None, "4"]]
    preds = correction.apply_correction(recs, [0.9, 0.1, 0.5, 0.3], {1: "3"}, mode="none", tau=0.5)
    assert preds == ["1", "2", None, "4"]
    assert correction.apply_correction(recs, [0.9, 0.1, 0.5, 0.3], {1: "3"}, mode="switch_view", tau=0) == preds


def test_correction_switch_top_risk_only_and_stable():
    recs = [dict(rec(pred="1"), qid=i) for i in range(4)]
    alt = {0: "9", 1: "9", 2: "9", 3: "9"}
    out = correction.apply_correction(recs, [0.2, 0.8, 0.8, 0.1], alt, mode="switch_view", tau=0.25)
    assert out == ["1", "9", "1", "1"]     # 并列时按原顺序取第一个
    with pytest.raises(ValueError):
        correction.apply_correction(recs, [0] * 4, alt, mode="bogus", tau=0.5)


# ---------- 边界情况 ----------

def test_contrastive_pred_none_or_unknown():
    own, rival, margin = detect.contrastive_features(None, {"1": 0.3, "2": 0.6})
    assert own == 0.0 and rival == 0.6 and margin == -0.6
    own, rival, margin = detect.contrastive_features("9", {"1": 0.3, "2": 0.6})
    assert own == 0.0 and margin == -0.6


def test_contrastive_ties():
    own, rival, margin = detect.contrastive_features("1", {"1": 0.5, "2": 0.5, "3": 0.5})
    assert own == rival == 0.5 and margin == 0.0


def test_feature_dict_empty_retrieval_and_missing_pred():
    r = rec(pred=None, retrieved=(), evidence=(), nli={"1": (0.2, 0.1), "2": (0.6, 0.3)},
            slm={"1": 0.8, "2": 0.9})
    f = detect.feature_dict(r)
    assert f["rerank_top"] == 0.0 and f["cites_evidence"] == 0.0
    assert f["nli_contra"] == 1.0           # 预测缺失时按最高风险处理
    assert f["slm_argmax_agree"] == 0.0


def test_logit_saturation_is_finite():
    assert np.isfinite(detect._logit(1.0)) and np.isfinite(detect._logit(0.0))
    assert detect._logit(0.999) > detect._logit(0.99) > detect._logit(0.9)


def test_parse_response_edge_cases():
    opts = {"1": "a", "2": "b", "3": "c"}
    assert generate.parse_response("Answer: 2\nEvidence: [1], [3]\nExplanation: e", opts)["pred"] == "2"
    assert generate.parse_response("", opts)["pred"] is None
    assert generate.parse_response("Answer: 7", opts)["pred"] is None
    assert generate.parse_response("<think>Answer: 1</think>Answer: 3", opts)["pred"] == "3"
    assert generate.parse_response("Answer: option 2", opts)["evidence"] == []


def test_split_claims_rules():
    item = {"question": "Q?", "options": {"1": "A"}}
    fb = atomic.fallback_claim(item, "1")
    assert fb == "Q? A"
    assert atomic.split_claims("", fb) == [fb]
    assert atomic.split_claims("too short", fb) == [fb]
    c = atomic.split_claims("The UE sends a report [2], and the gNB applies the configuration; done", fb)
    assert c == ["The UE sends a report", "the gNB applies the configuration"]
    assert atomic.fallback_claim(item, None) == "Q?"


def test_view_features_and_sc_agreement():
    r = {"qid": 3, "pred": "2"}
    f = view_features(r, {"closed": {3: "2"}, "bm25": {3: "1"}, "dense": {}})
    assert f["agree_closed"] == 1.0 and f["agree_bm25"] == 0.0 and f["agree_dense"] == 0.0
    assert f["view_vote"] == pytest.approx(2 / 4)       # 最终答案本身 + closed 共 2 票，共 4 票
    assert sc_agreement("2", ["2", "2", "1", None, "2"]) == 0.6
    assert sc_agreement("2", []) == 0.0


# ---------- 高峰期与价格 ----------

def test_is_peak():
    fri = dt.datetime(2026, 10, 30, tzinfo=BJ)
    assert llm.is_peak(fri.replace(hour=9, minute=0))
    assert llm.is_peak(fri.replace(hour=11, minute=59))
    assert not llm.is_peak(fri.replace(hour=12, minute=0))
    assert llm.is_peak(fri.replace(hour=14, minute=0))
    assert not llm.is_peak(fri.replace(hour=18, minute=0))
    assert not llm.is_peak(fri.replace(hour=8, minute=59))
    assert not llm.is_peak(dt.datetime(2026, 10, 31, 10, 0, tzinfo=BJ))   # 周六


def test_cost_cny():
    c = llm.cost_cny("deepseek-flash", 1_000_000, 1_000_000, 0, peak=False)
    assert c == pytest.approx(1.0 + 4.0)
    assert llm.cost_cny("deepseek-flash", 1_000_000, 0, 1_000_000, peak=True) == pytest.approx(0.04)
    assert np.isnan(llm.cost_cny("unknown", 1, 1))


# ---------- API 缓存 ----------

class _Boom:
    """任何网络调用都会让测试失败。"""
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                raise AssertionError("不应发起网络请求")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    monkeypatch.setattr(llm, "resolve", lambda p: tmp_path / Path(p).name)
    cfg = {"model": "deepseek-flash", "base_url": "http://localhost", "api_key_env": "FAKE_KEY",
           "temperature": 0.0, "max_tokens": 10, "concurrency": 1, "price_input_per_m": 0.0,
           "price_output_per_m": 0.0}
    c = llm.LLMClient(cfg)
    c.client = _Boom()
    return c


def test_cache_hit_makes_no_request(client):
    msgs = [{"role": "user", "content": "hi"}]
    k = client._key(msgs, 10, 0.0, 0)
    client.db.execute("INSERT INTO cache VALUES (?, ?)", (k, json.dumps(
        {"text": "Answer: 1", "prompt_tokens": 3, "completion_tokens": 2, "latency": 0.1})))
    out = client.chat(msgs)
    assert out["cached"] and out["text"] == "Answer: 1"
    assert client.usage["cached"] == 1 and client.usage["calls"] == 0


def test_no_api_flag_blocks_cache_miss(client, monkeypatch):
    monkeypatch.setenv("TELERAG_NO_API", "1")
    with pytest.raises(RuntimeError, match="TELERAG_NO_API"):
        client.chat([{"role": "user", "content": "miss"}])


def test_peak_guard_blocks_cache_miss(client, monkeypatch):
    monkeypatch.delenv("TELERAG_NO_API", raising=False)
    monkeypatch.delenv("TELERAG_ALLOW_PEAK", raising=False)
    monkeypatch.setattr(llm, "is_peak", lambda t=None: True)
    with pytest.raises(RuntimeError, match="高峰"):
        client.chat([{"role": "user", "content": "miss"}])


def test_sample_index_changes_cache_key_only_when_set(client):
    msgs = [{"role": "user", "content": "x"}]
    assert client._key(msgs, 10, 0.0, 0) == client._key(msgs, 10, 0.0, 0)
    assert client._key(msgs, 10, 0.7, 1) != client._key(msgs, 10, 0.7, 2)
    assert client._key(msgs, 10, 0.7, 0) != client._key(msgs, 10, 0.7, 1)


# ---------- 数据划分 ----------

def test_split_seed3407_disjoint_and_complete():
    from telerag.config import load_config
    from telerag.data import load_questions

    if not (ROOT / "data/raw/teleqna/kf_full.parquet").exists():
        pytest.skip("缺少 TeleQnA 数据")
    cfg = load_config(overrides=["data.seed=3407", "data.dev_size=500"])
    a = load_questions(cfg)
    b = load_questions(cfg)
    val = {x["qid"] for x in a if x["split"] == "dev"}
    test = {x["qid"] for x in a if x["split"] == "test"}
    assert len(val) == 500 and len(test) == 2500 and not val & test
    assert [x["split"] for x in a] == [x["split"] for x in b]      # 可复现
