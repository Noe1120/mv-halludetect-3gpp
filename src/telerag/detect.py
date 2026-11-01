"""幻觉检测。

设定：大模型给出的答案如果不正确，就视为一次幻觉（答案没有被检索证据或事实支持）。
检测器只能看到题目、检索到的证据和大模型的输出，看不到标准答案。

本文方法（选项对比式证据核验，Option-Contrastive Evidence Verification）：
不仅核验"所选答案是否被证据支持"，还核验"其他选项是否被支持得更好"。
如果证据更支持另一个选项，或支持度差距很小，判为高风险。
核验器用小模型实现：DeBERTa NLI（0.2B）和 Qwen3-1.7B，都在本地运行。

基线：检索分数、大模型自身给出的置信度（LLM-as-a-Judge，需要调用 API）。
"""
import re

import numpy as np

from .corpus import chunk_text


def _hypothesis(item, opt):
    return f"{item['question']} {item['options'][opt]}"


class NLIVerifier:
    """对每个（证据片段, 题目+选项）计算蕴含概率，取证据片段上的最大值。"""

    def __init__(self, dcfg):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = dcfg["device"]
        self.tok = AutoTokenizer.from_pretrained(dcfg["nli_model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            dcfg["nli_model"], dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent, self.con = labels["entailment"], labels["contradiction"]

    def score(self, item, retrieved, chunks, batch_size=32):
        """返回 {选项: (最大蕴含概率, 最大矛盾概率)}。"""
        opts = list(item["options"])
        pairs = [(chunks[cid]["text"], _hypothesis(item, o)) for o in opts for cid, _ in retrieved]
        probs = []
        for i in range(0, len(pairs), batch_size):
            p, h = zip(*pairs[i:i + batch_size])
            enc = self.tok(list(p), list(h), truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs.append(self.torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        probs = np.concatenate(probs).reshape(len(opts), len(retrieved), -1)
        return {o: (float(probs[j, :, self.ent].max()), float(probs[j, :, self.con].max()))
                for j, o in enumerate(opts)}

    def entail_matrix(self, premises, hypotheses, batch_size=32):
        """返回形状 (len(hypotheses), len(premises)) 的蕴含概率矩阵。用于原子事实核查。"""
        if not premises or not hypotheses:
            return np.zeros((len(hypotheses), len(premises)))
        pairs = [(p, h) for h in hypotheses for p in premises]
        out = []
        for i in range(0, len(pairs), batch_size):
            p, h = zip(*pairs[i:i + batch_size])
            enc = self.tok(list(p), list(h), truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out.append(self.torch.softmax(self.model(**enc).logits.float(), -1)[:, self.ent].cpu().numpy())
        return np.concatenate(out).reshape(len(hypotheses), len(premises))


SLM_PROMPT = """Evidence from 3GPP specifications:
{context}

Question: {question}
Proposed answer: {answer}

Is the proposed answer correct and supported by the evidence? Reply with Yes or No only."""


class SLMVerifier:
    """用小语言模型判断答案是否被证据支持，取下一个 token 为 Yes 的概率（相对 No）。"""

    def __init__(self, dcfg, max_context_tokens=1800):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = dcfg["device"]
        self.tok = AutoTokenizer.from_pretrained(dcfg["slm_model"])
        self.tok.padding_side = "left"
        # 以权重自带的 bf16 直接加载到显卡，避免在内存里做精度转换（笔记本内存紧张时会报页面文件太小）
        self.model = AutoModelForCausalLM.from_pretrained(
            dcfg["slm_model"], dtype="auto" if self.device == "cuda" else torch.float32,
            device_map=self.device).eval()
        self.yes = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
        self.no = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
        self.max_ctx = max_context_tokens

    def _prompt(self, item, opt, context):
        msg = SLM_PROMPT.format(context=context, question=item["question"],
                                answer=f"{opt}. {item['options'][opt]}")
        kwargs = {"enable_thinking": False} if "qwen3" in self.tok.name_or_path.lower() else {}
        return self.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                            add_generation_prompt=True, **kwargs)

    def _context(self, retrieved, chunks):
        parts, used = [], 0
        for i, (cid, _) in enumerate(retrieved):
            text = f"[{i + 1}] {chunk_text(chunks[cid])}"
            n = len(self.tok.encode(text, add_special_tokens=False))
            if used + n > self.max_ctx:
                break
            parts.append(text)
            used += n
        return "\n\n".join(parts)

    def score(self, item, retrieved, chunks):
        """返回 {选项: P(Yes)}。同一题的各个选项放在一个 batch 里。"""
        context = self._context(retrieved, chunks)
        out = {}
        for o in item["options"]:   # 逐个选项算，只对最后一个位置求 logits，8GB 显存也够用
            enc = self.tok(self._prompt(item, o, context), return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                hidden = self.model.model(**enc).last_hidden_state[:, -1, :]
                logits = self.model.lm_head(hidden).float()
            lp = self.torch.log_softmax(logits, -1)[0]
            yes = self.torch.logsumexp(lp[self.yes], -1)
            no = self.torch.logsumexp(lp[self.no], -1)
            out[o] = float(self.torch.sigmoid(yes - no))
        return out


JUDGE_PROMPT = """Evidence from 3GPP specifications:
{context}

Question: {question}
Options:
{options}
Proposed answer: {answer}

How confident are you that the proposed answer is correct, given the evidence and your knowledge? Reply with a single integer from 0 to 100."""


def llm_judge_messages(item, pred, retrieved, chunks):
    from .data import format_options
    from .generate import SYSTEM, format_context

    ans = f"{pred}. {item['options'][pred]}" if pred else "(no valid answer)"
    user = JUDGE_PROMPT.format(context=format_context(retrieved, chunks), question=item["question"],
                               options=format_options(item["options"]), answer=ans)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_judge(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\d+", text)
    return min(100, int(m.group())) / 100 if m else 0.5


def contrastive_features(pred, support):
    """support: {选项: 支持度}。返回所选选项的支持度、最强竞争选项的支持度和差值。"""
    if pred is None or pred not in support:
        return 0.0, max(support.values()), -max(support.values())
    own = support[pred]
    rival = max((v for k, v in support.items() if k != pred), default=0.0)
    return own, rival, own - rival


def feature_dict(rec):
    """从一条记录抽取全部特征。

    - 检索：rerank_top
    - 生成自带：cites_evidence（大模型是否声明引用了证据）
    - 证据核验：nli_*、slm_*（含选项对比特征 rival、margin）
    - 多视角一致性：agree_*、view_agree、view_vote（由 consistency.view_features 预先写入 rec["views"]）
    - 自洽性采样：sc_agree（预先写入 rec["sc_agree"]）
    """
    pred = rec["pred"]
    f = {"rerank_top": rec["retrieved"][0][1] if rec.get("retrieved") else 0.0,
         "cites_evidence": float(bool(rec.get("evidence")))}
    if "nli" in rec:
        ent = {k: v[0] for k, v in rec["nli"].items()}
        f["nli_own"], f["nli_rival"], f["nli_margin"] = contrastive_features(pred, ent)
        f["nli_contra"] = rec["nli"][pred][1] if pred in rec["nli"] else 1.0
    if "slm" in rec:
        f["slm_own"], f["slm_rival"], f["slm_margin"] = contrastive_features(pred, rec["slm"])
        f["slm_argmax_agree"] = float(pred is not None and max(rec["slm"], key=rec["slm"].get) == pred)
        # E1：logit 尺度。Qwen3-0.6B 的 P(Yes) 多数接近 1，概率空间的差值被压扁；logit 尺度能拉开
        lg = {k: _logit(v) for k, v in rec["slm"].items()}
        f["slm_own_logit"], f["slm_rival_logit"], f["slm_margin_logit"] = contrastive_features(pred, lg)
    f.update(rec.get("views", {}))
    if "sc_agree" in rec:
        f["sc_agree"] = rec["sc_agree"]
    # E2：原子事实核查特征；E3：RAGAS 式忠实度（由 atomic.py / faithfulness.py 预先写入）
    f.update(rec.get("atomic", {}))
    return f


def _logit(p, eps=1e-6):
    p = min(max(float(p), eps), 1 - eps)
    return float(np.log(p / (1 - p)))


class RiskModel:
    """逻辑回归，把多个特征融合成一个幻觉风险分数（0 到 1）。"""

    def __init__(self, features):
        self.features = features

    def _X(self, records):
        rows = [feature_dict(r) for r in records]
        missing = [k for k in self.features if k not in rows[0]]
        if missing:
            raise KeyError(f"记录中缺少特征 {missing}")
        return np.array([[row[k] for k in self.features] for row in rows], dtype=np.float32)

    def fit(self, records):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = np.array([r["hallucinated"] for r in records], dtype=int)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced"))
        self.clf.fit(self._X(records), y)
        return self

    def predict(self, records):
        return self.clf.predict_proba(self._X(records))[:, 1]

    def coefficients(self):
        """标准化后的逻辑回归系数，用于论文里分析各特征的贡献（正值表示越大越可能是幻觉）。"""
        return dict(zip(self.features, self.clf[-1].coef_[0].tolist()))
