"""检索：BM25、稠密向量、RRF 混合，以及 cross-encoder 重排。"""
import numpy as np

from .config import config_hash, resolve
from .corpus import chunk_text


def _fp16_kwargs(device):
    """GPU 上直接以 fp16 加载，内存峰值减半（笔记本内存紧张时有用）。"""
    import torch

    return {"dtype": torch.float16} if device == "cuda" else {}


def rrf_fuse(rankings, k=60):
    """Reciprocal Rank Fusion。rankings: 多个按相关度排好序的 cid 列表。"""
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class BM25Index:
    def __init__(self, chunks, cache_dir):
        import bm25s
        import Stemmer

        self.stemmer = Stemmer.Stemmer("english")
        cache_dir.mkdir(parents=True, exist_ok=True)
        if (cache_dir / "params.index.json").exists():
            self.retriever = bm25s.BM25.load(str(cache_dir))
        else:
            tokens = bm25s.tokenize([chunk_text(c) for c in chunks],
                                    stopwords="en", stemmer=self.stemmer, show_progress=False)
            self.retriever = bm25s.BM25()
            self.retriever.index(tokens, show_progress=False)
            self.retriever.save(str(cache_dir))

    def search(self, queries, k):
        import bm25s

        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False)
        ids, _ = self.retriever.retrieve(tokens, k=k, show_progress=False)
        return [list(map(int, row)) for row in ids]


class DenseIndex:
    def __init__(self, chunks, rcfg, cache_file):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(rcfg["embed_model"], device=rcfg["device"],
                                         model_kwargs=_fp16_kwargs(rcfg["device"]))
        # 命令行 --set retrieval.query_prefix= 会被 YAML 解析成 None（bge-m3 不用查询前缀）
        self.prefix = rcfg.get("query_prefix") or ""
        if self.model.max_seq_length > 512:
            # bge-m3 默认最长 8192 token；块本身不超过 300 词，截到 512 不影响结果，只是控制显存
            self.model.max_seq_length = 512
        if cache_file.exists():
            self.emb = np.load(cache_file)
        else:
            # 分段编码并直接写入 fp16 的内存映射文件，避免在内存里一次持有全部 embedding
            # （E6 一次性编码时因内存不足被系统中止）。分段不改变编码结果。
            from tqdm import tqdm

            part = cache_file.with_suffix(".partial.npy")
            dim = self.model.get_sentence_embedding_dimension()
            mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(len(chunks), dim))
            bs = 16 if dim > 768 else 64
            for s in tqdm(range(0, len(chunks), 4096), desc="稠密编码"):
                mm[s:s + 4096] = self.model.encode([chunk_text(c) for c in chunks[s:s + 4096]], batch_size=bs,
                                                   normalize_embeddings=True, show_progress_bar=False,
                                                   convert_to_numpy=True)
            mm.flush()
            del mm
            part.replace(cache_file)
            self.emb = np.load(cache_file)
        import torch
        self.emb_t = torch.from_numpy(self.emb).to(rcfg["device"])

    def search(self, queries, k):
        import torch

        q = self.model.encode([self.prefix + x for x in queries], normalize_embeddings=True,
                              convert_to_tensor=True).to(self.emb_t.dtype)
        with torch.no_grad():
            top = torch.topk(q @ self.emb_t.T, k=k, dim=1).indices.cpu().tolist()
        return top


class Reranker:
    def __init__(self, rcfg):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(rcfg["rerank_model"], device=rcfg["device"], max_length=512,
                                  model_kwargs=_fp16_kwargs(rcfg["device"]))

    def rerank(self, query, cids, chunks):
        pairs = [(query, chunk_text(chunks[c])) for c in cids]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        return [cids[i] for i in order], [float(scores[i]) for i in order]


class Retriever:
    def __init__(self, cfg, chunks, chunk_file):
        self.cfg = cfg["retrieval"]
        self.chunks = chunks
        tag = chunk_file.stem
        cache = resolve("data/processed/index")
        cache.mkdir(parents=True, exist_ok=True)
        method = self.cfg["method"]
        self.bm25 = BM25Index(chunks, cache / f"bm25_{tag}") if method in ("bm25", "hybrid") else None
        emb_tag = config_hash({"m": self.cfg["embed_model"], "c": tag})
        self.dense = (DenseIndex(chunks, self.cfg, cache / f"dense_{emb_tag}.npy")
                      if method in ("dense", "hybrid") else None)
        self.reranker = Reranker(self.cfg) if self.cfg["rerank"] else None

    def search(self, queries):
        """返回每个查询的 [(cid, score), ...]，长度为 top_k。"""
        n = self.cfg["candidates"]
        runs = []
        if self.bm25:
            runs.append(self.bm25.search(queries, n))
        if self.dense:
            runs.append(self.dense.search(queries, n))
        results = []
        for qi, query in enumerate(queries):
            if len(runs) == 1:
                cand = runs[0][qi]
            else:
                cand = rrf_fuse([r[qi] for r in runs], self.cfg["rrf_k"])[:n]
            if self.reranker:
                cand, scores = self.reranker.rerank(query, cand, self.chunks)
            else:
                scores = [1.0 / (i + 1) for i in range(len(cand))]
            k = self.cfg["top_k"]
            results.append(list(zip(cand[:k], scores[:k])))
        return results


def build_query(item, mode):
    if mode == "question+options":
        return item["question"] + " " + " ".join(item["options"].values())
    return item["question"]
