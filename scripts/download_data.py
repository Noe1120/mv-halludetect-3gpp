"""下载实验用到的公开数据，到 data/raw/。数据只从原始来源下载、不在本仓库里重新分发。

每个文件固定了来源版本（commit/revision）和 SHA-256，下载后逐一校验，保证和论文实验用的是同一份数据。

| 数据 | 来源 | 许可 |
|---|---|---|
| TeleQnA（10,000 题全集） | Hugging Face `ketchup123/TeleQnA_full` | 原始数据集为 MIT |
| 3GPP Release 18 语料 | Hugging Face `netop/3GPP-R18`（Telco-RAG 项目） | MIT |
| Telco-RAG 术语表与系列描述 | GitHub `netop-team/Telco-RAG` | MIT |
| RAGTruth | GitHub `ParticleMedia/RAGTruth` | MIT |

用法：
  python scripts/download_data.py            # 全部下载
  python scripts/download_data.py --only teleqna corpus
  set HF_ENDPOINT=https://hf-mirror.com       # 国内网络可以先设置这个镜像
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
GH = "https://raw.githubusercontent.com"
TELCO_RAG_REV = "main"      # 上游没有打 tag；表中 SHA-256 对应 2026-10-30 下载的版本

FILES = {
    "teleqna": [
        (f"{HF}/datasets/ketchup123/TeleQnA_full/resolve/064b77d19a68/data/train-00000-of-00001.parquet",
         "teleqna/kf_full.parquet", "b1fb1ca2968a31817ff0365bf9f7554b8e8373ec698826d9ddfa7f109e9b9487"),
    ],
    "corpus": [
        (f"{HF}/datasets/netop/3GPP-R18/resolve/b8d598e50cad/Documents.db",
         "Documents.db", "8ac27d76aa9018e53cae6e29f64b6ab43ad312e4c575abddd6cbcb8db2528d17"),
    ],
    "telco_rag": [
        (f"{GH}/netop-team/Telco-RAG/{TELCO_RAG_REV}/Telco-RAG_api/src/resources/3GPP_vocabulary.docx",
         "telco_rag/3GPP_vocabulary.docx", "94025058fccd7ab2f91c4f0c46471a85b3fbf3d70f74f33614d4dfa4c6d96f05"),
        (f"{GH}/netop-team/Telco-RAG/{TELCO_RAG_REV}/Telco-RAG_api/src/resources/series_description.json",
         "telco_rag/series_description.json", "b4a494d1dcc0d4e0fdb316272c54aca95ba04c64adc0e1faae15a8b53ca72fe3"),
    ],
    "ragtruth": [
        (f"{GH}/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
         "ragtruth/response.jsonl", "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073"),
        (f"{GH}/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
         "ragtruth/source_info.jsonl", "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b"),
    ],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dst, expected):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (expected is None or sha256(dst) == expected):
        print(f"[skip] {dst.relative_to(ROOT)} 已存在且校验通过")
        return True
    print(f"[get ] {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    got = sha256(tmp)
    if expected and got != expected:
        tmp.unlink()
        print(f"[FAIL] {dst.name}：SHA-256 不一致\n       期望 {expected}\n       实际 {got}")
        return False
    tmp.replace(dst)
    print(f"[ ok ] {dst.relative_to(ROOT)}  sha256={got[:12]}…")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=list(FILES), help="只下载其中几项")
    args = p.parse_args()
    ok = True
    for name in args.only or FILES:
        for url, rel, digest in FILES[name]:
            ok &= fetch(url, RAW / rel, digest)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
