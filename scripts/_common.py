"""脚本公用的命令行参数。"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/claude-home/cache/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None, help="配置文件，默认 configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="覆盖配置项，例如 --set retrieval.method=bm25 llm.model=qwen-plus")
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    return p


def load(args):
    from telerag.config import load_config
    from telerag.data import load_questions, select

    cfg = load_config(args.config, args.set)
    items = select(load_questions(cfg), args.split, args.limit)
    return cfg, items
