"""实验结果的读写。每次运行把逐题记录保存为 jsonl，便于后续分析和复现。"""
import json

from .config import resolve


def run_path(name):
    p = resolve(f"outputs/runs/{name}.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_records(name, records):
    p = run_path(name)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def load_records(name):
    with run_path(name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
