"""实验目录与统一日志。

- 每个实验一个独立目录 outputs/exp/<日期时间>_<实验名>/；目录已存在时自动加序号，不覆盖旧产物。
- 所有运行追加写入同一个 outputs/log.jsonl，不覆盖旧日志。
"""
import json
import sys
import time

import numpy as np
import yaml

from .config import config_hash, resolve


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


class Experiment:
    def __init__(self, name, cfg, extra=None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = resolve("outputs/exp")
        base.mkdir(parents=True, exist_ok=True)
        d = base / f"{stamp}_{name}"
        i = 1
        while d.exists():
            d = base / f"{stamp}_{name}_{i}"
            i += 1
        d.mkdir()
        self.dir, self.name, self.t0 = d, name, time.time()
        self.cfg = cfg
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self.meta = {"exp": name, "dir": str(d.relative_to(resolve("."))), "config_hash": config_hash(cfg),
                     "argv": sys.argv, "start": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}

    def save_json(self, fname, obj):
        p = self.dir / fname
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
        return p

    def save_records(self, fname, records):
        p = self.dir / fname
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=_jsonable) + "\n")
        return p

    def save_array(self, fname, arr, columns=None):
        np.save(self.dir / fname, np.asarray(arr))
        if columns is not None:
            (self.dir / (fname + ".columns.json")).write_text(json.dumps(list(columns)), encoding="utf-8")

    def finish(self, summary=None, cost_cny=0.0):
        entry = {**self.meta, "seconds": round(time.time() - self.t0, 1), "cost_cny": round(cost_cny, 4),
                 "summary": summary or {}}
        with resolve("outputs/log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=_jsonable) + "\n")
        return entry
