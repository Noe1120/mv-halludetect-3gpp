"""配置加载：YAML 文件 + 命令行 --set 覆盖 + .env 环境变量。"""
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _parse_value(s):
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(path=None, overrides=()):
    """overrides 形如 ["retrieval.method=bm25", "llm.model=qwen-plus"]。"""
    _load_dotenv()
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(val)
    return cfg


def resolve(p):
    """把配置里的相对路径解析为项目根目录下的绝对路径。"""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def config_hash(section):
    """对配置的某一部分求哈希，用于缓存文件命名。"""
    s = json.dumps(section, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def sub(cfg, *keys):
    return copy.deepcopy({k: cfg[k] for k in keys})
