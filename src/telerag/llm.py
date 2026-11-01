"""OpenAI 兼容接口的大模型调用：带磁盘缓存（同一请求不重复花钱）、并发和 token 计数。"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .config import resolve

# DeepSeek 官方价格（元 / 百万 token，空闲时段），2026-10-30 抓取自 api-docs.deepseek.com/zh-cn/quick_start/pricing。
# 高峰时段（北京时间工作日 9:00–12:00、14:00–18:00）价格翻倍。
OFFICIAL_PRICES_CNY = {
    "deepseek-flash": {"hit": 0.02, "miss": 1.0, "out": 4.0},
    "deepseek-v4-pro": {"hit": 0.15, "miss": 4.5, "out": 13.5},
}


def is_peak(t=None):
    """北京时间工作日 9:00–12:00、14:00–18:00 为高峰（不处理法定节假日，节假日按高峰保守处理）。"""
    import datetime as dt

    now = t or dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return 9 * 60 <= hm < 12 * 60 or 14 * 60 <= hm < 18 * 60


def cost_cny(model, prompt_tokens, completion_tokens, cache_hit_tokens=0, peak=False):
    p = OFFICIAL_PRICES_CNY.get(model)
    if p is None:
        return float("nan")
    m = 2.0 if peak else 1.0
    return m * ((prompt_tokens - cache_hit_tokens) * p["miss"] + cache_hit_tokens * p["hit"]
                + completion_tokens * p["out"]) / 1e6


class LLMClient:
    def __init__(self, lcfg):
        from openai import OpenAI

        self.cfg = lcfg
        key = os.environ.get(lcfg["api_key_env"])
        if not key:
            raise RuntimeError(
                f"未找到环境变量 {lcfg['api_key_env']}。请在项目根目录的 .env 中写入 "
                f"{lcfg['api_key_env']}=你的密钥（参考 .env.example）。")
        self.client = OpenAI(api_key=key, base_url=lcfg["base_url"], timeout=120)
        path = resolve("data/processed/llm_cache.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0}

    def _key(self, messages, max_tokens, temperature, sample):
        payload = [self.cfg["model"], self.cfg["base_url"], temperature, max_tokens, messages,
                   self.cfg.get("extra_body")]
        if sample:                      # 自洽性采样：同一提示词的第 i 次采样单独缓存
            payload.append(sample)
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()

    def chat(self, messages, max_tokens=None, temperature=None, sample=0):
        """返回 {'text', 'prompt_tokens', 'completion_tokens', 'latency', 'cached'}。

        temperature 默认用配置里的值；sample > 0 表示同一提示词的第几次随机采样。
        """
        max_tokens = max_tokens or self.cfg["max_tokens"]
        temperature = self.cfg["temperature"] if temperature is None else temperature
        k = self._key(messages, max_tokens, temperature, sample)
        with self.lock:
            row = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            out = json.loads(row[0])
            out["cached"] = True
            with self.lock:
                self.usage["cached"] += 1
                self.usage["prompt_tokens"] += out["prompt_tokens"]
                self.usage["completion_tokens"] += out["completion_tokens"]
            return out

        if os.environ.get("TELERAG_NO_API") == "1":
            # 只允许读缓存，防止意外产生 API 费用
            raise RuntimeError("TELERAG_NO_API=1：该请求不在缓存中，已阻止付费调用")
        if is_peak() and os.environ.get("TELERAG_ALLOW_PEAK") != "1":
            # 高峰时段价格翻倍；需要时设置 TELERAG_ALLOW_PEAK=1 放行
            raise RuntimeError("当前为 DeepSeek 高峰时段（北京时间工作日 9–12、14–18 点），已阻止新的付费调用")
        for attempt in range(6):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.cfg["model"], messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                    extra_body=self.cfg.get("extra_body"))
                break
            except Exception as e:  # 限流或网络抖动时指数退避重试
                if attempt == 5:
                    raise
                wait = 2 ** attempt
                print(f"[llm] {type(e).__name__}: {str(e)[:120]}，{wait}s 后重试")
                time.sleep(wait)
        u = resp.usage
        out = {
            "text": resp.choices[0].message.content or "",
            # DeepSeek 对重复的提示词前缀按缓存命中价计费（约为正常价的 3%）
            "cache_hit_tokens": getattr(u, "prompt_cache_hit_tokens", 0) or 0 if u else 0,
            "prompt_tokens": u.prompt_tokens if u else 0,
            "completion_tokens": u.completion_tokens if u else 0,
            "latency": time.time() - t0,
        }
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, json.dumps(out, ensure_ascii=False)))
            self.db.commit()
            self.usage["calls"] += 1
            self.usage["cache_hit_tokens"] += out["cache_hit_tokens"]
            self.usage["prompt_tokens"] += out["prompt_tokens"]
            self.usage["completion_tokens"] += out["completion_tokens"]
        out["cached"] = False
        out["peak"] = is_peak()
        with self.lock:
            self.usage["cost_cny"] = self.usage.get("cost_cny", 0.0) + cost_cny(
                self.cfg["model"], out["prompt_tokens"], out["completion_tokens"], out["cache_hit_tokens"], out["peak"])
        return out

    def map(self, fn, items):
        """并发地对 items 执行 fn（fn 内部调用 self.chat），保持顺序。"""
        with ThreadPoolExecutor(self.cfg["concurrency"]) as ex:
            return list(ex.map(fn, items))

    def cost(self, prompt_tokens, completion_tokens, cache_hit_tokens=0):
        hit_price = self.cfg.get("price_cache_hit_per_m", self.cfg["price_input_per_m"])
        return ((prompt_tokens - cache_hit_tokens) * self.cfg["price_input_per_m"]
                + cache_hit_tokens * hit_price
                + completion_tokens * self.cfg["price_output_per_m"]) / 1e6
