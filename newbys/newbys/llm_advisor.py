from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests


BAYESIAN_SYSTEM_PROMPT = """你是“贝叶斯短线分析”助手，只分析同花顺热榜前5只股票的T+1短线机会。

你的方法：
1. 把市场周期、指数方向、市场情绪作为动态先验概率。
2. 把近五日K线形态、量能结构、消息驱动强度、题材强度、人气排名作为似然证据。
3. 用贝叶斯思想比较 P(证据|盈利) 与 P(证据|亏损)，再判断 T+1 盈利后验概率是否可信。
4. 不要把热度等同于买点；涨停、高位追涨、爆量分歧、退潮期要降低仓位或回避。
5. 输出必须是可执行的短线决策：轻仓试错、观察、回避，并说明主要证据和风险。

限制：
- 只基于用户提供的当前数据分析，不编造新闻或财务事实。
- 不承诺收益，不给满仓建议。
- 如果模型概率和证据冲突，要明确指出冲突。
- 使用中文，结构清晰。
"""


def load_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def load_llm_config(base_dir: str | os.PathLike[str] | None = None) -> dict[str, str]:
    root = Path(base_dir) if base_dir else Path.cwd()
    env_values = load_env_file(root / ".env")
    return {
        "freechat_url": os.environ.get("freechat_url") or os.environ.get("FREECHAT_URL") or env_values.get("freechat_url", ""),
        "freechat_key": os.environ.get("freechat_key") or os.environ.get("FREECHAT_KEY") or env_values.get("freechat_key", ""),
        "freechat_model": os.environ.get("freechat_model") or os.environ.get("FREECHAT_MODEL") or env_values.get("freechat_model", ""),
    }


def mask_config(config: dict[str, str]) -> dict[str, str]:
    return {key: ("***" if "key" in key.lower() and value else value) for key, value in config.items()}


def build_llm_payload(market: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    compact_items = items[:5]
    user_payload = {
        "market": market,
        "stocks": compact_items,
        "task": "请根据贝叶斯短线分析框架，对热榜前5只股票给出T+1决策、排序、主要证据、主要风险和仓位建议。",
    }
    return {
        "messages": [
            {"role": "system", "content": BAYESIAN_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "temperature": 0.2,
    }


class LlmAdvisor:
    def __init__(
        self,
        config: dict[str, str] | None = None,
        session: requests.Session | None = None,
        timeout: int = 30,
    ):
        self.config = config or load_llm_config(Path(__file__).resolve().parents[1])
        self.session = session or requests.Session()
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.config.get("freechat_url") and self.config.get("freechat_key") and self.config.get("freechat_model"))

    def analyze(self, market: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.is_configured():
            return {
                "enabled": False,
                "content": "",
                "error": "LLM is not configured. Set freechat_url, freechat_key and freechat_model in .env.",
                "config": mask_config(self.config),
            }
        url = self._chat_url()
        payload = build_llm_payload(market, items)
        payload["model"] = self.config["freechat_model"]
        try:
            response = self.session.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config['freechat_key']}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "enabled": True,
                "content": content,
                "error": "",
                "model": self.config["freechat_model"],
            }
        except Exception as exc:
            return {
                "enabled": True,
                "content": "",
                "error": str(exc),
                "model": self.config["freechat_model"],
            }

    def _chat_url(self) -> str:
        base = self.config["freechat_url"].rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"
