from __future__ import annotations

import random
import re
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import requests

from .models import StockSnapshot


class DataSource(Protocol):
    source_name: str

    def get_stocks(self, top_n: int = 50) -> list[StockSnapshot]:
        ...


class StaticDataSource:
    source_name = "static"

    def __init__(self, stocks: list[StockSnapshot]):
        self.stocks = stocks

    def get_stocks(self, top_n: int = 50) -> list[StockSnapshot]:
        return self.stocks[:top_n]


class MockDataSource:
    source_name = "mock"

    def get_stocks(self, top_n: int = 50) -> list[StockSnapshot]:
        base = [
            ("300033", "同花顺", 118.0, 3.8, 13.0, 2.2, 22, ["AI", "金融科技"]),
            ("300750", "宁德时代", 198.0, 1.2, 2.8, 1.3, 38, ["新能源"]),
            ("002230", "科大讯飞", 48.0, 4.2, 7.2, 1.9, 12, ["AI", "教育"]),
            ("600519", "贵州茅台", 1420.0, -0.5, 0.4, 0.8, 80, []),
            ("688981", "中芯国际", 72.0, 5.1, 11.5, 2.6, 6, ["芯片", "国产替代"]),
        ]
        rng = random.Random(datetime.now().strftime("%Y%m%d%H"))
        stocks = []
        for code, name, price, change, turnover, vr, rank, tags in base:
            noise = rng.uniform(-0.4, 0.4)
            stocks.append(
                StockSnapshot(
                    code=code,
                    name=name,
                    price=round(price * (1 + noise * 0.01), 2),
                    change_pct=round(change + noise, 2),
                    turnover_rate=round(max(0.1, turnover + noise), 2),
                    volume_ratio=round(max(0.1, vr + noise * 0.2), 2),
                    amount=round((5 + rank % 20) * 100_000_000, 0),
                    hot_rank=rank,
                    concept_tags=list(tags),
                )
            )
        stocks.sort(key=lambda s: s.hot_rank if s.hot_rank > 0 else 999)
        return stocks[:top_n]


@dataclass(frozen=True)
class HotRankEntry:
    rank: int
    code: str
    name: str
    heat_score: float = 0.0
    concept_tags: list[str] | None = None


class TencentQuoteDataSource:
    source_name = "tencent_quotes"
    TENCENT_URL = "http://qt.gtimg.cn/q={}"
    DEFAULT_POOL = [
        "300033", "688981", "002230", "300750", "002594", "601127", "300059", "600030",
        "600519", "000858", "002475", "603986", "600570", "000001", "600036",
    ]

    def __init__(self, codes: list[str] | None = None, timeout: int = 8):
        self.codes = codes or self.DEFAULT_POOL
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_stocks(self, top_n: int = 50) -> list[StockSnapshot]:
        return self.get_stocks_for_codes(self.codes)[:top_n]

    def get_stocks_for_codes(self, codes: list[str]) -> list[StockSnapshot]:
        rows = self._query(codes)
        stocks = []
        for idx, fields in enumerate(rows, 1):
            stock = self._parse(fields, idx)
            if stock:
                stocks.append(stock)
        stocks.sort(key=lambda s: (s.hot_rank if s.hot_rank > 0 else 999, -s.turnover_rate))
        return stocks

    def _query(self, codes: list[str]) -> list[list[str]]:
        if not codes:
            return []
        qstr = ",".join(("sh" if code.startswith("6") else "sz") + code for code in codes)
        response = self.session.get(self.TENCENT_URL.format(qstr), timeout=self.timeout)
        response.encoding = "gbk"
        rows = []
        for line in response.text.splitlines():
            if "=" not in line:
                continue
            fields = line.split("=", 1)[1].strip('";').split("~")
            if len(fields) >= 46 and fields[2]:
                rows.append(fields)
        return rows

    @staticmethod
    def _parse(fields: list[str], rank: int) -> StockSnapshot | None:
        try:
            price = float(fields[3] or 0)
            pre_close = float(fields[4] or 0)
            if price <= 0 or pre_close <= 0:
                return None
            change_pct = round((price - pre_close) / pre_close * 100, 2)
            amount = float(fields[37] or 0) * 10_000 if len(fields) > 37 else 0.0
            return StockSnapshot(
                code=fields[2],
                name=fields[1],
                price=price,
                change_pct=change_pct,
                turnover_rate=float(fields[38] or 0),
                volume_ratio=1.0,
                amount=amount,
                hot_rank=rank,
                concept_tags=[],
            )
        except (ValueError, IndexError):
            return None


class TickerLabHotRankDataSource:
    source_name = "tickerlab_ths_hot_rank"
    HOT_RANK_URL = "https://tickerlab.org/v1/ranking/hot-stock"

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        quote_source: TencentQuoteDataSource | None = None,
        timeout: int = 10,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("TICKERLAB_API_KEY", "")
        self.session = session or requests.Session()
        self.quote_source = quote_source or TencentQuoteDataSource(timeout=timeout)
        self.timeout = timeout

    def fetch_hot_rank_entries(self, top_n: int = 50) -> list[HotRankEntry]:
        if not self.api_key:
            raise RuntimeError("TICKERLAB_API_KEY is required for real THS hot rank")
        response = self.session.get(
            self.HOT_RANK_URL,
            params={"limit": top_n},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-API-Key": self.api_key,
                "User-Agent": "Mozilla/5.0",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        entries = []
        used_ranks = set()
        for idx, item in enumerate(self._extract_items(response.json()), 1):
            code = self._clean_code(self._first_value(item, ("code", "stock_code", "symbol", "ticker")))
            name = str(self._first_value(item, ("name", "stock_name", "short_name"), "")).strip()
            if not code or not name:
                continue
            rank = self._to_int(self._first_value(item, ("rank", "ranking", "position"), idx), idx)
            if rank in used_ranks:
                rank = idx
            used_ranks.add(rank)
            heat = self._to_float(self._first_value(item, ("heat", "hot", "hot_value", "score", "heat_score"), 0.0))
            entries.append(HotRankEntry(rank=rank, code=code, name=name, heat_score=heat, concept_tags=[]))
        entries.sort(key=lambda entry: entry.rank)
        return entries[:top_n]

    def get_stocks(self, top_n: int = 50) -> list[StockSnapshot]:
        entries = self.fetch_hot_rank_entries(top_n=top_n)
        codes = [entry.code for entry in entries]
        quotes = {stock.code: stock for stock in self.quote_source.get_stocks_for_codes(codes)}
        snapshots = []
        for entry in entries:
            quote = quotes.get(entry.code)
            if quote:
                snapshots.append(
                    StockSnapshot(
                        code=quote.code,
                        name=quote.name or entry.name,
                        price=quote.price,
                        change_pct=quote.change_pct,
                        turnover_rate=quote.turnover_rate,
                        volume_ratio=quote.volume_ratio,
                        amount=quote.amount,
                        hot_rank=entry.rank,
                        concept_tags=entry.concept_tags or quote.concept_tags,
                    )
                )
            else:
                snapshots.append(
                    StockSnapshot(
                        code=entry.code,
                        name=entry.name,
                        price=0.0,
                        change_pct=0.0,
                        turnover_rate=0.0,
                        volume_ratio=1.0,
                        amount=0.0,
                        hot_rank=entry.rank,
                        concept_tags=entry.concept_tags or [],
                    )
                )
        return snapshots[:top_n]

    @staticmethod
    def _extract_items(payload: object) -> list[dict]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        data = payload.get("data", payload)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("items", "list", "stocks", "rows", "result"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _first_value(item: dict, keys: tuple[str, ...], default=None):
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
        return default

    @staticmethod
    def _clean_code(raw_code: object) -> str:
        code = str(raw_code or "").strip().upper()
        if "." in code:
            left, right = code.split(".", 1)
            code = right if left in {"SH", "SZ", "BJ"} else left
        return re.sub(r"\D", "", code)

    @staticmethod
    def _to_int(value: object, default: int) -> int:
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float(value: object) -> float:
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return 0.0


class ThsAppHotRankDataSource:
    source_name = "ths_app_hot_rank"
    HOT_RANK_URL = "https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock"

    def __init__(
        self,
        session: requests.Session | None = None,
        quote_source: TencentQuoteDataSource | None = None,
        timeout: int = 10,
    ):
        self.session = session or requests.Session()
        self.quote_source = quote_source or TencentQuoteDataSource(timeout=timeout)
        self.timeout = timeout

    def fetch_hot_rank_entries(self, top_n: int = 50) -> list[HotRankEntry]:
        response = self.session.get(
            self.HOT_RANK_URL,
            params={"stock_type": "a", "type": "hour", "list_type": "normal"},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.10jqka.com.cn/",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("status_code") not in (None, 0):
            raise RuntimeError(f"THS App hot rank error: {payload.get('status_msg')}")
        stock_list = payload.get("data", {}).get("stock_list", []) if isinstance(payload, dict) else []
        entries = []
        for idx, item in enumerate(stock_list[:top_n], 1):
            if not isinstance(item, dict):
                continue
            code = TickerLabHotRankDataSource._clean_code(item.get("code"))
            name = str(item.get("name", "")).strip()
            if not code or not name:
                continue
            rank = TickerLabHotRankDataSource._to_int(item.get("order"), idx)
            heat = TickerLabHotRankDataSource._to_float(item.get("rate"))
            tag = item.get("tag", {})
            entries.append(
                HotRankEntry(
                    rank=rank,
                    code=code,
                    name=name,
                    heat_score=heat,
                    concept_tags=self._extract_concept_tags(tag),
                )
            )
        entries.sort(key=lambda entry: entry.rank)
        return entries[:top_n]

    def get_stocks(self, top_n: int = 50) -> list[StockSnapshot]:
        entries = self.fetch_hot_rank_entries(top_n=top_n)
        quotes = {stock.code: stock for stock in self.quote_source.get_stocks_for_codes([entry.code for entry in entries])}
        snapshots = []
        for entry in entries:
            quote = quotes.get(entry.code)
            if quote:
                snapshots.append(
                    StockSnapshot(
                        code=quote.code,
                        name=quote.name or entry.name,
                        price=quote.price,
                        change_pct=quote.change_pct,
                        turnover_rate=quote.turnover_rate,
                        volume_ratio=quote.volume_ratio,
                        amount=quote.amount,
                        hot_rank=entry.rank,
                        concept_tags=entry.concept_tags or quote.concept_tags,
                    )
                )
            else:
                snapshots.append(
                    StockSnapshot(
                        code=entry.code,
                        name=entry.name,
                        price=0.0,
                        change_pct=0.0,
                        turnover_rate=0.0,
                        volume_ratio=1.0,
                        amount=0.0,
                        hot_rank=entry.rank,
                        concept_tags=entry.concept_tags or [],
                    )
                )
        return snapshots[:top_n]

    @staticmethod
    def _extract_concept_tags(tag: object) -> list[str]:
        if not isinstance(tag, dict):
            return []
        concepts = tag.get("concept_tag", [])
        if isinstance(concepts, str):
            return [item.strip() for item in re.split(r"[,;，；]", concepts) if item.strip()]
        if isinstance(concepts, list):
            return [str(item).strip() for item in concepts if str(item).strip()]
        return []


def create_data_source(force_mock: bool = False) -> DataSource:
    if force_mock:
        return MockDataSource()
    try:
        source = ThsAppHotRankDataSource()
        if source.get_stocks(3):
            return source
    except Exception:
        pass
    if os.environ.get("TICKERLAB_API_KEY"):
        try:
            source = TickerLabHotRankDataSource()
            if source.get_stocks(3):
                return source
        except Exception:
            pass
    try:
        source = TencentQuoteDataSource()
        if source.get_stocks(3):
            return source
    except Exception:
        pass
    return MockDataSource()
