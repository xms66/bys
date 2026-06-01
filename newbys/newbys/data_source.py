from __future__ import annotations

import random
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
        rows = self._query(self.codes)
        stocks = []
        for idx, fields in enumerate(rows, 1):
            stock = self._parse(fields, idx)
            if stock:
                stocks.append(stock)
        stocks.sort(key=lambda s: (s.hot_rank if s.hot_rank > 0 else 999, -s.turnover_rate))
        return stocks[:top_n]

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


def create_data_source(force_mock: bool = False) -> DataSource:
    if force_mock:
        return MockDataSource()
    try:
        source = TencentQuoteDataSource()
        if source.get_stocks(3):
            return source
    except Exception:
        pass
    return MockDataSource()

