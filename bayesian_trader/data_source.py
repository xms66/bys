#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时数据源 - 调用腾讯/新浪行情API获取A股实时数据，按热度（换手率/成交额/涨幅）排序
"""

import re
import json
import time
import os
import requests
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


# ============================================================
# 数据模型
# ============================================================

@dataclass
class RealTimeStock:
    """实时股票数据"""
    code: str           # 股票代码（如 600519）
    name: str           # 股票名称
    price: float        # 最新价
    change_pct: float   # 涨跌幅(%)
    change_amt: float   # 涨跌额
    volume: float       # 成交量(手)
    amount: float       # 成交额(元)
    high: float         # 最高
    low: float          # 最低
    open_px: float      # 今开
    pre_close: float    # 昨收
    turnover_rate: float  # 换手率(%)
    volume_ratio: float # 量比
    pe: float           # 动态市盈率
    total_mv: float     # 总市值
    market_cap: float   # 流通市值

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConceptBoard:
    """概念板块"""
    code: str
    name: str
    change_pct: float
    up_count: int
    total_count: int
    leader_stock: str = ""


@dataclass
class HotRankEntry:
    """外部热度榜条目"""
    rank: int
    code: str
    name: str
    heat_score: float = 0.0
    concept_tags: List[str] = None
    popularity_tag: str = ""

    def __post_init__(self):
        if self.concept_tags is None:
            self.concept_tags = []


class ThsAppHotRankDataSource:
    """截图口径的同花顺 App 热榜：热股 / 大家都在看 / 1小时。"""

    HOT_RANK_URL = "https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock"
    source_name = "ths_app_hot_rank"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        timeout: int = 10,
        quote_source: Optional["TencentDataSource"] = None,
    ):
        self.session = session or requests.Session()
        self.timeout = timeout
        self.quote_source = quote_source or TencentDataSource(timeout=timeout)

    @staticmethod
    def _clean_code(raw_code: object) -> str:
        return re.sub(r"\D", "", str(raw_code or ""))

    @staticmethod
    def _extract_concept_tags(tag: object) -> List[str]:
        if not isinstance(tag, dict):
            return []
        concepts = tag.get("concept_tag", [])
        if isinstance(concepts, str):
            return [x.strip() for x in re.split(r"[;,，；]", concepts) if x.strip()]
        if isinstance(concepts, list):
            return [str(x).strip() for x in concepts if str(x).strip()]
        return []

    def fetch_hot_rank_entries(self, top_n: int = 50) -> List[HotRankEntry]:
        response = self.session.get(
            self.HOT_RANK_URL,
            params={"stock_type": "a", "type": "hour", "list_type": "normal"},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Referer": "https://www.10jqka.com.cn/",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("status_code") not in (None, 0):
            raise RuntimeError(f"同花顺热榜接口异常: {payload.get('status_msg')}")

        stock_list = payload.get("data", {}).get("stock_list", []) if isinstance(payload, dict) else []
        entries: List[HotRankEntry] = []
        for idx, item in enumerate(stock_list[:top_n], 1):
            if not isinstance(item, dict):
                continue
            code = self._clean_code(item.get("code"))
            name = str(item.get("name", "")).strip()
            if not code or not name:
                continue
            tag = item.get("tag", {})
            entries.append(HotRankEntry(
                rank=idx,
                code=code,
                name=name,
                heat_score=max(0.0, float(top_n - idx + 1)),
                concept_tags=self._extract_concept_tags(tag),
                popularity_tag=str(tag.get("popularity_tag", "")).strip() if isinstance(tag, dict) else "",
            ))
        return entries

    def get_stock_list(
        self,
        top_n: int = 50,
        sort_by: str = "hot_rank",
        custom_pool: Optional[List[str]] = None,
    ) -> Tuple[List[RealTimeStock], str]:
        entries = self.fetch_hot_rank_entries(top_n=top_n if custom_pool is None else max(top_n, 50))
        if custom_pool:
            wanted = {self._clean_code(code) for code in custom_pool}
            entries = [entry for entry in entries if entry.code in wanted]

        raw_data = self.quote_source._query_tencent([entry.code for entry in entries])
        stocks: List[RealTimeStock] = []
        for entry in entries:
            fields = raw_data.get(entry.code)
            stock = self.quote_source.parse_tencent_stock(entry.code, fields) if fields else None
            if stock is None:
                stock = RealTimeStock(
                    code=entry.code,
                    name=entry.name,
                    price=0.0,
                    change_pct=0.0,
                    change_amt=0.0,
                    volume=0.0,
                    amount=0.0,
                    high=0.0,
                    low=0.0,
                    open_px=0.0,
                    pre_close=0.0,
                    turnover_rate=0.0,
                    volume_ratio=1.0,
                    pe=0.0,
                    total_mv=0.0,
                    market_cap=0.0,
                )
            setattr(stock, "hot_rank", entry.rank)
            setattr(stock, "hot_score", entry.heat_score)
            setattr(stock, "concept_tags", entry.concept_tags)
            setattr(stock, "popularity_tag", entry.popularity_tag)
            stocks.append(stock)

        if sort_by != "hot_rank":
            stocks = TencentDataSource._sort_stocks(stocks, sort_by)
        return stocks[:top_n], datetime.now().strftime("%H:%M:%S")

    def get_hot_rank(self, top_n: int = 50) -> List[RealTimeStock]:
        return self.get_stock_list(top_n=top_n, sort_by="hot_rank")[0]

    def get_market_overview(self) -> dict:
        stocks, _ = self.get_stock_list(top_n=50)
        up = sum(1 for s in stocks if s.change_pct > 0)
        down = sum(1 for s in stocks if s.change_pct < 0)
        flat = len(stocks) - up - down
        return {
            "total": len(stocks),
            "up": up,
            "down": down,
            "flat": flat,
            "avg_turnover": round(sum(s.turnover_rate for s in stocks) / max(len(stocks), 1), 2),
        }

    def get_concept_boards(self) -> List[ConceptBoard]:
        return TencentDataSource().get_concept_boards()


class ThsHotRankDataSource:
    """
    真实同花顺热股榜数据源。

    第一版使用 TickerLab 的同花顺热门股票排行榜兼容接口。没有 API Key 时
    明确报错，避免把模拟数据伪装成真实同花顺热度前50。
    """

    HOT_RANK_URL = "https://tickerlab.org/v1/ranking/hot-stock"
    source_name = "ths_hot_rank"

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: int = 10,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("TICKERLAB_API_KEY", "")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.quote_source = TencentDataSource(timeout=timeout)

    @staticmethod
    def _clean_code(raw_code: object) -> str:
        code = str(raw_code or "").strip().upper()
        if "." in code:
            left, right = code.split(".", 1)
            code = right if left in {"SH", "SZ", "BJ"} else left
        return re.sub(r"\D", "", code)

    @staticmethod
    def _extract_items(payload: object) -> List[dict]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if not isinstance(payload, dict):
            return []

        data = payload.get("data", payload)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("items", "list", "stocks", "rows", "result"):
                value = data.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        return []

    @staticmethod
    def _first_value(item: dict, keys: Tuple[str, ...], default=None):
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
        return default

    def fetch_hot_rank_entries(self, top_n: int = 50) -> List[HotRankEntry]:
        if not self.api_key:
            raise RuntimeError("真实同花顺热度榜需要配置 TICKERLAB_API_KEY")

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
        items = self._extract_items(response.json())

        entries: List[HotRankEntry] = []
        used_ranks = set()
        for idx, item in enumerate(items, 1):
            code = self._clean_code(self._first_value(item, ("code", "stock_code", "symbol", "ticker")))
            name = str(self._first_value(item, ("name", "stock_name", "short_name"), "")).strip()
            if not code or not name:
                continue
            rank_raw = self._first_value(item, ("rank", "ranking", "position"), idx)
            heat_raw = self._first_value(item, ("heat", "hot", "hot_value", "score", "heat_score"), 0.0)
            try:
                rank = int(float(rank_raw))
            except (TypeError, ValueError):
                rank = idx
            if rank in used_ranks:
                rank = idx
            used_ranks.add(rank)
            try:
                heat_score = float(heat_raw)
            except (TypeError, ValueError):
                heat_score = 0.0
            entries.append(HotRankEntry(rank=rank, code=code, name=name, heat_score=heat_score))

        entries.sort(key=lambda e: e.rank)
        return entries[:top_n]

    def get_stock_list(
        self,
        top_n: int = 50,
        sort_by: str = "hot_rank",
        custom_pool: Optional[List[str]] = None,
    ) -> Tuple[List[RealTimeStock], str]:
        entries = self.fetch_hot_rank_entries(top_n=top_n if custom_pool is None else len(custom_pool))
        if custom_pool:
            wanted = {self._clean_code(code) for code in custom_pool}
            entries = [entry for entry in entries if entry.code in wanted]

        rank_by_code = {entry.code: entry for entry in entries}
        raw_data = self.quote_source._query_tencent([entry.code for entry in entries])
        stocks: List[RealTimeStock] = []
        for entry in entries:
            fields = raw_data.get(entry.code)
            stock = self.quote_source.parse_tencent_stock(entry.code, fields) if fields else None
            if stock is None:
                stock = RealTimeStock(
                    code=entry.code,
                    name=entry.name,
                    price=0.0,
                    change_pct=0.0,
                    change_amt=0.0,
                    volume=0.0,
                    amount=0.0,
                    high=0.0,
                    low=0.0,
                    open_px=0.0,
                    pre_close=0.0,
                    turnover_rate=0.0,
                    volume_ratio=1.0,
                    pe=0.0,
                    total_mv=0.0,
                    market_cap=0.0,
                )
            setattr(stock, "hot_rank", rank_by_code[entry.code].rank)
            setattr(stock, "hot_score", rank_by_code[entry.code].heat_score)
            stocks.append(stock)

        if sort_by != "hot_rank":
            return TencentDataSource._sort_stocks(stocks, sort_by)[:top_n], datetime.now().strftime("%H:%M:%S")
        return stocks[:top_n], datetime.now().strftime("%H:%M:%S")

    def get_hot_rank(self, top_n: int = 50) -> List[RealTimeStock]:
        return self.get_stock_list(top_n=top_n, sort_by="hot_rank")[0]

    def get_market_overview(self) -> dict:
        stocks, _ = self.get_stock_list(top_n=50)
        up = sum(1 for s in stocks if s.change_pct > 0)
        down = sum(1 for s in stocks if s.change_pct < 0)
        flat = len(stocks) - up - down
        return {
            "total": len(stocks),
            "up": up,
            "down": down,
            "flat": flat,
            "avg_turnover": round(sum(s.turnover_rate for s in stocks) / max(len(stocks), 1), 2),
        }

    def get_concept_boards(self) -> List[ConceptBoard]:
        return TencentDataSource().get_concept_boards()


# ============================================================
# 活跃股票池（热门短线关注标的）
# 每日换手率高、交易活跃的股票
# ============================================================

# 沪深300代表性成分 + 近期热门活跃股
HOT_STOCK_POOL = [
    "600519","000858","000568","600809","000799",  # 白酒
    "002594","601127","600733","000625","002920",  # 新能源车
    "300750","300274","688223","601012","002459",  # 光伏/电池
    "600036","601166","000001","002142","600016",  # 银行
    "601318","600030","601688","600837","002736",  # 保险/券商
    "600276","300760","300015","600196","002007",  # 医药
    "002230","300308","300124","688036","002415",  # AI/科技
    "600585","002271","600031","000651","000333",  # 基建/家电
    "002475","603986","600745","600703","688981",  # 芯片/半导体
    "300059","600570","300033","002410","002555",  # 金融科技/软件
    # 活跃妖股/短线热门（示例）
    "002584","002316","002296","002888","603017",
    "603007","603131","603200","603123","603090",
    "300208","300303","300313","300323","300330",
    "002174","002176","002178","002181","002183",
    "600661","600662","600663","600664","600665",
    "000301","000302","000303","000304","000305",
]


class TencentDataSource:
    """
    腾讯行情数据源 (qt.gtimg.cn)
    备用: 新浪行情 (hq.sinajs.cn)
    """

    TENCENT_URL = "http://qt.gtimg.cn/q={}"
    SINA_URL = "http://hq.sinajs.cn/list={}"
    source_name = "tencent_quotes_pool"

    # 腾讯行情字段索引
    TENCENT_FIELDS = {
        "name": 1,       # 股票名称
        "code": 2,       # 代码
        "price": 3,      # 当前价
        "pre_close": 4,  # 昨收
        "open": 5,       # 今开
        "volume": 6,     # 成交量(手)
        "amount": 7,     # 成交额
        "high": 33,      # 最高
        "low": 34,       # 最低
        "turnover": 38,  # 换手率(%)
        "pe": 39,        # 市盈率
        "amplitude": 43, # 振幅(%)
        "circulating_mv": 44, # 流通市值
        "total_mv": 45,  # 总市值
    }

    def __init__(self, use_sina: bool = False, timeout: int = 10):
        self.use_sina = use_sina
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        self.stock_pool = HOT_STOCK_POOL

    @staticmethod
    def _sort_stocks(stocks: List[RealTimeStock], sort_by: str) -> List[RealTimeStock]:
        sorted_stocks = list(stocks)
        if sort_by == "turnover":
            sorted_stocks.sort(key=lambda s: s.turnover_rate, reverse=True)
        elif sort_by == "change":
            sorted_stocks.sort(key=lambda s: abs(s.change_pct), reverse=True)
        elif sort_by == "amount":
            sorted_stocks.sort(key=lambda s: s.amount, reverse=True)
        elif sort_by == "volume_ratio":
            sorted_stocks.sort(key=lambda s: s.volume_ratio, reverse=True)
        elif sort_by == "hot_rank":
            sorted_stocks.sort(key=lambda s: getattr(s, "hot_rank", 999999))
        else:
            sorted_stocks.sort(key=lambda s: s.turnover_rate, reverse=True)
        return sorted_stocks

    def _query_tencent(self, codes: List[str]) -> Dict[str, List[str]]:
        """批量查询腾讯行情"""
        if not codes:
            return {}
        
        # 分组查询，每批最多50只
        batch_size = 50
        result_map = {}
        
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]
            qstr = ",".join(
                f"sh{c}" if c.startswith("6") else f"sz{c}"
                for c in batch
            )
            try:
                r = self.session.get(
                    self.TENCENT_URL.format(qstr),
                    timeout=self.timeout
                )
                r.encoding = "gbk"
                for line in r.text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # 格式: v_sh600519="..."
                    # 腾讯返回字段用 ~ 分隔
                    parts = line.split("=", 1)
                    if len(parts) < 2:
                        continue
                    fields = parts[1].strip('";').split("~")
                    if len(fields) >= 46:
                        code = fields[2]
                        result_map[code] = fields
            except Exception:
                continue
        
        return result_map

    def _query_sina(self, codes: List[str]) -> Dict[str, List[str]]:
        """批量查询新浪行情"""
        if not codes:
            return {}
        
        batch_size = 50
        result_map = {}
        
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]
            qstr = ",".join(
                f"sh{c}" if c.startswith("6") else f"sz{c}"
                for c in batch
            )
            try:
                r = self.session.get(
                    self.SINA_URL.format(qstr),
                    headers={"Referer": "https://finance.sina.com.cn"},
                    timeout=self.timeout
                )
                r.encoding = "gbk"
                for line in r.text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # 格式: var hq_str_sh600519="..."
                    parts = line.split("=", 1)
                    if len(parts) < 2:
                        continue
                    fields = parts[1].strip('";').split(",")
                    if len(fields) >= 32:
                        # 新浪: 名称(0), 今开(1), 昨收(2), 当前(3), 高(4), 低(5)
                        # 买1价(6), 卖1价(7), 成交量(8手), 成交额(9元)
                        # 涨停(44?), 换手率需要另外处理
                        # 新浪不直接提供换手率，量比，需要计算
                        # 提取代码: sh600519 -> 600519
                        raw_code = parts[0].split("_")[-1]
                        result_map[raw_code] = fields
            except Exception:
                continue
        
        return result_map

    def parse_tencent_stock(self, code: str, fields: List[str]) -> Optional[RealTimeStock]:
        """解析腾讯行情数据为 RealTimeStock"""
        try:
            name = fields[1]
            price = float(fields[3]) if fields[3] else 0.0
            pre_close = float(fields[4]) if fields[4] else 0.0
            open_px = float(fields[5]) if fields[5] else 0.0
            volume = float(fields[6]) if fields[6] else 0.0  # 手
            amount = float(fields[7]) if fields[7] else 0.0  # 元
            high = float(fields[33]) if len(fields) > 33 and fields[33] else 0.0
            low = float(fields[34]) if len(fields) > 34 and fields[34] else 0.0
            turnover_rate = float(fields[38]) if len(fields) > 38 and fields[38] else 0.0
            pe = float(fields[39]) if len(fields) > 39 and fields[39] else 0.0
            total_mv = float(fields[45]) if len(fields) > 45 and fields[45] else 0.0
            market_cap = float(fields[44]) if len(fields) > 44 and fields[44] else 0.0

            if price == 0.0 or pre_close == 0.0:
                return None

            change_amt = round(price - pre_close, 2)
            change_pct = round((change_amt / pre_close) * 100, 2)
            amplitude = round((high - low) / pre_close * 100, 2) if pre_close > 0 else 0.0

            # 计算量比(腾讯无直接量比，用成交额/5日均量的近似值)
            # 简化处理：用换手率与平均换手率的关系估算
            volume_ratio = round(turnover_rate / 3.0, 2) if turnover_rate > 0 else 1.0
            volume_ratio = max(0.1, min(10.0, volume_ratio))

            return RealTimeStock(
                code=code,
                name=name,
                price=price,
                change_pct=change_pct,
                change_amt=change_amt,
                volume=volume,
                amount=amount,
                high=high,
                low=low,
                open_px=open_px,
                pre_close=pre_close,
                turnover_rate=turnover_rate,
                volume_ratio=volume_ratio,
                pe=pe,
                total_mv=total_mv,
                market_cap=market_cap,
            )
        except (ValueError, IndexError):
            return None

    def get_stock_list(
        self,
        top_n: int = 50,
        sort_by: str = "turnover",
        custom_pool: Optional[List[str]] = None
    ) -> Tuple[List[RealTimeStock], str]:
        """
        获取实时股票列表

        Args:
            top_n: 返回前N只
            sort_by: 排序方式 (turnover, change, amount, volume_ratio)
            custom_pool: 自定义股票池，不传则使用内置热门池

        Returns:
            (排序后的股票列表, 时间戳)
        """
        pool = custom_pool or self.stock_pool
        
        if self.use_sina:
            raw_data = self._query_sina(pool)
            stocks = self._parse_sina_batch(raw_data)
        else:
            raw_data = self._query_tencent(pool)
            stocks = []
            for code, fields in raw_data.items():
                stock = self.parse_tencent_stock(code, fields)
                if stock:
                    stocks.append(stock)

        if not stocks:
            return [], datetime.now().strftime("%H:%M:%S")

        # 按指定字段排序
        if sort_by == "turnover":
            stocks.sort(key=lambda s: s.turnover_rate, reverse=True)
        elif sort_by == "change":
            stocks.sort(key=lambda s: abs(s.change_pct), reverse=True)
        elif sort_by == "amount":
            stocks.sort(key=lambda s: s.amount, reverse=True)
        elif sort_by == "volume_ratio":
            stocks.sort(key=lambda s: s.volume_ratio, reverse=True)
        else:
            stocks.sort(key=lambda s: s.turnover_rate, reverse=True)

        ts = datetime.now().strftime("%H:%M:%S")
        return stocks[:top_n], ts

    def get_hot_rank(self, top_n: int = 50) -> List[RealTimeStock]:
        """按换手率排名（热度）"""
        return self.get_stock_list(top_n=top_n, sort_by="turnover")[0]

    def get_market_overview(self) -> dict:
        """获取市场概况（上涨/下跌数）"""
        stocks, _ = self.get_stock_list(top_n=300, sort_by="turnover")
        up = sum(1 for s in stocks if s.change_pct > 0)
        down = sum(1 for s in stocks if s.change_pct < 0)
        flat = len(stocks) - up - down
        return {
            "total": len(stocks),
            "up": up,
            "down": down,
            "flat": flat,
            "avg_turnover": round(sum(s.turnover_rate for s in stocks) / max(len(stocks), 1), 2),
        }

    def get_concept_boards(self) -> List[ConceptBoard]:
        """
        概念板块数据（从内置热门板块模拟）
        腾讯API不直接提供概念板块数据，这里返回模拟热门概念
        """
        # 热门概念列表，后续可考虑从东方财富页面抓取
        concepts = [
            ("BK0991", "人工智能", 2.5, 35),
            ("BK0445", "新能源车", 1.8, 42),
            ("BK0477", "半导体", 1.6, 38),
            ("BK0816", "低空经济", 3.2, 28),
            ("BK0900", "国产芯片", 1.4, 45),
            ("BK0705", "机器人", 1.9, 32),
            ("BK0806", "算力", 2.8, 25),
            ("BK0818", "数据要素", 2.1, 30),
            ("BK0992", "无人驾驶", 1.5, 36),
            ("BK0446", "光伏", 1.2, 40),
            ("BK0455", "储能", 0.8, 38),
            ("BK0860", "创新药", 0.5, 35),
            ("BK0891", "国企改革", 0.3, 50),
            ("BK0993", "消费电子", 1.1, 42),
            ("BK0701", "ChatGPT概念", 2.6, 22),
        ]
        
        boards = []
        for code, name, chg, count in concepts:
            up_count = int(count * (0.5 + chg / 10))
            up_count = max(0, min(count, up_count))
            boards.append(ConceptBoard(
                code=code, name=name, change_pct=chg,
                up_count=up_count, total_count=count
            ))
        
        return boards

    def _parse_sina_batch(self, raw_data: Dict[str, List[str]]) -> List[RealTimeStock]:
        """解析新浪行情数据"""
        stocks = []
        for code, fields in raw_data.items():
            try:
                if len(fields) < 32:
                    continue
                name = fields[0]
                open_px = float(fields[1]) if fields[1] else 0.0
                pre_close = float(fields[2]) if fields[2] else 0.0
                price = float(fields[3]) if fields[3] else 0.0
                high = float(fields[4]) if fields[4] else 0.0
                low = float(fields[5]) if fields[5] else 0.0
                volume = float(fields[8]) if fields[8] else 0.0  # 手
                amount = float(fields[9]) if fields[9] else 0.0  # 元

                if price == 0.0 or pre_close == 0.0:
                    continue

                change_amt = round(price - pre_close, 2)
                change_pct = round((change_amt / pre_close) * 100, 2)
                amplitude = round((high - low) / pre_close * 100, 2) if pre_close > 0 else 0.0

                stock = RealTimeStock(
                    code=code,
                    name=name,
                    price=price,
                    change_pct=change_pct,
                    change_amt=change_amt,
                    volume=volume,
                    amount=amount,
                    high=high,
                    low=low,
                    open_px=open_px,
                    pre_close=pre_close,
                    turnover_rate=0.0,      # 新浪不直接提供
                    volume_ratio=1.0,
                    pe=0.0,
                    total_mv=0.0,
                    market_cap=0.0,
                )
                stocks.append(stock)
            except (ValueError, IndexError):
                continue
        return stocks


# ============================================================
# 数据与贝叶斯整合
# ============================================================

def convert_to_stock_info(stock: RealTimeStock) -> "StockInfo":
    """
    将实时数据转换为 bayesian_trader 的 StockInfo 模型
    """
    from bayesian_trader.models import StockInfo

    vr = stock.volume_ratio
    turnover = stock.turnover_rate
    chg = stock.change_pct
    hot_rank = int(getattr(stock, "hot_rank", 0) or 0)

    # K线形态推理（基于量价关系）
    if vr >= 1.8 and chg > 5:
        candle_pattern = "强势突破"
        candle_score = 0.85
    elif vr >= 1.5 and chg > 2:
        candle_pattern = "放量上攻"
        candle_score = 0.75
    elif vr >= 1.0 and chg > 0:
        candle_pattern = "弱转强"
        candle_score = 0.65
    elif chg < -3 and turnover > 10:
        candle_pattern = "放量下跌"
        candle_score = 0.25
    elif chg < -5:
        candle_pattern = "破位下行"
        candle_score = 0.15
    else:
        candle_pattern = "震荡"
        candle_score = 0.50

    # 概念传播度
    concept_spread = min(0.95, (turnover / 15 + vr / 3) / 2)

    if 1 <= hot_rank <= 50:
        hot_rank_score = round((51 - hot_rank) / 50, 4)
    else:
        hot_rank_score = 0.5

    turnover_score = min(max(turnover, 0.0) / 20.0, 1.0)
    volume_ratio_score = min(max(vr, 0.0) / 3.0, 1.0)
    amount_score = min((stock.amount / 1e8) / 20.0, 1.0) if stock.amount > 0 else 0.0
    volume_activity_score = round(
        max(0.0, min(1.0, turnover_score * 0.45 + volume_ratio_score * 0.35 + amount_score * 0.20)),
        4,
    )

    stock_info = StockInfo(
        code=stock.code,
        name=stock.name,
        price=stock.price,
        change_pct=stock.change_pct,
        turnover_rate=stock.turnover_rate,
        volume_ratio=stock.volume_ratio,
        market_cap=stock.market_cap / 1e8 if stock.market_cap > 0 else 0,
        hot_rank=hot_rank,
        hot_rank_score=hot_rank_score,
        volume_activity_score=volume_activity_score,
        concept_spread=concept_spread,
        candle_pattern=candle_pattern,
        candle_score=min(0.95, max(0.05, candle_score)),
    )
    setattr(stock_info, "concept_tags", getattr(stock, "concept_tags", []))
    setattr(stock_info, "popularity_tag", getattr(stock, "popularity_tag", ""))
    return stock_info


def fetch_and_convert(top_n: int = 50) -> Tuple[List, "MarketState"]:
    """
    一站式获取实时数据并转换为贝叶斯需要的格式

    Returns:
        (stock_info列表, MarketState对象)
    """
    from bayesian_trader.models import MarketState

    ds = TencentDataSource()
    stocks, ts = ds.get_stock_list(top_n=top_n, sort_by="turnover")

    if not stocks:
        # 降级：返回空
        return [], MarketState(heat="未知", heat_score=0.5, cycle_phase=2)

    # 计算市场统计
    up_count = sum(1 for s in stocks if s.change_pct > 0)
    down_count = sum(1 for s in stocks if s.change_pct <= 0)
    avg_turnover = sum(s.turnover_rate for s in stocks) / max(len(stocks), 1)
    avg_change = sum(s.change_pct for s in stocks) / max(len(stocks), 1)

    # 判定市场热度
    if avg_turnover > 8 and avg_change > 2:
        heat_name, heat_score = "火热", 0.90
        cycle_phase = 3
    elif avg_turnover > 4 and avg_change > 0.5:
        heat_name, heat_score = "偏暖", 0.70
        cycle_phase = 2
    elif avg_turnover > 2 and avg_change > -0.5:
        heat_name, heat_score = "中性", 0.50
        cycle_phase = 1
    elif avg_turnover > 1:
        heat_name, heat_score = "偏冷", 0.30
        cycle_phase = 1
    else:
        heat_name, heat_score = "冰点", 0.10
        cycle_phase = 0

    market_state = MarketState(
        heat=heat_name,
        heat_score=heat_score,
        cycle_phase=cycle_phase,
        up_count=up_count,
        down_count=down_count,
        limit_up_count=0,
        limit_down_count=0,
        total_volume=sum(s.amount for s in stocks) / 1e8,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    stock_infos = [convert_to_stock_info(s) for s in stocks]
    return stock_infos, market_state


# ============================================================
# 本地 Mock 数据源（无网络时用）
# ============================================================

class MockDataSource:
    """离线模拟数据源（使用内置数据，无需网络）"""
    source_name = "mock"

    def __init__(self):
        self.mock_stocks = self._build_mock_stocks()

    def _build_mock_stocks(self) -> List[RealTimeStock]:
        """构建模拟数据"""
        base_stocks = [
            # (代码, 名称, 价格, 涨幅%, 换手率%, 量比, 成交额亿)
            ("600519", "贵州茅台", 1420.0, 1.2, 0.5, 0.8, 50),
            ("000858", "五粮液", 145.0, 2.1, 1.2, 1.1, 15),
            ("002594", "比亚迪", 268.0, 3.5, 3.8, 2.1, 45),
            ("601127", "赛力斯", 88.5, 5.2, 8.5, 3.2, 35),
            ("002230", "科大讯飞", 48.0, 4.8, 6.2, 2.8, 28),
            ("300750", "宁德时代", 198.0, 2.5, 2.8, 1.5, 40),
            ("300059", "东方财富", 22.5, 6.8, 12.5, 3.5, 55),
            ("600030", "中信证券", 19.0, 3.2, 4.5, 2.2, 25),
            ("601318", "中国平安", 48.0, 1.5, 1.8, 0.9, 20),
            ("600036", "招商银行", 35.0, 0.8, 0.6, 0.7, 12),
            ("688981", "中芯国际", 72.0, 7.2, 15.8, 4.2, 65),
            ("002475", "立讯精密", 36.0, 4.5, 5.8, 2.5, 22),
            ("300308", "中际旭创", 148.0, 6.5, 10.2, 3.8, 38),
            ("603986", "兆易创新", 88.0, 5.8, 9.5, 3.0, 18),
            ("600745", "闻泰科技", 42.0, 3.8, 7.2, 2.6, 12),
            ("002415", "海康威视", 32.0, 1.8, 2.5, 1.2, 15),
            ("000001", "平安银行", 11.0, 0.5, 1.5, 0.8, 8),
            ("002142", "宁波银行", 25.0, 1.2, 2.0, 1.0, 6),
            ("600276", "恒瑞医药", 45.0, 2.8, 3.5, 1.8, 14),
            ("300760", "迈瑞医疗", 285.0, 1.5, 1.2, 0.9, 10),
            ("000568", "泸州老窖", 185.0, 2.5, 2.8, 1.5, 16),
            ("002371", "北方华创", 320.0, 4.5, 7.8, 2.8, 25),
            ("688041", "海光信息", 88.0, 6.2, 14.5, 4.5, 32),
            ("601012", "隆基绿能", 18.0, -1.5, 3.5, 1.2, 20),
            ("300274", "阳光电源", 75.0, 3.2, 5.5, 2.0, 18),
            ("000333", "美的集团", 65.0, 1.0, 1.5, 0.8, 12),
            ("000651", "格力电器", 42.0, 0.5, 1.2, 0.7, 8),
            ("600585", "海螺水泥", 28.0, -0.8, 2.0, 1.1, 6),
            ("600031", "三一重工", 16.0, 1.5, 3.0, 1.3, 10),
            ("002920", "德赛西威", 95.0, 4.2, 6.8, 2.5, 12),
            ("002410", "广联达", 38.0, 3.5, 4.5, 2.2, 8),
            ("300033", "同花顺", 118.0, 8.5, 16.5, 4.8, 22),
            ("600570", "恒生电子", 28.0, 4.2, 6.5, 2.8, 10),
            ("688111", "金山办公", 260.0, 3.8, 2.5, 1.8, 15),
            ("002555", "三七互娱", 18.0, 2.5, 5.5, 1.8, 8),
            ("600703", "三安光电", 16.0, 3.2, 4.8, 2.0, 10),
            ("600809", "山西汾酒", 210.0, 2.0, 1.8, 1.2, 12),
            ("000799", "酒鬼酒", 52.0, 3.5, 4.2, 1.8, 5),
            ("001979", "招商蛇口", 12.0, 1.0, 2.5, 1.1, 8),
            ("688036", "传音控股", 88.0, 2.5, 3.0, 1.5, 6),
            ("300124", "汇川技术", 62.0, 3.0, 4.5, 2.0, 12),
            ("300015", "爱尔眼科", 15.0, 2.8, 3.5, 1.6, 10),
            ("002007", "华兰生物", 22.0, 1.5, 2.0, 1.2, 4),
            ("002304", "洋河股份", 98.0, 1.2, 1.5, 0.9, 6),
            ("300413", "芒果超媒", 28.0, 3.8, 5.5, 2.2, 5),
            ("002602", "世纪华通", 6.0, 4.5, 8.5, 3.0, 12),
            ("002174", "游族网络", 12.0, 5.0, 12.0, 3.5, 8),
            ("603444", "吉比特", 195.0, 2.5, 3.8, 1.5, 4),
            ("300502", "新易盛", 82.0, 7.5, 14.0, 4.5, 28),
            ("688012", "中微公司", 158.0, 5.5, 8.5, 3.2, 20),
        ]

        stocks = []
        base_date = datetime.now().strftime("%Y-%m-%d")
        seed = int(datetime.now().timestamp()) % 1000
        rng = random.Random(seed)

        for code, name, base_price, chg, turn, vr, amt_100m in base_stocks:
            # 添加随机波动
            noise = rng.uniform(-0.5, 0.5)
            price = round(base_price * (1 + noise * 0.02), 2)
            pct = round(chg + noise * 2, 2)
            turnover = round(turn + noise * 3, 2)
            vol_ratio = round(vr + noise * 0.8, 2)
            amount = round(amt_100m * 1e8, 0)

            pre_close = round(price / (1 + pct / 100), 2)
            change_amt = round(price - pre_close, 2)
            high = round(price * (1 + abs(pct) / 150), 2)
            low = round(price * (1 - abs(pct) / 150), 2)
            volume = round(amount / price, 0) if price > 0 else 0

            stock = RealTimeStock(
                code=code, name=name, price=price, change_pct=pct,
                change_amt=change_amt, volume=volume, amount=amount,
                high=high, low=low, open_px=round(price * (1 - pct / 300), 2),
                pre_close=pre_close, turnover_rate=turnover,
                volume_ratio=vol_ratio, pe=round(rng.uniform(10, 80), 2),
                total_mv=round(price * rng.uniform(10, 100) * 1e8, 0),
                market_cap=round(price * rng.uniform(5, 60) * 1e8, 0),
            )
            stocks.append(stock)

        # 按换手率排序
        stocks.sort(key=lambda s: s.turnover_rate, reverse=True)
        return stocks

    def get_hot_rank(self, top_n: int = 50) -> List[RealTimeStock]:
        return self.mock_stocks[:top_n]

    def get_stock_list(self, top_n: int = 50, sort_by: str = "turnover", **kw) -> Tuple[List[RealTimeStock], str]:
        sorted_stocks = list(self.mock_stocks)
        if sort_by == "turnover":
            sorted_stocks.sort(key=lambda s: s.turnover_rate, reverse=True)
        elif sort_by == "change":
            sorted_stocks.sort(key=lambda s: abs(s.change_pct), reverse=True)
        elif sort_by == "amount":
            sorted_stocks.sort(key=lambda s: s.amount, reverse=True)
        ts = datetime.now().strftime("%H:%M:%S")
        return sorted_stocks[:top_n], ts

    def get_market_overview(self) -> dict:
        up = sum(1 for s in self.mock_stocks if s.change_pct > 0)
        down = sum(1 for s in self.mock_stocks if s.change_pct <= 0)
        return {"total": len(self.mock_stocks), "up": up, "down": down, "flat": 0,
                "avg_turnover": round(sum(s.turnover_rate for s in self.mock_stocks) / len(self.mock_stocks), 2)}

    def get_concept_boards(self) -> List[ConceptBoard]:
        return TencentDataSource().get_concept_boards()


# ============================================================
# 智能选择：自动检测网络并切换
# ============================================================

def create_data_source(force_mock: bool = False):
    """
    智能创建数据源

    优先使用截图口径的同花顺 App 热榜，若网络不可用则回退到备用源。
    """
    if force_mock:
        return MockDataSource()

    try:
        ds = ThsAppHotRankDataSource()
        stocks, _ = ds.get_stock_list(top_n=3)
        if stocks and len(stocks) > 0:
            return ds
        raise ConnectionError("No THS App hot-rank data received")
    except Exception as exc:
        print(f"[数据源] 同花顺App热榜不可用: {exc}; 尝试备用行情源")

    if os.environ.get("TICKERLAB_API_KEY"):
        try:
            ds = ThsHotRankDataSource()
            stocks, _ = ds.get_stock_list(top_n=3)
            if stocks and len(stocks) > 0:
                return ds
            raise ConnectionError("No TickerLab hot-rank data received")
        except Exception as exc:
            print(f"[数据源] TickerLab热榜不可用: {exc}; 尝试腾讯行情池")

    try:
        ds = TencentDataSource()
        stocks, _ = ds.get_stock_list(top_n=3)
        if stocks and len(stocks) > 0:
            return ds
        raise ConnectionError("No data received")
    except Exception:
        print("[数据源] 网络不可用，使用离线模拟数据")
        return MockDataSource()


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("贝叶斯短线 - 实时数据源测试")
    print("=" * 60)

    # 测试1: 腾讯行情API
    print("\n>> 测试1: 腾讯行情API")
    ds = TencentDataSource()
    stocks, ts = ds.get_stock_list(5)
    if stocks:
        print(f"  获取成功! 时间: {ts}")
        for i, s in enumerate(stocks[:5], 1):
            print(f"  #{i} {s.name}({s.code}) {s.price:.2f} "
                  f"涨幅={s.change_pct:+.2f}% "
                  f"换手={s.turnover_rate:.1f}% "
                  f"量比={s.volume_ratio:.1f} "
                  f"成交额={s.amount/1e8:.1f}亿")
    else:
        print("  × 腾讯行情失败")

    # 测试2: 离线Mock
    print("\n>> 测试2: 离线Mock数据源")
    mock = MockDataSource()
    mocks = mock.get_hot_rank(10)
    for i, s in enumerate(mocks[:5], 1):
        print(f"  #{i} {s.name}({s.code}) {s.price:.2f} "
              f"涨幅={s.change_pct:+.2f}% "
              f"换手={s.turnover_rate:.1f}%")

    # 测试3: 智能选择
    print("\n>> 测试3: 智能选择数据源")
    ds2 = create_data_source()
    print(f"  数据源类型: {type(ds2).__name__}")

    # 测试4: 概念板块
    print("\n>> 测试4: 概念板块")
    boards = ds2.get_concept_boards()
    for b in boards[:5]:
        print(f"  {b.name} 涨幅={b.change_pct:+.1f}%  ({b.up_count}/{b.total_count})")

    # 测试5: 市场概况
    print("\n>> 测试5: 市场概况")
    overview = ds2.get_market_overview()
    print(f"  统计: 上涨={overview['up']} 下跌={overview['down']} "
          f"换手率均值={overview['avg_turnover']:.1f}%")

    # 测试6: 贝叶斯格式转换
    print("\n>> 测试6: 转换为贝叶斯格式")
    # 使用离线Mock模拟转换
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results = fetch_and_convert(10)
    if results[0]:
        print(f"  转换完成: {len(results[0])} 只股票")
        print(f"  市场状态: {results[1].heat} (得分={results[1].heat_score})")
        sample = results[0][0]
        print(f"  样本: {sample.name}({sample.code}) "
              f"形态={sample.candle_pattern} "
              f"传播度={sample.concept_spread:.3f}")
    else:
        print("  × 转换失败")

    print("\n" + "=" * 60)
    print("测试完成")
