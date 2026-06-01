"""
数据模型模块 - 定义股票数据和交易信号的数据结构
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class StockInfo:
    """单只股票的信息"""
    code: str                      # 股票代码
    name: str                      # 股票名称
    price: float = 0.0             # 当前价格
    change_pct: float = 0.0        # 涨跌幅%
    turnover_rate: float = 0.0     # 换手率%
    volume_ratio: float = 1.0      # 量比
    market_cap: float = 0.0        # 流通市值(亿)
    hot_rank: int = 0              # 同花顺热度排名
    hot_rank_score: float = 0.5    # 同花顺热度排名分 0-1
    volume_activity_score: float = 0.5  # 量价活跃度 0-1
    concept_spread: float = 0.5    # 概念传播度 0-1
    candle_pattern: str = "震荡"   # K线形态描述
    candle_score: float = 0.5      # K线形态评分 0-1
    rank: int = 0                  # 排名
    timestamp: str = ""            # 数据时间戳

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class MarketState:
    """市场状态"""
    heat: str = "中性"             # 市场热度描述
    heat_score: float = 0.5        # 热度评分 0-1
    cycle_phase: int = 1           # 周期阶段 0-4
    cycle_phase_name: str = "混沌期"  # 周期阶段名称
    up_count: int = 0              # 上涨家数
    down_count: int = 0            # 下跌家数
    limit_up_count: int = 0        # 涨停家数
    limit_down_count: int = 0      # 跌停家数
    total_volume: float = 0.0      # 总成交额(亿)
    timestamp: str = ""

    CYCLE_NAMES = ["退潮期", "混沌期", "上升期", "主升期", "高潮期"]

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cycle_phase_name = self.CYCLE_NAMES[self.cycle_phase]


@dataclass
class BayesianSignal:
    """贝叶斯推理后的交易信号"""
    stock: StockInfo
    posterior_prob: float = 0.0        # 后验概率 P(赚钱|证据)
    signal_strength: str = "观望"       # 信号强度
    evidence_detail: dict = field(default_factory=dict)  # 各特征贡献详情
    rank_score: float = 0.0            # 综合排名分

    def __post_init__(self):
        if not self.evidence_detail:
            self.evidence_detail = {
                "prior": 0.0,
                "heat_likelihood": 0.0,
                "cycle_likelihood": 0.0,
                "concept_likelihood": 0.0,
                "pattern_likelihood": 0.0,
                "evidence": 0.0,
            }


@dataclass
class TradeRecord:
    """交易记录"""
    stock_code: str
    stock_name: str
    buy_date: str
    buy_price: float
    signal_prob: float            # 买入时的后验概率
    sell_date: Optional[str] = None
    sell_price: Optional[float] = None
    profit_pct: Optional[float] = None
    hold_days: Optional[int] = None
    status: str = "持有"          # 持有/已卖/止损/止盈
    reason: str = ""
