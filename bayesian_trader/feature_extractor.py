"""
特征提取器模块
负责从原始数据中提取和计算贝叶斯模型所需的特征
模拟从同花顺API获取的数据进行特征工程
"""

import numpy as np
from typing import List, Optional
from datetime import datetime

from .config import HEAT_SCORE_MAP, CANDLE_PATTERN_SCORE_MAP
from .models import StockInfo, MarketState


class FeatureExtractor:
    """特征提取器 - 将原始数据转换为模型特征"""

    @staticmethod
    def infer_heat_score(heat_label: str) -> float:
        """将市场热度标签转为数值评分 0-1"""
        return HEAT_SCORE_MAP.get(heat_label, 0.5)

    @staticmethod
    def infer_candle_score(pattern_label: str) -> float:
        """将K线形态标签转为数值评分 0-1"""
        return CANDLE_PATTERN_SCORE_MAP.get(pattern_label, 0.5)

    @staticmethod
    def infer_cycle_phase(
        up_ratio: float,
        limit_up_ratio: float,
        limit_down_ratio: float,
        volume_change: float
    ) -> int:
        """
        根据市场数据推断当前周期阶段

        Args:
            up_ratio: 上涨家数比例
            limit_up_ratio: 涨停比例
            limit_down_ratio: 跌停比例
            volume_change: 成交量变化率

        Returns:
            周期阶段 0-4
        """
        score = 0.0
        score += up_ratio * 2.0              # 上涨比例越高越好
        score += limit_up_ratio * 3.0        # 涨停越多越好
        score -= limit_down_ratio * 3.0      # 跌停越少越好
        score += volume_change * 0.5         # 放量加分

        if score < 0.3:
            return 0  # 退潮期
        elif score < 0.6:
            return 1  # 混沌期
        elif score < 1.2:
            return 2  # 上升期
        elif score < 2.0:
            return 3  # 主升期
        else:
            return 4  # 高潮期

    @staticmethod
    def compute_concept_spread(
        concept_count: int,
        concept_limit_up: int,
        concept_leader: bool = False
    ) -> float:
        """
        计算概念传播度 0-1

        Args:
            concept_count: 概念内股票数量
            concept_limit_up: 概念内涨停数量
            concept_leader: 是否为龙头股

        Returns:
            概念传播度评分
        """
        spread = 0.0

        # 概念内涨停比例
        if concept_count > 0:
            limit_up_ratio = concept_limit_up / concept_count
            spread += limit_up_ratio * 0.5

        # 龙头股加成
        if concept_leader:
            spread += 0.3

        # 概念规模适中最佳（太大太杂，太小没合力）
        if 5 <= concept_count <= 20:
            spread += 0.2
        elif 3 <= concept_count < 5 or 20 < concept_count <= 30:
            spread += 0.1

        return np.clip(spread, 0.0, 1.0)

    @staticmethod
    def create_stock_from_raw(
        raw: dict
    ) -> StockInfo:
        """
        从原始数据字典创建StockInfo对象

        模拟同花顺API返回的数据格式
        """
        candle_pattern = raw.get("candle_pattern", "震荡")
        candle_score = FeatureExtractor.infer_candle_score(candle_pattern)

        return StockInfo(
            code=raw.get("code", ""),
            name=raw.get("name", ""),
            price=float(raw.get("price", 0)),
            change_pct=float(raw.get("change_pct", 0)),
            turnover_rate=float(raw.get("turnover_rate", 0)),
            volume_ratio=float(raw.get("volume_ratio", 1.0)),
            market_cap=float(raw.get("market_cap", 100)),
            concept_spread=float(raw.get("concept_spread", 0.5)),
            candle_pattern=candle_pattern,
            candle_score=candle_score,
            rank=int(raw.get("rank", 0)),
        )

    @staticmethod
    def create_market_from_raw(
        raw: dict
    ) -> MarketState:
        """
        从原始数据字典创建MarketState对象
        """
        heat_label = raw.get("heat", "中性")
        heat_score = FeatureExtractor.infer_heat_score(heat_label)

        market = MarketState(
            heat=heat_label,
            heat_score=heat_score,
            cycle_phase=int(raw.get("cycle_phase", 1)),
            up_count=int(raw.get("up_count", 0)),
            down_count=int(raw.get("down_count", 0)),
            limit_up_count=int(raw.get("limit_up_count", 0)),
            limit_down_count=int(raw.get("limit_down_count", 0)),
            total_volume=float(raw.get("total_volume", 0)),
        )

        # 如果没有直接给出周期阶段，则根据数据推断
        if "cycle_phase" not in raw or raw["cycle_phase"] is None:
            total = market.up_count + market.down_count
            up_ratio = market.up_count / total if total > 0 else 0.5
            limit_up_ratio = market.limit_up_count / total if total > 0 else 0.0
            limit_down_ratio = market.limit_down_count / total if total > 0 else 0.0
            volume_change = raw.get("volume_change", 0.0)
            market.cycle_phase = FeatureExtractor.infer_cycle_phase(
                up_ratio, limit_up_ratio, limit_down_ratio, volume_change
            )

        market.cycle_phase_name = MarketState.CYCLE_NAMES[market.cycle_phase]
        return market