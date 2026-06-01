"""
贝叶斯推理引擎 - 核心模块
基于朴素贝叶斯(Naive Bayes)计算P(赚钱|特征)的后验概率

核心公式:
P(赚钱|特征1,特征2,...) = [P(赚钱) * P(特征1|赚钱) * P(特征2|赚钱) * ...] / P(特征1,特征2,...)

假设各特征条件独立
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .config import (
    PRIOR_PROFIT, HEAT_WEIGHT, CYCLE_WEIGHT,
    CONCEPT_SPREAD_WEIGHT, CANDLE_PATTERN_WEIGHT,
    HEAT_SCORE_MAP, CANDLE_PATTERN_SCORE_MAP
)
from .models import StockInfo, MarketState, BayesianSignal


class BayesianEngine:
    """
    贝叶斯推理引擎

    朴素贝叶斯假设各特征条件独立，计算后验概率:
    P(profit|evidence) = P(profit) * ∏P(evidence_i|profit) / P(evidence)
    其中 P(evidence) = P(profit)*∏P(e_i|profit) + P(loss)*∏P(e_i|loss)
    """

    def __init__(self):
        # 可学习的先验概率（可从数据库加载）
        self.prior_profit = PRIOR_PROFIT
        self.prior_loss = 1.0 - PRIOR_PROFIT

        # 加载各特征的权重参数
        self.heat_params = HEAT_WEIGHT
        self.cycle_params = CYCLE_WEIGHT
        self.concept_params = CONCEPT_SPREAD_WEIGHT
        self.pattern_params = CANDLE_PATTERN_WEIGHT

    def update_prior(self, new_prior: float):
        """更新先验概率（在线学习）"""
        self.prior_profit = np.clip(new_prior, 0.05, 0.95)
        self.prior_loss = 1.0 - self.prior_profit

    def _get_heat_likelihood(self, heat_score: float, given_profit: bool) -> float:
        """
        计算市场热度的条件概率似然

        Args:
            heat_score: 市场热度评分 0-1
            given_profit: 是否给定赚钱条件

        Returns:
            条件概率值
        """
        if given_profit:
            base = self.heat_params["prior_heat_given_profit"]
        else:
            base = self.heat_params["prior_heat_given_loss"]

        # 用热度评分调整似然值
        # 热度高时放大赚钱条件下的概率，热度低时缩小
        if given_profit:
            likelihood = base * (0.5 + heat_score)  # heat 0->0.5*base, heat 1->1.5*base
        else:
            likelihood = base * (1.5 - heat_score)  # heat 0->1.5*base, heat 1->0.5*base

        return np.clip(likelihood, 0.05, 0.95)

    def _get_cycle_likelihood(self, cycle_phase: int, given_profit: bool) -> float:
        """
        计算市场周期阶段的条件概率似然

        Args:
            cycle_phase: 周期阶段 0-4
            given_profit: 是否给定赚钱条件

        Returns:
            条件概率值
        """
        cycle_phase = np.clip(cycle_phase, 0, 4)
        if given_profit:
            return self.cycle_params["prior_cycle_given_profit"][cycle_phase]
        else:
            return self.cycle_params["prior_cycle_given_loss"][cycle_phase]

    def _get_concept_likelihood(self, concept_spread: float, given_profit: bool) -> float:
        """
        计算概念传播度的条件概率似然

        Args:
            concept_spread: 概念传播度 0-1
            given_profit: 是否给定赚钱条件

        Returns:
            条件概率值
        """
        if given_profit:
            base = self.concept_params["prior_concept_given_profit"]
            # 传播度越高，赚钱似然越大
            likelihood = base * (0.5 + concept_spread)
        else:
            base = self.concept_params["prior_concept_given_loss"]
            # 传播度越高，亏钱似然越小
            likelihood = base * (1.5 - concept_spread)

        return np.clip(likelihood, 0.05, 0.95)

    def _get_pattern_likelihood(self, candle_score: float, given_profit: bool) -> float:
        """
        计算K线形态的条件概率似然

        Args:
            candle_score: K线形态评分 0-1
            given_profit: 是否给定赚钱条件

        Returns:
            条件概率值
        """
        if given_profit:
            base = self.pattern_params["prior_pattern_given_profit"]
            # 形态越好，赚钱似然越大
            likelihood = base * (0.3 + 0.7 * candle_score)
        else:
            base = self.pattern_params["prior_pattern_given_loss"]
            # 形态越好，亏钱似然越小
            likelihood = base * (1.3 - 0.7 * candle_score)

        return np.clip(likelihood, 0.05, 0.95)

    def _get_hot_rank_likelihood(self, hot_rank_score: float, given_profit: bool) -> float:
        """
        计算同花顺热度排名的条件概率似然。

        hot_rank_score 越高，说明越靠近热度榜前排。第一版把热度视为短线
        流动性和关注度证据，但只作为概率因子之一，不直接等同于买入信号。
        """
        hot_rank_score = float(np.clip(hot_rank_score, 0.0, 1.0))
        if given_profit:
            likelihood = 0.45 + 0.45 * hot_rank_score
        else:
            likelihood = 0.65 - 0.35 * hot_rank_score
        return np.clip(likelihood, 0.05, 0.95)

    def _get_volume_activity_likelihood(self, activity_score: float, given_profit: bool) -> float:
        """计算量价活跃度的条件概率似然。"""
        activity_score = float(np.clip(activity_score, 0.0, 1.0))
        if given_profit:
            likelihood = 0.40 + 0.45 * activity_score
        else:
            likelihood = 0.65 - 0.35 * activity_score
        return np.clip(likelihood, 0.05, 0.95)

    def _calibrate_short_term_probability(
        self,
        raw_posterior: float,
        stock: StockInfo,
        market: MarketState
    ) -> float:
        """
        将朴素贝叶斯原始后验校准到更现实的短线胜率区间。

        朴素贝叶斯会因为独立性假设把多个强证据相乘后推到过高概率。
        在没有多年历史样本校准前，T+1 概率应向 50% 收缩，并扣除追高、
        极端量价和弱市场环境风险。
        """
        cycle_score = float(np.clip(market.cycle_phase / 4.0, 0.0, 1.0))
        market_score = (float(np.clip(market.heat_score, 0.0, 1.0)) + cycle_score) / 2.0

        if -3.0 <= stock.change_pct <= 5.0:
            change_quality = 0.65
        elif 5.0 < stock.change_pct < 9.0:
            change_quality = 0.55
        elif stock.change_pct >= 9.0:
            change_quality = 0.45
        elif -7.0 <= stock.change_pct < -3.0:
            change_quality = 0.35
        else:
            change_quality = 0.25

        turnover_quality = 1.0 - min(abs(stock.turnover_rate - 12.0) / 24.0, 1.0)
        stock_score = (
            0.22 * float(np.clip(stock.hot_rank_score, 0.0, 1.0))
            + 0.22 * float(np.clip(stock.concept_spread, 0.0, 1.0))
            + 0.20 * float(np.clip(stock.candle_score, 0.0, 1.0))
            + 0.20 * float(np.clip(stock.volume_activity_score, 0.0, 1.0))
            + 0.08 * turnover_quality
            + 0.08 * change_quality
        )

        raw_component = (float(raw_posterior) - 0.50) * 0.18
        p = 0.38 + 0.18 * stock_score + 0.08 * market_score + raw_component

        if market.heat_score >= 0.75 and market.cycle_phase >= 2:
            p += 0.025
        elif market.heat_score <= 0.35 or market.cycle_phase <= 1:
            p -= 0.025

        if stock.change_pct >= 9.0:
            p -= 0.035
        elif stock.change_pct >= 7.0:
            p -= 0.02

        if stock.change_pct <= -7.0:
            p -= 0.025

        if stock.turnover_rate > 35.0:
            p -= 0.03
        elif stock.turnover_rate > 25.0:
            p -= 0.015

        if stock.volume_ratio > 6.0:
            p -= 0.02

        return float(np.clip(p, 0.30, 0.68))

    def compute_posterior(
        self,
        stock: StockInfo,
        market: MarketState
    ) -> BayesianSignal:
        """
        计算后验概率 P(赚钱 | 所有特征)

        公式:
        P(profit|E) = P(profit) * P(E_heat|profit) * P(E_cycle|profit) 
                      * P(E_concept|profit) * P(E_pattern|profit) / Z

        其中 Z = P(profit) * ∏P(E_i|profit) + P(loss) * ∏P(E_i|loss)

        Args:
            stock: 股票信息
            market: 市场状态

        Returns:
            BayesianSignal 包含后验概率和信号强度
        """
        # 1. 计算各特征的条件概率
        # 市场热度似然
        heat_likelihood_profit = self._get_heat_likelihood(
            market.heat_score, given_profit=True
        )
        heat_likelihood_loss = self._get_heat_likelihood(
            market.heat_score, given_profit=False
        )

        # 周期阶段似然
        cycle_likelihood_profit = self._get_cycle_likelihood(
            market.cycle_phase, given_profit=True
        )
        cycle_likelihood_loss = self._get_cycle_likelihood(
            market.cycle_phase, given_profit=False
        )

        # 概念传播度似然
        concept_likelihood_profit = self._get_concept_likelihood(
            stock.concept_spread, given_profit=True
        )
        concept_likelihood_loss = self._get_concept_likelihood(
            stock.concept_spread, given_profit=False
        )

        # K线形态似然
        pattern_likelihood_profit = self._get_pattern_likelihood(
            stock.candle_score, given_profit=True
        )
        pattern_likelihood_loss = self._get_pattern_likelihood(
            stock.candle_score, given_profit=False
        )

        # 同花顺热度排名似然
        hot_rank_likelihood_profit = self._get_hot_rank_likelihood(
            stock.hot_rank_score, given_profit=True
        )
        hot_rank_likelihood_loss = self._get_hot_rank_likelihood(
            stock.hot_rank_score, given_profit=False
        )

        # 量价活跃度似然
        volume_activity_likelihood_profit = self._get_volume_activity_likelihood(
            stock.volume_activity_score, given_profit=True
        )
        volume_activity_likelihood_loss = self._get_volume_activity_likelihood(
            stock.volume_activity_score, given_profit=False
        )

        # 2. 计算分子: P(profit) * ∏P(E_i|profit)
        numerator_profit = (
            self.prior_profit
            * heat_likelihood_profit
            * cycle_likelihood_profit
            * concept_likelihood_profit
            * pattern_likelihood_profit
            * hot_rank_likelihood_profit
            * volume_activity_likelihood_profit
        )

        # 3. 计算分子: P(loss) * ∏P(E_i|loss)
        numerator_loss = (
            self.prior_loss
            * heat_likelihood_loss
            * cycle_likelihood_loss
            * concept_likelihood_loss
            * pattern_likelihood_loss
            * hot_rank_likelihood_loss
            * volume_activity_likelihood_loss
        )

        # 4. 归一化常数（证据因子）
        evidence = numerator_profit + numerator_loss

        # 5. 后验概率
        if evidence > 0:
            raw_posterior = numerator_profit / evidence
        else:
            raw_posterior = self.prior_profit
        posterior = self._calibrate_short_term_probability(raw_posterior, stock, market)

        # 6. 构建信号
        signal = BayesianSignal(stock=stock)
        signal.posterior_prob = round(posterior, 4)
        signal.evidence_detail = {
            "prior": round(self.prior_profit, 4),
            "heat_likelihood": round(heat_likelihood_profit, 4),
            "cycle_likelihood": round(cycle_likelihood_profit, 4),
            "concept_likelihood": round(concept_likelihood_profit, 4),
            "pattern_likelihood": round(pattern_likelihood_profit, 4),
            "hot_rank_likelihood": round(hot_rank_likelihood_profit, 4),
            "volume_activity_likelihood": round(volume_activity_likelihood_profit, 4),
            "heat_likelihood_loss": round(heat_likelihood_loss, 4),
            "cycle_likelihood_loss": round(cycle_likelihood_loss, 4),
            "concept_likelihood_loss": round(concept_likelihood_loss, 4),
            "pattern_likelihood_loss": round(pattern_likelihood_loss, 4),
            "hot_rank_likelihood_loss": round(hot_rank_likelihood_loss, 4),
            "volume_activity_likelihood_loss": round(volume_activity_likelihood_loss, 4),
            "evidence": round(evidence, 4),
            "numerator_profit": round(numerator_profit, 4),
            "numerator_loss": round(numerator_loss, 4),
            "raw_posterior": round(float(raw_posterior), 4),
            "calibrated": True,
        }

        # 7. 判断信号强度
        if posterior >= 0.64:
            signal.signal_strength = "强买入"
        elif posterior >= 0.57:
            signal.signal_strength = "买入"
        elif posterior >= 0.50:
            signal.signal_strength = "关注"
        else:
            signal.signal_strength = "观望"

        return signal

    def compute_rank_score(self, signal: BayesianSignal) -> float:
        """
        计算综合排名分（用于排序同花顺前50）

        综合考虑后验概率和额外的因子（换手率、量比等）
        """
        stock = signal.stock

        # 基础分 = 后验概率
        base_score = signal.posterior_prob

        # 换手率加分（适度换手率 = 活跃，太高或太低不好）
        turnover_score = 0.0
        if 5.0 <= stock.turnover_rate <= 30.0:
            turnover_score = 0.05
        elif 30.0 < stock.turnover_rate <= 50.0:
            turnover_score = 0.03

        # 量比加分（量比1.0-3.0为健康放量）
        volume_score = 0.0
        if 1.0 <= stock.volume_ratio <= 3.0:
            volume_score = 0.05
        elif 0.5 <= stock.volume_ratio < 1.0:
            volume_score = 0.02

        # 市值加分（小市值更适合短线）
        cap_score = 0.0
        if stock.market_cap < 50:
            cap_score = 0.05
        elif stock.market_cap < 100:
            cap_score = 0.03
        elif stock.market_cap < 200:
            cap_score = 0.01

        # 涨幅加分（微涨或微跌的票弹性好，大涨的票追高风险大）
        change_score = 0.0
        if -3.0 <= stock.change_pct <= 3.0:
            change_score = 0.03
        elif -5.0 <= stock.change_pct < -3.0:
            change_score = 0.02
        elif 3.0 < stock.change_pct <= 7.0:
            change_score = 0.02

        rank_score = base_score + turnover_score + volume_score + cap_score + change_score
        signal.rank_score = round(rank_score, 4)
        return signal.rank_score

    def rank_stocks(
        self,
        stocks: List[StockInfo],
        market: MarketState
    ) -> List[BayesianSignal]:
        """
        对一组股票进行贝叶斯推理并排名

        Args:
            stocks: 股票列表（如同花顺热门前50）
            market: 当前市场状态

        Returns:
            按排名分降序排列的BayesianSignal列表
        """
        signals = []
        for stock in stocks:
            signal = self.compute_posterior(stock, market)
            self.compute_rank_score(signal)
            signals.append(signal)

        # 按排名分降序排列
        signals.sort(key=lambda s: s.rank_score, reverse=True)
        return signals
