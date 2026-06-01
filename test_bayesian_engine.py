# -*- coding: utf-8 -*-

from bayesian_trader.bayesian_engine import BayesianEngine
from bayesian_trader.models import MarketState, StockInfo


def test_better_evidence_produces_higher_t1_profit_probability():
    engine = BayesianEngine()
    hot_market = MarketState(heat="活跃", heat_score=0.8, cycle_phase=3)

    strong = StockInfo(
        code="300033",
        name="同花顺",
        change_pct=2.5,
        turnover_rate=12.0,
        volume_ratio=2.0,
        market_cap=80,
        concept_spread=0.85,
        candle_pattern="强势突破",
        candle_score=0.85,
        hot_rank=1,
        hot_rank_score=1.0,
        volume_activity_score=0.8,
    )
    weak = StockInfo(
        code="000001",
        name="平安银行",
        change_pct=-2.0,
        turnover_rate=0.8,
        volume_ratio=0.6,
        market_cap=3000,
        concept_spread=0.2,
        candle_pattern="震荡",
        candle_score=0.4,
        hot_rank=50,
        hot_rank_score=0.02,
        volume_activity_score=0.2,
    )

    strong_signal = engine.compute_posterior(strong, hot_market)
    weak_signal = engine.compute_posterior(weak, hot_market)

    assert strong_signal.posterior_prob > weak_signal.posterior_prob
    assert "hot_rank_likelihood" in strong_signal.evidence_detail
    assert "volume_activity_likelihood" in strong_signal.evidence_detail


def test_short_term_probability_is_conservatively_calibrated():
    engine = BayesianEngine()
    hot_market = MarketState(heat="火热", heat_score=0.9, cycle_phase=3)
    strong = StockInfo(
        code="600863",
        name="华能蒙电",
        change_pct=8.49,
        turnover_rate=11.94,
        volume_ratio=3.98,
        market_cap=80,
        concept_spread=0.95,
        candle_pattern="强势突破",
        candle_score=0.85,
        hot_rank=1,
        hot_rank_score=1.0,
        volume_activity_score=0.62,
    )

    signal = engine.compute_posterior(strong, hot_market)

    assert 0.45 <= signal.posterior_prob <= 0.68
    assert signal.evidence_detail["raw_posterior"] > signal.posterior_prob
    assert signal.evidence_detail["calibrated"] is True


def test_rank_score_is_separate_from_posterior_probability():
    engine = BayesianEngine()
    market = MarketState(heat="活跃", heat_score=0.8, cycle_phase=3)
    stock = StockInfo(
        code="300033",
        name="同花顺",
        turnover_rate=12.0,
        volume_ratio=2.0,
        market_cap=80,
        concept_spread=0.85,
        candle_score=0.85,
        hot_rank=1,
        hot_rank_score=1.0,
        volume_activity_score=0.8,
    )

    signal = engine.compute_posterior(stock, market)
    posterior = signal.posterior_prob
    rank_score = engine.compute_rank_score(signal)

    assert signal.posterior_prob == posterior
    assert rank_score != posterior
