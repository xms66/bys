from newbys.engine import SubjectiveBayesEngine
from newbys.models import (
    EvidenceInput,
    MarketContext,
    StockSnapshot,
)


def test_strong_short_term_evidence_produces_higher_t1_probability():
    engine = SubjectiveBayesEngine()
    market = MarketContext(
        cycle_phase="markup",
        index_trend="up",
        sentiment_score=0.78,
        limit_up_count=65,
        limit_down_count=3,
    )
    strong = EvidenceInput(
        stock=StockSnapshot(
            code="300033",
            name="Example Strong",
            price=100.0,
            change_pct=3.2,
            turnover_rate=13.0,
            volume_ratio=2.1,
            amount=1_800_000_000,
            hot_rank=3,
            concept_tags=["AI", "data"],
        ),
        market=market,
        five_day_pattern="breakout_after_base",
        volume_pattern="healthy_expansion",
        message_strength="strong",
        concept_strength="strong",
        popularity="top10",
    )
    weak = EvidenceInput(
        stock=StockSnapshot(
            code="000001",
            name="Example Weak",
            price=12.0,
            change_pct=-2.5,
            turnover_rate=1.2,
            volume_ratio=0.7,
            amount=200_000_000,
            hot_rank=45,
            concept_tags=[],
        ),
        market=market,
        five_day_pattern="weak_downtrend",
        volume_pattern="shrinking_or_inactive",
        message_strength="none",
        concept_strength="weak",
        popularity="normal",
    )

    strong_result = engine.infer(strong)
    weak_result = engine.infer(weak)

    assert strong_result.posterior_profit > weak_result.posterior_profit
    assert 0.5 < strong_result.posterior_profit <= 0.72
    assert weak_result.posterior_profit < 0.5
    assert 0.55 < strong_result.prior_profit <= 0.68
    assert strong_result.evidence[0].name == "five_day_pattern"


def test_market_cycle_changes_dynamic_prior_before_evidence():
    engine = SubjectiveBayesEngine()
    stock = StockSnapshot(
        code="600000",
        name="Same Stock",
        price=10.0,
        change_pct=1.0,
        turnover_rate=5.0,
        volume_ratio=1.2,
        amount=500_000_000,
        hot_rank=20,
        concept_tags=["robot"],
    )
    shared_kwargs = dict(
        stock=stock,
        five_day_pattern="sideways",
        volume_pattern="neutral",
        message_strength="medium",
        concept_strength="medium",
        popularity="top50",
    )
    hot_market = MarketContext(cycle_phase="markup", index_trend="up", sentiment_score=0.8)
    cold_market = MarketContext(cycle_phase="retreat", index_trend="down", sentiment_score=0.2)

    hot_result = engine.infer(EvidenceInput(market=hot_market, **shared_kwargs))
    cold_result = engine.infer(EvidenceInput(market=cold_market, **shared_kwargs))

    assert hot_result.prior_profit > cold_result.prior_profit
    assert hot_result.posterior_profit > cold_result.posterior_profit


def test_result_explains_profit_and_loss_likelihoods():
    result = SubjectiveBayesEngine().infer(
        EvidenceInput(
            stock=StockSnapshot(
                code="002000",
                name="Explainable",
                price=20.0,
                change_pct=4.0,
                turnover_rate=9.0,
                volume_ratio=1.8,
                amount=900_000_000,
                hot_rank=8,
                concept_tags=["low altitude"],
            ),
            market=MarketContext(cycle_phase="warmup", index_trend="up", sentiment_score=0.62),
            five_day_pattern="breakout_after_base",
            volume_pattern="healthy_expansion",
            message_strength="strong",
            concept_strength="medium",
            popularity="top10",
        )
    )

    names = [item.name for item in result.evidence]
    assert names == [
        "five_day_pattern",
        "volume_pattern",
        "message_strength",
        "concept_strength",
        "popularity",
    ]
    assert all(0 < item.profit_likelihood < 1 for item in result.evidence)
    assert all(0 < item.loss_likelihood < 1 for item in result.evidence)
    assert result.raw_posterior >= result.posterior_profit
