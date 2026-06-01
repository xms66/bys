from newbys.features import build_evidence_from_snapshot, infer_market_context
from newbys.models import StockSnapshot


def test_feature_builder_infers_evidence_categories_from_quote_snapshot():
    stocks = [
        StockSnapshot(
            code="300001",
            name="Hot One",
            price=20.0,
            change_pct=4.5,
            turnover_rate=12.0,
            volume_ratio=2.2,
            amount=1_200_000_000,
            hot_rank=5,
            concept_tags=["AI", "robot"],
        ),
        StockSnapshot(
            code="000001",
            name="Cold One",
            price=10.0,
            change_pct=-1.0,
            turnover_rate=1.0,
            volume_ratio=0.7,
            amount=100_000_000,
            hot_rank=80,
            concept_tags=[],
        ),
    ]
    market = infer_market_context(stocks)
    evidence = build_evidence_from_snapshot(stocks[0], market)

    assert market.cycle_phase in {"warmup", "markup", "mixed", "retreat", "ice"}
    assert evidence.popularity == "top10"
    assert evidence.volume_pattern == "healthy_expansion"
    assert evidence.five_day_pattern == "breakout_after_base"
    assert evidence.concept_strength == "strong"


def test_manual_evidence_overrides_inferred_categories():
    stock = StockSnapshot(
        code="300002",
        name="Manual",
        price=30.0,
        change_pct=8.0,
        turnover_rate=35.0,
        volume_ratio=5.0,
        amount=2_000_000_000,
        hot_rank=2,
        concept_tags=["AI"],
    )
    market = infer_market_context([stock])
    evidence = build_evidence_from_snapshot(
        stock,
        market,
        overrides={
            "five_day_pattern": "first_pullback",
            "volume_pattern": "explosive_divergence",
            "message_strength": "medium",
        },
    )

    assert evidence.five_day_pattern == "first_pullback"
    assert evidence.volume_pattern == "explosive_divergence"
    assert evidence.message_strength == "medium"
