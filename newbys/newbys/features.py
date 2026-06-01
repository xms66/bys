from __future__ import annotations

from .models import EvidenceInput, MarketContext, StockSnapshot


def infer_market_context(stocks: list[StockSnapshot]) -> MarketContext:
    if not stocks:
        return MarketContext(cycle_phase="mixed", index_trend="flat", sentiment_score=0.5)

    avg_change = sum(s.change_pct for s in stocks) / len(stocks)
    avg_turnover = sum(s.turnover_rate for s in stocks) / len(stocks)
    up_count = sum(1 for s in stocks if s.change_pct > 0)
    up_ratio = up_count / len(stocks)

    sentiment = max(0.0, min(1.0, 0.25 + up_ratio * 0.45 + min(avg_change / 8.0, 0.25) + min(avg_turnover / 40.0, 0.10)))
    if avg_change >= 2.0 and avg_turnover >= 8.0 and up_ratio >= 0.60:
        cycle = "markup"
    elif avg_change >= 0.5 and up_ratio >= 0.52:
        cycle = "warmup"
    elif avg_change <= -2.0 or up_ratio <= 0.35:
        cycle = "retreat"
    elif avg_change <= -4.0:
        cycle = "ice"
    else:
        cycle = "mixed"

    if avg_change > 0.7:
        trend = "up"
    elif avg_change < -0.7:
        trend = "down"
    else:
        trend = "flat"

    return MarketContext(
        cycle_phase=cycle,
        index_trend=trend,
        sentiment_score=round(sentiment, 4),
    )


def build_evidence_from_snapshot(
    stock: StockSnapshot,
    market: MarketContext,
    overrides: dict[str, str] | None = None,
) -> EvidenceInput:
    overrides = overrides or {}
    return EvidenceInput(
        stock=stock,
        market=market,
        five_day_pattern=overrides.get("five_day_pattern", infer_five_day_pattern(stock)),
        volume_pattern=overrides.get("volume_pattern", infer_volume_pattern(stock)),
        message_strength=overrides.get("message_strength", infer_message_strength(stock)),
        concept_strength=overrides.get("concept_strength", infer_concept_strength(stock)),
        popularity=overrides.get("popularity", infer_popularity(stock)),
    )


def infer_five_day_pattern(stock: StockSnapshot) -> str:
    if stock.change_pct >= 2.0 and 1.2 <= stock.volume_ratio <= 3.5:
        return "breakout_after_base"
    if -3.0 <= stock.change_pct <= 1.5 and stock.turnover_rate >= 5.0:
        return "first_pullback"
    if stock.change_pct >= 7.0 or stock.volume_ratio >= 4.5:
        return "extended_chase"
    if stock.change_pct <= -2.0:
        return "weak_downtrend"
    return "sideways"


def infer_volume_pattern(stock: StockSnapshot) -> str:
    if 1.2 <= stock.volume_ratio <= 3.2 and 4.0 <= stock.turnover_rate <= 22.0:
        return "healthy_expansion"
    if stock.volume_ratio >= 4.0 or stock.turnover_rate >= 28.0:
        return "explosive_divergence"
    if stock.volume_ratio < 0.9 or stock.turnover_rate < 2.0:
        return "shrinking_or_inactive"
    return "neutral"


def infer_message_strength(stock: StockSnapshot) -> str:
    if len(stock.concept_tags) >= 3 or stock.hot_rank <= 10:
        return "strong"
    if len(stock.concept_tags) >= 1 or stock.hot_rank <= 50:
        return "medium"
    return "none"


def infer_concept_strength(stock: StockSnapshot) -> str:
    if len(stock.concept_tags) >= 2:
        return "strong"
    if len(stock.concept_tags) == 1:
        return "medium"
    return "weak"


def infer_popularity(stock: StockSnapshot) -> str:
    if 1 <= stock.hot_rank <= 10:
        return "top10"
    if 11 <= stock.hot_rank <= 50:
        return "top50"
    if stock.hot_rank > 0:
        return "normal"
    return "cold"
