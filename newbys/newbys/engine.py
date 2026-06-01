from __future__ import annotations

from .config import (
    BASE_PRIORS,
    DEFAULT_LIKELIHOOD,
    LIKELIHOODS,
    POSTERIOR_CEILING,
    POSTERIOR_FLOOR,
    TREND_ADJUSTMENT,
)
from .models import EvidenceContribution, EvidenceInput, InferenceResult, MarketContext


class SubjectiveBayesEngine:
    """Naive subjective Bayesian engine for current short-term evidence."""

    def dynamic_prior(self, market: MarketContext) -> float:
        base = BASE_PRIORS.get(market.cycle_phase, BASE_PRIORS["mixed"])
        trend = TREND_ADJUSTMENT.get(market.index_trend, 0.0)
        sentiment = (max(0.0, min(1.0, market.sentiment_score)) - 0.5) * 0.10
        limit_balance = 0.0
        total_limits = market.limit_up_count + market.limit_down_count
        if total_limits > 0:
            limit_balance = ((market.limit_up_count - market.limit_down_count) / total_limits) * 0.03
        return round(max(0.18, min(0.68, base + trend + sentiment + limit_balance)), 4)

    def infer(self, data: EvidenceInput) -> InferenceResult:
        prior_profit = self.dynamic_prior(data.market)
        prior_loss = 1.0 - prior_profit
        evidence = self._build_evidence(data)

        profit_score = prior_profit
        loss_score = prior_loss
        for item in evidence:
            profit_score *= item.profit_likelihood
            loss_score *= item.loss_likelihood

        denominator = profit_score + loss_score
        raw = profit_score / denominator if denominator else prior_profit
        posterior = max(POSTERIOR_FLOOR, min(POSTERIOR_CEILING, raw))
        posterior = round(posterior, 4)

        return InferenceResult(
            stock=data.stock,
            market=data.market,
            prior_profit=prior_profit,
            prior_loss=round(prior_loss, 4),
            raw_posterior=round(raw, 4),
            posterior_profit=posterior,
            posterior_loss=round(1.0 - posterior, 4),
            profit_score=round(profit_score, 8),
            loss_score=round(loss_score, 8),
            evidence=evidence,
            signal=self._signal(posterior),
            action=self._action(posterior),
        )

    def _build_evidence(self, data: EvidenceInput) -> list[EvidenceContribution]:
        pairs = [
            ("five_day_pattern", data.five_day_pattern),
            ("volume_pattern", data.volume_pattern),
            ("message_strength", data.message_strength),
            ("concept_strength", data.concept_strength),
            ("popularity", data.popularity),
        ]
        result = []
        for name, value in pairs:
            profit, loss = LIKELIHOODS.get(name, {}).get(value, DEFAULT_LIKELIHOOD)
            result.append(
                EvidenceContribution(
                    name=name,
                    value=value,
                    profit_likelihood=profit,
                    loss_likelihood=loss,
                    ratio=round(profit / loss, 4) if loss else 0.0,
                )
            )
        return result

    @staticmethod
    def _signal(probability: float) -> str:
        if probability >= 0.66:
            return "strong"
        if probability >= 0.58:
            return "positive"
        if probability >= 0.50:
            return "watch"
        return "avoid"

    @staticmethod
    def _action(probability: float) -> str:
        if probability >= 0.66:
            return "small_position_only"
        if probability >= 0.58:
            return "watch_for_entry"
        if probability >= 0.50:
            return "observe"
        return "avoid"
