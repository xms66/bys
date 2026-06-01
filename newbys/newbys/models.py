from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class StockSnapshot:
    code: str
    name: str
    price: float
    change_pct: float
    turnover_rate: float
    volume_ratio: float
    amount: float
    hot_rank: int = 0
    concept_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketContext:
    cycle_phase: str = "mixed"
    index_trend: str = "flat"
    sentiment_score: float = 0.5
    limit_up_count: int = 0
    limit_down_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceInput:
    stock: StockSnapshot
    market: MarketContext
    five_day_pattern: str
    volume_pattern: str
    message_strength: str
    concept_strength: str
    popularity: str


@dataclass(frozen=True)
class EvidenceContribution:
    name: str
    value: str
    profit_likelihood: float
    loss_likelihood: float
    ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceResult:
    stock: StockSnapshot
    market: MarketContext
    prior_profit: float
    prior_loss: float
    raw_posterior: float
    posterior_profit: float
    posterior_loss: float
    profit_score: float
    loss_score: float
    evidence: list[EvidenceContribution]
    signal: str
    action: str
    model: str = "subjective_bayes_v1"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data

