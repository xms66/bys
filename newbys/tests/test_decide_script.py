import json

from newbys.decide_hot50 import build_decision_payload
from newbys.models import MarketContext, StockSnapshot


class FakeAdvisor:
    def analyze(self, market, items):
        return {
            "enabled": True,
            "content": json.dumps(
                {
                    "decision": "buy",
                    "selected": {"code": "300001", "name": "Hot One", "rank": 1},
                    "bayes_probability": 0.66,
                    "position": "light",
                    "reason": ["test"],
                    "risks": [],
                    "cancel_conditions": [],
                },
                ensure_ascii=False,
            ),
            "error": "",
            "model": "fake",
        }


def test_build_decision_payload_uses_all_items_and_parses_decision():
    stocks = [
        StockSnapshot(
            code=f"3000{i:02d}",
            name=f"Stock {i}",
            price=10,
            change_pct=2,
            turnover_rate=8,
            volume_ratio=1.5,
            amount=100000000,
            hot_rank=i,
            concept_tags=["AI"],
        )
        for i in range(1, 11)
    ]

    payload = build_decision_payload(stocks, FakeAdvisor())

    assert payload["candidate_count"] == 10
    assert payload["decision"]["decision"] == "buy"
    assert payload["decision"]["selected"]["code"] == "300001"
    assert len(payload["items"]) == 10
