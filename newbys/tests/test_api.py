from newbys.app import create_app
from newbys.data_source import StaticDataSource
from newbys.models import StockSnapshot


def test_analysis_api_returns_subjective_bayes_result():
    advisor = FakeAdvisor()
    app = create_app(
        data_source=StaticDataSource(
            [
                StockSnapshot(
                    code="300033",
                    name="Hot",
                    price=100.0,
                    change_pct=3.0,
                    turnover_rate=12.0,
                    volume_ratio=2.0,
                    amount=1_000_000_000,
                    hot_rank=1,
                    concept_tags=["AI", "data"],
                )
            ]
        ),
        llm_advisor=advisor,
    )
    client = app.test_client()

    response = client.get("/api/analysis?top_n=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model"] == "subjective_bayes_v1"
    assert payload["items"][0]["posterior_profit"] > 0.5
    assert payload["items"][0]["evidence"][0]["name"] == "five_day_pattern"


def test_manual_analysis_accepts_explicit_evidence():
    app = create_app(data_source=StaticDataSource([]))
    client = app.test_client()

    response = client.post(
        "/api/infer",
        json={
            "stock": {
                "code": "002000",
                "name": "Manual",
                "price": 20,
                "change_pct": 2.5,
                "turnover_rate": 8,
                "volume_ratio": 1.8,
                "amount": 800000000,
                "hot_rank": 8,
                "concept_tags": ["robot"],
            },
            "market": {
                "cycle_phase": "markup",
                "index_trend": "up",
                "sentiment_score": 0.75,
            },
            "evidence": {
                "five_day_pattern": "breakout_after_base",
                "volume_pattern": "healthy_expansion",
                "message_strength": "strong",
                "concept_strength": "medium",
                "popularity": "top10",
            },
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["posterior_profit"] > 0.5
    assert payload["prior_profit"] > 0.55


class FakeAdvisor:
    def __init__(self):
        self.calls = []

    def analyze(self, market, items):
        self.calls.append((market, items))
        return {"enabled": True, "content": "LLM decision", "error": "", "model": "fake"}


def test_analysis_defaults_to_top5_and_includes_llm_advice():
    stocks = [
        StockSnapshot(
            code=f"30000{i}",
            name=f"Stock {i}",
            price=10.0,
            change_pct=1.0,
            turnover_rate=5.0,
            volume_ratio=1.2,
            amount=100_000_000,
            hot_rank=i,
            concept_tags=["AI"],
        )
        for i in range(1, 8)
    ]
    advisor = FakeAdvisor()
    app = create_app(data_source=StaticDataSource(stocks), llm_advisor=advisor)
    client = app.test_client()

    response = client.get("/api/analysis")

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["items"]) == 5
    assert payload["llm_advice"]["content"] == "LLM decision"
    assert len(advisor.calls[0][1]) == 5
