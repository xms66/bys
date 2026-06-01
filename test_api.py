# -*- coding: utf-8 -*-

from bayesian_trader.data_source import RealTimeStock
from bayesian_trader import server


class FakeHotDataSource:
    source_name = "ths_app_hot_rank"

    def __init__(self):
        self.stock = RealTimeStock(
            code="300033",
            name="同花顺",
            price=100.0,
            change_pct=2.5,
            change_amt=2.5,
            volume=1000000,
            amount=100000000.0,
            high=103.0,
            low=98.0,
            open_px=99.0,
            pre_close=97.5,
            turnover_rate=12.0,
            volume_ratio=2.0,
            pe=20.0,
            total_mv=50000000000.0,
            market_cap=8000000000.0,
        )
        setattr(self.stock, "hot_rank", 1)
        setattr(self.stock, "hot_score", 9988.0)
        setattr(self.stock, "concept_tags", ["超超临界发电", "煤炭概念"])
        setattr(self.stock, "popularity_tag", "6天3板")

    def get_stock_list(self, top_n=50, sort_by="hot_rank", custom_pool=None):
        return [self.stock], "10:30:00"

    def get_market_overview(self):
        return {"total": 1, "up": 1, "down": 0, "flat": 0, "avg_turnover": 12.0}

    def get_concept_boards(self):
        return []


def setup_module(_module):
    server.data_source = FakeHotDataSource()


def test_hot_rank_api_returns_source_and_ths_rank():
    client = server.app.test_client()

    response = client.get("/api/hot_rank?top_n=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source"] == "ths_app_hot_rank"
    assert payload["stocks"][0]["rank"] == 1
    assert payload["stocks"][0]["hot_rank"] == 1
    assert payload["stocks"][0]["hot_score"] == 9988.0
    assert payload["stocks"][0]["concept_tags"] == ["超超临界发电", "煤炭概念"]
    assert payload["stocks"][0]["popularity_tag"] == "6天3板"


def test_analysis_api_returns_evidence_and_features():
    client = server.app.test_client()

    response = client.get("/api/analysis?top_n=1")

    assert response.status_code == 200
    payload = response.get_json()
    item = payload["analysis"][0]
    assert "evidence_detail" in item
    assert "decision_steps" in item
    assert "risk_control" in item
    assert "model_notice" in item
    assert item["risk_control"]["max_position_pct"] <= 0.1
    assert item["risk_control"]["stop_loss_pct"] < 0
    assert "hot_rank_likelihood" in item["evidence_detail"]
    assert item["features"]["hot_rank"] == 1
    assert item["features"]["hot_rank_score"] == 1.0
    assert item["features"]["volume_activity_score"] > 0
    assert item["features"]["concept_tags"] == ["超超临界发电", "煤炭概念"]
