from newbys.app import create_app
from newbys.data_source import StaticDataSource
from newbys.models import StockSnapshot
from newbys.trade_planner import TradePlanStore


class PlanAdvisor:
    def is_configured(self):
        return True

    def analyze(self, market, items):
        return {
            "enabled": True,
            "content": (
                '{"decision":"buy","selected":{"code":"300001","name":"Plan Stock","rank":1},'
                '"bayes_probability":0.67,"position":"light","reason":["ok"],'
                '"risks":["risk"],"cancel_conditions":["cancel"]}'
            ),
            "error": "",
            "model": "fake",
        }


def test_generate_plan_saves_llm_decision(tmp_path):
    store = TradePlanStore(str(tmp_path / "plans.db"))
    app = create_app(
        data_source=StaticDataSource(
            [
                StockSnapshot(
                    code="300001",
                    name="Plan Stock",
                    price=10,
                    change_pct=2,
                    turnover_rate=8,
                    volume_ratio=1.5,
                    amount=100000000,
                    hot_rank=1,
                    concept_tags=["AI"],
                )
            ]
        ),
        llm_advisor=PlanAdvisor(),
        plan_store=store,
    )
    client = app.test_client()

    response = client.post("/api/plans/generate", json={"plan_date": "2026-06-01"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["plan"]["code"] == "300001"
    assert payload["plan"]["buy_date"] == "2026-06-02"
    assert payload["plan"]["sell_date"] == "2026-06-03"


def test_today_actions_and_mark_execution_api(tmp_path):
    store = TradePlanStore(str(tmp_path / "plans.db"))
    app = create_app(
        data_source=StaticDataSource([]),
        llm_advisor=PlanAdvisor(),
        plan_store=store,
    )
    client = app.test_client()
    plan = client.post("/api/plans/generate", json={"plan_date": "2026-06-01"}).get_json()["plan"]

    actions = client.get("/api/plans/today-actions?date=2026-06-02").get_json()
    assert actions["buy"]["code"] == "300001"

    buy_response = client.post(f"/api/plans/{plan['id']}/mark-buy", json={"price": 10.0})
    sell_response = client.post(f"/api/plans/{plan['id']}/mark-sell", json={"price": 10.5})

    assert buy_response.status_code == 200
    assert sell_response.get_json()["plan"]["status"] == "sold"
