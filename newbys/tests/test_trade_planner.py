from datetime import date

from newbys.trade_planner import (
    TradePlanStore,
    create_trade_plan_from_llm,
    get_today_actions,
    next_trade_dates,
    parse_llm_decision,
)


def test_next_trade_dates_use_calendar_days_for_initial_version():
    buy_date, sell_date = next_trade_dates(date(2026, 6, 1))

    assert buy_date == "2026-06-02"
    assert sell_date == "2026-06-03"


def test_store_creates_plan_and_today_actions(tmp_path):
    db_path = tmp_path / "plans.db"
    store = TradePlanStore(str(db_path))
    plan = create_trade_plan_from_llm(
        plan_date="2026-06-01",
        llm_decision={
            "decision": "buy",
            "selected": {"code": "002081", "name": "金螳螂", "rank": 1},
            "bayes_probability": 0.72,
            "position": "light",
            "reason": ["题材强"],
            "risks": ["高位"],
            "cancel_conditions": ["竞价低于预期"],
        },
        raw_llm_text='{"decision":"buy"}',
    )

    saved = store.save_plan(plan)
    actions = get_today_actions(store, "2026-06-02")

    assert saved["decision"] == "buy"
    assert actions["buy"]["code"] == "002081"
    assert actions["sell"] is None


def test_today_actions_roll_sell_then_buy(tmp_path):
    store = TradePlanStore(str(tmp_path / "plans.db"))
    store.save_plan(
        create_trade_plan_from_llm(
            plan_date="2026-06-01",
            llm_decision={
                "decision": "buy",
                "selected": {"code": "002081", "name": "金螳螂", "rank": 1},
                "bayes_probability": 0.65,
            },
            raw_llm_text="old",
        )
    )
    store.save_plan(
        create_trade_plan_from_llm(
            plan_date="2026-06-02",
            llm_decision={
                "decision": "buy",
                "selected": {"code": "600863", "name": "华能蒙电", "rank": 2},
                "bayes_probability": 0.66,
            },
            raw_llm_text="new",
        )
    )

    actions = get_today_actions(store, "2026-06-03")

    assert actions["sell"]["code"] == "002081"
    assert actions["buy"]["code"] == "600863"


def test_mark_execution_updates_buy_and_sell_prices(tmp_path):
    store = TradePlanStore(str(tmp_path / "plans.db"))
    plan = store.save_plan(
        create_trade_plan_from_llm(
            plan_date="2026-06-01",
            llm_decision={
                "decision": "buy",
                "selected": {"code": "002081", "name": "金螳螂", "rank": 1},
                "bayes_probability": 0.65,
            },
            raw_llm_text="raw",
        )
    )

    store.mark_buy_executed(plan["id"], price=6.78)
    store.mark_sell_executed(plan["id"], price=7.10)
    updated = store.get_plan(plan["id"])

    assert updated["status"] == "sold"
    assert updated["buy_price"] == 6.78
    assert updated["sell_price"] == 7.10
    assert round(updated["profit_pct"], 4) == 0.0472


def test_parse_llm_decision_marks_invalid_output_as_parse_error():
    decision = parse_llm_decision("自然语言报告，不是JSON")

    assert decision["decision"] == "parse_error"
    assert "not valid JSON" in decision["reason"][0]
