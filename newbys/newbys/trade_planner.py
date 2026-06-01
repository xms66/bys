from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TradePlan:
    plan_date: str
    buy_date: str
    sell_date: str
    decision: str
    code: str | None
    name: str | None
    rank: int | None
    bayes_probability: float | None
    position: str
    reason_json: str
    risks_json: str
    cancel_conditions_json: str
    raw_llm_text: str
    status: str = "planned"
    buy_price: float | None = None
    sell_price: float | None = None
    profit_pct: float | None = None


def next_trade_dates(plan_day: date) -> tuple[str, str]:
    buy = plan_day + timedelta(days=1)
    sell = plan_day + timedelta(days=2)
    return buy.isoformat(), sell.isoformat()


def create_trade_plan_from_llm(plan_date: str, llm_decision: dict[str, Any], raw_llm_text: str) -> TradePlan:
    plan_day = date.fromisoformat(plan_date)
    buy_date, sell_date = next_trade_dates(plan_day)
    selected = llm_decision.get("selected") or {}
    decision = str(llm_decision.get("decision", "hold_cash"))
    return TradePlan(
        plan_date=plan_date,
        buy_date=buy_date,
        sell_date=sell_date,
        decision=decision,
        code=selected.get("code") if decision == "buy" else None,
        name=selected.get("name") if decision == "buy" else None,
        rank=int(selected.get("rank")) if decision == "buy" and selected.get("rank") is not None else None,
        bayes_probability=float(llm_decision.get("bayes_probability")) if llm_decision.get("bayes_probability") is not None else None,
        position=str(llm_decision.get("position", "none" if decision != "buy" else "light")),
        reason_json=json.dumps(llm_decision.get("reason", []), ensure_ascii=False),
        risks_json=json.dumps(llm_decision.get("risks", []), ensure_ascii=False),
        cancel_conditions_json=json.dumps(llm_decision.get("cancel_conditions", []), ensure_ascii=False),
        raw_llm_text=raw_llm_text,
    )


def parse_llm_decision(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        return {"decision": "hold_cash", "reason": ["LLM did not return content"]}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"decision": "hold_cash", "reason": ["LLM output was not valid JSON"], "raw": raw_text}


class TradePlanStore:
    def __init__(self, db_path: str = "data/trade_plans.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_date TEXT NOT NULL,
                    buy_date TEXT NOT NULL,
                    sell_date TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    code TEXT,
                    name TEXT,
                    rank INTEGER,
                    bayes_probability REAL,
                    position TEXT NOT NULL,
                    reason_json TEXT NOT NULL,
                    risks_json TEXT NOT NULL,
                    cancel_conditions_json TEXT NOT NULL,
                    raw_llm_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    buy_price REAL,
                    sell_price REAL,
                    profit_pct REAL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save_plan(self, plan: TradePlan) -> dict[str, Any]:
        values = asdict(plan)
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO trade_plans (
                    plan_date, buy_date, sell_date, decision, code, name, rank,
                    bayes_probability, position, reason_json, risks_json,
                    cancel_conditions_json, raw_llm_text, status, buy_price,
                    sell_price, profit_pct, created_at
                ) VALUES (
                    :plan_date, :buy_date, :sell_date, :decision, :code, :name, :rank,
                    :bayes_probability, :position, :reason_json, :risks_json,
                    :cancel_conditions_json, :raw_llm_text, :status, :buy_price,
                    :sell_price, :profit_pct, :created_at
                )
                """,
                {**values, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            )
            conn.commit()
            return self.get_plan(cursor.lastrowid)

    def get_plan(self, plan_id: int) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM trade_plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            raise KeyError(f"trade plan {plan_id} not found")
        return _row_to_dict(row)

    def latest_plan_for_buy_date(self, buy_date: str) -> dict[str, Any] | None:
        return self._latest_for_date("buy_date", buy_date)

    def latest_plan_for_sell_date(self, sell_date: str) -> dict[str, Any] | None:
        return self._latest_for_date("sell_date", sell_date)

    def _latest_for_date(self, column: str, value: str) -> dict[str, Any] | None:
        if column not in {"buy_date", "sell_date"}:
            raise ValueError("invalid date column")
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT * FROM trade_plans WHERE {column} = ? AND decision = 'buy' ORDER BY id DESC LIMIT 1",
                (value,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def mark_buy_executed(self, plan_id: int, price: float) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE trade_plans SET buy_price = ?, status = 'bought' WHERE id = ?",
                (float(price), plan_id),
            )
            conn.commit()
        return self.get_plan(plan_id)

    def mark_sell_executed(self, plan_id: int, price: float) -> dict[str, Any]:
        plan = self.get_plan(plan_id)
        buy_price = float(plan["buy_price"] or 0)
        sell_price = float(price)
        profit_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else None
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE trade_plans SET sell_price = ?, profit_pct = ?, status = 'sold' WHERE id = ?",
                (sell_price, profit_pct, plan_id),
            )
            conn.commit()
        return self.get_plan(plan_id)


def get_today_actions(store: TradePlanStore, today: str | None = None) -> dict[str, Any]:
    day = today or date.today().isoformat()
    sell_plan = store.latest_plan_for_sell_date(day)
    buy_plan = store.latest_plan_for_buy_date(day)
    return {
        "date": day,
        "sell": _action_from_plan(sell_plan) if sell_plan else None,
        "buy": _action_from_plan(buy_plan) if buy_plan else None,
    }


def _action_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": plan["id"],
        "code": plan["code"],
        "name": plan["name"],
        "rank": plan["rank"],
        "position": plan["position"],
        "bayes_probability": plan["bayes_probability"],
        "status": plan["status"],
    }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["reason"] = json.loads(data.pop("reason_json") or "[]")
    data["risks"] = json.loads(data.pop("risks_json") or "[]")
    data["cancel_conditions"] = json.loads(data.pop("cancel_conditions_json") or "[]")
    return data
