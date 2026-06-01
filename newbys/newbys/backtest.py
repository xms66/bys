from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any


INITIAL_CAPITAL = 100000.0


class BacktestStore:
    def __init__(self, db_path: str = "data/backtest.db"):
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
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL,
                    position_code TEXT,
                    position_name TEXT,
                    shares INTEGER NOT NULL,
                    avg_price REAL,
                    total_value REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    code TEXT,
                    name TEXT,
                    action TEXT NOT NULL,
                    price REAL,
                    shares INTEGER,
                    capital REAL NOT NULL,
                    profit_pct REAL,
                    decision_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            exists = conn.execute("SELECT id FROM portfolio_state WHERE id = 1").fetchone()
            if not exists:
                conn.execute(
                    """
                    INSERT INTO portfolio_state (
                        id, cash, position_code, position_name, shares, avg_price, total_value, updated_at
                    ) VALUES (1, ?, NULL, NULL, 0, NULL, ?, ?)
                    """,
                    (INITIAL_CAPITAL, INITIAL_CAPITAL, _now()),
                )
            conn.commit()

    def get_state(self) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM portfolio_state WHERE id = 1").fetchone()
        return dict(row)

    def update_state(
        self,
        cash: float,
        position_code: str | None,
        position_name: str | None,
        shares: int,
        avg_price: float | None,
        total_value: float,
    ) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE portfolio_state
                SET cash = ?, position_code = ?, position_name = ?, shares = ?,
                    avg_price = ?, total_value = ?, updated_at = ?
                WHERE id = 1
                """,
                (cash, position_code, position_name, shares, avg_price, total_value, _now()),
            )
            conn.commit()
        return self.get_state()

    def add_history(
        self,
        date: str,
        decision: str,
        code: str | None,
        name: str | None,
        action: str,
        price: float | None,
        shares: int | None,
        capital: float,
        profit_pct: float | None,
        decision_text: str,
    ) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO backtest_history (
                    date, decision, code, name, action, price, shares,
                    capital, profit_pct, decision_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (date, decision, code, name, action, price, shares, capital, profit_pct, decision_text, _now()),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM backtest_history WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def history(self, limit: int = 200) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_history ORDER BY date DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


class PortfolioEngine:
    def __init__(self, store: BacktestStore):
        self.store = store

    def execute_buy(self, trade_date: str, decision_text: str, code: str, name: str, price: float) -> dict[str, Any]:
        state = self.store.get_state()
        cash = float(state["cash"])
        shares = int(cash / price / 100) * 100
        cost = shares * price
        cash_after = round(cash - cost, 2)
        total_value = round(cash_after + shares * price, 2)
        self.store.update_state(cash_after, code, name, shares, price, total_value)
        return self.store.add_history(
            date=trade_date,
            decision="buy",
            code=code,
            name=name,
            action="buy",
            price=price,
            shares=shares,
            capital=total_value,
            profit_pct=None,
            decision_text=decision_text,
        ) | {"cash_after": cash_after}

    def execute_sell(self, trade_date: str, price: float) -> dict[str, Any] | None:
        state = self.store.get_state()
        shares = int(state["shares"])
        if shares <= 0 or not state["position_code"]:
            return None
        cash_after = round(shares * price + float(state["cash"]), 2)
        avg_price = float(state["avg_price"] or 0)
        profit_pct = (price - avg_price) / avg_price if avg_price > 0 else None
        code = state["position_code"]
        name = state["position_name"]
        self.store.update_state(cash_after, None, None, 0, None, cash_after)
        return self.store.add_history(
            date=trade_date,
            decision="sell",
            code=code,
            name=name,
            action="sell",
            price=price,
            shares=shares,
            capital=cash_after,
            profit_pct=profit_pct,
            decision_text="open-to-open sell",
        ) | {"cash_after": cash_after}

    def execute_daily_roll(
        self,
        trade_date: str,
        open_prices: dict[str, float],
        decision: dict[str, Any],
        decision_text: str,
    ) -> dict[str, Any]:
        current = self.store.get_state()
        sell_result = None
        if current["position_code"]:
            sell_price = open_prices.get(current["position_code"])
            if sell_price is None:
                raise ValueError(f"missing open price for sell {current['position_code']}")
            sell_result = self.execute_sell(trade_date, sell_price)

        buy_result = None
        selected = decision.get("selected") or {}
        if decision.get("decision") == "buy" and selected.get("code"):
            code = selected["code"]
            buy_price = open_prices.get(code)
            if buy_price is None:
                raise ValueError(f"missing open price for buy {code}")
            buy_result = self.execute_buy(trade_date, decision_text, code, selected.get("name", ""), buy_price)
        elif decision.get("decision"):
            state = self.store.get_state()
            self.record_decision(
                trade_date,
                decision.get("decision", "hold_cash"),
                None,
                None,
                float(state["total_value"]),
                decision_text,
            )

        return {"date": trade_date, "sell": sell_result, "buy": buy_result}

    def record_decision(
        self,
        trade_date: str,
        decision: str,
        code: str | None,
        name: str | None,
        capital: float,
        decision_text: str,
    ) -> dict[str, Any]:
        return self.store.add_history(
            date=trade_date,
            decision=decision,
            code=code,
            name=name,
            action="decision",
            price=None,
            shares=None,
            capital=capital,
            profit_pct=None,
            decision_text=decision_text,
        )


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
