from newbys.backtest import BacktestStore, PortfolioEngine


def test_portfolio_starts_with_100k_cash(tmp_path):
    store = BacktestStore(str(tmp_path / "bt.db"))

    state = store.get_state()

    assert state["cash"] == 100000.0
    assert state["position_code"] is None
    assert state["total_value"] == 100000.0


def test_full_position_buy_and_sell_updates_capital(tmp_path):
    store = BacktestStore(str(tmp_path / "bt.db"))
    engine = PortfolioEngine(store)

    buy = engine.execute_buy(
        trade_date="2026-06-02",
        decision_text="buy 300001",
        code="300001",
        name="Test Stock",
        price=10.0,
    )
    sell = engine.execute_sell(trade_date="2026-06-03", price=11.0)
    state = store.get_state()

    assert buy["shares"] == 10000
    assert buy["cash_after"] == 0.0
    assert sell["cash_after"] == 110000.0
    assert round(sell["profit_pct"], 4) == 0.1
    assert state["cash"] == 110000.0
    assert state["position_code"] is None


def test_daily_roll_sells_existing_position_then_buys_new_decision(tmp_path):
    store = BacktestStore(str(tmp_path / "bt.db"))
    engine = PortfolioEngine(store)
    engine.execute_buy("2026-06-02", "buy old", "300001", "Old", 10.0)

    result = engine.execute_daily_roll(
        trade_date="2026-06-03",
        open_prices={"300001": 11.0, "300002": 22.0},
        decision={
            "decision": "buy",
            "selected": {"code": "300002", "name": "New", "rank": 1},
        },
        decision_text="buy new",
    )

    assert result["sell"]["code"] == "300001"
    assert result["buy"]["code"] == "300002"
    assert result["buy"]["shares"] == 5000
    assert store.get_state()["position_code"] == "300002"


def test_hold_cash_only_sells_existing_position(tmp_path):
    store = BacktestStore(str(tmp_path / "bt.db"))
    engine = PortfolioEngine(store)
    engine.execute_buy("2026-06-02", "buy old", "300001", "Old", 10.0)

    result = engine.execute_daily_roll(
        trade_date="2026-06-03",
        open_prices={"300001": 9.0},
        decision={"decision": "hold_cash", "selected": None},
        decision_text="hold cash",
    )

    assert result["sell"]["cash_after"] == 90000.0
    assert result["buy"] is None
    assert store.get_state()["position_code"] is None


def test_history_table_rows_include_date_decision_and_capital(tmp_path):
    store = BacktestStore(str(tmp_path / "bt.db"))
    engine = PortfolioEngine(store)
    engine.record_decision("2026-06-01", "buy", "300001", "Test", 100000.0, "buy test")

    rows = store.history()

    assert rows[0]["date"] == "2026-06-01"
    assert rows[0]["decision"] == "buy"
    assert rows[0]["capital"] == 100000.0
