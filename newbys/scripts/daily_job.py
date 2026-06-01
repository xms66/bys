import argparse
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from newbys.backtest import BacktestStore, PortfolioEngine
from newbys.decide_hot50 import run as decide_hot50


def generate_decision(args):
    payload = decide_hot50(top_n=args.top_n, pretty=False)
    decision = payload["decision"]
    store = BacktestStore()
    engine = PortfolioEngine(store)
    selected = decision.get("selected") or {}
    state = store.get_state()
    engine.record_decision(
        args.date or date.today().isoformat(),
        decision.get("decision", "hold_cash"),
        selected.get("code"),
        selected.get("name"),
        float(state["total_value"]),
        json.dumps(decision, ensure_ascii=False),
    )
    print(json.dumps({"decision": decision, "capital": state["total_value"]}, ensure_ascii=False, indent=2))


def execute_roll(args):
    decision_text = _read_json_argument(args.decision_json, args.decision_file, "decision")
    open_prices_text = _read_json_argument(args.open_prices_json, args.open_prices_file, "open prices")
    decision = json.loads(decision_text)
    open_prices = {str(k): float(v) for k, v in json.loads(open_prices_text).items()}
    engine = PortfolioEngine(BacktestStore())
    result = engine.execute_daily_roll(
        trade_date=args.date or date.today().isoformat(),
        open_prices=open_prices,
        decision=decision,
        decision_text=args.decision_json,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _read_json_argument(raw_value: str, file_path: str, label: str) -> str:
    if file_path:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return f.read()
    if raw_value:
        return raw_value
    raise SystemExit(f"--{label.replace(' ', '-')}-json or --{label.replace(' ', '-')}-file is required")


def main():
    parser = argparse.ArgumentParser(description="Daily decision and full-position backtest job.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    decide = sub.add_parser("decide", help="Run evening LLM hot50 decision and record it.")
    decide.add_argument("--date", default=None)
    decide.add_argument("--top-n", type=int, default=50)
    decide.set_defaults(func=generate_decision)

    execute = sub.add_parser("execute", help="Execute morning sell/buy using explicit open prices.")
    execute.add_argument("--date", default=None)
    execute.add_argument("--decision-json", default="")
    execute.add_argument("--decision-file", default="")
    execute.add_argument("--open-prices-json", default="")
    execute.add_argument("--open-prices-file", default="")
    execute.set_defaults(func=execute_roll)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
