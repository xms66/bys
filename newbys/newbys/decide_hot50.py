from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

from .data_source import create_data_source
from .engine import SubjectiveBayesEngine
from .features import build_evidence_from_snapshot, infer_market_context
from .llm_advisor import LlmAdvisor
from .models import StockSnapshot
from .trade_planner import parse_llm_decision


def build_decision_payload(stocks: list[StockSnapshot], advisor: LlmAdvisor) -> dict[str, Any]:
    engine = SubjectiveBayesEngine()
    market = infer_market_context(stocks)
    items = []
    for stock in stocks:
        evidence_input = build_evidence_from_snapshot(stock, market)
        items.append(engine.infer(evidence_input).to_dict())
    market_dict = market.to_dict()
    llm_advice = advisor.analyze(market_dict, items)
    decision = parse_llm_decision(llm_advice.get("content", ""))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "ths_app_hot_rank",
        "candidate_count": len(items),
        "market": market_dict,
        "decision": decision,
        "llm_advice": llm_advice,
        "items": items,
    }


def run(top_n: int = 50, pretty: bool = True) -> dict[str, Any]:
    source = create_data_source()
    stocks = source.get_stocks(top_n)
    payload = build_decision_payload(stocks, LlmAdvisor())
    payload["source"] = source.source_name
    if pretty:
        print_human_summary(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return payload


def print_human_summary(payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    print("=" * 80)
    print("同花顺热榜50 LLM 贝叶斯决策")
    print("=" * 80)
    print(f"时间: {payload['generated_at']}")
    print(f"数据源: {payload['source']}")
    print(f"候选数量: {payload['candidate_count']}")
    print(f"市场: {payload['market'].get('cycle_phase')} / {payload['market'].get('index_trend')} / 情绪 {payload['market'].get('sentiment_score')}")
    print("-" * 80)
    print(f"最终决策: {decision.get('decision')}")
    selected = decision.get("selected")
    if selected:
        print(f"推荐: {selected.get('rank')}. {selected.get('code')} {selected.get('name')}")
        print(f"贝叶斯概率: {decision.get('bayes_probability')}")
        print(f"仓位: {decision.get('position')}")
    print("理由:")
    for reason in decision.get("reason", []):
        print(f"- {reason}")
    print("风险:")
    for risk in decision.get("risks", []):
        print(f"- {risk}")
    print("放弃条件:")
    for condition in decision.get("cancel_conditions", []):
        print(f"- {condition}")
    if decision.get("decision") == "parse_error":
        print("-" * 80)
        print("LLM原文:")
        print(payload.get("llm_advice", {}).get("content", ""))
    print("-" * 80)
    print("前10候选:")
    for item in payload["items"][:10]:
        stock = item["stock"]
        print(
            f"{stock['hot_rank']:>2}. {stock['code']} {stock['name']} "
            f"涨跌 {stock['change_pct']:.2f}% 概率 {item['posterior_profit']:.2%} "
            f"{item['signal']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze THS hot rank top 50 with Bayesian evidence and LLM decision.")
    parser.add_argument("--top-n", type=int, default=50, help="Number of hot-rank stocks to analyze.")
    parser.add_argument("--json", action="store_true", help="Print full JSON only.")
    args = parser.parse_args()
    run(top_n=args.top_n, pretty=not args.json)


if __name__ == "__main__":
    main()
