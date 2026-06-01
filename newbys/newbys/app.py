from __future__ import annotations

import os
from dataclasses import fields

from flask import Flask, jsonify, request, send_from_directory

try:
    from flask_cors import CORS
except ModuleNotFoundError:
    CORS = None

from .data_source import DataSource, create_data_source
from .engine import SubjectiveBayesEngine
from .features import build_evidence_from_snapshot, infer_market_context
from .llm_advisor import LlmAdvisor
from .models import EvidenceInput, MarketContext, StockSnapshot


def create_app(data_source: DataSource | None = None, llm_advisor: LlmAdvisor | None = None) -> Flask:
    app = Flask(__name__, static_folder="../web", static_url_path="")
    if CORS:
        CORS(app)
    engine = SubjectiveBayesEngine()
    source = data_source or create_data_source()
    advisor = llm_advisor or LlmAdvisor()

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/status")
    def status():
        return jsonify({
            "status": "ok",
            "model": "subjective_bayes_v1",
            "source": source.source_name,
            "llm_configured": advisor.is_configured() if hasattr(advisor, "is_configured") else True,
        })

    @app.get("/api/analysis")
    def analysis():
        top_n = min(int(request.args.get("top_n", 5)), 5)
        stocks = source.get_stocks(top_n)
        market = infer_market_context(stocks)
        items = []
        for stock in stocks:
            evidence_input = build_evidence_from_snapshot(stock, market)
            result = engine.infer(evidence_input).to_dict()
            result["posterior_profit"] = result["posterior_profit"]
            items.append(result)
        market_dict = market.to_dict()
        llm_advice = advisor.analyze(market_dict, items)
        return jsonify({
            "model": "subjective_bayes_v1",
            "source": source.source_name,
            "market": market_dict,
            "items": items,
            "llm_advice": llm_advice,
        })

    @app.post("/api/infer")
    def infer_manual():
        payload = request.get_json(force=True)
        stock = _stock_from_payload(payload.get("stock", {}))
        market = _market_from_payload(payload.get("market", {}))
        evidence = payload.get("evidence", {})
        evidence_input = EvidenceInput(
            stock=stock,
            market=market,
            five_day_pattern=evidence.get("five_day_pattern", "sideways"),
            volume_pattern=evidence.get("volume_pattern", "neutral"),
            message_strength=evidence.get("message_strength", "medium"),
            concept_strength=evidence.get("concept_strength", "medium"),
            popularity=evidence.get("popularity", "top50"),
        )
        return jsonify(engine.infer(evidence_input).to_dict())

    return app


def _stock_from_payload(data: dict) -> StockSnapshot:
    allowed = {field.name for field in fields(StockSnapshot)}
    values = {key: value for key, value in data.items() if key in allowed}
    return StockSnapshot(
        code=str(values.get("code", "")),
        name=str(values.get("name", "")),
        price=float(values.get("price", 0)),
        change_pct=float(values.get("change_pct", 0)),
        turnover_rate=float(values.get("turnover_rate", 0)),
        volume_ratio=float(values.get("volume_ratio", 1)),
        amount=float(values.get("amount", 0)),
        hot_rank=int(values.get("hot_rank", 0)),
        concept_tags=list(values.get("concept_tags", [])),
    )


def _market_from_payload(data: dict) -> MarketContext:
    return MarketContext(
        cycle_phase=str(data.get("cycle_phase", "mixed")),
        index_trend=str(data.get("index_trend", "flat")),
        sentiment_score=float(data.get("sentiment_score", 0.5)),
        limit_up_count=int(data.get("limit_up_count", 0)),
        limit_down_count=int(data.get("limit_down_count", 0)),
    )


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5010"))
    app.run(host="0.0.0.0", port=port, debug=True)
