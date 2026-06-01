#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯短线交易系统 - Flask API 后端

提供RESTful API供前端面板调用
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict

# 确保可以导入bayesian_trader模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request
from flask_cors import CORS

from bayesian_trader import config as cfg
from bayesian_trader.models import StockInfo, MarketState
from bayesian_trader.database import Database
from bayesian_trader.bayesian_engine import BayesianEngine
from bayesian_trader.feature_extractor import FeatureExtractor
from bayesian_trader.strategy import TradingStrategy
from bayesian_trader.data_source import (
    create_data_source, RealTimeStock, ConceptBoard,
    convert_to_stock_info, fetch_and_convert, MockDataSource
)

# ============================================================
# 应用初始化
# ============================================================

app = Flask(__name__)
CORS(app)

db = Database(cfg.DB_PATH)
engine = BayesianEngine()
strategy = TradingStrategy(engine, db)

# 数据源（自动检测网络）
data_source = None


def get_data_source():
    """获取数据源（懒加载）"""
    global data_source
    if data_source is None:
        data_source = create_data_source()
    return data_source


# ============================================================
# 辅助函数
# ============================================================

def _infer_market_state(stock_infos: List[StockInfo]) -> MarketState:
    """从股票数据推断市场状态"""
    up_count = sum(1 for s in stock_infos if s.change_pct > 0)
    down_count = sum(1 for s in stock_infos if s.change_pct <= 0)
    n = max(len(stock_infos), 1)
    avg_turnover = sum(s.turnover_rate for s in stock_infos) / n
    avg_change = sum(s.change_pct for s in stock_infos) / n

    if avg_turnover > 8 and avg_change > 2:
        heat_name, heat_score, cycle_phase = "火热", 0.90, 3
    elif avg_turnover > 4 and avg_change > 0.5:
        heat_name, heat_score, cycle_phase = "偏暖", 0.70, 2
    elif avg_turnover > 2 and avg_change > -0.5:
        heat_name, heat_score, cycle_phase = "中性", 0.50, 1
    else:
        heat_name, heat_score, cycle_phase = "偏冷", 0.30, 1

    return MarketState(
        heat=heat_name, heat_score=heat_score, cycle_phase=cycle_phase,
        up_count=up_count, down_count=down_count,
        total_volume=sum(s.market_cap for s in stock_infos),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _signal_to_dict(signal) -> dict:
    """将BayesianSignal转为前端可用字典"""
    from bayesian_trader.models import BayesianSignal
    evidence = signal.evidence_detail
    risk_reasons = []
    stock = signal.stock
    if stock.change_pct >= 9:
        risk_reasons.append("涨幅接近涨停，T+1追高风险高")
    if stock.change_pct <= -7:
        risk_reasons.append("当日大跌，可能存在趋势破位")
    if stock.turnover_rate > 25:
        risk_reasons.append("换手率过高，筹码分歧较大")
    if stock.volume_ratio > 6:
        risk_reasons.append("量比过高，次日波动和滑点风险高")
    if signal.posterior_prob < 0.57:
        risk_reasons.append("校准后概率未达到买入阈值")

    if signal.posterior_prob >= 0.64 and len(risk_reasons) <= 1:
        risk_level = "medium"
        action = "轻仓试错"
        max_position_pct = 0.08
    elif signal.posterior_prob >= 0.57 and len(risk_reasons) <= 2:
        risk_level = "medium"
        action = "观察或小仓"
        max_position_pct = 0.05
    elif signal.posterior_prob >= 0.50:
        risk_level = "high"
        action = "观察"
        max_position_pct = 0.02
    else:
        risk_level = "high"
        action = "回避"
        max_position_pct = 0.0

    if not risk_reasons:
        risk_reasons.append("无极端量价风险，但仍需按热度股纪律交易")

    risk_control = {
        "risk_level": risk_level,
        "action": action,
        "max_position_pct": max_position_pct,
        "stop_loss_pct": -0.05,
        "take_profit_pct": 0.08,
        "reasons": risk_reasons,
    }
    decision_steps = [
        {"label": "赚钱先验", "profit": evidence.get("prior", 0), "loss": round(1 - evidence.get("prior", 0), 4)},
        {"label": "市场热度", "profit": evidence.get("heat_likelihood", 0), "loss": evidence.get("heat_likelihood_loss", 0)},
        {"label": "市场周期", "profit": evidence.get("cycle_likelihood", 0), "loss": evidence.get("cycle_likelihood_loss", 0)},
        {"label": "热榜排名", "profit": evidence.get("hot_rank_likelihood", 0), "loss": evidence.get("hot_rank_likelihood_loss", 0)},
        {"label": "概念传播", "profit": evidence.get("concept_likelihood", 0), "loss": evidence.get("concept_likelihood_loss", 0)},
        {"label": "K线形态", "profit": evidence.get("pattern_likelihood", 0), "loss": evidence.get("pattern_likelihood_loss", 0)},
        {"label": "量价活跃", "profit": evidence.get("volume_activity_likelihood", 0), "loss": evidence.get("volume_activity_likelihood_loss", 0)},
        {"label": "T+1后验", "profit": signal.posterior_prob, "loss": round(1 - signal.posterior_prob, 4)},
    ]
    return {
        "code": signal.stock.code,
        "name": signal.stock.name,
        "price": signal.stock.price,
        "change_pct": signal.stock.change_pct,
        "turnover_rate": signal.stock.turnover_rate,
        "volume_ratio": signal.stock.volume_ratio,
        "hot_rank": signal.stock.hot_rank,
        "hot_rank_score": signal.stock.hot_rank_score,
        "volume_activity_score": signal.stock.volume_activity_score,
        "buy_prob": signal.posterior_prob,
        "sell_prob": 1 - signal.posterior_prob,
        "concept_spread": signal.stock.concept_spread,
        "concept_tags": getattr(signal.stock, "concept_tags", []),
        "popularity_tag": getattr(signal.stock, "popularity_tag", ""),
        "candle_pattern": signal.stock.candle_pattern,
        "candle_score": signal.stock.candle_score,
        "signal": signal.signal_strength,
        "signal_score": signal.rank_score,
        "evidence_detail": signal.evidence_detail,
        "decision_steps": decision_steps,
        "risk_control": risk_control,
        "model_notice": "概率已做保守校准；未经过2-3年历史热榜样本回测前，只能作为短线决策辅助，不代表确定收益。",
        "features": {
            "market_cap": signal.stock.market_cap,
            "hot_rank": signal.stock.hot_rank,
            "hot_rank_score": signal.stock.hot_rank_score,
            "concept_tags": getattr(signal.stock, "concept_tags", []),
            "popularity_tag": getattr(signal.stock, "popularity_tag", ""),
            "concept_spread": signal.stock.concept_spread,
            "candle_pattern": signal.stock.candle_pattern,
            "candle_score": signal.stock.candle_score,
            "volume_activity_score": signal.stock.volume_activity_score,
        },
        "reason": f"贝叶斯概率={signal.posterior_prob:.2%}, {signal.signal_strength}",
    }


# ============================================================
# 静态页面
# ============================================================

@app.route("/")
def index():
    """提供前端页面"""
    from flask import send_file
    return send_file(os.path.join(os.path.dirname(__file__), "..", "index.html"))


# ============================================================
# API 端点
# ============================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    """服务器状态"""
    ds_type = type(get_data_source()).__name__
    return jsonify({
        "status": "ok",
        "message": "贝叶斯短线交易系统运行中",
        "version": "1.0.0",
        "data_source": ds_type,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/hot_rank", methods=["GET"])
def api_hot_rank():
    """
    获取热门排名（按换手率排序）

    Query params:
        top_n: int = 50 - 返回前N只
        sort: str = "turnover" - 排序方式 (turnover, change, amount)
    """
    top_n = int(request.args.get("top_n", 50))
    sort_by = request.args.get("sort", "turnover")

    ds = get_data_source()
    try:
        stocks, ts = ds.get_stock_list(top_n=top_n, sort_by=sort_by)
        source = getattr(ds, "source_name", type(ds).__name__)
        result = []
        for i, s in enumerate(stocks, 1):
            hot_rank = int(getattr(s, "hot_rank", i) or i)
            result.append({
                "rank": hot_rank,
                "display_rank": i,
                "code": s.code,
                "name": s.name,
                "price": s.price,
                "change_pct": s.change_pct,
                "change_amt": s.change_amt,
                "turnover_rate": s.turnover_rate,
                "volume_ratio": s.volume_ratio,
                "amount": s.amount,
                "volume": s.volume,
                "high": s.high,
                "low": s.low,
                "pe": s.pe,
                "market_cap": round(s.market_cap / 1e8, 2) if s.market_cap > 0 else 0,
                "hot_rank": hot_rank,
                "hot_score": float(getattr(s, "hot_score", 0.0) or 0.0),
                "concept_tags": getattr(s, "concept_tags", []),
                "popularity_tag": getattr(s, "popularity_tag", ""),
            })
        return jsonify({"stocks": result, "timestamp": ts, "total": len(result), "source": source})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/market_overview", methods=["GET"])
def api_market_overview():
    """市场概况"""
    ds = get_data_source()
    try:
        overview = ds.get_market_overview()
        return jsonify(overview)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/concept_boards", methods=["GET"])
def api_concept_boards():
    """概念板块"""
    ds = get_data_source()
    try:
        boards = ds.get_concept_boards()
        return jsonify({
            "boards": [
                {"code": b.code, "name": b.name, "change_pct": b.change_pct,
                 "up_count": b.up_count, "total_count": b.total_count}
                for b in boards
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analysis", methods=["GET"])
def api_analysis():
    """
    获取贝叶斯分析结果

    Query params:
        top_n: int = 20 - 分析前N只热门股
        mock: bool = False - 是否使用模拟数据
    """
    top_n = int(request.args.get("top_n", 20))
    force_mock = request.args.get("mock", "false").lower() == "true"

    try:
        if force_mock:
            ds = MockDataSource()
            stocks_raw, ts = ds.get_stock_list(top_n=top_n)
        else:
            ds = get_data_source()
            stocks_raw, ts = ds.get_stock_list(top_n=top_n)
        source = getattr(ds, "source_name", type(ds).__name__)

        if not stocks_raw:
            return jsonify({"error": "无法获取股票数据"}), 500

        # 转换为贝叶斯模型
        stock_infos = [convert_to_stock_info(s) for s in stocks_raw]

        # 推断市场状态
        market_state = _infer_market_state(stock_infos)

        # 贝叶斯推理排名
        signals = engine.rank_stocks(stock_infos, market_state)

        # 获取交易决策
        strong_buy, buy, watch = strategy.get_trading_decisions(signals)

        # 格式化为前端可用
        analysis_results = [_signal_to_dict(s) for s in signals]

        # 策略评分摘要（按信号强度排序）
        buy_list = [_signal_to_dict(s) for s in strong_buy[:5]]
        watch_list = [_signal_to_dict(s) for s in buy[:5]]
        monitor_list = [_signal_to_dict(s) for s in watch[:5]]

        return jsonify({
            "analysis": analysis_results[:top_n],
            "market_state": {
                "heat": market_state.heat,
                "heat_score": market_state.heat_score,
                "cycle_phase": market_state.cycle_phase,
                "up_count": market_state.up_count,
                "down_count": market_state.down_count,
                "total_volume": market_state.total_volume,
                "avg_turnover": sum(s.turnover_rate for s in stock_infos) / max(len(stock_infos), 1),
                "avg_change": sum(s.change_pct for s in stock_infos) / max(len(stock_infos), 1),
            },
            "strategies": [
                {"name": "强买入", "count": len(strong_buy), "score": 0.85, "desc": "后验概率≥70%"},
                {"name": "买入", "count": len(buy), "score": 0.65, "desc": "后验概率≥55%"},
                {"name": "关注", "count": len(watch), "score": 0.45, "desc": "后验概率≥40%"},
            ],
            "buy_signals": buy_list,
            "sell_signals": watch_list,
            "timestamp": ts,
            "source": source,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/<code>", methods=["GET"])
def api_stock_detail(code: str):
    """单只股票详细分析"""
    ds = get_data_source()
    try:
        stocks, _ = ds.get_stock_list(top_n=100, custom_pool=[code])
        if not stocks:
            return jsonify({"error": f"股票 {code} 未找到"}), 404

        stock = stocks[0]
        stock_info = convert_to_stock_info(stock)

        # 生成市场状态
        market = MarketState(
            heat="中性", heat_score=0.5, cycle_phase=1,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        signals = engine.rank_stocks([stock_info], market)

        if not signals:
            return jsonify({"error": "分析失败"}), 500

        s = signals[0]
        return jsonify({
            "code": stock.code,
            "name": stock.name,
            "price": stock.price,
            "change_pct": stock.change_pct,
            "change_amt": stock.change_amt,
            "turnover_rate": stock.turnover_rate,
            "volume_ratio": stock.volume_ratio,
            "amount": stock.amount,
            "volume": stock.volume,
            "high": stock.high,
            "low": stock.low,
            "pre_close": stock.pre_close,
            "open": stock.open_px,
            "pe": stock.pe,
            "market_cap": round(stock.market_cap / 1e8, 2) if stock.market_cap > 0 else 0,
            "analysis": {
                "buy_prob": s.posterior_prob,
                "sell_prob": 1 - s.posterior_prob,
                "candle_pattern": s.stock.candle_pattern,
                "candle_score": s.stock.candle_score,
                "concept_spread": s.stock.concept_spread,
                "signal": s.signal_strength,
                "signal_score": s.rank_score,
                "reason": f"贝叶斯概率={s.posterior_prob:.2%}, {s.signal_strength}",
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trade_log", methods=["GET"])
def api_trade_log():
    """获取交易记录"""
    try:
        logs = db.get_trade_logs()
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trade_log", methods=["POST"])
def api_add_trade_log():
    """添加交易记录"""
    try:
        data = request.get_json()
        db.add_trade_log(
            code=data["code"],
            name=data["name"],
            action=data["action"],
            price=float(data["price"]),
            shares=float(data.get("shares", 0)),
            reason=data.get("reason", ""),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/demo", methods=["GET"])
def api_demo():
    """
    快速演示：运行完整分析流程并返回结果
    """
    try:
        from bayesian_trader.demo import run_demo
        result = run_demo(top_n=15, save_charts=False)
        return jsonify({
            "status": "ok",
            "message": "演示运行完成",
            "result": result,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def api_search():
    """
    搜索股票

    Query params:
        q: str - 搜索关键词（代码或名称）
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"stocks": []})

    ds = get_data_source()
    try:
        stocks, _ = ds.get_stock_list(top_n=100)
        matches = []
        for s in stocks:
            if q.upper() in s.code or q in s.name:
                matches.append({
                    "code": s.code,
                    "name": s.name,
                    "price": s.price,
                    "change_pct": s.change_pct,
                })
        return jsonify({"stocks": matches[:20]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 启动服务
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"

    print("=" * 60)
    print("贝叶斯短线交易系统 - API 服务")
    print("=" * 60)
    print(f"  监听端口: {port}")
    print(f"  Debug模式: {debug}")
    print(f"  数据源: {type(get_data_source()).__name__}")
    print()
    print("  接口列表:")
    print("    GET /api/status           - 服务状态")
    print("    GET /api/hot_rank         - 热门排名")
    print("    GET /api/market_overview  - 市场概况")
    print("    GET /api/concept_boards   - 概念板块")
    print("    GET /api/analysis         - 贝叶斯分析结果")
    print("    GET /api/stock/<code>     - 个股详情")
    print("    GET /api/trade_log        - 交易记录")
    print("    POST /api/trade_log       - 添加交易记录")
    print("    GET /api/demo             - 运行演示")
    print("    GET /api/search           - 搜索股票")
    print("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=debug)
