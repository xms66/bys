#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯短线交易策略 - 演示入口

展示三种市场场景下的贝叶斯推理结果:
1. 主升期-题材炒作热火朝天
2. 退潮期-高位股大分歧
3. 混沌期-板块轮动快
"""

import json
import os
import sys
from typing import List, Dict
from datetime import datetime

# 确保能导入 bayesian_trader 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesian_trader.bayesian_engine import BayesianEngine
from bayesian_trader.feature_extractor import FeatureExtractor
from bayesian_trader.strategy import TradingStrategy
from bayesian_trader.models import StockInfo, MarketState, BayesianSignal
from bayesian_trader.visualizer import Visualizer, print_report_index


def load_test_data() -> dict:
    """加载测试数据"""
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "test_data.json"
    )
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_header(title: str):
    """打印分隔标题"""
    width = 70
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_market_state(market: MarketState):
    """打印市场状态"""
    print(f"\n📊 市场状态:")
    print(f"   ├ 热度: {market.heat} ({market.heat_score:.2f})")
    print(f"   ├ 周期: {market.cycle_phase_name} (阶段{market.cycle_phase})")
    print(f"   ├ 上涨/下跌: {market.up_count}/{market.down_count}")
    print(f"   ├ 涨停/跌停: {market.limit_up_count}/{market.limit_down_count}")
    print(f"   └ 总成交额: {market.total_volume:.0f}亿")


def print_signal_detail(signal: BayesianSignal, rank: int):
    """打印单个信号的详细信息"""
    stock = signal.stock
    ed = signal.evidence_detail

    print(f"\n  #{rank} {stock.name}({stock.code})")
    print(f"   ├ 价格: {stock.price:.2f} | 涨幅: {stock.change_pct:+.1f}%")
    print(f"   ├ 换手率: {stock.turnover_rate:.1f}% | 量比: {stock.volume_ratio:.1f}")
    print(f"   ├ 流通市值: {stock.market_cap:.0f}亿")
    print(f"   ├ 概念传播度: {stock.concept_spread:.2f}")
    print(f"   ├ K线形态: {stock.candle_pattern} (评分: {stock.candle_score:.2f})")
    print(f"   ├── 贝叶斯推理过程 ──")
    print(f"   ├  先验概率 P(赚钱) = {ed['prior']:.2%}")
    print(f"   ├  P(热度|赚钱)   = {ed['heat_likelihood']:.2%}")
    print(f"   ├  P(周期|赚钱)   = {ed['cycle_likelihood']:.2%}")
    print(f"   ├  P(概念|赚钱)   = {ed['concept_likelihood']:.2%}")
    print(f"   ├  P(形态|赚钱)   = {ed['pattern_likelihood']:.2%}")
    print(f"   ├  P(热度|亏钱)   = {ed['heat_likelihood_loss']:.2%}")
    print(f"   ├  P(周期|亏钱)   = {ed['cycle_likelihood_loss']:.2%}")
    print(f"   ├  P(概念|亏钱)   = {ed['concept_likelihood_loss']:.2%}")
    print(f"   ├  P(形态|亏钱)   = {ed['pattern_likelihood_loss']:.2%}")
    print(f"   ├  分子(赚钱)     = {ed['numerator_profit']:.5f}")
    print(f"   ├  分子(亏钱)     = {ed['numerator_loss']:.5f}")
    print(f"   ├  证据因子       = {ed['evidence']:.5f}")
    print(f"   ├── 结果 ──")
    print(f"   ├  后验概率: {signal.posterior_prob:.2%}")
    print(f"   ├  综合排名分: {signal.rank_score:.4f}")
    print(f"   └  信号强度: {signal.signal_strength}")


def run_scenario_analysis(scenario_name: str, scenario_data: dict, engine: BayesianEngine):
    """运行单个场景分析"""
    print_header(f"📈 场景: {scenario_name}")
    print(f"  {scenario_data['description']}")

    # 解析数据
    market = FeatureExtractor.create_market_from_raw(scenario_data["market_state"])
    stocks = [
        FeatureExtractor.create_stock_from_raw(s)
        for s in scenario_data["stocks"]
    ]

    # 打印市场状态
    print_market_state(market)

    # 运行贝叶斯推理
    strategy = TradingStrategy(engine)
    signals = strategy.generate_signals(stocks, market)
    strong_buy, buy, watch = strategy.get_trading_decisions(signals)

    # 打印信号统计
    print(f"\n📋 信号统计:")
    print(f"   ├ 强买入: {len(strong_buy)} 只")
    print(f"   ├ 买入:   {len(buy)} 只")
    print(f"   ├ 关注:   {len(watch)} 只")
    print(f"   └ 观望:   {len(signals) - len(strong_buy) - len(buy) - len(watch)} 只")

    # 打印排名前5的详细推理过程
    print(f"\n🔍 Top 5 股票详细分析:")
    for i, signal in enumerate(signals[:5]):
        print_signal_detail(signal, i + 1)

    # 打印排名后3的
    print(f"\n⏬ 排名后3的股票:")
    for i, signal in enumerate(signals[-3:]):
        stock = signal.stock
        print(f"  #{len(signals)-2+i} {stock.name}({stock.code}) "
              f"后验概率={signal.posterior_prob:.2%} "
              f"信号={signal.signal_strength}")

    # 贝叶斯概率分布
    probs = [s.posterior_prob for s in signals]
    print(f"\n📊 概率分布统计:")
    print(f"   ├ 最高概率: {max(probs):.2%}")
    print(f"   ├ 最低概率: {min(probs):.2%}")
    print(f"   ├ 平均概率: {sum(probs)/len(probs):.2%}")
    print(f"   └ 中位数:  {sorted(probs)[len(probs)//2]:.2%}")


def scenario_comparison(scenarios: dict, engine: BayesianEngine):
    """三种场景对比分析"""
    print_header("🔄 三场景对比分析")

    results = []
    for scenario in scenarios["scenarios"]:
        market = FeatureExtractor.create_market_from_raw(scenario["market_state"])
        stocks = [
            FeatureExtractor.create_stock_from_raw(s)
            for s in scenario["stocks"]
        ]

        signals = engine.rank_stocks(stocks, market)
        strong_buy, buy, watch = [], [], []
        for s in signals:
            if s.posterior_prob >= 0.70:
                strong_buy.append(s)
            elif s.posterior_prob >= 0.55:
                buy.append(s)
            elif s.posterior_prob >= 0.40:
                watch.append(s)

        avg_prob = sum(s.posterior_prob for s in signals) / len(signals)
        top3_avg = sum(s.posterior_prob for s in signals[:3]) / 3

        results.append({
            "name": scenario["name"],
            "heat": market.heat,
            "cycle": market.cycle_phase_name,
            "strong_buy": len(strong_buy),
            "buy": len(buy),
            "watch": len(watch),
            "avg_prob": avg_prob,
            "top3_avg": top3_avg,
            "max_prob": max(s.posterior_prob for s in signals),
        })

    # 打印对比表
    print(f"\n{'场景':<25} {'热度':<6} {'周期':<6} {'强买':<6} {'买入':<6} {'关注':<6} {'平均概率':<10} {'Top3平均':<10}")
    print(f"{'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*10}")
    for r in results:
        print(f"{r['name']:<25} {r['heat']:<6} {r['cycle']:<6} "
              f"{r['strong_buy']:<6} {r['buy']:<6} {r['watch']:<6} "
              f"{r['avg_prob']:<10.2%} {r['top3_avg']:<10.2%}")

    print(f"\n💡 分析结论:")
    print(f"  • 主升期强买入信号最多，概率集中度高，适合积极操作")
    print(f"  • 退潮期几乎无买入信号，概率集中在低区间，应空仓观望")
    print(f"  • 混沌期信号分散，需精选个股，控制仓位")


def main():
    """主入口"""
    print_header("🎯 贝叶斯短线交易策略系统 v1.0")
    print(f"  基于朴素贝叶斯定理的短线交易概率推理框架")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  核心公式:")
    print(f"  P(赚钱|证据) = P(赚钱)×P(热度|赚钱)×P(周期|赚钱)×P(概念|赚钱)×P(形态|赚钱)")
    print(f"                ────────────────────────────────────────────────────────")
    print(f"                P(赚钱)×∏P(eᵢ|赚钱) + P(亏钱)×∏P(eᵢ|亏钱)")

    # 加载数据
    data = load_test_data()

    # 初始化贝叶斯引擎和可视化器
    engine = BayesianEngine()
    viz = Visualizer()
    print(f"\n⚙️  模型参数:")
    print(f"   ├ 先验概率 P(赚钱) = {engine.prior_profit:.0%}")
    print(f"   ├ 特征数量: 4 (市场热度、周期阶段、概念传播度、K线形态)")
    print(f"   └ 信号阈值: 强买入≥70% | 买入≥55% | 关注≥40%")

    # 逐个场景分析 + 收集可视化数据
    viz_scenarios = []
    for scenario in data["scenarios"]:
        market = FeatureExtractor.create_market_from_raw(scenario["market_state"])
        stocks = [FeatureExtractor.create_stock_from_raw(s) for s in scenario["stocks"]]
        signals = engine.rank_stocks(stocks, market)
        viz_scenarios.append({
            "name": scenario["name"],
            "signals": signals,
            "market": market,
        })
        run_scenario_analysis(scenario["name"], scenario, engine)

    # 场景对比
    scenario_comparison(data, engine)

    # ===== 生成可视化图表 =====
    print_header("📊 生成可视化报告")
    chart_paths = viz.generate_full_report(viz_scenarios, engine)
    print_report_index(chart_paths)

    print_header("✅ 演示完成")
    print(f"  项目代码结构:")
    print(f"  ├ bayesian_trader/")
    print(f"  │  ├ config.py           - 配置参数")
    print(f"  │  ├ models.py           - 数据模型")
    print(f"  │  ├ database.py         - SQLite数据库")
    print(f"  │  ├ bayesian_engine.py  - 贝叶斯推理引擎(核心)")
    print(f"  │  ├ feature_extractor.py- 特征提取器")
    print(f"  │  ├ strategy.py         - 交易策略+回测引擎")
    print(f"  │  ├ demo.py             - 演示入口")
    print(f"  │  └ data/test_data.json - 模拟测试数据")
    print(f"  └ requirements.txt       - Python依赖")


if __name__ == "__main__":
    main()