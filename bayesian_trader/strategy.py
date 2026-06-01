"""
交易策略模块
基于贝叶斯信号生成交易决策，含回测框架
"""

import random
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from tabulate import tabulate

from .config import (
    STRONG_BUY_THRESHOLD, BUY_THRESHOLD, WATCH_THRESHOLD,
    HOLD_DAYS, STOP_LOSS, TAKE_PROFIT
)
from .models import StockInfo, MarketState, BayesianSignal, TradeRecord
from .bayesian_engine import BayesianEngine
from .feature_extractor import FeatureExtractor
from .database import Database


class TradingStrategy:
    """
    交易策略 - 根据贝叶斯信号生成交易决策
    """

    def __init__(self, engine: BayesianEngine, db: Optional[Database] = None):
        self.engine = engine
        self.db = db

    def generate_signals(
        self,
        stocks: List[StockInfo],
        market: MarketState
    ) -> List[BayesianSignal]:
        """
        生成所有股票的交易信号

        Args:
            stocks: 股票列表
            market: 市场状态

        Returns:
            排序后的信号列表
        """
        ranked_signals = self.engine.rank_stocks(stocks, market)

        # 保存到数据库
        if self.db:
            for signal in ranked_signals:
                self.db.save_signal(signal, market)

        return ranked_signals

    def get_trading_decisions(
        self,
        signals: List[BayesianSignal]
    ) -> Tuple[List[BayesianSignal], List[BayesianSignal], List[BayesianSignal]]:
        """
        根据信号强度分类交易决策

        Returns:
            (strong_buy, buy, watch) 三个级别的列表
        """
        strong_buy = []
        buy = []
        watch = []
        skip = []

        for s in signals:
            if s.posterior_prob >= STRONG_BUY_THRESHOLD:
                strong_buy.append(s)
            elif s.posterior_prob >= BUY_THRESHOLD:
                buy.append(s)
            elif s.posterior_prob >= WATCH_THRESHOLD:
                watch.append(s)
            else:
                skip.append(s)

        return strong_buy, buy, watch

    def execute_trade(
        self,
        signal: BayesianSignal,
        market: MarketState,
        date: Optional[str] = None
    ) -> Optional[TradeRecord]:
        """
        执行一笔模拟交易

        Args:
            signal: 贝叶斯信号
            market: 市场状态
            date: 交易日期

        Returns:
            TradeRecord 交易记录
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        record = TradeRecord(
            stock_code=signal.stock.code,
            stock_name=signal.stock.name,
            buy_date=date,
            buy_price=signal.stock.price,
            signal_prob=signal.posterior_prob,
            reason=f"贝叶斯概率={signal.posterior_prob:.2%}, "
                   f"信号={signal.signal_strength}, "
                   f"市场={market.heat}, 周期={market.cycle_phase_name}"
        )

        if self.db:
            self.db.save_trade_record(record)

        return record

    def simulate_sell(
        self,
        record: TradeRecord,
        days_held: int = 1,
        actual_return: Optional[float] = None
    ) -> TradeRecord:
        """
        模拟卖出（回测用）

        Args:
            record: 买入记录
            days_held: 持仓天数
            actual_return: 实际收益率（回测用）；None则随机模拟

        Returns:
            更新后的交易记录
        """
        sell_date = datetime.strptime(record.buy_date, "%Y-%m-%d")
        sell_date += timedelta(days=days_held)
        record.sell_date = sell_date.strftime("%Y-%m-%d")
        record.hold_days = days_held

        if actual_return is not None:
            # 回测用实际收益率
            record.profit_pct = round(actual_return, 4)
        else:
            # 模拟：基于买入概率随机生成收益率
            prob = record.signal_prob
            # 概率越高，赚钱概率越大，收益期望越高
            expected_return = (prob - 0.5) * 0.1  # 例如 prob=0.7 -> 期望2%收益
            std = 0.03  # 3%标准差
            record.profit_pct = round(
                random.gauss(expected_return, std), 4
            )

        # 判断是否触发止损/止盈
        if record.profit_pct <= STOP_LOSS:
            record.status = "止损"
            record.profit_pct = STOP_LOSS
        elif record.profit_pct >= TAKE_PROFIT:
            record.status = "止盈"
            record.profit_pct = TAKE_PROFIT
        elif record.profit_pct > 0:
            record.status = "盈利卖出"
        else:
            record.status = "亏损卖出"

        record.sell_price = round(
            record.buy_price * (1 + record.profit_pct), 2
        )

        if self.db:
            self.db.save_trade_record(record)

        return record


class BacktestEngine:
    """
    回测引擎 - 对贝叶斯策略进行历史回测
    """

    def __init__(self, engine: BayesianEngine, db: Optional[Database] = None):
        self.strategy = TradingStrategy(engine, db)
        self.engine = engine
        self.trade_records: List[TradeRecord] = []
        self.portfolio_value = 1_000_000  # 初始资金100万
        self.cash = self.portfolio_value
        self.positions = {}  # code -> (record, shares)

    def run_backtest(
        self,
        market_data: List[dict],
        top_n: int = 50
    ):
        """
        运行回测

        Args:
            market_data: 历史市场数据列表
                每项包含 {date, market_state, stocks}
            top_n: 每次选前N名
        """
        print(f"\n{'='*60}")
        print(f"  回测开始 | 初始资金: {self.portfolio_value:,.0f}元")
        print(f"{'='*60}\n")

        for day_idx, day_data in enumerate(market_data):
            date = day_data["date"]
            market_raw = day_data["market_state"]
            stocks_raw = day_data["stocks"]

            # 解析数据
            market = FeatureExtractor.create_market_from_raw(market_raw)
            stocks = [FeatureExtractor.create_stock_from_raw(s) for s in stocks_raw]

            # 生成信号
            signals = self.strategy.generate_signals(stocks, market)
            strong_buy, buy, watch = self.strategy.get_trading_decisions(signals)

            # 交易决策：取强买入的前top_n%进行模拟买入
            candidates = strong_buy[:max(1, len(strong_buy) // 3)]

            print(f"📅 {date} | {market.heat} | {market.cycle_phase_name} | "
                  f"信号: 强买入{len(strong_buy)} 买入{len(buy)} 关注{len(watch)}")

            # 模拟买入
            for signal in candidates[:5]:  # 每次最多买5只
                if self.cash < 10000:
                    break

                record = self.strategy.execute_trade(signal, market, date)
                # 每只票分配20%仓位
                position_value = min(self.cash * 0.2, self.portfolio_value * 0.2)
                shares = int(position_value / signal.stock.price / 100) * 100  # 整手买入
                if shares < 100:
                    continue

                cost = shares * signal.stock.price
                self.cash -= cost
                self.positions[signal.stock.code] = (record, shares, cost)

                print(f"   ✅ 买入 {signal.stock.name}({signal.stock.code}) "
                      f"概率={signal.posterior_prob:.1%} 仓位={cost:,.0f}元")

            # 模拟第二日卖出
            to_remove = []
            for code, (record, shares, cost) in self.positions.items():
                # 简单模拟：次日涨跌幅
                actual_return = random.gauss(0.005, 0.025)
                self.strategy.simulate_sell(record, days_held=1, actual_return=actual_return)

                # 计算收益
                sell_value = cost * (1 + actual_return)
                self.cash += sell_value
                self.trade_records.append(record)
                to_remove.append(code)

                # 只打印触发止损止盈的
                if record.status in ("止损", "止盈"):
                    print(f"   {record.status} {record.stock_name} "
                          f"收益={record.profit_pct:+.2%}")

            for code in to_remove:
                del self.positions[code]

        # 回测结果
        self._print_summary()

    def _print_summary(self):
        """打印回测总结"""
        if not self.trade_records:
            print("\n没有交易记录")
            return

        total_trades = len(self.trade_records)
        win_trades = [t for t in self.trade_records if t.profit_pct and t.profit_pct > 0]
        loss_trades = [t for t in self.trade_records if t.profit_pct and t.profit_pct <= 0]

        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
        avg_profit = sum(t.profit_pct or 0 for t in win_trades) / len(win_trades) if win_trades else 0
        avg_loss = sum(t.profit_pct or 0 for t in loss_trades) / len(loss_trades) if loss_trades else 0

        total_return = sum(t.profit_pct or 0 for t in self.trade_records)

        print(f"\n{'='*60}")
        print(f"  回测结果汇总")
        print(f"{'='*60}")
        print(f"  总交易次数: {total_trades}")
        print(f"  盈利次数: {len(win_trades)} | 亏损次数: {len(loss_trades)}")
        print(f"  胜率: {win_rate:.2%}")
        print(f"  平均盈利: {avg_profit:+.2%} | 平均亏损: {avg_loss:+.2%}")
        print(f"  总收益率之和: {total_return:+.2%}")
        print(f"  最终资产: {self.cash:,.0f}元")
        print(f"{'='*60}\n")