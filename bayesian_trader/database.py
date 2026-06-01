"""
数据库模块 - SQLite数据持久化
存储交易信号、持仓记录、市场状态等历史数据
"""

import sqlite3
import json
from typing import List, Optional
from datetime import datetime
from .config import DB_PATH
from .models import StockInfo, MarketState, BayesianSignal, TradeRecord


class Database:
    """SQLite数据库管理器"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """初始化数据库表"""
        cursor = self.conn.cursor()

        # 股票快照表 - 记录每日热门前50数据
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                name TEXT NOT NULL,
                price REAL,
                change_pct REAL,
                turnover_rate REAL,
                volume_ratio REAL,
                market_cap REAL,
                concept_spread REAL,
                candle_pattern TEXT,
                candle_score REAL,
                rank INTEGER,
                timestamp TEXT NOT NULL
            )
        """)

        # 市场状态表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                heat TEXT,
                heat_score REAL,
                cycle_phase INTEGER,
                cycle_phase_name TEXT,
                up_count INTEGER,
                down_count INTEGER,
                limit_up_count INTEGER,
                limit_down_count INTEGER,
                total_volume REAL,
                timestamp TEXT NOT NULL
            )
        """)

        # 贝叶斯信号表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bayesian_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                name TEXT NOT NULL,
                posterior_prob REAL,
                signal_strength TEXT,
                evidence_detail TEXT,
                rank_score REAL,
                market_heat TEXT,
                cycle_phase INTEGER,
                timestamp TEXT NOT NULL
            )
        """)

        # 交易记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                buy_date TEXT,
                buy_price REAL,
                signal_prob REAL,
                sell_date TEXT,
                sell_price REAL,
                profit_pct REAL,
                hold_days INTEGER,
                status TEXT,
                reason TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        # 贝叶斯模型参数表 - 存储可学习的参数
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                param_name TEXT UNIQUE NOT NULL,
                param_value REAL NOT NULL,
                update_time TEXT NOT NULL
            )
        """)

        self.conn.commit()

    # ==================== 数据插入 ====================

    def save_stock_snapshot(self, stock: StockInfo):
        """保存股票快照"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO stock_snapshots 
            (code, name, price, change_pct, turnover_rate, volume_ratio, 
             market_cap, concept_spread, candle_pattern, candle_score, rank, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stock.code, stock.name, stock.price, stock.change_pct,
            stock.turnover_rate, stock.volume_ratio, stock.market_cap,
            stock.concept_spread, stock.candle_pattern, stock.candle_score,
            stock.rank, stock.timestamp
        ))
        self.conn.commit()

    def save_market_state(self, market: MarketState):
        """保存市场状态"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_states
            (heat, heat_score, cycle_phase, cycle_phase_name, up_count, down_count,
             limit_up_count, limit_down_count, total_volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market.heat, market.heat_score, market.cycle_phase,
            market.cycle_phase_name, market.up_count, market.down_count,
            market.limit_up_count, market.limit_down_count,
            market.total_volume, market.timestamp
        ))
        self.conn.commit()

    def save_signal(self, signal: BayesianSignal, market_state: MarketState):
        """保存贝叶斯信号"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO bayesian_signals
            (code, name, posterior_prob, signal_strength, evidence_detail,
             rank_score, market_heat, cycle_phase, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.stock.code, signal.stock.name, signal.posterior_prob,
            signal.signal_strength, json.dumps(signal.evidence_detail, ensure_ascii=False),
            signal.rank_score, market_state.heat, market_state.cycle_phase,
            signal.stock.timestamp
        ))
        self.conn.commit()

    def save_trade_record(self, record: TradeRecord):
        """保存交易记录"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trade_records
            (stock_code, stock_name, buy_date, buy_price, signal_prob,
             sell_date, sell_price, profit_pct, hold_days, status, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.stock_code, record.stock_name, record.buy_date,
            record.buy_price, record.signal_prob, record.sell_date,
            record.sell_price, record.profit_pct, record.hold_days,
            record.status, record.reason, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        self.conn.commit()

    def update_trade_record(self, record_id: int, **kwargs):
        """更新交易记录"""
        allowed_fields = {"sell_date", "sell_price", "profit_pct", "hold_days", "status", "reason"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [record_id]
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE trade_records SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    # ==================== 参数管理 ====================

    def save_param(self, name: str, value: float):
        """保存模型参数"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO model_params (param_name, param_value, update_time)
            VALUES (?, ?, ?)
            ON CONFLICT(param_name) DO UPDATE SET
                param_value = excluded.param_value,
                update_time = excluded.update_time
        """, (name, value, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    def get_param(self, name: str, default: float = 0.4) -> float:
        """获取模型参数"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT param_value FROM model_params WHERE param_name = ?", (name,))
        row = cursor.fetchone()
        return row["param_value"] if row else default

    # ==================== 数据查询 ====================

    def get_latest_signals(self, limit: int = 50) -> List[dict]:
        """获取最新信号"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM bayesian_signals 
            ORDER BY timestamp DESC, rank_score DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_trade_history(self, limit: int = 50) -> List[dict]:
        """获取交易历史"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trade_records 
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_signal_stats(self) -> dict:
        """获取信号统计"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN signal_strength = '强买入' THEN 1 ELSE 0 END) as strong_buy,
                SUM(CASE WHEN signal_strength = '买入' THEN 1 ELSE 0 END) as buy,
                SUM(CASE WHEN signal_strength = '关注' THEN 1 ELSE 0 END) as watch,
                AVG(posterior_prob) as avg_prob
            FROM bayesian_signals
        """)
        return dict(cursor.fetchone())

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()