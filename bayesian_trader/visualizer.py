#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化面板模块
用 matplotlib 生成贝叶斯策略分析图表
"""

import matplotlib
matplotlib.use("Agg")  # 非交互后端，兼容无GUI环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from typing import List, Optional
from pathlib import Path

from .models import BayesianSignal, MarketState

# ===== 中文字体配置 =====
# 尝试加载系统中的中文字体
_CN_FONTS = [
    "Microsoft YaHei",           # Win10/11
    "SimHei",                    # Windows
    "WenQuanYi Micro Hei",       # Linux
    "Noto Sans CJK SC",          # Linux/macOS
    "PingFang SC",               # macOS
    "Source Han Sans SC",        # 通用
]
_FONT_PROPS = None
for fname in _CN_FONTS:
    try:
        _FONT_PROPS = fm.FontProperties(family=fname)
        # 验证字体可用
        plt.text(0, 0, "测", fontproperties=_FONT_PROPS)
        plt.close("all")
        break
    except Exception:
        _FONT_PROPS = None

# 如果都没有找到，回退到英文
if _FONT_PROPS is None:
    plt.rcParams["font.family"] = "sans-serif"
else:
    plt.rcParams["font.family"] = _FONT_PROPS.get_name()

plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """可视化面板 - 生成贝叶斯策略分析图表"""

    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, name: str) -> str:
        """保存图表并返回路径"""
        path = self.output_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(path)

    def plot_probability_distribution(
        self,
        signals: List[BayesianSignal],
        scenario_name: str
    ) -> str:
        """
        概率分布直方图 + 累积分布曲线

        展示股票后验概率的分布形态
        """
        probs = [s.posterior_prob * 100 for s in signals]
        names = [s.stock.name for s in signals]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(
            f"后验概率分布 - {scenario_name}",
            fontsize=15, fontweight="bold", y=1.02
        )

        # --- 左图: 直方图 ---
        ax = axes[0]
        bins = np.arange(0, 101, 5)
        n, _, patches = ax.hist(probs, bins=bins, edgecolor="white",
                                color="#2196F3", alpha=0.75)
        # 按概率段着色: 低(红) 中(黄) 高(绿)
        for patch, bin_left in zip(patches, bins[:-1]):
            if bin_left < 40:
                patch.set_facecolor("#f44336")
            elif bin_left < 55:
                patch.set_facecolor("#FF9800")
            elif bin_left < 70:
                patch.set_facecolor("#FFC107")
            else:
                patch.set_facecolor("#4CAF50")

        ax.axvline(np.mean(probs), color="blue", linestyle="--",
                   linewidth=1.5, label=f"均值={np.mean(probs):.1f}%")
        ax.axvline(np.median(probs), color="red", linestyle=":",
                   linewidth=1.5, label=f"中位数={np.median(probs):.1f}%")
        ax.set_xlabel("后验概率 (%)", fontsize=11)
        ax.set_ylabel("股票数量", fontsize=11)
        ax.set_title("概率分布直方图", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # --- 右图: 概率排名 ---
        ax = axes[1]
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = np.array(probs)[sorted_indices]
        sorted_names = np.array(names)[sorted_indices]

        x = range(len(sorted_probs))
        colors = ["#4CAF50" if p >= 70 else "#FFC107" if p >= 55
                  else "#FF9800" if p >= 40 else "#f44336"
                  for p in sorted_probs]

        bars = ax.bar(x, sorted_probs, color=colors, edgecolor="white", width=0.7)
        ax.set_xlabel("股票排名", fontsize=11)
        ax.set_ylabel("后验概率 (%)", fontsize=11)
        ax.set_title("概率排名条形图", fontsize=12)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=7)
        ax.axhline(70, color="green", linestyle="--", linewidth=1, alpha=0.7, label="强买入(70%)")
        ax.axhline(55, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="买入(55%)")
        ax.axhline(40, color="red", linestyle="--", linewidth=1, alpha=0.7, label="关注(40%)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return self._save(fig, f"distribution_{scenario_name[:4]}.png")

    def plot_evidence_breakdown(
        self,
        signals: List[BayesianSignal],
        top_n: int = 5
    ) -> str:
        """
        证据因子分解堆叠图

        展示 Top N 股票的贝叶斯各因子贡献对比
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(
            "贝叶斯证据因子分解 (Top 5)",
            fontsize=15, fontweight="bold", y=1.02
        )

        top_signals = signals[:top_n]
        names = [s.stock.name for s in top_signals]

        # 提取各因子的条件概率 (赚钱条件下的似然)
        heat = [s.evidence_detail["heat_likelihood"] * 100 for s in top_signals]
        cycle = [s.evidence_detail["cycle_likelihood"] * 100 for s in top_signals]
        concept = [s.evidence_detail["concept_likelihood"] * 100 for s in top_signals]
        pattern = [s.evidence_detail["pattern_likelihood"] * 100 for s in top_signals]
        prior = [s.evidence_detail["prior"] * 100 for s in top_signals]
        posterior = [s.posterior_prob * 100 for s in top_signals]

        x = np.arange(len(names))
        width = 0.13

        # 分组柱状图
        ax.bar(x - 2*width, heat, width, label="P(热度|赚钱)", color="#2196F3", alpha=0.8)
        ax.bar(x - width, cycle, width, label="P(周期|赚钱)", color="#9C27B0", alpha=0.8)
        ax.bar(x, concept, width, label="P(概念|赚钱)", color="#FF9800", alpha=0.8)
        ax.bar(x + width, pattern, width, label="P(形态|赚钱)", color="#00BCD4", alpha=0.8)
        ax.bar(x + 2*width, posterior, width, label="后验概率", color="#4CAF50", alpha=0.9)

        # 标注先验概率线
        prior_val = prior[0]
        ax.axhline(prior_val, color="red", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"先验={prior_val:.0f}%")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("概率 (%)", fontsize=11)
        ax.set_title("各特征似然 vs 先验 vs 后验", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        return self._save(fig, "evidence_breakdown.png")

    def plot_scenario_comparison(self, scenario_results: List[dict]) -> str:
        """
        三场景对比雷达图 + 柱状图

        Args:
            scenario_results: 每个场景的统计数据
                [{"name", "avg_prob", "top3_avg", "strong_buy", "buy", "watch", "heat"}]
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 subplot_kw=dict(polar=True) if False else None)
        fig.suptitle(
            "三种市场场景对比分析",
            fontsize=15, fontweight="bold", y=1.02
        )

        # --- 左图: 多指标对比柱状图 ---
        ax1 = axes[0] if hasattr(axes, '__getitem__') else axes
        # 重新布局: 因为polar可能有兼容问题，改为两个柱状图
        fig.clear()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(
            "三种市场场景对比分析",
            fontsize=15, fontweight="bold", y=1.02
        )

        ax1 = axes[0]
        names = [r["name"][:6] for r in scenario_results]
        x = np.arange(len(names))

        metrics = [
            ("avg_prob", "平均概率 (%)", "#2196F3"),
            ("top3_avg", "Top3平均 (%)", "#4CAF50"),
            ("max_prob", "最高概率 (%)", "#FF9800"),
        ]

        width = 0.25
        for i, (key, label, color) in enumerate(metrics):
            values = [r[key] * 100 if key in ("avg_prob", "top3_avg", "max_prob")
                      else r[key] for r in scenario_results]
            ax1.bar(x + (i - 1) * width, values, width,
                    label=label, color=color, alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels(names, fontsize=10)
        ax1.set_ylabel("概率 (%)", fontsize=11)
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        # --- 右图: 信号数量对比 ---
        ax2 = axes[1]
        bottom = np.zeros(len(names))
        signal_types = [
            ("strong_buy", "强买入", "#4CAF50"),
            ("buy", "买入", "#FFC107"),
            ("watch", "关注", "#FF9800"),
        ]
        for key, label, color in signal_types:
            values = [r[key] for r in scenario_results]
            ax2.bar(x, values, 0.5, bottom=bottom, label=label, color=color, alpha=0.8)
            bottom += values

        ax2.set_xticks(x)
        ax2.set_xticklabels(names, fontsize=10)
        ax2.set_ylabel("股票数量", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_title("各场景信号分布", fontsize=12)

        # 在柱子上标数字
        for i, r in enumerate(scenario_results):
            total = r["strong_buy"] + r["buy"] + r["watch"]
            ax2.text(i, total + 0.3, str(total), ha="center", fontsize=10, fontweight="bold")

        plt.tight_layout()
        return self._save(fig, "scenario_comparison.png")

    def plot_heat_cycle_impact(self, engine) -> str:
        """
        参数敏感性分析: 热度/周期对后验概率的影响
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "参数敏感性分析",
            fontsize=15, fontweight="bold", y=1.02
        )

        # --- 左图: 市场热度影响 ---
        ax1 = axes[0]
        heat_values = np.linspace(0, 1, 50)
        # 固定其他参数为中等水平
        probs = []
        cycle_phases = [0, 2, 4]
        colors = ["#f44336", "#FFC107", "#4CAF50"]
        labels = ["退潮期(0)", "上升期(2)", "高潮期(4)"]

        for phase, color, label in zip(cycle_phases, colors, labels):
            probs_phase = []
            for h in heat_values:
                # 用简化的似然计算
                heat_profit = 0.4 * (0.5 + h)
                heat_loss = 0.4 * (1.5 - h)
                cycle_profit = [0.1, 0.2, 0.5, 0.7, 0.8][phase]
                cycle_loss = [0.5, 0.4, 0.3, 0.2, 0.1][phase]
                concept_profit = 0.5 * (0.5 + 0.6)
                concept_loss = 0.3 * (1.5 - 0.6)
                pattern_profit = 0.3 * (0.3 + 0.7 * 0.5)
                pattern_loss = 0.2 * (1.3 - 0.7 * 0.5)
                prior_p = 0.4
                prior_l = 0.6
                num_p = prior_p * heat_profit * cycle_profit * concept_profit * pattern_profit
                num_l = prior_l * heat_loss * cycle_loss * concept_loss * pattern_loss
                evidence = num_p + num_l
                posterior = num_p / evidence if evidence > 0 else prior_p
                probs_phase.append(posterior * 100)

            ax1.plot(heat_values, probs_phase, color=color,
                     label=label, linewidth=2)

        ax1.set_xlabel("市场热度评分", fontsize=11)
        ax1.set_ylabel("后验概率 (%)", fontsize=11)
        ax1.set_title("热度 vs 后验概率 (不同周期)", fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # --- 右图: 概念传播度影响 ---
        ax2 = axes[1]
        concept_values = np.linspace(0, 1, 50)
        heat_labels = ["冰点(0.1)", "偏冷(0.3)", "火热(0.9)"]
        heat_scores = [0.1, 0.3, 0.9]
        colors2 = ["#2196F3", "#FF9800", "#f44336"]

        for heat_s, color, label in zip(heat_scores, colors2, heat_labels):
            probs_c = []
            for c in concept_values:
                heat_profit = 0.4 * (0.5 + heat_s)
                heat_loss = 0.4 * (1.5 - heat_s)
                cycle_profit = 0.5
                cycle_loss = 0.3
                concept_profit = 0.5 * (0.5 + c)
                concept_loss = 0.3 * (1.5 - c)
                pattern_profit = 0.3 * (0.3 + 0.7 * 0.5)
                pattern_loss = 0.2 * (1.3 - 0.7 * 0.5)
                prior_p = 0.4
                prior_l = 0.6
                num_p = prior_p * heat_profit * cycle_profit * concept_profit * pattern_profit
                num_l = prior_l * heat_loss * cycle_loss * concept_loss * pattern_loss
                evidence = num_p + num_l
                posterior = num_p / evidence if evidence > 0 else prior_p
                probs_c.append(posterior * 100)

            ax2.plot(concept_values, probs_c, color=color,
                     label=label, linewidth=2)

        ax2.set_xlabel("概念传播度评分", fontsize=11)
        ax2.set_ylabel("后验概率 (%)", fontsize=11)
        ax2.set_title("概念传播度 vs 后验概率 (不同热度)", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return self._save(fig, "sensitivity_analysis.png")

    def generate_full_report(
        self,
        scenarios_data: list,
        engine
    ) -> List[str]:
        """
        生成完整可视化报告

        Args:
            scenarios_data: 每个场景的数据
                [{"name", "signals", "market"}]
            engine: BayesianEngine实例

        Returns:
            生成的图片路径列表
        """
        paths = []

        # 1. 逐场景概率分布图
        for sd in scenarios_data:
            p = self.plot_probability_distribution(sd["signals"], sd["name"])
            paths.append(p)

        # 2. 证据因子分解图 (用第一个场景的Top5)
        if scenarios_data:
            p = self.plot_evidence_breakdown(scenarios_data[0]["signals"])
            paths.append(p)

        # 3. 场景对比图
        scenario_results = []
        for sd in scenarios_data:
            signals = sd["signals"]
            strong_buy = sum(1 for s in signals if s.posterior_prob >= 0.70)
            buy = sum(1 for s in signals if 0.55 <= s.posterior_prob < 0.70)
            watch = sum(1 for s in signals if 0.40 <= s.posterior_prob < 0.55)
            avg_prob = np.mean([s.posterior_prob for s in signals])
            top3_avg = np.mean([s.posterior_prob for s in signals[:3]])
            max_prob = max(s.posterior_prob for s in signals)

            scenario_results.append({
                "name": sd["name"],
                "avg_prob": avg_prob,
                "top3_avg": top3_avg,
                "max_prob": max_prob,
                "strong_buy": strong_buy,
                "buy": buy,
                "watch": watch,
            })
        p = self.plot_scenario_comparison(scenario_results)
        paths.append(p)

        # 4. 参数敏感性分析
        p = self.plot_heat_cycle_impact(engine)
        paths.append(p)

        return paths


def print_report_index(paths: List[str]):
    """打印图表索引"""
    print(f"\n共生成 {len(paths)} 张图表:")
    print(f"{'='*60}")
    for p in paths:
        print(f"  {p}")
    print(f"{'='*60}\n")