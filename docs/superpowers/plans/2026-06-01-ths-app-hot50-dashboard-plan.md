# 同花顺 App 热榜前50贝叶斯面板 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the real THS App hot-stock list shown in the screenshot and expose a front-end dashboard for hot 50 stocks, Bayesian decision process, and T+1 profit probability.

**Architecture:** Add a first-class `ThsAppHotRankDataSource` that calls the THS App hot-list endpoint and enriches rows with existing quote data. Extend API serialization with `decision_steps`. Replace the current garbled two-table page with a dense three-panel dashboard.

**Tech Stack:** Python, Flask, requests, pytest, plain HTML/CSS/JavaScript.

---

### Task 1: Real THS App Hot Rank Source

**Files:**
- Modify: `test_data_source.py`
- Modify: `bayesian_trader/data_source.py`

- [ ] Write failing tests for the THS App endpoint URL, params, hot rank order, concept tags, and popularity tags.
- [ ] Run `python -m pytest test_data_source.py -q` and verify the new test fails because `ThsAppHotRankDataSource` is missing.
- [ ] Implement `ThsAppHotRankDataSource` with URL `https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock` and params `stock_type=a&type=hour&list_type=normal`.
- [ ] Run `python -m pytest test_data_source.py -q` and verify it passes.

### Task 2: API Decision Process

**Files:**
- Modify: `test_api.py`
- Modify: `bayesian_trader/server.py`
- Modify: `bayesian_trader/data_source.py`

- [ ] Write failing API tests asserting `/api/hot_rank?top_n=50` includes `source=ths_app_hot_rank`, `concept_tags`, and `popularity_tag`.
- [ ] Write failing API tests asserting `/api/analysis?top_n=50` includes `decision_steps`.
- [ ] Run `python -m pytest test_api.py -q` and verify failure.
- [ ] Add decision-step serialization in `_signal_to_dict`.
- [ ] Make `create_data_source()` prefer `ThsAppHotRankDataSource`.
- [ ] Run `python -m pytest test_api.py test_data_source.py test_bayesian_engine.py -q`.

### Task 3: Frontend Dashboard

**Files:**
- Modify: `index.html`

- [ ] Replace the current garbled UI with a three-panel dashboard.
- [ ] Load `hot_rank?top_n=50&sort=hot_rank` and `analysis?top_n=50`.
- [ ] Render all 50 hot stocks, probability rows, and selected-stock decision steps.
- [ ] Keep mobile responsive layout with no overlapping text.

### Task 4: Verification

**Files:**
- Modify only if verification reveals a defect.

- [ ] Run `python -m py_compile bayesian_trader\data_source.py bayesian_trader\server.py bayesian_trader\models.py bayesian_trader\bayesian_engine.py`.
- [ ] Run `python -m pytest -q`.
- [ ] Start the server on a free port and verify `/api/status`, `/api/hot_rank?top_n=50`, and `/api/analysis?top_n=50`.
