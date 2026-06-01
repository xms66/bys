# 同花顺热度前50贝叶斯策略 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simulated hot stock pool with a real THS hot-stock source and compute explainable T+1 profit probabilities for the hot 50.

**Architecture:** Add a THS hot-rank provider that reads TickerLab-compatible data, then enriches ranked stocks with existing quote parsing. Split posterior probability from trading rank score by adding explicit evidence features and likelihoods. Keep offline mock mode visible, never labeled as real THS data.

**Tech Stack:** Python, Flask, requests, pytest, existing dataclass models.

---

### Task 1: Fix test collection and add THS source contract tests

**Files:**
- Modify: `test_data_source.py`
- Modify: `bayesian_trader/data_source.py`

- [ ] **Step 1: Write failing tests**

Create pytest tests in `test_data_source.py` that assert:

```python
from bayesian_trader.data_source import ThsHotRankDataSource


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("bad status")


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return FakeResponse(self.payload)


def test_ths_hot_rank_preserves_external_rank_and_code():
    payload = {
        "data": [
            {"rank": 1, "code": "300033", "name": "同花顺", "heat": 9988},
            {"rank": 2, "stock_code": "000001.SZ", "stock_name": "平安银行", "hot": 9000},
        ]
    }
    session = FakeSession(payload)
    ds = ThsHotRankDataSource(api_key="k", session=session)

    entries = ds.fetch_hot_rank_entries(2)

    assert [e.rank for e in entries] == [1, 2]
    assert [e.code for e in entries] == ["300033", "000001"]
    assert [e.name for e in entries] == ["同花顺", "平安银行"]
    assert entries[0].heat_score == 9988


def test_ths_hot_rank_requires_api_key_for_real_source():
    ds = ThsHotRankDataSource(api_key="")

    try:
        ds.fetch_hot_rank_entries(50)
    except RuntimeError as exc:
        assert "TICKERLAB_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing API key error")
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest test_data_source.py -q`

Expected: FAIL because `ThsHotRankDataSource` does not exist.

- [ ] **Step 3: Implement minimal THS hot-rank source**

Add `HotRankEntry` and `ThsHotRankDataSource` to `bayesian_trader/data_source.py`. It should call `https://api.tickerlab.org/v1/ranking/hot-stock`, pass `limit`, accept Bearer or `X-API-Key` auth, normalize code/name/rank/heat fields, and raise a clear error without an API key.

- [ ] **Step 4: Verify tests pass**

Run: `python -m pytest test_data_source.py -q`

Expected: PASS.

### Task 2: Add evidence features to Bayesian analysis

**Files:**
- Modify: `bayesian_trader/models.py`
- Modify: `bayesian_trader/bayesian_engine.py`
- Test: `test_bayesian_engine.py`

- [ ] **Step 1: Write failing tests**

Create tests showing hot rank, concept spread, candle score, and market state change posterior probability, while rank score remains separate from posterior.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest test_bayesian_engine.py -q`

Expected: FAIL because hot-rank evidence does not exist.

- [ ] **Step 3: Implement minimal model changes**

Add `hot_rank`, `hot_rank_score`, and `volume_activity_score` to `StockInfo`. Add likelihood functions for hot rank and volume activity in `BayesianEngine.compute_posterior`. Keep `compute_rank_score` as a separate trading preference layer.

- [ ] **Step 4: Verify tests pass**

Run: `python -m pytest test_bayesian_engine.py test_data_source.py -q`

Expected: PASS.

### Task 3: Wire real THS hot50 into API

**Files:**
- Modify: `bayesian_trader/data_source.py`
- Modify: `bayesian_trader/server.py`
- Test: `test_api.py`

- [ ] **Step 1: Write failing Flask tests**

Use Flask test client and a fake data source to assert `/api/hot_rank` and `/api/analysis` return `source`, `rank`, `buy_prob`, `evidence_detail`, and `features`.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest test_api.py -q`

Expected: FAIL because API does not return those fields.

- [ ] **Step 3: Implement wiring**

`create_data_source()` should prefer `ThsHotRankDataSource` when `TICKERLAB_API_KEY` exists. `api_hot_rank` and `api_analysis` should include `source`, real THS rank, and evidence fields. If no real key exists, status should show the fallback source name.

- [ ] **Step 4: Verify tests pass**

Run: `python -m pytest test_api.py test_data_source.py test_bayesian_engine.py -q`

Expected: PASS.

### Task 4: Verification

**Files:**
- Modify only if tests reveal defects.

- [ ] **Step 1: Run syntax checks**

Run: `python -m py_compile bayesian_trader/data_source.py bayesian_trader/server.py bayesian_trader/models.py bayesian_trader/bayesian_engine.py`

Expected: no output and exit code 0.

- [ ] **Step 2: Run all tests**

Run: `python -m pytest -q`

Expected: all tests pass without import-time network calls.

- [ ] **Step 3: Start Flask server**

Run: `Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-Command','cd /d d:\workbase\贝叶斯短线; python bayesian_trader\server.py'`

Expected: server starts on `http://localhost:5000`.
