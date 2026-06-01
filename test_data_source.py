# -*- coding: utf-8 -*-

import pytest

from bayesian_trader.data_source import ThsAppHotRankDataSource, ThsHotRankDataSource


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
            {"rank": 3, "symbol": "sh.600410", "stock_name": "华胜天成", "hot_value": 8000},
        ]
    }
    session = FakeSession(payload)
    ds = ThsHotRankDataSource(api_key="k", session=session)

    entries = ds.fetch_hot_rank_entries(3)

    assert [e.rank for e in entries] == [1, 2, 3]
    assert [e.code for e in entries] == ["300033", "000001", "600410"]
    assert [e.name for e in entries] == ["同花顺", "平安银行", "华胜天成"]
    assert entries[0].heat_score == 9988
    assert session.requests[0][0] == "https://tickerlab.org/v1/ranking/hot-stock"
    assert session.requests[0][1]["params"]["limit"] == 3


def test_ths_hot_rank_requires_api_key_for_real_source():
    ds = ThsHotRankDataSource(api_key="")

    with pytest.raises(RuntimeError, match="TICKERLAB_API_KEY"):
        ds.fetch_hot_rank_entries(50)


def test_ths_hot_rank_uses_position_when_provider_repeats_rank():
    payload = {
        "data": [
            {"rank": 1, "symbol": "sh.600410", "stock_name": "华胜天成", "hot_value": 813359},
            {"rank": 1, "symbol": "sh.601991", "stock_name": "大唐发电", "hot_value": 128874},
            {"rank": 1, "symbol": "sz.002131", "stock_name": "利欧股份", "hot_value": 770445},
        ]
    }
    ds = ThsHotRankDataSource(api_key="k", session=FakeSession(payload))

    entries = ds.fetch_hot_rank_entries(3)

    assert [e.rank for e in entries] == [1, 2, 3]


def test_ths_app_hot_rank_uses_screenshot_endpoint_and_tags():
    payload = {
        "status_code": 0,
        "data": {
            "stock_list": [
                {
                    "code": "600863",
                    "name": "华能蒙电",
                    "tag": {
                        "concept_tag": ["超超临界发电", "煤炭概念"],
                        "popularity_tag": "6天3板",
                    },
                },
                {
                    "code": "601991",
                    "name": "大唐发电",
                    "tag": {
                        "concept_tag": ["绿色电力", "风电"],
                        "popularity_tag": "持续上榜",
                    },
                },
            ]
        },
    }
    session = FakeSession(payload)
    ds = ThsAppHotRankDataSource(session=session)

    entries = ds.fetch_hot_rank_entries(2)

    assert session.requests[0][0] == "https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock"
    assert session.requests[0][1]["params"] == {
        "stock_type": "a",
        "type": "hour",
        "list_type": "normal",
    }
    assert [e.rank for e in entries] == [1, 2]
    assert [e.code for e in entries] == ["600863", "601991"]
    assert entries[0].name == "华能蒙电"
    assert entries[0].concept_tags == ["超超临界发电", "煤炭概念"]
    assert entries[0].popularity_tag == "6天3板"
