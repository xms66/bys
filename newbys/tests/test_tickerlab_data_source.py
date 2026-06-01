from newbys.data_source import TickerLabHotRankDataSource


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload)


class FakeQuoteSource:
    def __init__(self, stocks):
        self.stocks = stocks
        self.codes = []

    def get_stocks_for_codes(self, codes):
        self.codes = codes
        return [self.stocks[code] for code in codes if code in self.stocks]


def test_tickerlab_hot_rank_parses_top_entries_and_preserves_rank():
    payload = {
        "data": {
            "items": [
                {"symbol": "SH.600001", "name": "Alpha", "rank": 2, "heat": 9000},
                {"code": "000002", "stock_name": "Beta", "ranking": 1, "score": 10000},
            ]
        }
    }
    session = FakeSession(payload)
    source = TickerLabHotRankDataSource(api_key="secret", session=session)

    entries = source.fetch_hot_rank_entries(top_n=50)

    assert [entry.code for entry in entries] == ["000002", "600001"]
    assert [entry.rank for entry in entries] == [1, 2]
    assert entries[0].name == "Beta"
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer secret"


def test_tickerlab_source_uses_quote_data_and_returns_ranked_snapshots():
    payload = [
        {"code": "300001", "name": "Hot One", "rank": 1, "heat_score": 99},
        {"code": "300002", "name": "Hot Two", "rank": 2, "heat_score": 90},
    ]
    from newbys.models import StockSnapshot

    quote_source = FakeQuoteSource(
        {
            "300001": StockSnapshot("300001", "Hot One", 10, 1, 5, 1.2, 100000000, 0, []),
            "300002": StockSnapshot("300002", "Hot Two", 20, 2, 6, 1.5, 200000000, 0, []),
        }
    )
    source = TickerLabHotRankDataSource(
        api_key="secret",
        session=FakeSession(payload),
        quote_source=quote_source,
    )

    stocks = source.get_stocks(top_n=50)

    assert quote_source.codes == ["300001", "300002"]
    assert [stock.hot_rank for stock in stocks] == [1, 2]
    assert [stock.code for stock in stocks] == ["300001", "300002"]
