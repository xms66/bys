from newbys.llm_advisor import (
    LlmAdvisor,
    build_llm_payload,
    load_env_file,
    mask_config,
)
from newbys.models import MarketContext, StockSnapshot


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"summary":"ok","decisions":[]}'
                        }
                    }
                ]
            }
        )


def test_load_env_file_reads_freechat_config_without_exporting_secret(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "freechat_url=https://example.test/v1/chat/completions\n"
        "freechat_key=secret-key\n"
        "freechat_model=test-model\n",
        encoding="utf-8",
    )

    config = load_env_file(str(env_path))

    assert config["freechat_url"] == "https://example.test/v1/chat/completions"
    assert config["freechat_key"] == "secret-key"
    assert mask_config(config)["freechat_key"] == "***"


def test_build_llm_payload_uses_bayesian_system_prompt_and_top5_inputs():
    market = MarketContext(cycle_phase="markup", index_trend="up", sentiment_score=0.8)
    items = [
        {
            "stock": StockSnapshot(
                code=f"30000{i}",
                name=f"Stock {i}",
                price=10,
                change_pct=1,
                turnover_rate=5,
                volume_ratio=1.2,
                amount=100000000,
                hot_rank=i,
                concept_tags=["AI"],
            ).to_dict(),
            "posterior_profit": 0.6,
            "prior_profit": 0.55,
            "signal": "positive",
            "action": "watch_for_entry",
            "evidence": [],
        }
        for i in range(1, 7)
    ]

    payload = build_llm_payload(market.to_dict(), items)

    assert "贝叶斯短线分析" in payload["messages"][0]["content"]
    assert "open-to-open" in payload["messages"][0]["content"]
    assert "次日09:30买入" in payload["messages"][0]["content"]
    assert "300005" in payload["messages"][1]["content"]
    assert "300006" in payload["messages"][1]["content"]
    assert "只返回JSON" in payload["messages"][0]["content"]
    assert "hold_cash" in payload["messages"][0]["content"]


def test_build_llm_payload_can_limit_display_count_for_front_page():
    market = MarketContext(cycle_phase="markup", index_trend="up", sentiment_score=0.8)
    items = [
        {
            "stock": StockSnapshot(
                code=f"30000{i}",
                name=f"Stock {i}",
                price=10,
                change_pct=1,
                turnover_rate=5,
                volume_ratio=1.2,
                amount=100000000,
                hot_rank=i,
                concept_tags=[],
            ).to_dict(),
            "posterior_profit": 0.6,
            "prior_profit": 0.55,
            "signal": "positive",
            "action": "watch_for_entry",
            "evidence": [],
        }
        for i in range(1, 7)
    ]

    payload = build_llm_payload(market.to_dict(), items, max_items=5)

    assert "300005" in payload["messages"][1]["content"]
    assert "300006" not in payload["messages"][1]["content"]


def test_llm_advisor_posts_openai_compatible_request():
    session = FakeSession()
    advisor = LlmAdvisor(
        config={
            "freechat_url": "https://example.test",
            "freechat_key": "secret-key",
            "freechat_model": "test-model",
        },
        session=session,
    )

    result = advisor.analyze({"cycle_phase": "mixed"}, [])

    assert result["enabled"] is True
    assert result["content"] == '{"summary":"ok","decisions":[]}'
    url, kwargs = session.calls[0]
    assert url == "https://example.test/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer secret-key"
    assert kwargs["json"]["model"] == "test-model"
