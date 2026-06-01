import json

from scripts.daily_job import _read_json_argument


def test_read_json_argument_prefers_file(tmp_path):
    path = tmp_path / "decision.json"
    path.write_text(json.dumps({"decision": "hold_cash"}), encoding="utf-8")

    result = _read_json_argument("", str(path), "decision")

    assert json.loads(result)["decision"] == "hold_cash"
