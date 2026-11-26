import json

from finqa_agent.report import summarize, wilson


def test_summary_counts_pass_rate_tokens_and_stops(tmp_path):
    rows = [
        {"reward": 1.0, "prompt_tokens": 100, "completion_tokens": 20, "stop": "submitted"},
        {"reward": 0.0, "prompt_tokens": 300, "completion_tokens": 60, "stop": "max_steps"},
    ]
    base = {
        "model": "m",
        "agent": "baseline",
        "env_steps": 4,
        "llm_calls": 5,
        "seconds": 2.0,
        "error": None,
        "selfcheck_changed": None,
    }
    path = tmp_path / "m__baseline.jsonl"
    path.write_text("".join(json.dumps({**base, **r}) + "\n" for r in rows))
    s = summarize(str(path))
    assert (s["n"], s["correct"], s["pass_at_1"]) == (2, 1, 0.5)
    assert s["mean_tokens"] == 240
    assert s["stops"] == {"submitted": 1, "max_steps": 1}


def test_wilson_interval_matches_known_value():
    # 50/100 -> 95% Wilson interval 0.404 to 0.596.
    lo, hi = wilson(50, 100)
    assert round(lo, 3) == 0.404 and round(hi, 3) == 0.596
