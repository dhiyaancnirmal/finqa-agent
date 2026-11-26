"""Agent loops against the real environment code and a scripted model."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finqa_agent.agents import HarnessConfig, run_baseline, run_harness
from finqa_agent.env import FinQAEnv
from finqa_agent.llm import Usage

DATA = str(Path(__file__).parent / "fixtures" / "data")
TABLE = "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock"


def call(name, **args):
    return SimpleNamespace(id=f"c{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedLLM:
    """Replays a fixed list of assistant turns and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools, usage, **kw):
        usage.add(SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        self.seen.append({"messages": list(messages), "tools": [t["function"]["name"] for t in tools]})
        content, calls = self.turns.pop(0)
        return SimpleNamespace(content=content, tool_calls=calls)


@pytest.fixture
def env():
    return FinQAEnv(DATA)


def test_env_grades_with_the_snorkel_grader(env):
    ep = env.episode(env.questions[0])
    ep.step("submit_answer", {"answer": "6.118"})
    assert ep.done and ep.reward == 1.0


def test_env_stops_at_max_steps(env):
    ep = FinQAEnv(DATA, max_steps=2).episode(env.questions[0])
    ep.step("get_descriptions", {"company_name": "alphabet"})
    res = ep.step("get_descriptions", {"company_name": "alphabet"})
    assert res.done and res.reward == 0.0 and "Max steps" in res.result


def test_baseline_treats_a_text_reply_as_the_answer(env):
    llm = ScriptedLLM([(None, [call("get_descriptions", company_name="alphabet")]), ("6.118", None)])
    log = run_baseline(env.episode(env.questions[0]), llm, Usage())
    assert log.submitted == "6.118" and log.reward == 1.0 and log.env_steps == 2


def test_harness_full_path(env):
    q = '"2022"'
    llm = ScriptedLLM(
        [
            (None, [call("get_table_info", company_name="alphabet", table_name=TABLE)]),
            (
                None,
                [
                    call(
                        "sql_query",
                        company_name="alphabet",
                        table_name=TABLE,
                        query=f"SELECT operations, 2022 FROM {TABLE} WHERE operations != 'total'",
                    )
                ],
            ),
            (
                None,
                [
                    call(
                        "sql_query",
                        company_name="alphabet",
                        table_name=TABLE,
                        query=f"SELECT operations, {q} FROM {TABLE} WHERE operations != 'total'",
                    )
                ],
            ),
            (None, [call("calculate", expression="61307 / 10021")]),
            (None, [call("submit_answer", answer="about 6.118x")]),
            (None, [call("submit_answer", answer="6.118")]),
        ]
    )
    usage = Usage()
    log = run_harness(env.episode(env.questions[0]), llm, usage)
    tool_msgs = [m for m in llm.seen[-1]["messages"] if m["role"] == "tool"]

    # Schema discovery: the table list is in the first user message, from a real get_descriptions step.
    assert TABLE in llm.seen[0]["messages"][1]["content"]
    assert "calculate" in llm.seen[0]["tools"]
    # SQL lint caught the unquoted year; the quoted query returned the real value.
    assert tool_msgs[1]["content"].startswith("Warning:")
    assert "$61,307" in tool_msgs[2]["content"]
    # Self-check intercepted the first submit and showed the normalized answer.
    assert "check it" in tool_msgs[4]["content"] and "6.118" in tool_msgs[4]["content"]
    assert log.reward == 1.0 and log.submitted == "6.118" and log.selfcheck_changed is False
    # Env steps: get_descriptions, get_table_info, 2 x sql_query, submit (calculate is local).
    assert log.env_steps == 5 and log.calc_calls == 1


def test_harness_with_no_features_behaves_like_a_plain_loop(env):
    llm = ScriptedLLM([(None, [call("submit_answer", answer="about 6.118x")])])
    log = run_harness(env.episode(env.questions[0]), llm, Usage(), HarnessConfig.from_names("none"))
    assert "calculate" not in llm.seen[0]["tools"]
    assert log.submitted == "about 6.118x" and log.reward == 0.0 and log.env_steps == 1


def test_feature_names_are_validated():
    assert HarnessConfig.from_names("schema,calc") == HarnessConfig(True, False, True, False, False)
    with pytest.raises(ValueError):
        HarnessConfig.from_names("schema,typo")
