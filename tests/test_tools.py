import pytest

from finqa_agent.agents import lint_sql
from finqa_agent.calculator import calculate
from finqa_agent.normalize import normalize_answer
from finqa_agent.vendor.rewards import compute_reward


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("61307 / 10021", "6.1178525097"),
        ("(73,600 - 61,307) / 61,307", "0.2005154387"),
        ("12% * 200", "24.0"),
        ("2^10", "1024"),
        ("round(1/3, 3)", "0.333"),
    ],
)
def test_calculate(expr, expected):
    assert calculate(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "1/0", "9**9**9", "a + 1"])
def test_calculate_rejects_unsafe_or_bad_input(expr):
    assert calculate(expr).startswith("Error")


@pytest.mark.parametrize(
    ("raw", "truth"),
    [
        ("approximately 6.118x", r"\boxed{6.118}"),
        (r"\boxed{0.209}", r"\boxed{0.209}"),
        ("$1,234 million", r"\boxed{1234}"),
        ("(77)", r"\boxed{-77}"),
        ("−0.089", r"\boxed{-0.089}"),
        ("yes.", r"\boxed{Yes}"),
        ("2022: 0.933; 2023: 0.930", r"\boxed{2022: 0.933, 2023: 0.930}"),
    ],
)
def test_normalized_answers_pass_the_grader(raw, truth):
    assert compute_reward(normalize_answer(raw), truth) == 1.0


def test_normalization_is_needed_for_decorated_answers():
    # The grader alone rejects these, which is why the harness normalizes.
    assert compute_reward("approximately 6.118x", r"\boxed{6.118}") == 0.0
    assert compute_reward("$1,234 million", r"\boxed{1234}") == 0.0


def test_normalize_leaves_a_plain_number_alone():
    assert normalize_answer("0.757") == "0.757"


def test_lint_sql_flags_unquoted_year_columns():
    cols = ["operations", "2022", "2023"]
    warning = lint_sql("SELECT operations, 2022 FROM t WHERE operations = 'total'", cols)
    assert warning is not None and '"2022"' in warning
    assert lint_sql("SELECT \"2022\" FROM t WHERE operations = 'total'", cols) is None
    assert lint_sql("SELECT x FROM t WHERE label = '2022'", cols) is None
    assert lint_sql("SELECT 2022 FROM t WHERE a = 1", ["a", "b"]) is None
