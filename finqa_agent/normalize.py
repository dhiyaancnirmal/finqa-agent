"""Answer-format normalization applied by the harness before submit_answer.

The grader (vendor/rewards.py) parses one number, a "%" number, a fraction,
or a list of "label: value" parts. Models often add words, units or LaTeX
around a correct number ("approximately 6.12x", "$1,234 million",
"\\boxed{0.21}"), which the grader then cannot parse. This strips that
decoration without changing the value.
"""

from __future__ import annotations

import re

_NUM = r"[-−–]?\(?\$?\d[\d,]*(?:\.\d+)?\)?%?"
_WORDS = re.compile(
    r"\b(approximately|approx\.?|about|around|roughly|usd|dollars?|million|billion|thousand|"
    r"mn|bn|times|percent(?:age)?\s*points?|basis\s*points|bps)\b",
    re.IGNORECASE,
)


def _clean_number(token: str) -> str:
    t = token.replace("−", "-").replace("–", "-").replace("$", "")
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    t = t.replace("(", "").replace(")", "")
    return t.replace(",", "")


def normalize_answer(raw: str) -> str:
    """Return a grader-friendly form of the model's answer.

    Yes/no answers and strings with no number pass through stripped.
    Multi-value answers keep their "label: value" structure.
    """
    text = raw.strip()
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        text = ", ".join(boxed)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = text.replace("\\%", "%").replace("\\$", "$").replace("\\,", "")
    text = text.strip().strip("`*").strip()
    if re.fullmatch(r"(?i)(yes|no)\.?", text):
        return text.rstrip(".").capitalize()

    parts = [p.strip() for p in re.split(r"[;\n]|,\s+(?=[^,]*:)", text) if p.strip()]
    if len(parts) > 1 and all(":" in p for p in parts):
        out = []
        for p in parts:
            label, _, value = p.rpartition(":")
            out.append(f"{label.strip()}: {_single(value)}")
        return ", ".join(out)
    return _single(text)


def _single(text: str) -> str:
    text = _WORDS.sub(" ", text)
    text = re.sub(r"(?<=\d)\s*x\b", "", text)  # "6.1x" -> "6.1"
    nums = re.findall(_NUM, text)
    if len(nums) == 1:
        return _clean_number(nums[0])
    return text.strip().rstrip(".")
