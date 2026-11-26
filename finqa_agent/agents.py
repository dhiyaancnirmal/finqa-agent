"""Two agents over the same environment and model.

baseline: the reference loop and prompt from OpenEnv's examples/finqa_inference.py
          (Snorkel's published way to run this environment).
harness:  the same model with five harness changes, each switchable for ablations:
          schema    - list the company's tables up front and require get_table_info before SQL
          sql_lint  - warn when a year column like 2023 is used without quotes
          calc      - a calculator tool; all arithmetic goes through it
          normalize - strip words, units and LaTeX from the answer before submitting
          selfcheck - the first submit_answer is answered with a checklist; the model resubmits
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .calculator import calculate
from .env import Episode
from .llm import ChatClient, Usage
from .normalize import normalize_answer

# Tool descriptions are the docstrings the Space's MCP server exposes.
ENV_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_descriptions",
            "description": "Get a list of available table names for a company.\n\nArgs:\n    company_name: "
            "The name of the company\n\nReturns:\n    JSON list of table names",
            "parameters": {
                "type": "object",
                "properties": {"company_name": {"type": "string", "description": ""}},
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_info",
            "description": "Get table metadata: description, columns, types, unique values.\n\nArgs:\n    "
            "company_name: The name of the company\n    table_name: The name of the table\n\nReturns:\n"
            "    JSON string with table metadata",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": ""},
                    "table_name": {"type": "string", "description": ""},
                },
                "required": ["company_name", "table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Execute a SQL query on a table. Select * not allowed.\n\nFilters are required: "
            "WHERE, HAVING, IN, NOT IN, EXISTS, NOT EXISTS,\nANY, SOME, ALL, LIKE, NOT LIKE, BETWEEN, NOT "
            "BETWEEN, IS NULL,\nIS NOT NULL, CASE, FILTER.\n\nArgs:\n    company_name: The name of the "
            "company\n    table_name: The name of the table\n    query: SQL query to execute (must include "
            "filters)\n\nReturns:\n    JSON string with query results",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": ""},
                    "table_name": {"type": "string", "description": ""},
                    "query": {"type": "string", "description": ""},
                },
                "required": ["company_name", "table_name", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit a final answer for the question.\n\nArgs:\n    answer: The final answer to "
            "submit\n\nReturns:\n    Confirmation message",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": ""}},
                "required": ["answer"],
            },
        },
    },
]

CALC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression exactly, e.g. (61307 - 59972) / 59972. "
        "Supports + - * / ** ( ), abs, round, min, max, sqrt, log, exp. Thousands separators are allowed.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Arithmetic expression"}},
            "required": ["expression"],
        },
    },
}

# Verbatim from OpenEnv examples/finqa_inference.py.
BASELINE_SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about SEC 10-K filings.

Think and reason step by step. Iteratively gather data using the available tools until you have enough information to answer the question.

When submitting your final answer:
- Provide ONLY the numerical value. No explanations, units, or LaTeX formatting.
- Always express percentages, growth rates, and percentage point differences as decimal ratios by dividing by 100 (e.g., 22% → 0.22, -8.9% → -0.089, a 4.5 percentage point difference → 0.045).
- Submit numbers exactly as they appear in the query results. Do not convert units (e.g., if the table shows values in millions, submit the number as-is, not multiplied out).
- For multi-year answers, use: year: value, year: value (e.g., 2022: 0.933, 2023: 0.930, 2024: 0.931)
- For year-over-year changes, use: year to year: value (e.g., 2022 to 2023: 0.189, 2023 to 2024: 0.025)
- For single values, just submit the number (e.g., 0.895 or -77 or 63)
- If the question is yes/no, answer Yes or No
"""

HARNESS_WORKFLOW = """
Work in this order:
1. Pick the tables whose names match the question. Call get_table_info on each before querying it: it shows the column names and the exact row labels.
2. Query with sql_query. The SQL table name is the table_name itself. Wrap every column name in double quotes, especially year columns: "2023" is the column, 2023 is just the number. Use the exact row labels from get_table_info in WHERE clauses. SELECT * is not allowed and a WHERE clause is required.
3. Values come back as text such as "$61,307" or "(1,234)". Parentheses mean negative.
4. Do every calculation with the calculate tool, never in your head. Keep full precision; round only the final answer to about 3 significant decimals.
5. Submit with submit_answer. A ratio is a/b. A growth rate is (new - old) / old.
"""

SELFCHECK_PROMPT = """Before this answer is final, check it:
- Does it answer exactly what was asked (right line item, right year, right direction of the ratio or change)?
- Are percentages and growth rates written as decimal ratios (22% -> 0.22)?
- Are the values in the table's units, not multiplied out?
- Was every number computed with the calculate tool from values you actually queried?
If anything is wrong, fix it with more tool calls. Then call submit_answer again with the final answer only (it will be recorded as: {normalized})."""

SQL_HINT = (
    "\nHint: the SQL table name is the table_name. Wrap column names in double quotes. Call get_table_info "
    "for the exact column names and row labels. A WHERE clause is required and SELECT * is not allowed."
)

MAX_RESULT_CHARS = 8000


@dataclass
class HarnessConfig:
    schema: bool = True
    sql_lint: bool = True
    calc: bool = True
    normalize: bool = True
    selfcheck: bool = True

    @classmethod
    def from_names(cls, names: str) -> HarnessConfig:
        """'all', 'none', or a comma list of features to keep, e.g. 'schema,calc'."""
        keys = ["schema", "sql_lint", "calc", "normalize", "selfcheck"]
        if names == "all":
            return cls()
        on = set() if names == "none" else {n.strip() for n in names.split(",")}
        unknown = on - set(keys)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        return cls(**{k: k in on for k in keys})


@dataclass
class EpisodeLog:
    submitted: str | None = None
    reward: float = 0.0
    env_steps: int = 0
    calc_calls: int = 0
    selfcheck_changed: bool | None = None
    stop: str = ""
    transcript: list[dict[str, Any]] = field(default_factory=list)


def _parse_args(raw: str | None) -> dict[str, Any]:
    try:
        args = json.loads(raw or "{}")
        return args if isinstance(args, dict) else {}
    except json.JSONDecodeError:
        return {}


def _assistant_msg(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in tool_calls
        ],
    }


def run_baseline(ep: Episode, llm: ChatClient, usage: Usage, max_turns: int = 50) -> EpisodeLog:
    """OpenEnv's reference loop: tool_choice=required, first tool call per turn, no-tool reply = answer."""
    log = EpisodeLog()
    q = ep.question
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Company: {q.company}\nQuestion: {q.user_query}"},
    ]
    names = {t["function"]["name"] for t in ENV_TOOL_SCHEMAS}
    for _ in range(max_turns):
        if ep.done:
            break
        msg = llm.chat(messages, ENV_TOOL_SCHEMAS, usage, tool_choice="required")
        if not msg.tool_calls:
            name, args, call_id, raw = "submit_answer", {"answer": msg.content or "unknown"}, "none", None
        else:
            tc = msg.tool_calls[0]
            name, args, call_id, raw = (
                tc.function.name,
                _parse_args(tc.function.arguments),
                tc.id,
                tc.function.arguments,
            )
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args)},
                    }
                ],
            }
        )
        if name not in names:
            name, args = "submit_answer", {"answer": "unknown"}
        res = ep.step(name, args)
        log.transcript.append({"tool": name, "args": args, "result": res.result[:2000]})
        if not res.done:
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": str(res.result) or "No result"}
            )
    log.stop = "submitted" if ep.submitted is not None else ("max_steps" if ep.done else "max_turns")
    log.submitted, log.reward, log.env_steps = ep.submitted, ep.reward, ep.step_count
    return log


_BARE_NUMBER = re.compile(r"(?<![\w\"'.\[`])(\d{4})(?![\w\"'.\]`])")


def lint_sql(query: str, columns: list[str]) -> str | None:
    """Warn when a numeric column name (a year) appears unquoted, so SQL reads it as a number."""
    numeric_cols = {c for c in columns if c.isdigit()}
    if not numeric_cols:
        return None
    # Ignore anything inside quotes.
    stripped = re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`|\[[^\]]*\]", " ", query)
    bare = sorted({m for m in _BARE_NUMBER.findall(stripped) if m in numeric_cols})
    if not bare:
        return None
    cols = ", ".join(f'"{b}"' for b in bare)
    return (
        f"Warning: {', '.join(bare)} without double quotes is a number, not a column, so those result "
        f"values are just that number. Use {cols} to read the column."
    )


def _columns(ep: Episode, args: dict[str, Any]) -> list[str]:
    table = str(args.get("table_name", "")).replace(".json", "").replace(".txt", "")
    info = ep.tools.tables_cleaned.get(f"{args.get('company_name', '')}/{table}", {})
    return list(info.get("column_names", []))


def run_harness(
    ep: Episode, llm: ChatClient, usage: Usage, cfg: HarnessConfig | None = None, max_turns: int = 40
) -> EpisodeLog:
    cfg = cfg or HarnessConfig()
    log = EpisodeLog()
    q = ep.question
    tools = list(ENV_TOOL_SCHEMAS) + ([CALC_SCHEMA] if cfg.calc else [])
    system = BASELINE_SYSTEM_PROMPT + (HARNESS_WORKFLOW if cfg.schema else "")
    if not cfg.calc:
        system = system.replace("4. Do every calculation with the calculate tool, never in your head.", "4.")
    user = f"Company: {q.company}\nQuestion: {q.user_query}"
    if cfg.schema:
        res = ep.step("get_descriptions", {"company_name": q.company})
        log.transcript.append(
            {"tool": "get_descriptions", "args": {"company_name": q.company}, "result": res.result[:2000]}
        )
        user += f"\n\nTables available for {q.company} (from get_descriptions):\n{res.result}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    pending_check: str | None = None
    nudges = 0

    for _ in range(max_turns):
        if ep.done:
            break
        msg = llm.chat(messages, tools, usage, tool_choice="auto")
        calls = list(msg.tool_calls or [])
        if not calls:
            if nudges < 2:
                nudges += 1
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "Use the tools. When you have the answer, call "
                        "submit_answer with the number only.",
                    }
                )
                continue
            answer = normalize_answer(msg.content or "") if cfg.normalize else (msg.content or "")
            ep.step("submit_answer", {"answer": answer})
            log.transcript.append(
                {"tool": "submit_answer", "args": {"answer": answer}, "result": "(no tool call)"}
            )
            break
        messages.append(_assistant_msg(msg, calls))
        for tc in calls:
            name, args = tc.function.name, _parse_args(tc.function.arguments)
            if ep.done:
                result = "Episode already finished."
            elif name == "calculate" and cfg.calc:
                log.calc_calls += 1
                result = calculate(str(args.get("expression", "")))
            elif name == "submit_answer":
                raw = str(args.get("answer", ""))
                answer = normalize_answer(raw) if cfg.normalize else raw
                if cfg.selfcheck and pending_check is None:
                    pending_check = answer
                    result = SELFCHECK_PROMPT.format(normalized=answer)
                else:
                    if pending_check is not None:
                        log.selfcheck_changed = answer != pending_check
                    result = ep.step("submit_answer", {"answer": answer}).result
                    args = {"answer": answer, "raw": raw}
            else:
                result = ep.step(name, args).result
                if name == "sql_query":
                    if cfg.sql_lint:
                        warning = lint_sql(str(args.get("query", "")), _columns(ep, args))
                        if warning:
                            result = f"{warning}\n{result}"
                    if cfg.schema and result.startswith("Error"):
                        result += SQL_HINT
            if len(result) > MAX_RESULT_CHARS:
                result = result[:MAX_RESULT_CHARS] + "\n[truncated: add a narrower WHERE clause]"
            log.transcript.append({"tool": name, "args": args, "result": result[:2000]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result or "No result"})

    log.stop = "submitted" if ep.submitted is not None else ("max_steps" if ep.done else "max_turns")
    log.submitted, log.reward, log.env_steps = ep.submitted, ep.reward, ep.step_count
    return log
