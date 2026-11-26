"""In-process FinQA environment.

Mirrors snorkelai/finqa-env's FinQAEnvironment.step(): same four tools, same
grader, same 50-step cap. It runs in-process so an eval needs no Docker server,
and it iterates questions in a fixed order instead of shuffling.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any

from .vendor.rewards import compute_reward
from .vendor.tools import FinQATools

ENV_TOOLS = ["get_descriptions", "get_table_info", "sql_query", "submit_answer"]
DEFAULT_MAX_STEPS = 50  # FINQA_MAX_STEPS default in the Space


@dataclass
class Question:
    id: str
    company: str
    user_query: str
    answer: str


def load_questions(data_path: str) -> list[Question]:
    path = os.path.join(data_path, "benchmark_questions", "finqa.csv")
    with open(path, newline="") as f:
        return [Question(r["id"], r["company"], r["user_query"], r["answer"]) for r in csv.DictReader(f)]


@dataclass
class StepResult:
    result: str
    done: bool
    reward: float


@dataclass
class Episode:
    """One question. Call step() with env tools until done."""

    tools: FinQATools
    question: Question
    max_steps: int = DEFAULT_MAX_STEPS
    step_count: int = 0
    done: bool = False
    reward: float = 0.0
    submitted: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def step(self, tool_name: str, args: dict[str, Any]) -> StepResult:
        if self.done:
            raise RuntimeError("episode is over")
        self.step_count += 1
        if tool_name not in ENV_TOOLS:
            result = f"Error: Unknown tool '{tool_name}'"
        else:
            try:
                result, _ = self.tools.execute_tool(tool_name, args)
            except TypeError as e:  # wrong or missing arguments
                result = f"Error: {e}"
        self.calls.append({"tool": tool_name, "args": args})
        if tool_name == "submit_answer":
            self.submitted = str(args.get("answer", ""))
            self.reward = compute_reward(self.submitted, self.question.answer)
            self.done = True
        elif self.step_count >= self.max_steps:
            self.done = True
            result = f"{result}\nMax steps ({self.max_steps}) reached without submitting answer."
        return StepResult(result, self.done, self.reward)


class FinQAEnv:
    def __init__(self, data_path: str, max_steps: int = DEFAULT_MAX_STEPS):
        self.tools = FinQATools(data_path)
        self.questions = load_questions(data_path)
        self.max_steps = max_steps

    def episode(self, question: Question) -> Episode:
        return Episode(self.tools, question, self.max_steps)
