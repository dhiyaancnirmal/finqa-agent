"""Run an agent over FinQA questions and write one JSON line per episode.

    finqa-eval --model nvidia/nemotron-3.5-lightning-30b-a3b --agent baseline --out results/runs/x.jsonl

Runs are resumable: question ids already in --out are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import HarnessConfig, run_baseline, run_harness
from .env import FinQAEnv
from .llm import OpenAIChat, Usage


def load_dotenv(path: str) -> None:
    """Read KEY=VALUE lines into os.environ without overwriting or printing anything."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.removeprefix("export ").strip(), v.strip().strip("'\""))


def done_ids(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--agent", choices=["baseline", "harness"], required=True)
    p.add_argument("--features", default="all", help="harness features: all, none, or e.g. schema,calc")
    p.add_argument("--data", default=os.environ.get("FINQA_DATA_PATH", "data"))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--ids", default=None, help="comma-separated question ids")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--env-file", default=os.path.expanduser("~/.config/rec9/.env"))
    a = p.parse_args(argv)

    load_dotenv(a.env_file)
    env = FinQAEnv(a.data)
    questions = env.questions
    if a.ids:
        wanted = set(a.ids.split(","))
        questions = [q for q in questions if q.id in wanted]
    if a.limit:
        questions = questions[: a.limit]
    skip = done_ids(a.out)
    todo = [q for q in questions if q.id not in skip]
    cfg = HarnessConfig.from_names(a.features)
    llm = OpenAIChat(a.model, max_tokens=a.max_tokens)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    lock = threading.Lock()
    print(f"{len(todo)} to run, {len(skip)} already done -> {a.out}", flush=True)

    def one(q):
        usage, ep, t0 = Usage(), env.episode(q), time.time()
        error = None
        try:
            if a.agent == "baseline":
                log = run_baseline(ep, llm, usage)
            else:
                log = run_harness(ep, llm, usage, cfg)
        except Exception as e:  # noqa: BLE001 - record and move on; a crashed episode scores 0
            log, error = None, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        row = {
            "id": q.id,
            "company": q.company,
            "question": q.user_query,
            "truth": q.answer,
            "model": a.model,
            "agent": a.agent,
            "features": a.features if a.agent == "harness" else None,
            "submitted": ep.submitted,
            "reward": ep.reward,
            "env_steps": ep.step_count,
            "llm_calls": usage.llm_calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "seconds": round(time.time() - t0, 1),
            "calc_calls": log.calc_calls if log else 0,
            "selfcheck_changed": log.selfcheck_changed if log else None,
            "stop": log.stop if log else "error",
            "error": error,
            "transcript": log.transcript if log else [],
        }
        with lock, open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    n = correct = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for fut in as_completed([pool.submit(one, q) for q in todo]):
            row = fut.result()
            n += 1
            correct += row["reward"] > 0
            print(
                f"[{n}/{len(todo)}] {row['id']} reward={row['reward']} submitted={row['submitted']!r} "
                f"truth={row['truth']!r} ({correct}/{n})",
                flush=True,
            )


if __name__ == "__main__":
    main()
