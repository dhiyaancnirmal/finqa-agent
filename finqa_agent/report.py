"""Summarize results/runs/*.jsonl into results/summary.json and results/summary.md.

finqa-report [--runs results/runs] [--out results]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(path: str) -> dict:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    n = len(rows)
    k = sum(r["reward"] > 0 for r in rows)
    lo, hi = wilson(k, n)
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in rows]
    stops: dict[str, int] = {}
    for r in rows:
        stops[r["stop"]] = stops.get(r["stop"], 0) + 1
    changed = [r for r in rows if r.get("selfcheck_changed") is not None]
    return {
        "run": os.path.basename(path).removesuffix(".jsonl"),
        "model": rows[0]["model"] if rows else "",
        "agent": rows[0]["agent"] if rows else "",
        "features": rows[0].get("features") if rows else None,
        "n": n,
        "correct": k,
        "pass_at_1": k / n if n else 0.0,
        "ci95": [lo, hi],
        "mean_env_steps": statistics.mean(r["env_steps"] for r in rows) if rows else 0,
        "mean_llm_calls": statistics.mean(r["llm_calls"] for r in rows) if rows else 0,
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in rows) if rows else 0,
        "mean_completion_tokens": statistics.mean(r["completion_tokens"] for r in rows) if rows else 0,
        "mean_tokens": statistics.mean(tokens) if rows else 0,
        "median_seconds": statistics.median(r["seconds"] for r in rows) if rows else 0,
        "stops": stops,
        "errors": sum(r["error"] is not None for r in rows),
        "selfcheck_changed_answer": sum(bool(r["selfcheck_changed"]) for r in changed),
        "usd_per_task": 0.0,  # build.nvidia.com free tier
    }


def markdown(summaries: list[dict]) -> str:
    lines = [
        (
            "| Model | Agent | n | pass@1 | 95% CI | Env steps / task | Tokens / task | Median s / task "
            "| Hit 50-step cap | Cost / task |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        agent = s["agent"] if s["agent"] == "baseline" else f"harness ({s['features']})"
        lo, hi = s["ci95"]
        lines.append(
            f"| `{s['model'].removeprefix('nvidia/')}` | {agent} | {s['n']} | **{100 * s['pass_at_1']:.1f}%** "
            f"| {100 * lo:.1f}-{100 * hi:.1f}% | {s['mean_env_steps']:.1f} | {s['mean_tokens']:,.0f} "
            f"| {s['median_seconds']:.0f} | {s['stops'].get('max_steps', 0)} | $0 (free tier) |"
        )
    return "\n".join(lines) + "\n"


def _load(path: str) -> dict[str, dict]:
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def paired(runs_dir: str) -> list[dict]:
    """Per model, baseline vs harness on only the questions both runs have answered."""
    out = []
    for base_path in sorted(glob.glob(os.path.join(runs_dir, "*__baseline.jsonl"))):
        harness_path = base_path.replace("__baseline.jsonl", "__harness.jsonl")
        if not os.path.isfile(harness_path):
            continue
        b, h = _load(base_path), _load(harness_path)
        ids = sorted(set(b) & set(h))
        if not ids:
            continue

        def tok(rows: list[dict]) -> float:
            return statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows)

        out.append(
            {
                "model": b[ids[0]]["model"],
                "n": len(ids),
                "baseline_correct": sum(b[i]["reward"] > 0 for i in ids),
                "harness_correct": sum(h[i]["reward"] > 0 for i in ids),
                "harness_only": sum(h[i]["reward"] > b[i]["reward"] for i in ids),
                "baseline_only": sum(b[i]["reward"] > h[i]["reward"] for i in ids),
                "baseline_tokens": tok([b[i] for i in ids]),
                "harness_tokens": tok([h[i] for i in ids]),
            }
        )
    return out


def paired_markdown(rows: list[dict]) -> str:
    lines = [
        "Paired: the same questions, answered by both agents.",
        "",
        (
            "| Model | Questions | Baseline | Harness | Gain | Only harness right | Only baseline right "
            "| Tokens / task, baseline -> harness |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["n"]
        b, h = 100 * r["baseline_correct"] / n, 100 * r["harness_correct"] / n
        lines.append(
            f"| `{r['model'].removeprefix('nvidia/')}` | {n} | {r['baseline_correct']}/{n} ({b:.1f}%) "
            f"| {r['harness_correct']}/{n} ({h:.1f}%) | {h - b:+.1f} pts | {r['harness_only']} | "
            f"{r['baseline_only']} | {r['baseline_tokens']:,.0f} -> {r['harness_tokens']:,.0f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results")
    p.add_argument("--example", default=None, help="RUN:QUESTION_ID to render as results/example_episode.md")
    a = p.parse_args(argv)
    summaries = [summarize(path) for path in sorted(glob.glob(os.path.join(a.runs, "*.jsonl")))]
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    table = markdown(summaries) + "\n" + paired_markdown(paired(a.runs))
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(table)
    print(table)
    if a.example:
        run, qid = a.example.split(":")
        with open(os.path.join(a.runs, f"{run}.jsonl")) as f:
            row = next(r for r in map(json.loads, f) if r["id"] == qid)
        with open(os.path.join(a.out, "example_episode.md"), "w") as f:
            f.write(episode_markdown(row, run))


def episode_markdown(row: dict, run: str) -> str:
    """Render one episode's tool calls as a readable transcript."""
    out = [
        f"# Example episode: question {row['id']} ({run})",
        "",
        f"Question: {row['question']}",
        "",
        f"Ground truth: `{row['truth']}`. Submitted: `{row['submitted']}`. Reward: {row['reward']}.",
        "",
    ]
    for i, t in enumerate(row["transcript"], 1):
        args = {k: v for k, v in t["args"].items() if k != "company_name"}
        result = t["result"] if len(t["result"]) <= 600 else t["result"][:600] + " ..."
        out += [f"**{i}. `{t['tool']}`** `{json.dumps(args)}`", "", "```", result, "```", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
