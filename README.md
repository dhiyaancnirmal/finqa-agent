# finqa-agent

An agent that answers financial questions over SEC 10-K tables with SQL and a calculator, measured on
[Snorkel's FinQA environment](https://huggingface.co/spaces/snorkelai/finqa-env) (290 expert-written
questions, binary reward), using free NVIDIA Nemotron models from [build.nvidia.com](https://build.nvidia.com).

The same model runs twice: once in the reference loop Snorkel and OpenEnv publish for this environment,
and once inside this repo's harness. The harness changes only the scaffolding around the model, with no
training and no extra data.

## Results

All 290 questions, one attempt each (pass@1), graded by Snorkel's own grader
(`finqa_agent/vendor/rewards.py`, unchanged). Numbers come from
[`results/summary.md`](results/summary.md), built from the per-question logs in [`results/runs/`](results/runs).

**Status, 2026-11-26: runs in progress.** The free API is rate-limiting these models, so not all 290
questions have been answered yet. The table below covers only the questions both agents have answered, so
the two columns score the same questions. It is regenerated as the runs fill in.

| Model | Questions so far | Baseline (OpenEnv reference loop) | Harness | Tokens / task, baseline -> harness |
|---|---|---|---|---|
| Nemotron 3.5 Super 120B-A12B | 79 of 290 | 54/79 (68.4%) | 60/79 (75.9%) | 37,443 -> 22,651 |
| Nemotron 3 Ultra 550B-A55B | 8 of 290 | 5/8 (62.5%) | 8/8 (100.0%) | 41,857 -> 15,701 |
| Nemotron 3.5 Lightning 30B-A3B | 6 of 290 | 0/6 (0.0%) | 5/6 (83.3%) | 251,656 -> 51,195 |

On Super's 79 questions the harness was right on 9 the baseline missed and wrong on 3 the baseline got:
+7.6 points, with 39% fewer tokens per task. Super's baseline loses mostly on turns that end with neither a
tool call nor text, which the reference loop submits as "unknown" (10 of its first 80 answers). It also
loses on loops to the 50-step cap; question 118 calls `get_descriptions` 50 times. The smaller and larger
models fall into a different trap: they write `SELECT 2023` for a year column and read the number 2023
back as data. That happened in 11 of Ultra's first 20 baseline episodes and 4 of Lightning's first 6.
The Ultra and Lightning rows are too small to conclude from yet. Per-run rows with 95% intervals are in
[`results/summary.md`](results/summary.md).

Cost per task is $0 because build.nvidia.com's free tier serves these models. Tokens per task is the cost
measure that carries over to a paid endpoint. Wall time includes waiting on the free tier's rate limit.

### Next to published scores

| System | FinQA pass@1 | Setup | Source |
|---|---|---|---|
| Nemotron 3.5 Super 120B-A12B, this harness (partial) | 75.9% | 79 of 290 questions so far, no training | this repo |
| Nemotron 3.5 Super 120B-A12B, OpenEnv reference loop (partial) | 68.4% | same 79 questions | this repo |
| Qwen3.5-9B, RL-trained with Crusoe Managed RL | 73.9% | GRPO on 165 other companies, scored on the 290 | [Crusoe, 2026-11-18](https://www.crusoe.ai/resources/blog/managed-rl-9b-model-beats-1t-finqa) |
| Kimi K2.6 (1T) | 66.9% | Crusoe's harness, 20 tool turns | same |
| rLLM-FinQA-4B, RL-trained | 59.7% | rLLM's harness with a calculator tool | [rLLM, 2026-02-18](https://rllm-project.com/posts/finqa.md) |
| Qwen3.5-9B, untrained | 53.1% | Crusoe's harness | Crusoe |
| Qwen3-235B-A22B | 51.4% | rLLM's harness | rLLM |
| Llama 3.3 Nemotron Super 49B v1.5 | 44% | Snorkel's leaderboard | [Snorkel, 2026-03-30](https://research.snorkel.ai/thoughts/building-finqa-an-open-rl-environment-for-financial-reasoning-agents/) |

These were run with different harnesses, step limits and model versions, so read them as context, not a
controlled comparison. Only the two rows from this repo share a setup.

## One real run

[`results/example_episode.md`](results/example_episode.md) is a full harness episode, copied from the
results file. Condensed:

```
Q: For company `alphabet`: What is the ratio of Domestic Income to Foreign Income for continuing
   operations before income taxes in 2022?

(harness)    get_descriptions -> 27 table names, put in the first message
1. get_table_info  us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock
                -> columns ["operations","2022","2023","2024"], rows domestic_operations, foreign_operations, total
2. sql_query    SELECT "operations", "2022" FROM ... WHERE "operations" IN ('domestic_operations','foreign_operations')
                -> [{"operations":"domestic_operations","2022":"$61,307"},{"operations":"foreign_operations","2022":"$10,021"}]
3. calculate    61307 / 10021  -> 6.1178525097
4. submit_answer 6.118  -> self-check checklist
5. submit_answer 6.118  -> reward 1.0 (truth \boxed{6.118})
```

## How it works

`finqa_agent/env.py` runs the environment in-process: the four tools (`get_descriptions`,
`get_table_info`, `sql_query`, `submit_answer`) and the grader are vendored from the Space, with the
Space's 50-step limit. No Docker server is needed and no question is shuffled.

**Baseline** (`run_baseline`) is OpenEnv's
[`examples/finqa_inference.py`](https://github.com/meta-pytorch/OpenEnv/blob/main/examples/finqa_inference.py):
its system prompt verbatim, `tool_choice="required"`, the first tool call of each turn, and a text reply
taken as the answer.

**Harness** (`run_harness`) keeps that prompt and adds five changes. Each can be turned off with
`--features`:

1. **Schema first.** The harness calls `get_descriptions` itself and puts the company's table list in the
   first message. The prompt tells the model to read `get_table_info` before writing SQL, and failed SQL comes
   back with a hint (quote columns, a WHERE clause is required, no `SELECT *`).
2. **Year-column lint.** Most tables have columns named `2022`, `2023`. In SQLite, `SELECT 2023` returns
   the number 2023, not the column, and the query still succeeds. The model then reads 2023 as the value.
   The harness spots an unquoted year that matches a column and prepends a warning to the result.
3. **Calculator.** A `calculate` tool evaluates arithmetic through Python's AST (no `eval`). The prompt
   requires it for every computation.
4. **Answer normalization.** Before submitting, words, units, `x` suffixes and LaTeX are stripped
   ("approximately 6.12x" becomes "6.12"). The grader would otherwise score a correct number as wrong.
5. **Self-check.** The first `submit_answer` is not sent. The model gets a short checklist instead (right
   line item and year, ratios as decimals, units as in the table, every number from the calculator) and
   submits again.

The harness also re-prompts when a turn ends with neither a tool call nor text. The reference loop
submits "unknown" in that case.

### How the harness was developed

The dataset has no train or dev split, so I built the harness on a fixed slice of the 290: every 7th
question, 42 in all. I froze it at tag `harness-v1` (33/42 on Nemotron 3.5 Super) before running the
full set. Most dev-slice misses were label questions, not agent errors: a growth rate labeled
0.01585 where the data gives 1.585, a decline labeled as a positive change, and a yes/no question labeled
1416. I left those alone instead of tuning the prompt to them.

## Reproduce

```bash
uv sync
echo "NVIDIA_API_KEY=nvapi-..." > .env          # free key from build.nvidia.com; .env is gitignored
uvx --from huggingface_hub hf download snorkelai/finqa-data --repo-type dataset --local-dir data
uv run finqa-eval --model nvidia/nemotron-3.5-super-vl-120b-a12b --agent harness \
    --out results/runs/super-120b__harness.jsonl --concurrency 4 --rpm 15
uv run finqa-report
```

`scripts/reproduce.sh` runs every row. Runs resume from their results file. Episodes the API never served
(rate limits, outages) are not written, so a rerun retries them and they are never scored as wrong.

Tests (`uv run pytest`) use a one-table fixture and a scripted model, so they need no key or network. CI
runs ruff and pytest on every push.

## Limits

- One attempt per question. Nemotron's answers vary between runs even at temperature 0, so differences of a
  few points are within noise. The 95% intervals in the table show how much.
- The free API's per-model rate limits set the pace: on 2026-11-26 Super sustained about 7 requests a
  minute, and the Lightning endpoint was down for part of the day ("DEGRADED function cannot be invoked").
  Every run resumes from its results file with the same command.
- Snorkel's blog says the questions cover 22 companies. The downloaded data, and its own README, have 20.

## Credits

Environment, data and grader: Snorkel AI ([finqa-env](https://huggingface.co/spaces/snorkelai/finqa-env),
[finqa-data](https://huggingface.co/datasets/snorkelai/finqa-data), Apache-2.0), part of
[OpenEnv](https://github.com/meta-pytorch/OpenEnv) (BSD-3-Clause, see `finqa_agent/vendor/`).
