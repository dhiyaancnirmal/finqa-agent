#!/usr/bin/env bash
# Reproduce every run in results/runs, then rebuild results/summary.md.
# Needs NVIDIA_API_KEY (free key from build.nvidia.com) in the environment or in --env-file.
# Runs resume: rerunning skips questions already in each results file.
set -euo pipefail
cd "$(dirname "$0")/.."

export FINQA_DATA_PATH="${FINQA_DATA_PATH:-data}"
if [ ! -f "$FINQA_DATA_PATH/benchmark_questions/finqa.csv" ]; then
  uvx --from huggingface_hub hf download snorkelai/finqa-data --repo-type dataset --local-dir "$FINQA_DATA_PATH"
fi

SUPER=nvidia/nemotron-3.5-super-vl-120b-a12b
ULTRA=nvidia/nemotron-3-ultra-550b-a55b
run() { uv run finqa-eval --model "$1" --agent "$2" --features "${4:-all}" --concurrency 4 --rpm 15 --out "results/runs/$3.jsonl"; }

run "$SUPER" baseline super-120b__baseline
run "$SUPER" harness  super-120b__harness
run "$ULTRA" baseline ultra-550b__baseline
run "$ULTRA" harness  ultra-550b__harness

uv run finqa-report
