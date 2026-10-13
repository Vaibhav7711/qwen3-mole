#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"

"$python_bin" scripts/write_smoke_data.py data/smoke.jsonl
"$python_bin" -m qwen3_mole.prepare_data \
  --input data/smoke.jsonl \
  --validation-fraction 0.25
"$python_bin" -m qwen3_mole.train_router
printf '%s\n' \
  '{"prompt":"Write a Python function that parses JSON."}' \
  '{"prompt":"Prove that the square root of two is irrational."}' \
  '{"prompt":"Diagnose high disk latency on a Linux server."}' \
  '{"prompt":"Rewrite this paragraph in a concise professional tone."}' \
  | "$python_bin" -m qwen3_mole.route_prompts
