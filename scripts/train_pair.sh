#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 DOMAIN_GPU0 DOMAIN_GPU1 [SMOKE_STEPS]" >&2
  exit 2
fi

domain0="$1"
domain1="$2"
python_bin="${PYTHON_BIN:-python3}"
extra_args=()
if [[ $# -eq 3 ]]; then
  extra_args=(--smoke-steps "$3")
fi

CUDA_VISIBLE_DEVICES=0 "$python_bin" -m qwen3_mole.train_expert --domain "$domain0" "${extra_args[@]}" &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$python_bin" -m qwen3_mole.train_expert --domain "$domain1" "${extra_args[@]}" &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
exit "$status"
