#!/usr/bin/env bash
set -euo pipefail

adapter_root="${1:-artifacts/adapters}"

exec vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules \
    "code=${adapter_root}/code/final" \
    "math=${adapter_root}/math/final" \
    "systems=${adapter_root}/systems/final" \
    "instruction=${adapter_root}/instruction/final"
