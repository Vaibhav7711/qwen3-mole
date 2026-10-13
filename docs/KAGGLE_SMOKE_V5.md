# Kaggle dual-T4 smoke result

- Date: 2026-10-13
- Kernel: `vaibhav0383/qwen3-mole-gpu-smoke`, version 5
- Status: **complete**

## Hardware

- GPU 0: NVIDIA Tesla T4, 15,360 MiB
- GPU 1: NVIDIA Tesla T4, 15,360 MiB
- NVIDIA driver: 580.178.04

## Verified path

1. Installed the Kaggle requirements and editable project.
2. Passed all 10 CPU/control-path tests.
3. Prepared a balanced four-domain smoke split.
4. Trained and persisted the prompt router.
5. Loaded two independent 4-bit Qwen3-8B backbones concurrently.
6. Attached one rank-16 LoRA expert to each backbone.
7. Completed three optimizer steps for `code` and `math` simultaneously.
8. Saved both final adapters and tokenizers to Kaggle output.

Each worker reported 43,646,976 trainable parameters out of 8,234,382,336
total parameters (0.5301%).

## Timing and diagnostic metrics

| Measurement | Code/math concurrent run |
|---|---:|
| Full dual-GPU stage, including download/load/save | 181.0 s |
| Expert training runtime | 12.82-12.92 s |
| Optimizer steps per expert | 3 |
| Reported optimizer steps/s | 0.232-0.234 |
| Reported samples/s | 3.715-3.745 |
| Smoke train loss | 4.977 and 6.301 |

The losses are plumbing diagnostics from four synthetic training records per
domain, not quality results. No model-quality conclusion should be drawn from
them. The important result is that the full quantized load, backward,
optimizer, concurrent-GPU, and adapter-save path works on Kaggle's T4 x2
runtime.

## Issues found by the smoke gate

- Kaggle expands uploaded dataset archives and mounts them beneath a nested
  `/kaggle/input/datasets/...` path. The kernel now discovers the project root.
- Transformers 5 returns chat-template tokens as a `BatchEncoding`; masking now
  normalizes both mapping and raw-list outputs.
- The Kaggle image's existing Transformers API rejected `warmup_ratio`.
  Requirements now pin 5.2.0 and training passes a computed `warmup_steps`.

All three issues were fixed and covered by local tests before the successful
version 5 run.
