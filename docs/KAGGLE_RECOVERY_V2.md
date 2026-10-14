# Kaggle recovery v2 results

Kernel: `bhabhihunter/qwen3-mole-bounded-recovery`, version 2  
Final state: `COMPLETE`  
Hardware: two Tesla T4 GPUs with 15,360 MiB each

## Outcome

The bounded real-data pipeline completed end to end. It built and split the
curated dataset, trained and evaluated the router, and trained four independent
rank-16 QLoRA experts in two concurrent GPU waves. Final PEFT adapter
configurations were verified for code, math, systems, and instruction.

This run is an integration and feasibility benchmark. Two optimizer steps per
expert are enough to validate the complete system but are not enough to claim a
domain-quality improvement over the untouched Qwen3-8B base model.

## Data and router

- 256 accepted records: 64 per domain.
- Split: 232 train, 12 validation, 12 untouched test.
- Router validation accuracy: 83.3%; fallback rate: 41.7%.
- Router test accuracy: 83.3%; fallback rate: 50.0%.
- The validation and test sets contain only three examples per domain, so these
  accuracy values are pipeline checks rather than stable statistical estimates.

The data sources and pinned revisions are recorded in the generated manifest.
Code, math, and systems data are Apache-2.0; instruction data is MIT.

## Expert results

| Expert | Usable train | Skipped at 512 tokens | Runtime | Step losses | Validation loss, step 1 -> 2 |
| --- | ---: | ---: | ---: | --- | --- |
| Code | 56 | 2 | 70.22 s | 1.536, 1.759 | 1.491 -> 1.429 |
| Math | 58 | 0 | 72.54 s | 1.176, 1.077 | 1.013 -> 0.932 |
| Systems | 51 | 7 | 31.87 s | 2.999, 3.010 | 3.194 -> 2.926 |
| Instruction | 53 | 5 | 72.63 s | 1.862, 1.654 | 2.076 -> 1.966 |

Each expert trained 43,646,976 parameters, 0.5301% of the 8,234,382,336
parameters visible through the quantized Qwen3-8B model.

The code/math GPU wave took 210.3 seconds including two model loads, training,
evaluation, checkpointing, final saves, and artifact copies. The cached
systems/instruction wave took 111.9 seconds. Summed pipeline stages took 403.5
seconds, or about 6 minutes 44 seconds.

Training loss is a different mini-batch at each step and is not expected to be
monotonic in a two-step run. Validation loss decreased for all four experts,
which confirms that evaluation and optimizer updates executed correctly; it is
not a sufficient quality result by itself.

## Artifact verification

All four final adapters report:

- base model `Qwen/Qwen3-8B`;
- PEFT type `LORA` and task type `CAUSAL_LM`;
- rank 16, alpha 32, dropout 0.05;
- `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj` targets;
- inference mode enabled.

The router model, router report, source manifest, step checkpoints, final
adapter weights, tokenizer assets, and run manifests are retained in the
private Kaggle output.

Machine-readable headline results are in
`benchmarks/kaggle_recovery_v2/summary.json`.

## Remaining evaluation gate

Before making a model-quality claim, run a larger held-out evaluation comparing
the base model, oracle expert, routed expert, and wrong-expert control. The
serving benchmark should compare FP16 tensor parallelism with one- and two-GPU
AWQ replica serving, measuring time to first token, decode throughput, memory,
and adapter-switch overhead.
